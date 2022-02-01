# codes from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1

import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import math

import torch
from torch.autograd import Variable
import torch.nn as nn

from control.config import args
from builder.data.data_preprocess import get_data_preprocessed
from builder.models import get_detector_model
from builder.utils.metrics import Evaluator
from builder.utils.logger import Logger
from builder.trainer.trainer import *
from builder.utils.utils import set_seeds, set_devices
from builder.utils.binary_performance_estimator import binary_detector_evaluator

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
label_method_max = True
scheduler = None
optimizer = None
criterion = nn.CrossEntropyLoss(reduction='none')



def calc_hf(ref, hyp):

    ## collect start and stop times from input arg events
    #
    start_r_a = ref[0]
    stop_r_a = ref[1]
    start_h_a = hyp[0]
    stop_h_a = hyp[1]

    # initialize local variables
    #
    ref_dur = stop_r_a - start_r_a
    hyp_dur = stop_h_a - start_h_a
    hit = float(0)
    fa = float(0)

    # ----------------------------------------------------------------------
    # deal explicitly with the four types of overlaps that can occur
    # ----------------------------------------------------------------------

    # (1) for pre-prediction event
    #     ref:         <--------------------->
    #     hyp:   <---------------->
    #
    if start_h_a <= start_r_a and stop_h_a <= stop_r_a:
        hit = (stop_h_a - start_r_a) / ref_dur
        if ((start_r_a - start_h_a) / ref_dur) < 1.0:
            fa = ((start_r_a - start_h_a) / ref_dur)
        else:
            fa = float(1)

    # (2) for post-prediction event
    #     ref:         <--------------------->
    #     hyp:                  <-------------------->
    #
    elif start_h_a >= start_r_a and stop_h_a >= stop_r_a:

        hit = (stop_r_a - start_h_a) / ref_dur
        if ((stop_h_a - stop_r_a) / ref_dur) < 1.0:
            fa = ((stop_h_a - stop_r_a) / ref_dur)
        else:
            fa = float(1)


    # (3) for over-prediction event
    #     ref:              <------->
    #     hyp:        <------------------->
    #
    elif start_h_a < start_r_a and stop_h_a > stop_r_a:

        hit = 1.0
        fa = ((stop_h_a - stop_r_a) + (start_r_a - start_h_a)) / \
                ref_dur
        if fa > 1.0:
            fa = float(1)

    # (4) for under-prediction event
    #     ref:        <--------------------->
    #     hyp:            <------>
    #
    else:
        hit = (stop_h_a - start_h_a) / ref_dur

    # exit gracefully
    #
    return (hit, fa)

def anyovlp(ref, hyp):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1
    # create set for the ref/hyp events
    #
    refset = set(range(int(ref[0]), int(ref[1]) + 1))
    hypset = set(range(int(hyp[0]), int(hyp[1]) + 1))

    if len(refset.intersection(hypset)) != 0:
        return True

    # return gracefully
    #
    return False

def ovlp_hyp_seqs(ref, hyp, rind, hind,
                    refflag, hypflag):

    # define variables
    #
    p_miss = float(0)

    # calculate the parameters for the current event
    #
    p_hit, p_fa = calc_hf(ref[rind], hyp[hind])
    p_miss += float(1) - p_hit

    # update flags for already detected events
    #
    refflag[rind] = False
    hypflag[hind] = False

    # update the index since there could be multiple hyp events
    # overlapped with hyp event
    #
    #  <----------------------->
    #   <---->  <-->   <-->
    #
    hind += 1

    # look for hyp events overlapping with hyp event
    #
    for i in range(hind, len(hyp)):

        # update HMF according to the TAES score definition
        #
        if anyovlp(ref[rind], hyp[i]):
            # update the flags for processed events
            #
            hypflag[i] = False

            ovlp_hit, ovlp_fa \
                = calc_hf(ref[rind], hyp[i])

            p_hit += ovlp_hit
            p_miss -= ovlp_hit
            p_fa += ovlp_fa

        # move to the next event index
        #
        i += 1

    # return gracefully
    #
    return p_hit, p_miss, p_fa

def ovlp_ref_seqs(ref, hyp, rind, hind,
                    refflag, hypflag):

    # define variables
    #
    p_miss = float(0)

    # calculate the parameters for the current event
    #
    p_hit, p_fa = calc_hf(ref[rind], hyp[hind])
    p_miss += float(1) - p_hit

    # update flags for already detected events
    #
    hypflag[hind] = False
    refflag[rind] = False

    # update the index since there could be multiple ref events
    # overlapped with hyp event
    #
    #  <-->    <-->  <-->
    # <--------------------->
    #
    rind += 1

    # look for more ref events overlapping with hyp event
    #
    for i in range(rind, len(ref)):

        # update misses according to the TAES score definition
        #
        if anyovlp(ref[i], hyp[hind]):
            # update the flags for processed events
            #
            refflag[i] = False
            p_miss += 1

        # move to next event index
        #
        i += 1

    # exit gracefully
    #
    return p_hit, p_miss, p_fa

def compute_partial(ref, hyp, rind, hind, rflags, hflags):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1
    # check whether current reference event has any overlap
    # with the hyp event
    if not anyovlp(ref[rind], hyp[hind]):
        return (float(0), float(0), float(0))

    # check whether detected event stop time exceed the
    # reference stop time.
    #
    elif float(hyp[hind][1]) >= float(ref[rind][1]):

        #   <---->
        #     <---->
        #
        #   <---->
        # <-------->
        #
        # check whether multiple reference events are
        # overlapped with hypothesis event
        #
        #  <-->    <-->  <-->
        # <--------------------->
        #
        p_hit, p_mis, p_fal = ovlp_ref_seqs(ref, hyp, rind, hind, rflags, hflags)

    # check whether reference event stop time exceed the
    # detected stop time.
    #
    elif float(ref[rind][1]) > float(hyp[hind][1]):

        #   <------>
        # <----->
        #
        #   <------>
        #     <-->
        #
        # check whether multiple hypothesis events are
        # overlapped with reference event
        #
        #  <----------------------->
        #   <---->  <-->   <-->
        #
        p_hit, p_mis, p_fal \
            = ovlp_hyp_seqs(ref, hyp, rind, hind,
                                    rflags, hflags)

    # return gracefully
    #
    return (p_hit, p_mis, p_fal)


def taes_get_events(start, stop, events_a, hflags):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1
    # declare output variables
    #
    labels = []
    starts = []
    stops = []
    flags = []
    ind = []

    # loop over all events
    #
    for i in range(len(events_a)):

        # if the event overlaps partially with the interval,
        # it is a match. this means:
        #              start               stop
        #   |------------|<---------------->|-------------|
        #          |---------- event -----|
        #
        if (events_a[i][1] > start) and (events_a[i][0] < stop):
            starts.append(events_a[i][0])
            stops.append(events_a[i][1])
            labels.append(1)
            ind.append(i)
            flags.append(hflags[i])

    # exit gracefully
    #
    return [labels, starts, stops]


def ovlp_get_events(start, stop, events):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1
    # declare output variables
    
    labels = []
    starts = []
    stops = []

    # loop over all events
    #
    for event in events:

        # if the event overlaps partially with the interval,
        # it is a match. this means:
        #              start               stop
        #   |------------|<---------------->|-------------|
        #          |---------- event -----|
        #
        if (event[1] > start) and (event[0] < stop):
            starts.append(event[0])
            stops.append(event[1])
            labels.append(1)    # since only seizure or not

    # exit gracefully
    #
    return [labels, starts, stops]

def ovlp_get_events_with_latency(start, stop, events):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1
    # declare output variables
    
    labels = []
    starts = []
    stops = []
    latencies = []
    not_detected = 0

    # loop over all events
    #
    for event in events:

        # if the event overlaps partially with the interval,
        # it is a match. this means:
        #              start               stop
        #  ref |------------|<---------------->|-------------|
        #  hyp        |---------- event -----|
        #
        if (event[1] > start) and (event[0] < stop):
            starts.append(event[0])
            stops.append(event[1])
            labels.append(1)    # since only seizure or not

        if (event[0] >= start-2) and (event[0] <= stop+5):
            delayed_time = start - event[0]
            if delayed_time < 0:
                delayed_time = 0
            latencies.append(delayed_time)
        
        if (event[0] < start) and (event[1] > start):
            latencies.append(0)

    # exit gracefully
    #
    if len(latencies) == 0:
        latencies.append(stop-start)
        not_detected = 1
    else:
        not_detected = -1
    return [labels, starts, stops, min(latencies), not_detected]


def taes(ref_events, hyp_events):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1
    hit = 0
    mis = 0
    fal = 0
    refo = 0
    hypo = 0
    i = 0
    j = 0
    hflags = []
    rflags = []
    for i in range(len(hyp_events)):
        hflags.append(True)
    for i in range(len(ref_events)):
        rflags.append(True)

    for i, event in enumerate(ref_events):
        refo += 1
        labels, starts, stops = taes_get_events(event[0], event[1], hyp_events, hflags)
        # one event at a time, don't bother if ref/hyp labels don't overlap
        #
        if rflags[i]:

            # loop through all hyp events and calculate partial HMF
            #
            for j in range(len(hyp_events)):

                # compare hyp and ref event labels and hyp flags status;
                #
                if hflags[j]:
                    # print("ref_events: ", ref_events)
                    # print("hyp_events: ", hyp_events)
                    # print("i: ", i)
                    # print("j: ", j)
                    # print("rflags: ", rflags)
                    # print("hflags: ", hflags)
                    p_hit, p_miss, p_fa = compute_partial(ref_events, hyp_events, i, j, rflags, hflags)
                    # print("p_hit: ", p_hit)
                    # print("p_miss: ", p_miss)
                    # print("p_fa: ", p_fa)

                    # updat the HMF confusion matrix
                    #
                    hit += p_hit
                    mis += p_miss
                    fal += p_fa

                # update the hyp event index
                #
                j += 1

        # updated the ref event index
        #
        i += 1

    return hit, mis, fal, i, j

def ovlp(ref_events, hyp_events):
    # from "Objective evaluation metrics for automatic classification of EEG events " by Saeedeh Ziyabari1, Vinit Shah1, Meysam Golmohammadi2, Iyad Obeid1 and Joseph Picone1    
    hit = 0
    mis = 0
    fal = 0
    refo = 0
    hypo = 0
    latency_time = 0
    refo_minus_count = 0
    latency_time_of_detected = 0

    for event in ref_events:
        refo += 1
        labels, starts, stops, delayed_time, not_detected = ovlp_get_events_with_latency(event[0], event[1], hyp_events)
        if 1 in labels:
            hit += 1
        else:
            mis += 1
        latency_time += delayed_time
        if not_detected == 1:
            refo_minus_count += 1
        else:
            latency_time_of_detected += delayed_time

    for event in hyp_events:
        hypo += 1
        labels, starts, stops = ovlp_get_events(event[0], event[1], ref_events)
        if 1 not in labels:
            fal += 1
    
    return hit, mis, fal, refo, hypo, latency_time, latency_time_of_detected, refo_minus_count


ovlp_tprs_seeds = []
ovlp_tnrs_seeds = []
ovlp_fas24_seeds = []

taes_tprs_seeds = []
taes_tnrs_seeds = []
taes_fas24_seeds = []

latencies_seeds = []
detected_latencies_seeds = []
missed_for_latency_seeds = []
refos_for_latency_seeds = []

margin_3sec_rise_seeds = []
margin_3sec_fall_seeds = []
margin_5sec_rise_seeds = []
margin_5sec_fall_seeds = []

for seed_num in args.seed_list:
    args.seed = seed_num
    iteration = 1
    set_seeds(args)
    device = set_devices(args)
    logger = Logger(args)
    logger.loss = 0
    print("Project name is: ", args.project_name)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get Dataloader, Model
    train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)
    model = get_detector_model(args) 
    model = model(args, device).to(device)
    evaluator = Evaluator(args)
    name = args.project_name
    if args.last:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best_{}.pth'.format(str(args.seed))
    elif args.best:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best_{}.pth'.format(str(args.seed))
    if not os.path.exists(ckpt_path):
        exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k: v for k, v in ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()
    print('loaded model')
    print("Test type is: ", args.test_type)
    evaluator.reset()
    result_list = []
    evaluator.seizure_wise_eval_for_binary = True




    hyps = []
    hyps_list = []
    refs = []
    count = -1
    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            count += 1
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = test_batch
            test_x = test_x.to(device)
            
            iter_num = math.ceil(test_x.shape[1] / 6000)
            signal_len = test_x.shape[1]
            label_len = test_y.shape[1]
            for iter_idx in range(iter_num):
                sig_start = iter_idx * 6000
                lable_start = iter_idx * 1500

                if 6000 < (signal_len-sig_start):
                    sig_end = sig_start + 6000
                    label_end = lable_start + 1500
                else:
                    sig_end = signal_len
                    label_end = label_len                
                    if sig_end - sig_start < 400:
                        continue
                
                test_x_sliced = test_x[:, sig_start:sig_end, :] 
                test_y_sliced = test_y[:, lable_start:label_end]
                seq_lengths = [sig_end-sig_start]
                target_lengths = [label_end-lable_start]
                
                model, _ = sliding_window_v1(args, iteration, test_x_sliced, test_y_sliced, seq_lengths, 
                                            target_lengths, model, logger, device, scheduler,
                                            optimizer, criterion, signal_name_list=signal_name_list, flow_type="test")    # margin_test , test

            hyps.append(torch.stack(logger.pred_results).numpy()[:,1])
            refs.append(logger.ans_results)

            logger.pred_results = []
            logger.ans_results = []

            # print("count: ", count)
            # if count == 15:
            #     break

        logger.test_result_only()


    hyps_list = [list(hyp) for hyp in hyps]
    # refs_list = [list(ref) for ref in refs]
    # print("refs_list: ", len(refs_list))
    # print("hyps_list: ", len(hyps_list))
    # for i in range(len(refs_list)):
    #     if len(refs_list[i]) != len(hyps_list[i]):
    #         print(len(refs_list[i]))
    #         print(len(hyps_list[i]))
    #         print("error!!!")
    #     else:
    #         print("correct!!")
    # exit(1)
    print("##### margin test evaluation #####")
    target_stack = torch.tensor([item for sublist in refs for item in sublist])
    thresholds_margintest = list(logger.evaluator.thresholds_margintest)
    print("thresholds_margintest: ", thresholds_margintest)
    margin_threshold = 0
    for margin in args.margin_list:
        for threshold_idx, threshold in enumerate(thresholds_margintest):
            hyp_output = list([[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list])
            pred_stack = torch.tensor(list([item for sublist in hyp_output for item in sublist]))
            margin_threshold = threshold
            # print("pred_stack: ", pred_stack)
            # print("target_stack: ", target_stack)
            # target_stack = target_stack.permute(0,1)
            pred_stack2 = pred_stack.unsqueeze(1)
            target_stack2 = target_stack.unsqueeze(1)
            rise_true, rise_pred_correct, fall_true, fall_pred_correct = binary_detector_evaluator(pred_stack2, target_stack2, margin)
            print("Margin: {}, Threshold: {}, TPR: {}, TNR: {}".format(str(margin), str(threshold), str(logger.evaluator.picked_tprs[threshold_idx]), str(logger.evaluator.picked_tnrs[threshold_idx])))
            print("rise_accuarcy:{}, fall_accuracy:{}".format(str(np.round((rise_pred_correct/float(rise_true)), decimals=4)), str(np.round((fall_pred_correct/float(fall_true)), decimals=4))))
            if margin == 3:
                margin_3sec_rise_seeds.append(np.round((rise_pred_correct/float(rise_true)), decimals=4))
                margin_3sec_fall_seeds.append(np.round((fall_pred_correct/float(fall_true)), decimals=4))

            if margin == 5:
                margin_5sec_rise_seeds.append(np.round((rise_pred_correct/float(rise_true)), decimals=4))
                margin_5sec_fall_seeds.append(np.round((fall_pred_correct/float(fall_true)), decimals=4))

    ref_events = []
    t_dur = 0   # unit in second
    for ref in refs:
        ref.insert(0,0)
        ref.insert(len(ref),0)
        ref_diff = np.array(ref)-np.array([ref[0]]+ref[:-1])
        starts = np.where(ref_diff == 1)[0]
        ends = np.where(ref_diff == -1)[0]
        
        if (len(starts) == 0) and (len(ends) == 0):
            ref_events.append(list())
        else:
            ref_events.append([(starts[idx]-1, ends[idx]-1) for idx in range(len(starts))])
        t_dur += len(ref)
        t_dur += 3 # window size of 4

    hyps_list = [list(hyp) for hyp in hyps]

    threshold_num = 500

    tprs = []
    tnrs = []
    fprs = []
    fas = []
    latency_times = []
    detected_latency_times = []
    latency_0_95 = 0
    print("##### OVLP evaluation #####")
    for i in range(1, threshold_num):
        hyp_events = []
        threshold = float(round((1.0 / threshold_num) * i,3))
        hyp_output = [[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list]
        if threshold == 0.95:
            latency_0_95_threshold_idx = i-1
        for hyp_element in hyp_output:
            hyp_element.insert(0,0)
            hyp_element.insert(len(hyp_element),0)
            hyp_diff = np.array(hyp_element)-np.array([hyp_element[0]]+hyp_element[:-1])
            starts = np.where(hyp_diff == 1)[0]
            ends = np.where(hyp_diff == -1)[0]

            if (len(starts) == 0) and (len(ends) == 0):
                hyp_events.append(list())
            else:
                hyp_events.append([(starts[idx]-1, ends[idx]-1) for idx in range(len(starts))])

        hit_t = 0
        mis_t = 0
        fal_t = 0
        refo_t = 0
        hypo_t = 0
        latency = 0
        detected_latency = 0
        refo_minus = 0 
        for k in range(len(ref_events)):
            hit, mis, fal, refo, hypo, delayed_time, latency_time_of_detected, refo_minus_count = ovlp(ref_events[k], hyp_events[k])
            hit_t += hit
            mis_t += mis
            fal_t += fal
            refo_t += refo
            hypo_t += hypo
            latency += delayed_time
            detected_latency += latency_time_of_detected
            refo_minus += refo_minus_count
        # print("threshold: {}, hit: {}, mis: {}, fal: {}, refo: {}, hypo: {}".format(str(threshold), str(hit_t), str(mis_t), str(fal_t), str(refo_t), str(hypo_t)))
        if refo_t == 0:
            tprs.append(1)
        else:
            tprs.append(float(hit_t)/refo_t)
        if hypo_t == 0:
            tnrs.append(0)
        else:
            tnrs.append(1-(float(fal_t)/hypo_t))
        if hypo_t == 0:
            fprs.append(1)
        else:
            fprs.append(float(fal_t)/hypo_t)
        fas.append(fal_t)
        latency_times.append(latency)
        detected_latency_times.append((detected_latency, refo_minus))
    # print(tprs)
    # print(fprs)e
    best_threshold = np.argmax(np.array(tprs) + np.array(tnrs))
    fa_24_hours = (float(fas[best_threshold]) / t_dur) * (60 * 60 * 24)
    print("Best sensitivity: ", tprs[best_threshold])
    print("Best specificity: ", tnrs[best_threshold])
    print("Best FA/24hrs: ", fa_24_hours)

    ovlp_tprs_seeds.append(tprs[best_threshold])
    ovlp_tnrs_seeds.append(tnrs[best_threshold])
    ovlp_fas24_seeds.append(fa_24_hours)

    # plt.plot(tprs,fprs)
    # plt.show()
    print("##### latency evaluation #####")
    print("Latency in second: ", float(latency_times[latency_0_95_threshold_idx])/refo_t) # 모든 레이턴시
    if (refo_t-detected_latency_times[latency_0_95_threshold_idx][1]) != 0:
        print("Detected Latency in second: ", float(detected_latency_times[latency_0_95_threshold_idx][0])/(refo_t-detected_latency_times[latency_0_95_threshold_idx][1]))
        print("Detected Latency: {}, Missed Events: {}/{}".format(str(detected_latency_times[latency_0_95_threshold_idx][0]), str(detected_latency_times[latency_0_95_threshold_idx][1]), str(refo_t)))

    latencies_seeds.append(float(latency_times[latency_0_95_threshold_idx])/refo_t)
    if (refo_t-detected_latency_times[latency_0_95_threshold_idx][1]) != 0:
        detected_latencies_seeds.append(float(detected_latency_times[latency_0_95_threshold_idx][0])/(refo_t-detected_latency_times[latency_0_95_threshold_idx][1]))
    else:
        detected_latencies_seeds.append(0)
    missed_for_latency_seeds.append(detected_latency_times[latency_0_95_threshold_idx][1])
    refos_for_latency_seeds.append(refo_t)

    tprs = []
    tnrs = []
    fprs = []
    fas = []
    print("##### TAES evaluation #####")
    for i in range(1, threshold_num):
        hyp_events = []
        threshold = float(round((1.0 / threshold_num) * i,3))
        hyp_output = [[int(hyp_step > threshold) for hyp_step in hyp_one] for hyp_one in hyps_list]

        for hyp_element in hyp_output:
            hyp_element.insert(0,0)
            hyp_element.insert(len(hyp_element),0)
            hyp_diff = np.array(hyp_element)-np.array([hyp_element[0]]+hyp_element[:-1])
            starts = np.where(hyp_diff == 1)[0]
            ends = np.where(hyp_diff == -1)[0]

            if (len(starts) == 0) and (len(ends) == 0):
                hyp_events.append(list())
            else:
                hyp_events.append([(starts[idx]-1, ends[idx]-1) for idx in range(len(starts))])

        hit_t = 0
        mis_t = 0
        fal_t = 0
        refo_t = 0
        hypo_t = 0
        for k in range(len(ref_events)):
            hit, mis, fal, refo, hypo = taes(ref_events[k], hyp_events[k])
            hit_t += hit
            mis_t += mis
            fal_t += fal
            refo_t += refo
            hypo_t += hypo
        # print("threshold: {}, hit: {}, mis: {}, fal: {}, refo: {}, hypo: {}".format(str(threshold), str(hit_t), str(mis_t), str(fal_t), str(refo_t), str(hypo_t)))
        if refo_t == 0:
            tprs.append(1)
        else:
            tprs.append(float(hit_t)/refo_t)
        if hypo_t == 0:
            tnrs.append(0)
        else:
            tnrs.append(1-(float(fal_t)/hypo_t))
        if hypo_t == 0:
            fprs.append(1)
        else:
            fprs.append(float(fal_t)/hypo_t)
        fas.append(fal_t)
    # print(tprs)
    # print(fprs)
    best_threshold = np.argmax(np.array(tprs) + np.array(tnrs))
    fa_24_hours = (float(fas[best_threshold]) / t_dur) * (60 * 60 * 24)
    print("Best sensitivity: ", tprs[best_threshold])
    print("Best specificity: ", tnrs[best_threshold])
    print("Best FA/24hrs: ", fa_24_hours)
    taes_tprs_seeds.append(tprs[best_threshold])
    taes_tnrs_seeds.append(tnrs[best_threshold])
    taes_fas24_seeds.append(fa_24_hours)

os.system("echo  \'#######################################\'")
os.system("echo  \'##### Final test results per seed #####\'")
os.system("echo  \'#######################################\'")
os.system("echo  \'Total average -- ovlp_tpr: {}, ovlp_tnr: {}, ovlp_fas24: {}\'".format(str(np.mean(ovlp_tprs_seeds)), str(np.mean(ovlp_tnrs_seeds)), str(np.mean(ovlp_fas24_seeds))))
os.system("echo  \'Total std -- ovlp_tpr: {}, ovlp_tnr: {}, ovlp_fas24: {}\'".format(str(np.std(ovlp_tprs_seeds)), str(np.std(ovlp_tnrs_seeds)), str(np.std(ovlp_fas24_seeds))))

os.system("echo  \'Total average -- taes_tpr: {}, taes_tnr: {}, taes_fas24: {}\'".format(str(np.mean(taes_tprs_seeds)), str(np.mean(taes_tnrs_seeds)), str(np.mean(taes_fas24_seeds))))
os.system("echo  \'Total std -- taes_tpr: {}, taes_tnr: {}, taes_fas24: {}\'".format(str(np.std(taes_tprs_seeds)), str(np.std(taes_tnrs_seeds)), str(np.std(taes_fas24_seeds))))

os.system("echo  \'Total average -- latnecy: {}, d_latency: {}, missed: {}, refos: {}\'".format(str(np.mean(latencies_seeds)), str(np.mean(detected_latencies_seeds)), str(np.mean(missed_for_latency_seeds)), str(np.mean(refos_for_latency_seeds))))
os.system("echo  \'Total std -- latnecy: {}, d_latency: {}, missed: {}, refos: {}\'".format(str(np.std(latencies_seeds)), str(np.std(detected_latencies_seeds)), str(np.std(missed_for_latency_seeds)), str(np.std(refos_for_latency_seeds))))

os.system("echo  \'Total average -- 3sec_rise: {}, 3sec_fall: {}, 5sec_rise: {}, 5sec_fall: {}\'".format(str(np.mean(margin_3sec_rise_seeds)), str(np.mean(margin_3sec_fall_seeds)), str(np.mean(margin_5sec_rise_seeds)), str(np.mean(margin_5sec_fall_seeds))))
os.system("echo  \'Total std -- 3sec_rise: {}, 3sec_fall: {}, 5sec_rise: {}, 5sec_fall: {}\'".format(str(np.std(margin_3sec_rise_seeds)), str(np.std(margin_3sec_fall_seeds)), str(np.std(margin_5sec_rise_seeds)), str(np.std(margin_5sec_fall_seeds))))
