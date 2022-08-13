# -*- coding: utf-8 -*-
# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pyedflib import highlevel, EdfReader
from scipy.io.wavfile import write
from scipy import signal as sci_sig
from scipy.spatial.distance import pdist
from scipy.signal import stft, hilbert, butter, freqz, filtfilt, find_peaks
from builder.utils.process_util import run_multi_process
from builder.utils.utils import search_walk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import argparse
import torch
import glob
import pickle
import random
import mne 
from mne.io.edf.edf import _read_annotations_edf, _read_edf_header
from itertools import groupby

GLOBAL_DATA = {}
label_dict = {}
sample_rate_dict = {}
sev_label = {}


def label_sampling_tuh(labels, feature_samplerate):
    y_target = ""
    remained = 0
    feature_intv = 1/float(feature_samplerate)
    for i in labels:
        begin, end, label = i.split(" ")[:3]

        intv_count, remained = divmod(float(end) - float(begin) + remained, feature_intv)
        y_target += int(intv_count) * str(GLOBAL_DATA['disease_labels'][label])
    return y_target


def generate_training_data_leadwise_tuh_train(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    label_file = open(file_name + "." + GLOBAL_DATA['label_type'], 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
    y = label_file.readlines()
    y = list(y[2:])
    y_labels = list(set([i.split(" ")[2] for i in y]))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])
    
    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        y_sampled += y_sampled[-1] * diff

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1,0)

    max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_after_seiz_label = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_after_seiz_raw = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]

    start_raw_idx = 0
    start_label_idx = 0
    end_raw_idx = raw_data.size(0)
    end_label_idx = len(y_sampled)
    previous_bckg_len = 0
    
    sliced_raws = []
    sliced_labels = []
    pre_bckg_lens_label = []
    label_list_for_filename = []
    
    for idx, label in enumerate(label_order):
        # if last and the label is "bckg"
        if (len(label_order) == idx+1) and (label == "0"):
            sliced_raw_data = raw_data[start_raw_idx:].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx:]))).byte()

            if sliced_y1.size(0) < min_seg_len_label:
                continue
            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        # if not last and the label is "bckg"
        elif (len(label_order) != idx+1) and (label == "0"):
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            sliced_raw_data = raw_data[start_raw_idx:end_raw_idx].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx:end_label_idx]))).byte()
            previous_bckg_len = end_label_idx - start_label_idx

            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if sliced_y1.size(0) < min_seg_len_label:
                continue

            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        # if the first and the label is "seiz" 1 ~ 8       
        elif (idx == 0) and (label != "0"):
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            if len(y_sampled)-end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled)-end_label_idx
                post_len_raw = ((len(y_sampled)-end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw
            
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if len(y_sampled) < min_seg_len_label:
                continue

            sliced_raw_data = raw_data[:post_ictal_end_raw].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1,0)[:max_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            elif sliced_y1.size(0) >= min_seg_len_label:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            else:
                sliced_y2 = torch.Tensor(list(map(int,y_sampled[:min_seg_len_label]))).byte()
                sliced_raw_data2 = raw_data[:min_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)

        # the label is "seiz" 1 ~ 8
        elif label != "0":
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            if len(y_sampled)-end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled)-end_label_idx
                post_len_raw = ((len(y_sampled)-end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw

            if previous_bckg_len > max_seg_len_before_seiz_label:
                pre_seiz_label_len = max_seg_len_before_seiz_label
            else:
                pre_seiz_label_len = previous_bckg_len
            pre_seiz_raw_len = pre_seiz_label_len * GLOBAL_DATA['fsr_sr_ratio']

            sample_len = post_ictal_end_label - (start_label_idx-pre_seiz_label_len)
            if sample_len < min_seg_len_label:
                post_ictal_end_label = start_label_idx - pre_seiz_label_len + min_seg_len_label
                post_ictal_end_raw = start_raw_idx - pre_seiz_raw_len + min_seg_len_raw
            if len(y_sampled) < post_ictal_end_label:
                start_raw_idx = end_raw_idx
                start_label_idx = end_label_idx
                continue

            sliced_raw_data = raw_data[start_raw_idx-pre_seiz_raw_len:post_ictal_end_raw].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx-pre_seiz_label_len:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1,0)[:max_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            # elif sliced_y1.size(0) >= min_seg_len_label:
            else:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
                
        else:
            print("Error! Impossible!")
            exit(1)

    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        prelabel_len = pre_bckg_lens_label[data_idx]
        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_pre{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx), str(prelabel_len), str(len(sliced_y)), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)      
        new_data = {}

def generate_training_data_leadwise_tuh_train_final(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    label_file = open(file_name + "." + GLOBAL_DATA['label_type'], 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
    y = label_file.readlines()
    y = list(y[2:])
    y_labels = list(set([i.split(" ")[2] for i in y]))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

    # check if seizure patient or non-seizure patient
    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    patient_id = file_name.split("/")[-3]
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".tse_bi"})
    patient_bool = False
    for tse_bi_file in edf_list:
        label_file = open(tse_bi_file, 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
        y = label_file.readlines()
        y = list(y[2:])
        for line in y:
            if len(line) > 5:
                if line.split(" ")[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break
    
    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        y_sampled += y_sampled[-1] * diff

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1,0)
    raw_data = raw_data.type(torch.float16)
    
    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    min_binary_edge_seiz_label = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['feature_sample_rate']
    min_binary_edge_seiz_raw = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]
    label_change_idxs = np.append(label_change_idxs, np.array([len(y_sampled_np)-1]))

    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []
    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        y_sampled_2nd = list(y_sampled)
        raw_data_2nd = raw_data
        while len(y_sampled) >= min_seg_len_label:
            is_at_middle = False
            sliced_y = y_sampled[:min_seg_len_label]
            labels = [x[0] for x in groupby(sliced_y)]
                
            if len(labels) == 1 and "0" in labels:
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                if patient_bool:
                    label = "0_patT"
                else:
                    label = "0_patF"
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)
                
            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] != '0'):
                temp_sliced_y = list(sliced_y)
                temp_sliced_y.reverse()
                boundary_seizlen = temp_sliced_y.index("0") + 1
                if boundary_seizlen < min_binary_edge_seiz_label:
                    if len(y_sampled) > (min_seg_len_label + min_binary_edge_seiz_label):
                        sliced_y = y_sampled[min_binary_edge_seiz_label:min_seg_len_label+min_binary_edge_seiz_label]
                        sliced_raw_data = raw_data[min_binary_edge_seiz_raw:min_seg_len_raw+min_binary_edge_seiz_raw].permute(1,0)
                    else:
                        sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                else:
                    sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)

                y_sampled = y_sampled[min_seg_len_label:]
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                
                label = label + "_beg"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) != 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_whole"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) == 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_middle"
                label_list_for_filename.append(label)
                is_at_middle = True
            
            elif len(labels) != 1 and (sliced_y[0] != '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                
                label = label + "_end"
                label_list_for_filename.append(label)
            
            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_whole"
                label_list_for_filename.append(label)
            
            else:
                print("unexpected case")
                exit(1)
        if is_at_middle == True:
            sliced_y = y_sampled_2nd[-min_seg_len_label:]
            sliced_raw_data = raw_data_2nd[-min_seg_len_raw:].permute(1,0)
            
            if sliced_y[-1] == '0':
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                
                label = label + "_end"
                label_list_for_filename.append(label)
            else:
                pass
            
    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))
        sliced_y = torch.Tensor(sliced_y_map).byte()

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_label_{}.pkl".format(data_file_name, str(data_idx), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)      
        new_data = {}

def generate_training_data_leadwise_tuh_dev(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    label_file = open(file_name + "." + GLOBAL_DATA['label_type'], 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
    y = label_file.readlines()
    y = list(y[2:])
    y_labels = list(set([i.split(" ")[2] for i in y]))
    signal_sample_rate = int(signal_headers[0]['sample_rate'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])
    
    # check if seizure patient or non-seizure patient
    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".tse_bi"})
    patient_bool = False
    for tse_bi_file in edf_list:
        label_file = open(tse_bi_file, 'r') # EX) 00007235_s003_t003.tse or 00007235_s003_t003.tse_bi
        y = label_file.readlines()
        y = list(y[2:])
        for line in y:
            if len(line) > 5:
                if line.split(" ")[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break

    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_rate']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            signal_list.append(signal)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        y_sampled += y_sampled[-1] * diff

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.Tensor(signal_final_list_raw).permute(1,0)
    raw_data = raw_data.type(torch.float16)
    
    # max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    # max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    # min_end_margin_label = args.slice_end_margin_length * GLOBAL_DATA['feature_sample_rate']
    # min_end_margin_raw = args.slice_end_margin_length * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    # max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    # max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    
    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []

    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        while len(y_sampled) >= min_seg_len_label:
            one_left_slice = False
            sliced_y = y_sampled[:min_seg_len_label]
                
            if (sliced_y[-1] == '0'):
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                y_sampled = y_sampled[min_seg_len_label:]

                labels = [x[0] for x in groupby(sliced_y)]
                if (len(labels) == 1) and (labels[0] == '0'):
                    label = "0"
                else:
                    label = ("".join(labels)).replace("0", "")[0]
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)

            else:
                if '0' in y_sampled[min_seg_len_label:]:
                    end_1 = y_sampled[min_seg_len_label:].index('0')
                    temp_y_sampled = list(y_sampled[min_seg_len_label+end_1:])
                    temp_y_sampled_order = [x[0] for x in groupby(temp_y_sampled)]

                    if len(list(set(temp_y_sampled))) == 1:
                        end_2 = len(temp_y_sampled)
                        one_left_slice = True
                    else:
                        end_2 = temp_y_sampled.index(temp_y_sampled_order[1])

                    if end_2 >= min_end_margin_label:
                        temp_sec = random.randint(1,args.slice_end_margin_length)
                        temp_seg_len_label = int(min_seg_len_label + (temp_sec * args.feature_sample_rate) + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_sec * args.samplerate) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))
                    else:
                        if one_left_slice:
                            temp_label = end_2
                        else:
                            temp_label = end_2 // 2

                        temp_seg_len_label = int(min_seg_len_label + temp_label + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_label * GLOBAL_DATA['fsr_sr_ratio']) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))

                    sliced_y = y_sampled[:temp_seg_len_label]
                    sliced_raw_data = raw_data[:temp_seg_len_raw].permute(1,0)
                    raw_data = raw_data[temp_seg_len_raw:]
                    y_sampled = y_sampled[temp_seg_len_label:]

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
                else:
                    sliced_y = y_sampled[:]
                    sliced_raw_data = raw_data[:].permute(1,0)
                    raw_data = []
                    y_sampled = []

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
            
    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        new_data['RAW_DATA'] = [sliced_raw]
        new_data['LABEL1'] = [sliced_y]
        new_data['LABEL2'] = [sliced_y2]
        new_data['LABEL3'] = [sliced_y3]

        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx), str(len(sliced_y)), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f)      
        new_data = {}


def main(args):
    save_directory = args.save_directory
    data_type = args.data_type
    dataset = args.dataset
    label_type = args.label_type
    sample_rate = args.samplerate
    cpu_num = args.cpu_num
    feature_type = args.feature_type
    feature_sample_rate = args.feature_sample_rate
    task_type = args.task_type
    data_file_directory = save_directory + "/dataset-{}_task-{}_datatype-{}_v6".format(dataset, task_type, data_type)
    
    
    labels = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',  
                    'EEG C3', 'EEG C4', 'EEG CZ', 'EEG T3', 'EEG T4', 
                    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    eeg_data_directory = "$PATH_TO_EEG/{}".format(data_type)
    # eeg_data_directory = "/mnt/aitrics_ext/ext01/shared/edf/tuh_final/{}".format(data_type)
    
    if label_type == "tse":
        disease_labels =  {'bckg': 0, 'cpsz': 1, 'mysz': 2, 'gnsz': 3, 'fnsz': 4, 'tnsz': 5, 'tcsz': 6, 'spsz': 7, 'absz': 8}
    elif label_type == "tse_bi":
        disease_labels =  {'bckg': 0, 'seiz': 1}
    disease_labels_inv = {v: k for k, v in disease_labels.items()}
    
    edf_list1 = search_walk({'path': eeg_data_directory, 'extension': ".edf"})
    edf_list2 = search_walk({'path': eeg_data_directory, 'extension': ".EDF"})
    if edf_list2:
        edf_list = edf_list1 + edf_list2
    else:
        edf_list = edf_list1

    if os.path.isdir(data_file_directory):
        os.system("rm -rf {}".format(data_file_directory))
    os.system("mkdir {}".format(data_file_directory))

    GLOBAL_DATA['label_list'] = labels # 'EEG FP1', 'EEG FP2', 'EEG F3', ...
    GLOBAL_DATA['disease_labels'] = disease_labels #  {'bckg': 0, 'cpsz': 1, 'mysz': 2, ...
    GLOBAL_DATA['disease_labels_inv'] = disease_labels_inv #  {0:'bckg', 1:'cpsz', 2:'mysz', ...
    GLOBAL_DATA['data_file_directory'] = data_file_directory
    GLOBAL_DATA['label_type'] = label_type # "tse_bi" ...
    GLOBAL_DATA['feature_type'] = feature_type
    GLOBAL_DATA['feature_sample_rate'] = feature_sample_rate
    GLOBAL_DATA['sample_rate'] = sample_rate
    GLOBAL_DATA['fsr_sr_ratio'] = (sample_rate // feature_sample_rate)
    GLOBAL_DATA['min_binary_slicelength'] = args.min_binary_slicelength
    GLOBAL_DATA['min_binary_edge_seiz'] = args.min_binary_edge_seiz

    target_dictionary = {0:0}
    selected_diseases = []
    for idx, i in enumerate(args.disease_type):
        selected_diseases.append(str(disease_labels[i]))
        target_dictionary[disease_labels[i]] = idx + 1
    
    GLOBAL_DATA['disease_type'] = args.disease_type # args.disease_type == ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz']
    GLOBAL_DATA['target_dictionary'] = target_dictionary # {0: 0, 4: 1, 5: 2, 8: 3, 2: 4, 9: 5, 6: 6, 7: 7, 3: 8}
    GLOBAL_DATA['selected_diseases'] = selected_diseases # ['4', '5', '8', '2', '9', '6', '7', '3']
    GLOBAL_DATA['binary_target1'] = args.binary_target1
    GLOBAL_DATA['binary_target2'] = args.binary_target2

    print("########## Preprocessor Setting Information ##########")
    print("Number of EDF files: ", len(edf_list))
    for i in GLOBAL_DATA:
        print("{}: {}".format(i, GLOBAL_DATA[i]))
    with open(data_file_directory + '/preprocess_info.infopkl', 'wb') as pkl:
        pickle.dump(GLOBAL_DATA, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print("################ Preprocess begins... ################\n")
    
    if (task_type == "binary") and (args.data_type == "train"):
        run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
    elif (task_type == "binary") and (args.data_type == "dev"):
        run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
        
if __name__ == '__main__':
    # make sure all edf file name different!!! if not, additional coding is necessary
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-sd', type=int, default=1004,
                        help='Random seed number')
    parser.add_argument('--samplerate', '-sr', type=int, default=200,
                        help='Sample Rate')
    parser.add_argument('--save_directory', '-sp', type=str,
                        help='Path to save data')
    parser.add_argument('--label_type', '-lt', type=str,
                        default='tse',
                        help='tse_bi = global with binary label, tse = global with various labels, cae = severance CAE seizure label.')                      
    parser.add_argument('--cpu_num', '-cn', type=int,
                        default=32,
                        help='select number of available cpus')   
    parser.add_argument('--feature_type', '-ft', type=str,
                        default=['rawsignal'])   
    parser.add_argument('--feature_sample_rate', '-fsr', type=int,
                        default=50,
                        help='select features sample rate')   
    parser.add_argument('--dataset', '-st', type=str,
                        default='tuh',
                        choices=['tuh'])                   
    parser.add_argument('--data_type', '-dt', type=str,
                        default='train',
                        choices=['train', 'dev'])                   
    parser.add_argument('--task_type', '-tt', type=str,
                        default='binary',
                        choices=['anomaly', 'multiclassification', 'binary'])                   

    ##### Target Grouping #####
    parser.add_argument('--disease_type', type=list, default=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'], choices=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'])

    ### for binary detector ###
    # key numbers represent index of --disease_type + 1  ### -1 is "not being used"
    parser.add_argument('--binary_target1', type=dict, default={0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    parser.add_argument('--binary_target2', type=dict, default={0:0, 1:1, 2:2, 3:2, 4:2, 5:1, 6:3, 7:4, 8:5})
    parser.add_argument('--min_binary_slicelength', type=int, default=30)           
    parser.add_argument('--min_binary_edge_seiz', type=int, default=3) 
    args = parser.parse_args()
    main(args)

