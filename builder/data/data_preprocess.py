# Copyright (c) 2022, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import itertools
import speechpy
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from tqdm import tqdm
from scipy.signal import stft, hilbert, butter, freqz, filtfilt, find_peaks, iirnotch
from control.config import args
from itertools import groupby

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torchaudio

from builder.utils.utils import *


def bipolar_signals_func(signals):
    bipolar_signals = []
    bipolar_signals.append(signals[0]-signals[4]) #fp1-f7
    bipolar_signals.append(signals[1]-signals[5]) #fp2-f8
    bipolar_signals.append(signals[4]-signals[9]) #f7-t3
    bipolar_signals.append(signals[5]-signals[10]) #f8-t4
    bipolar_signals.append(signals[9]-signals[15]) #t3-t5
    bipolar_signals.append(signals[10]-signals[16]) #t4-t6
    bipolar_signals.append(signals[15]-signals[13]) #t5-o1
    bipolar_signals.append(signals[16]-signals[14]) #t6-o2
    bipolar_signals.append(signals[9]-signals[6]) #t3-c3
    bipolar_signals.append(signals[7]-signals[10]) #c4-t4
    bipolar_signals.append(signals[6]-signals[8]) #c3-cz
    bipolar_signals.append(signals[8]-signals[7]) #cz-c4
    bipolar_signals.append(signals[0]-signals[2]) #fp1-f3
    bipolar_signals.append(signals[1]-signals[3]) #fp2-f4
    bipolar_signals.append(signals[2]-signals[6]) #f3-c3
    bipolar_signals.append(signals[3]-signals[7]) #f4-c4
    bipolar_signals.append(signals[6]-signals[11]) #c3-p3
    bipolar_signals.append(signals[7]-signals[12]) #c4-p4
    bipolar_signals.append(signals[11]-signals[13]) #p3-o1
    bipolar_signals.append(signals[12]-signals[14]) #p4-o2

    return bipolar_signals


def eeg_binary_collate_fn(train_data):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    batch = []
    aug_list = []
    signal_name_list = []
    for input_seiz in train_data:
        with open(input_seiz, 'rb') as _f:
            data_pkl = pkl.load(_f)
            signals = data_pkl['RAW_DATA'][0]
            y = data_pkl[args.label_group][0]
     
            if args.eeg_type == "bipolar":
                bipolar_signals = bipolar_signals_func(signals)
                signals = torch.stack(bipolar_signals)
            elif args.eeg_type == "uni_bipolar":
                bipolar_signals = bipolar_signals_func(signals)
                signals = torch.cat((signals, torch.stack(bipolar_signals)))
            else:
                pass #unipolar

            batch.append((signals, y, input_seiz.split("/")[-1].split(".")[0]))
    pad_id = 0
    # batch = sorted(batch, key=lambda sample: sample[0][0].size(0), reverse=True)

    seq_lengths = torch.IntTensor([len(s[0][0]) for s in batch])
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(1)
    max_target_size = len(max_target_sample)

    batch_size = len(batch)
    eeg_type_size = len(batch[0][0])

    seqs = torch.zeros(batch_size, max_seq_size, eeg_type_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]

        seq_length = tensor[0].size(0)
        tensor = tensor.permute(1,0)
        # tensor = torch.reshape(tensor, (seq_length, eeg_type_size))
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        signal_name_list.append(sample[2])
        target = [int(i) for i in target]
        # ####################################
        # tensor1 = sample[0]
        # from itertools import groupby
        # target_check = list([x[0] for x in groupby(target)])
        # print(target_check)
        # ####################################
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
        # ####################################
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(21):
        #     plt.subplot(22,1,i+1)
        #     plt.plot(tensor1[i].detach().cpu().numpy())
        # plt.subplot(22,1,22)
        # plt.plot(target)
        # plt.show()
        # ####################################
    return seqs, targets, seq_lengths, target_lengths, aug_list, signal_name_list

class Detector_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_pkls, augment, data_type="training dataset"):
        self.data_type = data_type
        self._data_list = []
        self._type_list = []
        self._type_detail1 = []
        self._type_detail2 = []
        self.type_type = []
        num_normals = 0
        num_seizures_boundaries = 0
        num_seizures_middles = 0
        patient_dev_dict = {}
        for idx, pkl in enumerate(tqdm(data_pkls, desc="Loading edf files of {}".format(data_type))):
            type1, type2 = pkl.split("_")[-2:]
            if type1 == "8":
                if args.output_dim == 8 or args.binary_sampler_type == "30types":
                    continue
            if args.binary_sampler_type == "6types":
                label = pkl.split("_")[-1].split(".")[0]
            elif args.binary_sampler_type == "30types":
                label = "_".join(pkl.split("_")[-2:]).split(".")[0]                 
            else:
                print("Error! select correct binary data type...")
                exit(1)

            if "training dataset" != data_type:
                # if "middle" in pkl:
                #     continue
                pat_id = (pkl.split("/")[-1]).split("_")[0]
                if pat_id not in patient_dev_dict:
                    patient_dev_dict[pat_id] = [0, 0, 0] # normal, seizure, seiz_middle
                
                if (type1 == "0") and (patient_dev_dict[pat_id][0] >= args.dev_bckg_num):
                    continue
                if (type1 != "0") and (patient_dev_dict[pat_id][2] >= args.dev_bckg_num):
                    continue
                if type1 == "0":
                    patient_dev_dict[pat_id][0] += 1
                elif "middle" in pkl:
                    patient_dev_dict[pat_id][2] += 1
                else:
                    patient_dev_dict[pat_id][1] += 1

            if label not in self.type_type:
                self.type_type.append(label)
            type2 = type2.split(".")[0]
            self._type_detail1.append("_".join([type1, type2]))
            self._type_detail2.append(type1)
            self._type_list.append(self.type_type.index(label))
            self._data_list.append(pkl)
            
        print("########## Summary of {} ##########".format(data_type))    
        print("Types of types for sampler: ", self.type_type)  
        print("Number of types for sampler: ", len(self.type_type))
        print("--- Normal Slices Info ---")
        print("Patient normal slices size: ", self._type_detail1.count("0_patT"))
        print("Non-Patient normal slices size: ", self._type_detail1.count("0_patF"))
        print("Total normal slices size: ", self._type_detail2.count("0"))
        print("--- Seizure Slices Info ---")
        total_seiz_slices_num = 0
        for idx, seizure in enumerate(args.seiz_classes):
            seiz_num = args.seizure_to_num[seizure] 
            beg_slice_num = self._type_detail1.count(seiz_num + "_beg")
            middle_slice_num = self._type_detail1.count(seiz_num + "_middle")
            end_slice_num = self._type_detail1.count(seiz_num + "_end")
            whole_slice_num = self._type_detail1.count(seiz_num + "_whole")
            total_seiz_num = self._type_detail2.count(seiz_num)
            total_seiz_slices_num += total_seiz_num
            print("Number of {} slices: total:{} - beg:{}, middle:{}, end:{}, whole:{}".format(seizure, str(total_seiz_num), str(beg_slice_num), str(middle_slice_num), str(end_slice_num), str(whole_slice_num)))
        print("Total seizure slices: ", str(total_seiz_slices_num))
        print("Dataset Prepared...\n")
        
        if "training dataset" != data_type:
            print("Number of patients: ", len(patient_dev_dict))
            for pat_info in patient_dev_dict:
                pat_normal, pat_seiz, pat_middle = patient_dev_dict[pat_info]
                print("(Non-)Patient: {} has normals:{}, seizures:{}, mid_seizures:{}".format(pat_info, str(pat_normal), str(pat_seiz), str(pat_middle)))
                num_normals += pat_normal
                num_seizures_boundaries += pat_seiz
                num_seizures_middles += pat_middle
            
            print("Total normals:{}, seizures with boundaries:{}, seizures with middles:{}".format(str(num_normals), str(num_seizures_boundaries), str(num_seizures_middles)))

    def __repr__(self):
        return (f"Data path: {self._data_pkl}")

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        _input = self._data_list[index]
        return _input


def get_data_preprocessed(args, mode="train"):
   
    print("Preparing data for bianry detector...")
    train_data_path = args.data_path + "/dataset-tuh_task-binary_datatype-train_v6"
    # dev_data_path = args.data_path + "/dataset-tuh_task-binary_datatype-dev_v6"
    dev_data_path = args.data_path + "/dataset-tuh_task-binary_noslice_datatype-dev_v6"
    train_dir = search_walk({"path": train_data_path, "extension": ".pkl"})
    dev_dir = search_walk({"path": dev_data_path, "extension": ".pkl"})
    random.shuffle(train_dir)
    random.shuffle(dev_dir)

    aug_train = ["0"] * len(train_dir)
    if args.augmentation == True:
        train_dir += train_dir
        aug_train = ["1"] * len(train_dir)

    # # get one spsz and one tnsz from training data to dev data in order to distribute at least one seizure type to each group
    # patid_to_transfer = ["00008527", "00009044"]
    # for pkl1 in train_dir:
    #     type1, type2 = pkl1.split("_")[-2:]
    #     pat_id = (pkl1.split("/")[-1]).split("_")[0]
    #     if pat_id in patid_to_transfer:
    #         dev_dir.append(pkl1)
    #         train_dir.remove(pkl1)
    #     if type1 == "8":
    #         train_dir.remove(pkl1)
    # # Validation data and Test data patient separation
    # pat_info = {}
    # val_dict = {}
    # test_dict = {}
    # val_dir = []
    # test_dir = []
    # for pkl2 in dev_dir:
    #     type1, type2 = pkl2.split("_")[-2:]
    #     pat_id = (pkl2.split("/")[-1]).split("_")[0]
    #     if pat_id not in pat_info:
    #         pat_info[pat_id] = [[],[],[]]
    #     pat_info[pat_id][2].append(pkl2)
    #     pat_info[pat_id][0].append(type1)
    #     pat_info[pat_id][1].append(type2)
    # for pat_id in pat_info:
    #     pat_info[pat_id][0] = list(set(pat_info[pat_id][0]))
    #     pat_info[pat_id][1] = list(set(pat_info[pat_id][1]))

    # val_list = ["00008527", "00008460", "00004671", "00009578", "00010062", "00009697", "00004087", "00006986", "00002289", "00010022", "00005479", "00009866", "00001640", "00005625", "00008889", "00010639", "00009842", "00010106", "00004594", "00000675", "00002297", "00005031", "00010547", "00008174", "00000795"]
    # test_list = ["00009044", "00006546", "00001981", "00009839", "00009570", "00008544", "00008453", "00007633", "00003306", "00005943", "00008479", "00008512", "00006059", "00010861", "00001770", "00001027", "00000629", "00000258", "00001278", "00003281", "00003635", "00005213", "00008550", "00006900", "00004151", "00001984"]
    # # val_list = ["00008460", "00004671", "00009578", "00010062", "00009697", "00004087", "00006986", "00002289", "00010022", "00005479", "00009866", "00001640", "00005625", "00008889", "00010639", "00009842", "00010106", "00004594", "00000675", "00002297", "00005031", "00010547", "00008174", "00000795"]
    # # test_list = ["00006546", "00001981", "00009839", "00009570", "00008544", "00008453", "00007633", "00003306", "00005943", "00008479", "00008512", "00006059", "00010861", "00001770", "00001027", "00000629", "00000258", "00001278", "00003281", "00003635", "00005213", "00008550", "00006900", "00004151", "00001984"]
    # for i in val_list:
    #     val_dict[i] = pat_info[i]
    # for i in test_list:
    #     test_dict[i] = pat_info[i]
    # # print(" ")
    # # for i in val_dict:
    # #     print("{}: {}".format(str(i), val_dict[i]))
    # # print(" ")
    # # for i in test_dict:
    # #     print("{}: {}".format(str(i), test_dict[i]))
    # # exit(1)
    # for i in val_dict:
    #     val_dir += val_dict[i][2]
    # for i in test_dict:
    #     test_dir += test_dict[i][2]

    half_dev_num = int(len(dev_dir) // 2)
    val_dir = dev_dir[:half_dev_num] 
    test_dir = dev_dir[half_dev_num:]
    aug_val = ["0"] * len(val_dir)
    aug_test = ["0"] * len(test_dir)


    train_data = Detector_Dataset(args, data_pkls=train_dir, augment=aug_train, data_type="training dataset")
    class_sample_count = np.unique(train_data._type_list, return_counts=True)[1]
    weight = 1. / class_sample_count
    ########## Change Dataloader Sampler Rate for each class Here ##########
    # abnor_nor_ratio = len(class_sample_count)-1
    # weight[0] = weight[0] * abnor_nor_ratio
    if args.binary_sampler_type == "6types":
        patT_idx = (train_data.type_type).index("patT")
        patF_idx = (train_data.type_type).index("patF")
        # weight[patT_idx] = weight[patT_idx] * 2
        # weight[patF_idx] = weight[patF_idx] * 2
    elif args.binary_sampler_type == "30types":
        patT_idx = (train_data.type_type).index("0_patT")
        patF_idx = (train_data.type_type).index("0_patF")
        # weight[patT_idx] = weight[patT_idx] * 14
        # weight[patF_idx] = weight[patF_idx] * 14
        weight[patT_idx] = weight[patT_idx] * 7
        weight[patF_idx] = weight[patF_idx] * 7
    else:
        print("No control on sampler rate")
    ########################################################################
    samples_weight = weight[train_data._type_list]
    # print("samples_weight: ", samples_weight)
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    val_data = Detector_Dataset(args, data_pkls=val_dir, augment=aug_val, data_type="validation dataset")
    test_data = Detector_Dataset(args, data_pkls=test_dir, augment=aug_test, data_type="test dataset")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, sampler=sampler, collate_fn=eeg_binary_collate_fn)     
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, collate_fn=eeg_binary_collate_fn)               
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True,
                    num_workers=1, pin_memory=True, collate_fn=eeg_binary_collate_fn)  


    info_dir = train_data_path + "/preprocess_info.infopkl"
    with open(info_dir, 'rb') as _f:
        data_info = pkl.load(_f)

        args.disease_labels = data_info["disease_labels"]
        args.disease_labels_inv = data_info["disease_labels_inv"]
        
        args.sample_rate = data_info["sample_rate"]
        args.feature_sample_rate = data_info["feature_sample_rate"]
        
        args.disease_type = data_info["disease_type"]

        args.target_dictionary = data_info["target_dictionary"]
        args.selected_diseases = data_info["selected_diseases"]
        
        args.window_size_label = args.feature_sample_rate * args.window_size
        args.window_shift_label = args.feature_sample_rate * args.window_shift
        args.window_size_sig = args.sample_rate * args.window_size
        args.window_shift_sig = args.sample_rate * args.window_shift
        
        args.fsr_sr_ratio = (args.sample_rate // args.feature_sample_rate)


    with open(train_dir[0], 'rb') as _f:
        data_pkl = pkl.load(_f)
        signals = data_pkl['RAW_DATA'][0]

    if args.eeg_type == "bipolar":
        args.num_channel = 20
    elif args.eeg_type == "uni_bipolar":
        args.num_channel = 20 + signals.size(0)
    else:
        args.num_channel = signals.size(0)
    
    ############################################################
    print("Number of training data: ", len(train_dir))
    print("Number of validation data: ", len(val_dir))
    print("Number of test data: ", len(test_dir))
    print("Selected seizures are: ", args.seiz_classes)
    print("Selected task type is: ", args.task_type)
    if args.task_type == "binary":
        print("Selected binary group is: ", args.num_to_seizure)
        print("Selected sampler type: ", args.binary_sampler_type)
        print("Max number of normal slices per patient: ", str(args.dev_bckg_num))
    print("label_sample_rate: ", args.feature_sample_rate)
    print("raw signal sample_rate: ", args.sample_rate)
    print("Augmentation: ", args.augmentation)     
    
    return train_loader, val_loader, test_loader, len(train_data._data_list), len(val_data._data_list), len(test_data._data_list)

