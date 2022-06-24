# Copyright (c) 2022, Kwanhyung Lee, Hyewon Jeong. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
# import shap
# import lime
import random
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from builder.utils.binary_performance_estimator import binary_detector_evaluator


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.n_labels = args.output_dim
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.batch_size = args.batch_size
        self.best_auc = 0
        self.labels_list = [i for i in range(self.n_labels)]
        self.seizure_wise_eval_for_binary = False
        self.y_true_multi = []
        self.y_pred_multi = []
        self.signal_info_list = []

        self.thresholds_margintest = []
        
        self.probability_list = []
        self.final_target_list = []

        self.picked_tnrs = []
        self.picked_tprs = []

        self.seizurewise_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    def binary_normalize(self, i):
        proba_list = [i[0], max(i[1:])]
        return np.array(proba_list)/sum(proba_list)

    def add_seizure_info(self, signal_info):
        self.signal_info_list.append(signal_info)

    def add_batch(self, y_true, y_pred_multi):
        y_pred_final = np.argmax(y_pred_multi, axis=1)
        y_true_multi = np.zeros((len(y_true), self.args.output_dim))
        y_true_multi[range(len(y_true)), y_true] = 1
        self.y_pred_multi.append(y_pred_multi)
        self.y_true_multi.append(y_true_multi)

        self.confusion_matrix += confusion_matrix(y_true, y_pred_final, labels=self.labels_list)

    def performance_metric_binary(self):
        self.y_true_multi = np.concatenate(self.y_true_multi, 0)
        self.y_pred_multi = np.concatenate(self.y_pred_multi, 0)

        auc = roc_auc_score (self.y_true_multi[:,1], self.y_pred_multi[:,1])
        apr = average_precision_score (self.y_true_multi[:,1], self.y_pred_multi[:,1])
        y_true_multi_array = np.argmax(self.y_true_multi, axis=1)

        f1 = 0
        for i in range(1, 200):
            threshold = float(i) / 200
            temp_output = np.array(self.y_pred_multi[:,1])
            temp_output[temp_output>=threshold] = 1
            temp_output[temp_output<threshold] = 0
            temp_score = f1_score(y_true_multi_array, temp_output, average="binary")
            if temp_score > f1:
                f1 = temp_score
            
        result = np.round(np.array([auc, apr, f1]), decimals=4)
        fpr, tpr, thresholds = roc_curve(y_true_multi_array, self.y_pred_multi[:,1], pos_label=1)
        fnr = 1 - tpr 
        tnr = 1 - fpr
        best_threshold = np.argmax(tpr + tnr)
        print("Best threshold is: ", thresholds[best_threshold])

        tnr_list = list(tnr)

        for tnr_one in self.args.tnr_for_margintest:
            picked_tnr = list([0 if x< tnr_one else x for x in tnr_list])
            picked_tnr_threshold = np.argmax(tpr + picked_tnr)        
            self.thresholds_margintest.append(thresholds[picked_tnr_threshold])
            self.picked_tnrs.append(np.round(tnr[picked_tnr_threshold], decimals=4))
            self.picked_tprs.append(np.round(tpr[picked_tnr_threshold], decimals=4))
        # print("TNRS: ", self.picked_tnrs)
        # print("TPRS: ", self.picked_tprs)
        # print("Selected Thresholds: ", self.thresholds_margintest)
        
        if self.args.seizure_wise_eval_for_binary:
            indx_by_seiz = [[], [], [], [], [], [], [], []]
            test_pat_dict = {}
            self.signal_info_list = np.concatenate(self.signal_info_list, 0)
            for idx, pkl in enumerate(self.signal_info_list):
                pat_id = pkl.split("_")[0]
                pat_bool = pkl.split("_")[-1]
                pat_seiz = pkl.split("_")[-2]
                pat_time = pkl.split("_")[1]

                if pat_bool == "patF":
                    continue
                if pat_id not in test_pat_dict:
                    test_pat_dict[pat_id] = []
                test_pat_dict[pat_id].append([int(pat_seiz), idx])
            for pat in test_pat_dict:
                seizure_types = list(set([seiz for seiz, idx in test_pat_dict[pat]]))
                seizure_types.remove(0)

                for seizure_type in seizure_types:
                    rest_types = list(seizure_types)
                    rest_types.remove(seizure_type)
                    indx_by_seiz[seizure_type-1].append(list([idx for seiz, idx in test_pat_dict[pat] if seiz not in rest_types]))

            for indx, t in enumerate(indx_by_seiz):
                indx_by_seiz[indx] = [item for sublist in t for item in sublist]
            lists_of_seizures_true = [[], [], [], [], [], [], [], []]
            lists_of_seizures_pred = [[], [], [], [], [], [], [], []]
            
            for indx, indxs_list  in enumerate(indx_by_seiz):
                lists_of_seizures_pred[indx] = [list(self.y_pred_multi[i]) for i in indxs_list]
                lists_of_seizures_true[indx] = [list(self.y_true_multi[i]) for i in indxs_list]

            for q in range(8):
                if len(lists_of_seizures_true[q]) == 0:
                    continue
                preds = np.array(lists_of_seizures_pred[q])
                trues = np.array(lists_of_seizures_true[q])
                auc = np.round(roc_auc_score(trues[:,1], preds[:,1]), decimals=4)
                apr = np.round(average_precision_score(trues[:,1], preds[:,1]), decimals=4)
                y_true_multi_array = np.argmax(trues, axis=1)
                f1 = 0
                for i in range(1, 200):
                    threshold = float(i) / 200
                    temp_output = np.array(preds[:,1])
                    temp_output[temp_output>=threshold] = 1
                    temp_output[temp_output<threshold] = 0
                    temp_score = f1_score(y_true_multi_array, temp_output, average="binary")
                    if temp_score > f1:
                        f1 = temp_score
                f1 = np.round(f1, decimals=4)
                fpr_seiz, tpr_seiz, thresholds_seiz = roc_curve(y_true_multi_array, preds[:,1], pos_label=1)
                fnr_seiz = 1 - tpr_seiz 
                tnr_seiz = 1 - fpr_seiz
                best_threshold_seiz = np.argmax(tpr_seiz + tnr_seiz)
                print("Seizure:{} - auc:{} apr:{} tpr:{} tnr:{}".format(self.args.seizure_to_num_inv[str(q+1)], str(auc), str(apr), str(tpr_seiz[best_threshold_seiz]), str(tnr_seiz[best_threshold_seiz])))
                self.seizurewise_list[q][0]=auc
                self.seizurewise_list[q][1]=apr
                self.seizurewise_list[q][2]=tpr_seiz[best_threshold_seiz]
                self.seizurewise_list[q][3]=tnr_seiz[best_threshold_seiz]

        if self.args.margin_test:
            target_stack = torch.stack(self.final_target_list)
            for margin in self.args.margin_list:
                for threshold_idx, threshold in enumerate(self.thresholds_margintest):
                    pred_stack = torch.stack(self.probability_list)
                    pred_stack = (pred_stack > threshold).int()
                    print("1: ", pred_stack.shape)
                    print("2: ", target_stack.shape)
                    rise_true, rise_pred_correct, fall_true, fall_pred_correct = binary_detector_evaluator(pred_stack, target_stack, margin)
                    print("Margin: {}, Threshold: {}, TPR: {}, TNR: {}".format(str(margin), str(threshold), str(self.picked_tprs[threshold_idx]), str(self.picked_tnrs[threshold_idx])))
                    # print("rise_t:{}, rise_cor:{}, fall_t:{}, fall_cor:{}".format(str(rise_true), str(rise_pred_correct), str(fall_true), str(fall_pred_correct)))    
                    print("rise_accuarcy:{}, fall_accuracy:{}".format(str(np.round((rise_pred_correct/float(rise_true)), decimals=4)), str(np.round((fall_pred_correct/float(fall_true)), decimals=4))))


        return result, np.round(tpr[best_threshold], decimals=4), np.round(fnr[best_threshold], decimals=4), np.round(tnr[best_threshold], decimals=4), np.round(fpr[best_threshold], decimals=4)

    def performance_metric_multi(self):
        print("bckg and {}".format(" ".join(self.args.seiz_classes)))
        print("Left: true, Top: pred")
        row_sums = self.confusion_matrix.sum(axis=1)
        confusion_matrix_proba = self.confusion_matrix / row_sums[:, np.newaxis]
        print(confusion_matrix_proba)

        self.y_true_multi = np.concatenate(self.y_true_multi, 0)
        self.y_pred_multi = np.concatenate(self.y_pred_multi, 0)

        multi_weighted_auc = roc_auc_score (self.y_true_multi, self.y_pred_multi, average="weighted")
        multi_unweighted_auc = roc_auc_score (self.y_true_multi, self.y_pred_multi, average="macro")
        multi_aucs = roc_auc_score (self.y_true_multi, self.y_pred_multi, average=None, multi_class='ovr')

        multi_weighted_apr = average_precision_score (self.y_true_multi, self.y_pred_multi, average="weighted")
        multi_unweighted_apr = average_precision_score (self.y_true_multi, self.y_pred_multi, average="macro")
        multi_aprs = average_precision_score (self.y_true_multi, self.y_pred_multi, average=None)

        y_true_multi_array = np.argmax(self.y_true_multi, axis=1)
        y_pred_multi_array = np.argmax(self.y_pred_multi, axis=1)
        multi_weighted_f1_score = f1_score(y_true_multi_array, y_pred_multi_array, average="weighted")
        multi_unweighted_f1_score = f1_score(y_true_multi_array, y_pred_multi_array, average="macro")
        multi_f1_scores = f1_score(y_true_multi_array, y_pred_multi_array, average=None)
            
        result = np.round(np.array([multi_weighted_auc, multi_unweighted_auc, multi_weighted_apr, multi_unweighted_apr, multi_weighted_f1_score, multi_unweighted_f1_score]), decimals=4)
        result_aucs = np.round(multi_aucs, decimals=4)
        result_aprs = np.round(multi_aprs, decimals=4)
        result_f1scores = np.round(multi_f1_scores, decimals=4)

        tprs = []
        fnrs = []
        tnrs = []
        fprs = []
        fdrs = []
        ppvs = []
        row_sums = self.confusion_matrix.sum(axis=1)
        column_sums = self.confusion_matrix.sum(axis=0)
        
        for i in range(self.args.output_dim):
            tp = float(self.confusion_matrix[i][i])
            fn = float(row_sums[i] - tp)
            fp = float(column_sums[i] - tp)
            tn = float(np.sum(self.confusion_matrix) - row_sums[i] - column_sums[i] + tp)

            if (tp + fn) == 0:
                tpr = 0
                fnr = 1
            else:
                tpr = tp / (tp + fn) #sensitivity (recall)
                fnr = fn / (tp + fn)

            if (tn + fp) == 0:
                tnr = 0
                fpr = 1
            else:
                tnr = tn / (tn + fp) #specificity
                fpr = fp / (fp + tn)
            
            if (tp + fp) == 0:
                fdr = 1
                ppv = 0
            else:
                fdr = fp / (tp + fp)
                ppv = tp / (tp + fp) 

            tprs.append( np.round(tpr, decimals=4))
            fnrs.append( np.round(fnr, decimals=4))
            tnrs.append( np.round(tnr, decimals=4))
            fprs.append( np.round(fpr, decimals=4))
            fdrs.append( np.round(fdr, decimals=4))
            ppvs.append( np.round(ppv, decimals=4))
            
        return result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs
       

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_labels, self.n_labels))
        self.y_true_multi = []
        self.y_pred_multi = []
        self.signal_info_list = []
        self.thresholds_margintest = []
        self.probability_list = []
        self.final_target_list = []
        self.seizurewise_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]




