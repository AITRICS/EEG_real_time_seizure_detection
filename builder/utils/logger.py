#!/usr/bin/env python3

# Copyright (c) 2022, Hyewon Jeong, Kwanhyung Lee. All rights reserved.
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
import sys
import shutil
import copy
import logging
import logging.handlers
from collections import OrderedDict
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from builder.utils.metrics import Evaluator


class Logger:
    def __init__(self, args):
        self.args = args
        self.args_save = copy.deepcopy(args)
        
        # Evaluator
        self.evaluator = Evaluator(self.args)
        
        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(self.args.dir_result, self.args.project_name)
        self.dir_log = os.path.join(self.dir_root, 'logs')
        self.dir_save = os.path.join(self.dir_root, 'ckpts')
        self.log_iter = args.log_iter

        if args.reset and os.path.exists(self.dir_root):
            shutil.rmtree(self.dir_root, ignore_errors=True)
        if not os.path.exists(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        elif os.path.exists(os.path.join(self.dir_save, 'last_{}.pth'.format(str(args.seed)))) and os.path.exists(self.dir_log):
            shutil.rmtree(self.dir_log, ignore_errors=True)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)
        
        # Log variables
        self.loss = 0
        self.val_loss = 0
        self.best_auc = 0
        self.best_iter = 0
        self.best_result_so_far = np.array([])
        self.best_results = []

        # results
        self.test_results = {}

        self.pred_results = []
        self.ans_results = []


    def log_tqdm(self, epoch, step, pbar):
        tqdm_log = "Epochs: {}, Iteration: {}, Loss: {}".format(str(epoch), str(step), str(self.loss / step))
        pbar.set_description(tqdm_log)
        
    def log_scalars(self, step):
        self.writer.add_scalar('train/loss', self.loss / step, global_step=step)
    
    def log_lr(self, lr, step):
        self.writer.add_scalar('train/lr', lr, global_step=step)

    def log_val_loss(self, val_step, step):
        self.writer.add_scalar('val/loss', self.val_loss / val_step, global_step=step)

    def add_validation_logs(self, step):
        if self.args.task_type == "binary":
            result, tpr, fnr, tnr, fpr = self.evaluator.performance_metric_binary()
            auc = result[0]
            os.system("echo  \'##### Current Validation results #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[1]), str(result[2])))
            os.system("echo  \'tpr: {}, fnr: {}, tnr: {}, fpr: {}\'".format(str(tpr), str(fnr), str(tnr), str(fpr)))

            self.writer.add_scalar('val/auc', result[0], global_step=step)
            self.writer.add_scalar('val/apr', result[1], global_step=step)
            self.writer.add_scalar('val/f1', result[2], global_step=step)
            self.writer.add_scalar('val/tpr', tpr, global_step=step)
            self.writer.add_scalar('val/fnr', fnr, global_step=step)
            self.writer.add_scalar('val/tnr', tnr, global_step=step)
            self.writer.add_scalar('val/fpr', fpr, global_step=step)

            if self.best_auc < auc:
                self.best_iter = step
                self.best_auc = auc
                self.best_result_so_far = result
                self.best_results = [tpr, fnr, tnr, fpr]

            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(self.best_result_so_far[0]), str(self.best_result_so_far[1]), str(self.best_result_so_far[2])))
            os.system("echo  \'tpr: {}, fnr: {}, tnr: {}, fpr: {}\'".format(str(self.best_results[0]), str(self.best_results[1]), str(self.best_results[2]), str(self.best_results[3])))

        else:
            result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = self.evaluator.performance_metric_multi()

            multi_weighted_auc = result[0]
            multi_unweighted_auc = result[1]
            multi_weighted_apr = result[2]
            multi_unweighted_apr = result[3]
            multi_weighted_f1_score = result[4]
            multi_unweighted_f1_score = result[5]

            os.system("echo  \'##### Current Validation results #####\'")
            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[2]), str(result[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[1]), str(result[3]), str(result[5])))
            os.system("echo  \'##### Each class Validation results #####\'")
            seizure_list = self.args.num_to_seizure_items
            results = []
            # results.append("Label:bckg auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(
            #                                                                     str(result_aucs[0]), str(result_aprs[0]), str(result_f1scores[0]), 
            #                                                                     str(tprs[0]), str(fnrs[0]), str(tnrs[0]), str(fprs[0]), str(fdrs[0]), str(ppvs[0])))
            for idx, seizure in enumerate(seizure_list):
                results.append("Label:{} auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(seizure,
                                                                                str(result_aucs[idx]), str(result_aprs[idx]), str(result_f1scores[idx]), 
                                                                                str(tprs[idx]), str(fnrs[idx]), str(tnrs[idx]), str(fprs[idx]), 
                                                                                str(fdrs[idx]), str(ppvs[idx])))

            for i in results:
                os.system("echo  \'{}\'".format(i))

            self.writer.add_scalar('val/multi_weighted_auc', multi_weighted_auc, global_step=step)
            self.writer.add_scalar('val/multi_weighted_apr', multi_weighted_apr, global_step=step)
            self.writer.add_scalar('val/multi_weighted_f1_score', multi_weighted_f1_score, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_auc', multi_unweighted_auc, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_apr', multi_unweighted_apr, global_step=step)
            self.writer.add_scalar('val/multi_unweighted_f1_score', multi_unweighted_f1_score, global_step=step)

            if self.best_auc < multi_weighted_auc:
                self.best_iter = step
                self.best_auc = multi_weighted_auc
                self.best_result_so_far = result
                self.best_results = results

            os.system("echo  \'##### Best Validation results in history #####\'")
            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(self.best_result_so_far[0]), str(self.best_result_so_far[2]), str(self.best_result_so_far[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(self.best_result_so_far[1]), str(self.best_result_so_far[3]), str(self.best_result_so_far[5])))
            for i in self.best_results:
                os.system("echo  \'{}\'".format(i))

        self.writer.flush()

    def save(self, model, optimizer, step, epoch, last=None):
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_step': step, 'last_step' : last, 'score' : self.best_auc, 'epoch' : epoch}
        
        if step == self.best_iter:
            self.save_ckpt(ckpt, 'best_{}.pth'.format(str(self.args.seed)))
            
        if last:
            self.evaluator.get_attributions()
            self.save_ckpt(ckpt, 'last_{}.pth'.format(str(self.args.seed)))
    
    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))

    def test_result_only(self):

        if self.args.task_type == "binary" or self.args.task_type == "binary_noslice":
            result, tpr, fnr, tnr, fpr = self.evaluator.performance_metric_binary()

            os.system("echo  \'##### Test results #####\'")
            os.system("echo  \'auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[1]), str(result[2])))
            os.system("echo  \'tpr: {}, fnr: {}, tnr: {}, fpr: {}\'".format(str(tpr), str(fnr), str(tnr), str(fpr)))

            # self.test_results.append("seed_case:{} -- auc: {}, apr: {}, f1_score: {}".format(str(self.args.seed), str(result[0]), str(result[1]), str(result[2])))
            self.test_results = list([[self.args.seed, result[0], result[1], result[2]], tpr, tnr])
        else:
            result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = self.evaluator.performance_metric_multi()

            multi_weighted_auc = result[0]
            multi_unweighted_auc = result[1]
            multi_weighted_apr = result[2]
            multi_unweighted_apr = result[3]
            multi_weighted_f1_score = result[4]
            multi_unweighted_f1_score = result[5]

            os.system("echo  \'##### Test results #####\'")
            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[2]), str(result[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[1]), str(result[3]), str(result[5])))
            os.system("echo  \'##### Each class Validation results #####\'")
            seizure_list = self.args.diseases_to_train
            results = []
            results.append("Label:bckg auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(
                                                                                str(result_aucs[0]), str(result_aprs[0]), str(result_f1scores[0]), 
                                                                                str(tprs[0]), str(fnrs[0]), str(tnrs[0]), str(fprs[0]), str(fdrs[0]), str(ppvs[0])))
            for idx, seizure in enumerate(seizure_list):
                results.append("Label:{} auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(seizure,
                                                                                str(result_aucs[idx+1]), str(result_aprs[idx+1]), str(result_f1scores[idx+1]), 
                                                                                str(tprs[idx+1]), str(fnrs[idx+1]), str(tnrs[idx+1]), str(fprs[idx+1]), 
                                                                                str(fdrs[idx+1]), str(ppvs[idx+1])))

            for i in results:
                os.system("echo  \'{}\'".format(i))


        