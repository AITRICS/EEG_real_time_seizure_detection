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

import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from control.config import args
import matplotlib.pyplot as plt

from builder.utils.nn_calibration import *
from builder.utils.nn_calibration import _ECELoss


def sliding_window_v1(args, iteration, train_x, train_y, seq_lengths, target_lengths, model, logger, device, scheduler=None, optimizer=None, criterion=None, signal_name_list=None, flow_type="train"):
    # target_lengths_tensor = torch.Tensor(target_lengths)
    target_lengths_tensor = torch.Tensor(target_lengths) -2
    train_x = train_x.permute(1, 0, 2)
    train_y = train_y.permute(1, 0)
    iter_loss = []
    val_loss = []

    answer_list = []
    prediction_list = []
    
    requirement_target1 = args.window_shift_label//2
    requirement_target2 = args.window_size_label//4

    if args.requirement_target is not None:
        requirement_target = int(args.requirement_target * args.feature_sample_rate)
    elif requirement_target1 >= requirement_target2:
        requirement_target = requirement_target1
    else:
        requirement_target = requirement_target2
    sincnet_extrawinsize=(args.sincnet_kernel_size -1)//2

    model.init_state(device)
    shift_num = math.ceil((train_y.shape[0]-args.window_size_label)/float(args.window_shift_label))

    if (args.enc_model == "sincnet") or (args.enc_model == "lfcc"):
        shift_start = 1
        shift_num -= 1
    else:
        shift_start = 0

    for i in range(shift_start, shift_num):
        x_idx = i * args.window_shift_sig
        y_idx = i * args.window_shift_label

        # seq_over = torch.ge(target_lengths_tensor, ((torch.tensor([y_idx+args.window_size_label-1])).repeat(args.batch_size)))
        # loss_zeros = torch.zeros(args.batch_size)

        if args.enc_model == "sincnet":
            slice_start = x_idx - sincnet_extrawinsize
            slice_end = x_idx + args.window_size_sig + sincnet_extrawinsize

        elif args.enc_model == "lfcc":
            slice_start = x_idx - args.window_shift_sig
            slice_end = x_idx + (2 * args.window_size_sig)

        else:
            slice_start = x_idx
            slice_end = x_idx+args.window_size_sig
        
        seq_slice = train_x[slice_start:slice_end].permute(1, 0, 2)
        # print("seq_slice: ", seq_slice[:,1,:10])
        train_y = train_y.type(torch.FloatTensor)
        target_temp = train_y[y_idx: y_idx+args.window_size_label]

        target, _ = torch.max(target_temp, 0)
        seiz_count = torch.count_nonzero(target_temp, dim=0)

        ### here we need to add the 0->1, 1->0 then seizure labelling process
        target[seiz_count < requirement_target] = 0

        final_target = torch.round(target).type(torch.LongTensor).squeeze()

        if flow_type == "train":
            optimizer.zero_grad()

        logits, maps = model(seq_slice)
        logits = logits.type(torch.FloatTensor)
        
        if flow_type == "train":
            loss = criterion(logits, final_target)
            if args.loss_decision == "max_division":
                loss = torch.div(torch.sum(loss), args.batch_size)
            elif args.loss_decision == "mean":
                loss = torch.mean(loss)
            else:
                print("Error! Select Correct args.loss_decision...")
                exit(1)
            
            iter_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step(iteration)    
            logger.log_lr(scheduler.get_lr()[0], iteration)

        else:
            
            proba = nn.functional.softmax(logits, dim=1)
            if args.batch_size == 1:
                final_target = final_target.unsqueeze(0)
            loss = criterion(logits, final_target)
            loss = torch.mean(loss)
            val_loss.append(loss.item())
            logger.evaluator.add_seizure_info(list(signal_name_list))

            if args.binary_target_groups == 1:
                final_target[final_target != 0] = 1
                re_proba = torch.cat((proba[:,0].unsqueeze(1), torch.sum(proba[:,1:], 1).unsqueeze(1)), 1)
                logger.evaluator.add_batch(np.array(final_target.cpu()), np.array(re_proba.cpu()))
            else:
                logger.evaluator.add_batch(np.array(final_target.cpu()), np.array(proba.cpu()))

            if args.margin_test:
                probability = proba[:, 1]
                logger.evaluator.probability_list.append(probability)
                logger.evaluator.final_target_list.append(final_target)
    
    if flow_type == "train":    
        return model, iter_loss
    else:
        return model, val_loss


