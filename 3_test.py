# Copyright (c) 2022, Kwanhyung Lee, Hyewon Jeong, Seyun Kim AITRICS. All rights reserved.
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
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import random
import time

import torch
from torch.autograd import Variable
import torch.nn as nn

from control.config import args
from builder.data.data_preprocess import get_data_preprocessed
from builder.models import get_detector_model, get_multiclassification_model, grad_cam, get_augmentation
from builder.utils.metrics import Evaluator
from builder.utils.logger import Logger
from builder.trainer.trainer import *
from builder.utils.utils import set_seeds, set_devices

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# test_mode
label_method_max = True
scheduler = None
optimizer = None
criterion = nn.CrossEntropyLoss(reduction='none')
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
print("args: ", args)
# Get Dataloader, Model
train_loader, val_loader, test_loader, len_train_dir, len_val_dir, len_test_dir = get_data_preprocessed(args)
model = get_detector_model(args) 

model = model(args, device).to(device)
evaluator = Evaluator(args)
names = [args.project_name]
average_speed_over = 10
time_taken = 0
num_windows = 30 - args.window_size

for name in names: 
    # Check if checkpoint exists
    if args.last:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best_{}.pth'.format(str(args.seed))
    elif args.best:
        ckpt_path = args.dir_result + '/' + name + '/ckpts/best_{}.pth'.format(str(args.seed))
    # if not os.path.exists(ckpt_path):
    #     continue

    ckpt_path = args.dir_result + '/' + name + '/ckpts/best_0.pth'
    ckpt = torch.load(ckpt_path, map_location=device)

    # state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    state = {k: v for k, v in ckpt['model'].items()}
    # print("state: ", state)
    model.load_state_dict(state)

    model.eval()
    print('loaded model')
    print("Test type is: ", args.test_type)
    evaluator.reset()
    result_list = []
    iteration = 0
    evaluator.seizure_wise_eval_for_binary = True

    with torch.no_grad():
        for test_batch in tqdm(test_loader, total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            test_x, test_y, seq_lengths, target_lengths, aug_list, signal_name_list = test_batch
            test_x = test_x.to(device)
            iteration += 1
            
            ### Model Structures
            print(f'iteration : {iteration}')
            iteration_start = time.time()
            if args.task_type == "binary": 
                model, _ = sliding_window_v1(args, iteration, test_x, test_y, seq_lengths, 
                                            target_lengths, model, logger, device, scheduler,
                                            optimizer, criterion, signal_name_list=signal_name_list, flow_type="test")    # margin_test , test

            else:
                print("Selected trainer is not prepared yet...")
                exit(1)
            
            if not args.ignore_model_speed:
                iteration_end = time.time()
                print("1: ", num_windows)
                print("used device: ", device)
                print("the number of cpu threads: {}".format(torch.get_num_threads()))

                print(f'Time taken to iterate once :    {(iteration_end-iteration_start)} seconds')
                print(f'Time taken per window slide :    {(iteration_end-iteration_start)/num_windows} seconds')
                exit(1)

    # print(f'Time taken to iterate once :    {(iteration_end-iteration_start)} seconds')
    # print(f'Time taken per window slide :    {(iteration_end-iteration_start)/num_windows} seconds')
    logger.test_result_only()


          