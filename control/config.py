import os
import yaml
import argparse
import torch

seed_list = [0, 1004, 911, 2021, 119]

### CONFIGURATIONS
parser = argparse.ArgumentParser()

# General Parameters
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--seed-list', type=list, default=[0])
# parser.add_argument('--seed-list', type=list, default=[10])
parser.add_argument('--device', type=int, default=1, nargs='+')
parser.add_argument('--cpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--project-name', type=str, default="test")
parser.add_argument('--checkpoint', '-cp', type=bool, default=False)

# Training Parameters e
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--l2-coeff', type=float, default=0.002)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--activation', help='activation function of the networks', choices=['selu','relu'], default='relu', type=str) #invase
parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'sgd_lars','adam', 'adam_lars','adamw', 'adamw_lars'])
parser.add_argument('--lr-scheduler', type=str, default="Single" , choices=["CosineAnnealing", "Single"])
parser.add_argument('--lr_init', type=float, default=1e-3) # not being used for CosineAnnealingWarmUpRestarts...
parser.add_argument('--lr_max', type=float, default=4e-3)
parser.add_argument('--t_0', '-tz', type=int, default=5, help='T_0 of cosine annealing scheduler')
parser.add_argument('--t_mult', '-tm', type=int, default=2, help='T_mult of cosine annealing scheduler')
parser.add_argument('--t_up', '-tup', type=int, default=1, help='T_up (warm up epochs) of cosine annealing scheduler')
parser.add_argument('--gamma', '-gam', type=float, default=0.5, help='T_up (warm up epochs) of cosine annealing scheduler')
parser.add_argument('--momentum', '-mo', type=float, default=0.9, help='Momentum of optimizer')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6, help='Weight decay of optimizer')
parser.add_argument('--loss-decision', type=str, default='max_division', choices=['mean', 'max_division'])

# Data Parameters
parser.add_argument('--val-data-ratio', type=float, default=0.2)
parser.add_argument('--test-data-ratio', type=float, default=0.2)
parser.add_argument('--normalization', type=str, default=None)

parser.add_argument('--val_data_ratio', type=float, default=0.2)
parser.add_argument('--test_data_ratio', type=float, default=0.2)
parser.add_argument('--window-size', type=int, default=1, help='unit is second')
parser.add_argument('--window-shift', type=int, default=1, help='unit is second')
parser.add_argument('--requirement-target', type=int, default=None, help='unit is second')

# Augment Option: --augmentation = augmentation right on raw-signal data, --spec-augmentation = augmentation after PSD feature
parser.add_argument('--spec-augmentation', type=bool, default=False)
parser.add_argument('--freq-mask-para', type=int, default=2) 
parser.add_argument('--time-mask-num', type=int, default=10)
parser.add_argument('--freq-mask-num', type=int, default=1)

parser.add_argument('--augmentation', type=bool, default=False)
parser.add_argument('--amplitude-min', type=float, default=0.5)
parser.add_argument('--amplitude-max', type=int, default=2)
parser.add_argument('--time-shift-min', type=int, default=-50, help='number of samples')
parser.add_argument('--time-shift-max', type=int, default=50, help='number of samples')
parser.add_argument('--DC-shift-min', type=int, default=-10)
parser.add_argument('--DC-shift-max', type=int, default=10)
parser.add_argument('--zero-masking-min', type=int, default=0)
parser.add_argument('--zero-masking-max', type=int, default=150)
parser.add_argument('--additive-gaussian-noise-min', type=float, default=0)
parser.add_argument('--additive-gaussian-noise-max', type=float, default=0.2)
parser.add_argument('--band-stop-filter-min', type=float, default=2.8)
parser.add_argument('--band-stop-filter-max', type=float, default=82.5)

# Model Parameters
# parser.add_argument('--trainer', type=str, default="binary_detector", choices=['binary_detector', 'multi_classification', 'prediction']) #structure name
parser.add_argument('--model', type=str, default="cnn2d_lstm_v1") #model name
parser.add_argument('--hyperopt-model-name', type=str, default="xgboost_classification") #model name

parser.add_argument('--enc-model', type=str, default="sincnet", choices= ['stft1', 'stft2', 'psd1', 'psd2', 'sincnet', 'raw', 'saliency', 'LFCC', 'downsampled'])
parser.add_argument('--sincnet-bandnum', type=int, default=20)
parser.add_argument('--sincnet-kernel-size', type=int, default=81, help="max is 101")
parser.add_argument('--sincnet-input-normalize', type=str, default="none", choices=["none","layernorm","batchnorm"])
parser.add_argument('--sincnet-layer-num', type=int, default=1, help="select int between 1 ~ 3")
parser.add_argument('--sincnet-stride', type=int, default=2)

# Architecture Parameters
parser.add_argument('--num-layers', type=float, default=2)
parser.add_argument('--hidden-dim', type=float, default=512)
parser.add_argument('--att-dim', type=int, default=25)
parser.add_argument('--cnn-maxpool', type=int, default=4)
parser.add_argument('--residual-block-type', type=str, default="standard", choices=["standard", "inverted_bottleneck"])
parser.add_argument('--block-reinforcement-methods', type=str, default=None, choices=["cbam", None])
parser.add_argument('--block-temporal-methods', type=str, default="stand_alone", choices=["time", "stand_alone"])
parser.add_argument('--extra-cnn-block-num', type=int, default=0)
parser.add_argument('--multi-head-num', type=int, default=4)
parser.add_argument('--self-att-layers-n', type=int, default=2)
parser.add_argument('--cross-attention', type=bool, default=True)
parser.add_argument('--lstm', type=bool, default=True)

# Metric learning
parser.add_argument('--centerloss', type=bool, default=False)
parser.add_argument('--centerloss-weight', type=float, default=0.3)

# Visualize / Logging Parameters
parser.add_argument('--log-iter', type=int, default=10)
parser.add_argument('--grad-cam', default=False, action='store_true')

# Test / Store Parameters
parser.add_argument('--best', default=True, action='store_true')
parser.add_argument('--last', default=False, action='store_true')
parser.add_argument('--test-type', type=str, default="test", choices=["test"])
parser.add_argument('--seizure-wise-eval-for-binary', type=bool, default=False)
parser.add_argument('--margin-test', type=bool, default=False)
parser.add_argument('--localization', type=bool, default=False)
# parser.add_argument('--start-margin', type=list, default=[1,2,3,4,5])
# parser.add_argument('--end-margin', type=list, default=[1,2,3,4,5])
parser.add_argument('--margin-list', type=list, default=[3,5])
# parser.add_argument('--tnr-for-margintest', type=list, default=[1.0, 0.95, 0.9, 0.85, 0.8])
parser.add_argument('--tnr-for-margintest', type=list, default=[0.95])
parser.add_argument('--calibration', type=bool, default=False)

# target groups options
# "1": '0':'bckg', '1':'gnsz', '2':'fnsz', '3':'spsz', '4':'cpsz', '5':'absz', '6':'tnsz', '7':'tcsz', '8':'mysz'
# "2": '0':'bckg', '1':'gnsz_fnsz_spsz_cpsz_absz_tnsz_tcsz_mysz'
# "4": '0':'bckg', '1':'gnsz_absz', '2':'fnsz_spsz_cpsz', '3':'tnsz', '4':'tcsz', '5':'mysz'
parser.add_argument('--binary-target-groups', type=int, default=2, choices= [1, 2, 3])
parser.add_argument('--eeg-type', type=str, default="bipolar", choices=["unipolar", "bipolar", "uni_bipolar"])
parser.add_argument('--task-type', '-tt', type=str, default='binary', choices=['anomaly', 'multiclassification', 'binary', 'binary_noslice'])       

parser.add_argument('--binary-sampler-type', type=str, default="6types", choices=["6types", "30types"])
parser.add_argument('--dev-bckg-num', type=int, default=10)

parser.add_argument('--get-model-summary', type=bool, default=False, help="print model summary before training")

args = parser.parse_args()
args.cnn_channel_sizes = [args.sincnet_bandnum, 10, 10]

if args.task_type == "anomaly":
    args.seiz_classes = ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz']
    args.num_to_seizure = {'0':'patient_nor', '1':'non_patient_nor', '2':'gnsz', '3':'fnsz', '4':'spsz', '5':'cpsz', '6':'absz', '7':'tnsz', '8':'tcsz', '9':'mysz'}
    args.label_group = "pre_pre-ict_label"
    args.output_dim = len(args.seiz_classes)

elif args.task_type == "multiclassification":
    args.seiz_classes = ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz']
    args.seiz_num = ['1', '2', '3', '4', '5', '6', '7', '8'] # exclude mysz
    args.num_to_seizure = {'0':'gnsz', '1':'fnsz', '2':'spsz', '3':'cpsz', '4':'absz', '5':'tnsz', '6':'tcsz', '7':'mysz'}
    args.label_group = "LABEL"
    args.output_dim = len(args.seiz_num)

elif args.task_type == "binary" or args.task_type == "binary_noslice":
    # args.seiz_classes = ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz']
    args.seiz_classes = ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz']
    # args.seizure_to_num = {'gnsz':'1', 'fnsz':'2', 'spsz':'3', 'cpsz':'4', 'absz':'5', 'tnsz':'6', 'tcsz':'7', 'mysz':'8'}
    args.seizure_to_num = {'gnsz':'1', 'fnsz':'2', 'spsz':'3', 'cpsz':'4', 'absz':'5', 'tnsz':'6', 'tcsz':'7'}
    # args.seizure_to_num_inv = {'1':'gnsz', '2':'fnsz', '3':'spsz', '4':'cpsz', '5':'absz', '6':'tnsz', '7':'tcsz', '8':'mysz'}
    args.seizure_to_num_inv = {'1':'gnsz', '2':'fnsz', '3':'spsz', '4':'cpsz', '5':'absz', '6':'tnsz', '7':'tcsz'}
    
    if args.binary_target_groups == 1:
        args.label_group = "LABEL1"
        args.output_dim = 8
        args.num_to_seizure = {'1':'gnsz', '2':'fnsz', '3':'spsz', '4':'cpsz', '5':'absz', '6':'tnsz', '7':'tcsz'}
    
    elif args.binary_target_groups == 2:
        args.label_group = "LABEL2"
        args.output_dim = 2
        # args.num_to_seizure = {'1':'gnsz_fnsz_spsz_cpsz_absz_tnsz_tcsz_mysz'}
        args.num_to_seizure = {'1':'gnsz_fnsz_spsz_cpsz_absz_tnsz_tcsz'}

    elif args.binary_target_groups == 3:
        args.label_group = "LABEL3"
        args.output_dim = 6
        args.num_to_seizure = {'1':'gnsz_absz', '2':'fnsz_spsz_cpsz', '3':'tnsz', '4':'tcsz', '5':'mysz'}
    else:
        print("Select Correct disease target group...")
        exit(1) 
else:
    print("Select Correct disease target group...")
    exit(1)

args.num_to_seizure_items = [v for k, v in args.num_to_seizure.items()]

# Dataset Path settings
with open('./control/path_configs.yaml') as f:
    path_configs = yaml.safe_load(f)
    args.data_path = path_configs['data_directory']['data_path']
    args.dir_root =  os.getcwd()
    args.dir_result = path_configs['dir_result']
