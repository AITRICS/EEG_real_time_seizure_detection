import os
import random
import numpy as np

import torch

def set_seeds(args):
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

def set_devices(args):
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	if args.cpu or not torch.cuda.is_available():
		return torch.device('cpu')
	else:
		return torch.device('cuda')

def search_walk(info):
    searched_list = []
    root_path = info.get('path')
    extension = info.get('extension')

    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = ('%s/%s' % (path, filename))
                searched_list.append(list_file)

    if searched_list:
        return searched_list
    else:
        return False
