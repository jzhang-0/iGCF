#! python
import numpy as np
import torch
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import yaml
def print_namespace(args):
    print(yaml.dump(args.__dict__))


import pickle

def save_class(cls, savepath):
    with open(savepath, "wb") as f:
        pickle.dump(cls, f)

def load_class(file):
    with open(file, "rb") as f:
        return pickle.load(f)
