# cd **/src

import yaml

model_list = [] # version

other_dir = ["Base"]
# other_dir = ["Base", "LinearBase", "NeuralBase"] 

dir_list= model_list + other_dir

import os

def add_dir(dirn):
    mkdir_model(dirn)

    with open("model/__init__.py", "a") as fp:
        fp.write(f"from .{dirn}_dir import *\n")

def add_model(mn):
    add_dir(mn)
    # with open("script/args_config.py", "a") as fa:
    #     fa.write(add_code_args_config(mn))
    
#     with open("script/run_"+ mn + ".py", "a") as fa:
#         fa.write("""
# from pack_import import *
#             """)


# def add_code_args_config(mn):
#     text = f"""
# def parse_args_{mn}(parser):
#     parser = parse_default_config(parser)

#     parser.add_argument('--modelname', default="{mn}", type=str, help='the name of model')
#     return parser    
#     """
#     return text

def mkdir(dirn):
    if not os.path.exists(dirn):
        os.mkdir(dirn)
        print(f"make directory {dirn}")
    
mkdir("model")


def mkdir_model(dirn):
    # dirn_low = dirn.lower()
    path = "model/" + f"{dirn}" + "_dir/"
    mkdir(path)

    filen = ["__init__.py", f"{dirn}_f.py"]
    # for fn in filen:
    with open(path + "__init__.py", "a") as fa:
        fa.write(f"""
from .{dirn}_f import *
        """) 
    
    with open(path + f"{dirn}_f.py", "a") as fa:
        fa.write(f"""
class {dirn}:
    pass
        """)

def mkdir_models(dir_list):
    path = "model/"
    for dir in dir_list:
        mkdir_model(dir)

        with open(path + "__init__.py", "a") as f:
            f.write(f"""
from .{dir}_dir import *            
            """)
    

def mkdir_script(model_list):
    mkdir("script")

    path = "script/"
    mkdir(path + "archive")

    filen = ["args_config.py", "load_data.py", "s_utils.py", "pack_import.py", "main.py"]

    for fn in filen:
        with open(path + fn, "a") as fa:
            pass 

    with open(path + "pack_import.py", "a") as fa:
        fa.write(f"""#! python
import sys
sys.path.append("src")
import numpy as np
from args_config import *
from load_data import *
from s_utils import *
        """)
    
    with open(path + "args_config.py", "a") as fa:
        fa.write("""
import torch,os

def parse_default_config(parser):
    parser.add_argument('--seed', default='123', type=int, help='random seed')         
    return parser

def parser_config(parser):
    parse_default_config(parser)

    parser.add_argument('--exp_id', default='test', type=str, help='exp id')     
    parser.add_argument('--datan', default='-1', type=str, help='the name of dataset')
    parser.add_argument('-m', '--modelname', default="-1", type=str, help='the name of model')

    return parser

def config_args(args):
    if args.cuda == -1:
        args.device = "cpu"
    else:
        args.device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        
    return args

        """)
    
    with open(path + "s_utils.py", "a") as fa:
        fa.write(f"""#! python
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

from mypackages.decorator import timefn
        """)

    with open(path + "main.py", "a") as fa:
        fa.write("""
import argparse
import sys
sys.path.append("src")
from load_data import LoadData
from args_config import parser_config, config_args
from model import *
from s_utils import same_seeds, print_namespace
import torch.optim as optim

parser = argparse.ArgumentParser(description='Initialize Parameters!')
parser = parser_config(parser)
args = parser.parse_args()
args = config_args(args)

para_dict = args.__dict__
modeln = args.modelname
same_seeds(args.seed)
dls = LoadData(args.datan)

## ex config
para_dict["argv"] = sys.argv[1:]

## main
def main():

    if modeln == "":
        pass
        # bandit_model = NeuralDropout(network, para_dict)
        # optimizer = optim.SGD(bandit_model.network.parameters(), lr = args.lr) 
        
        # bandit_model.run(dls, optimizer = optimizer)
    
    # elif modeln == "LinearTS":
    #     bandit_model = LinearTS(para_dict)
    #     bandit_model.run(dls)
    
    else:
        parser.print_help()
        assert False


if __name__ == "__main__":
    main()
        """)


#     for mn in model_list:
#         with open(path +"run_"+ mn + ".py", "a") as fa:
#             fa.write("""
# from pack_import import *
#             """)

        # with open(path + "args_config.py", "a") as fa:
        #     fa.write(add_code_args_config(mn))

if __name__ == "__main__":
    mkdir_models(dir_list)
    mkdir_script(model_list)

