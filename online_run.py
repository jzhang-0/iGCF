import sys
sys.path.append("src")
sys.path.append("src/script/")
import numpy as np
import argparse
from model import *
import torch.optim as optim
import pickle

parser = argparse.ArgumentParser(description='Initialize Parameters!')
parser.add_argument("--v",default='1', type = float, help='explore v')
parser.add_argument("--eid",default='0', type = str, help='eid')
args = parser.parse_args()

path = "Exp/ml-100k/GCNICF/F4_--K_4_1/save/"


v = args.v
eid = args.eid

def get_log_path(model):
    i = 1

    pl = f"online_{eid}_v_{v}_{i}.log"
    while os.path.exists(os.path.join(model.ec.exp_path, pl)):
        i += 1
    path = os.path.join(model.ec.exp_path, pl)
    print(path)
    return path

threads_num = "6"

os.environ["OMP_NUM_THREADS"] = threads_num 
os.environ["OPENBLAS_NUM_THREADS"] = threads_num 
os.environ["MKL_NUM_THREADS"] = threads_num 
os.environ["VECLIB_MAXIMUM_THREADS"] = threads_num 
os.environ["NUMEXPR_NUM_THREADS"] = threads_num 






dls,model,ui_cls = None, None,None

dls = load_class(path + "dls.class")
model = load_class(path + "model.class")
ui_cls = load_class(path + "ui_cls.class")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(get_log_path(model)),
        logging.StreamHandler(sys.stdout)
    ]
)
model.print = logging.info
ui_cls.print = logging.info

model.explore_v = v
model.online(ui_cls, data_cls=dls)
ui_cls.evaluate(dls)



