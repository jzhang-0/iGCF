import os
import multiprocessing
from datetime import datetime
import itertools
import time
import numpy as np
import re

GPU_USE = True
GPU_remove_list = []
threads_num = "4"
MAX_PROCESS = 6

exp_id_prefix = "1"
GPU_USE = False
param_dict = {
    "--datan":["KuaiRec", "ml-1m", "EachMovie"],    # ml-100k  ADS16  KuaiRec ml-1m EachMovie
    "-m":["Pop", "Pos", "Random"],  # LGCNICF LGCNICF_FixG_LTS  LGCNICF_FixG_VI LGCNICF_DynamicG_VI ICF Pop Pos Random MF PosPlus
    "-d":[128],
    "-v":[0.5],
    "-E":["UCB"],
    "-p":[0.5],
    "--max_iter":[20],
    "--online_rec_total_num":[120],
    "--rec_list_len":[3,5],
    "--task":["coldstart"],
    "--lr":[5e-2],
    "--lossfunc": ["reg"],
    "--online_iter":[50], 
}

exp_id_prefix = "F45_1_K"
param_dict = {
    "--datan":["KuaiRec"],    # ml-100k  ADS16  KuaiRec ml-1m EachMovie
    "-m":["GCNICF_Meta_V2"],  # LGCNICF LGCNICF_FixG_LTS  LGCNICF_FixG_VI LGCNICF_DynamicG_VI ICF Pop Pos Random MF PosPlus
    "-d":[128],
    "--lr":[1, 5],
    "-v":[1],
    "-E":["UCB"],
    "-p":[0.5],
    "--max_iter":[40000],
    "--epoch":[20],
    "--K":[0, 1, 2],  # For GCN
    "--save_cls":[0],
    "--lambda_u":[1],
    "--test_iters":[1000],
    
    "--online_rec_total_num":[120],
    "--rec_list_len":[1, 3],
    "--task":["coldstart"],    
    
    "--meta_update":["meta_prior"],

    "--LVI_iters":[3],
    "--online_iter":[50],
    "--lossfunc": ["reg"],
}

cmd_template = "python src/script/main.py {}"

import os

os.environ["OMP_NUM_THREADS"] = threads_num 
os.environ["OPENBLAS_NUM_THREADS"] = threads_num 
os.environ["MKL_NUM_THREADS"] = threads_num 
os.environ["VECLIB_MAXIMUM_THREADS"] = threads_num 
os.environ["NUMEXPR_NUM_THREADS"] = threads_num 
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

now = datetime.now()
s1 = f"{now.year}"[-2:]
s2 = f"{now.month}"
s3 = f"{now.day}"
s = f"{s1}.{s2}.{s3}"

log_path = f"./Exp/bash_script/{s}"
if not os.path.exists(log_path):
    os.makedirs(log_path)

log_name = f"{now.hour}.{now.minute}.{now.second}.log"
with open(f"{log_path}/{log_name}", "w") as f:
    print(param_dict, file=f)

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


expdir_list = []
path_list = []

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()

    key_list = []
    for key,values in kwargs.items():
        if key == "--datan" or key == "-m":
            continue

        if len(values) > 1:
            key_list.append(key)
        
    instance_list = []
    for instance in itertools.product(*vals):
        instance_args = " ".join(["{} {}".format(arg, value) for (arg, value) in zip(keys, instance)])

        parameters = {k:i for k,i in zip(keys,instance)}

        exp_id_suffix = "".join([f"{k}_{parameters[k]}" for k in key_list])
        exp_id = f"{exp_id_prefix}_{exp_id_suffix}"
        instance_args += f" --exp_id {exp_id}" 
        expdir_list.append(f"{exp_id}")
        m = parameters["-m"]
        datan = parameters["--datan"]
        path_list.append(f"Exp/{datan}/{m}/{exp_id}")

        instance_list.append(instance_args)
    return instance_list


import subprocess as sp

def get_gpu_info():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND1 = "nvidia-smi --query-gpu=memory.used --format=csv"
    COMMAND2 = "nvidia-smi --query-gpu=memory.total --format=csv"
    COMMAND3 = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND1.split(),stderr=sp.STDOUT))[1:]
        memory_total_info = output_to_list(sp.check_output(COMMAND2.split(),stderr=sp.STDOUT))[1:]
        gpu_utilization = output_to_list(sp.check_output(COMMAND3.split(),stderr=sp.STDOUT))[1:] 
         
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_info, memory_total_info, gpu_utilization

def available_gpu():
    m_use, m_total, utiliz = get_gpu_info()
    gpu_num = len(m_use)
    gpu_list = []

    utilize_list = []
    for i in range(gpu_num):
        m_percent = float(m_use[i][:-4]) / float(m_total[i][:-4])
        # if m_percent < 0.5 and float(utiliz[i][:-2]) < 50:
        if m_percent < 0.8 and float(utiliz[i][:-2]) < 80:
            gpu_list.append(i)
            utilize_list.append(float(utiliz[i][:-2]))
    ii = (-np.array(utilize_list)).argsort()
    gpu_list = list(np.array(gpu_list)[ii])
    return gpu_list

commands = [cmd_template.format(instance) for instance in product_dict(**param_dict)]
# print('\n'.join(commands))
print("# experiments = {}".format(len(commands)))

if GPU_USE:
    gpu_list = available_gpu()
    
    for i in GPU_remove_list:
        try:
            gpu_list.remove(i)
        except:
            pass
    while len(gpu_list) == 0:
        print(" no vailable gpu, waiting")
        time.sleep(300)
        gpu_list = available_gpu() 

    gpus = multiprocessing.Manager().list(gpu_list)
    proc_to_gpu_map = multiprocessing.Manager().dict()


def exp_runner(cmd):
    process_id = multiprocessing.current_process().name

    if GPU_USE:
        if process_id not in proc_to_gpu_map:
            proc_to_gpu_map[process_id] = gpus.pop()
            print("assign gpu {} to {}".format(proc_to_gpu_map[process_id], process_id))

        # print(cmd + ' -gpu {}'.format(proc_to_gpu_map[process_id]))
        return os.system(cmd + " --cuda {}".format(proc_to_gpu_map[process_id]))
    else:
        print(f"{process_id=}") 
        return os.system(cmd + " --cuda -1") 
if GPU_USE:
    c1 = len(gpus)
else:
    c1 = 100
c2 = MAX_PROCESS
p = multiprocessing.Pool(processes = min(c1,c2))
rets = p.map(exp_runner, commands)
print(rets)

with open(f"{log_path}/{log_name}", "a") as f:
    print(f"{now} complete!", file=f)


print(path_list)