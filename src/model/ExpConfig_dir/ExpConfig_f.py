import os,sys,json
import logging
import numpy as np
import shutil,datetime
import torch
import pandas as pd
import pickle

class ExpConfig:
    """
        Implement file read and write, record experimental data.
    """
    def __init__(self, para_dict) -> None:
        
        datan = para_dict["datan"]
        modeln = para_dict["modelname"]
        exp_id = para_dict["exp_id"]
        
        now = datetime.datetime.now()
        
        time_info = str(now).replace(" ","--")
        try_time = 1
        # self.exp_path = f"Exp/{modeln}/{datan}/{exp_id}_{time_info}/"
        self.exp_path = f"Exp/{datan}/{modeln}/{exp_id}_{try_time}/"

        while os.path.exists(self.exp_path):
            try_time += 1
            self.exp_path = f"Exp/{datan}/{modeln}/{exp_id}_{try_time}/"


        self.save_path = os.path.join(self.exp_path, "save")
        self.code_path = os.path.join(self.exp_path, "code") 
        self.log_path = f"{self.exp_path}.log"


        for p in [self.exp_path, self.save_path]:
            if not os.path.exists(p):
                os.makedirs(p)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.print = logging.info
        
        self.print(f"model:{modeln}")
        self.print(json.dumps(para_dict, indent=4))

        # save code
        shutil.copytree("src", f"{self.code_path}/src")
        argv_l = para_dict.get("argv", [0])
        s = "#! /bin/bash\npython src/script/main.py "
        for arg in argv_l:
            s = s + arg + " "

        with open(f"{self.code_path}/run.sh", "w") as f:
            f.write(s)


    def save_cls(self, cls, name):
        save_class(cls, f"{self.save_path}/{name}")


    def save_data(self, filen, data, savepath, **kwargs):
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        path = os.path.join(savepath, filen)
        np.save(path, data)


def save_class(cls, savepath):
    with open(savepath, "wb") as f:
        pickle.dump(cls, f)

def load_class(file):
    with open(file, "rb") as f:
        return pickle.load(f)

        