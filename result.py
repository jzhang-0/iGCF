
import matplotlib.pyplot as plt
import numpy as np
import re

import pandas as pd
# import dataframe_image as dfi

p5 = re.compile("Precsion@5:[0-9.]+")
p10 = re.compile("Precsion@10:[0-9.]+")
p20 = re.compile("Precsion@20:[0-9.]+")
p40 = re.compile("Precsion@20:[0-9.]+")

def pt(t):
    return re.compile(f"Precsion@{t}:[0-9.]+")

def pt_r(t):
    return re.compile(f"Recall@{t}:[0-9.]+")

import pandas as pd


def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]


def get_result(model_list, path_list):
    T_list = [5, 10, 20, 40]
    # model_list = ["Random", "Pop", "ICF-UCB-0", "ICF-UCB-0.1", "ICF-UCB-1", "ICF-UCB-5"]
    df = pd.DataFrame(columns=[f"P@{T}" for T in T_list] + [f"R@{T}" for T in T_list], index=model_list)

    # path_list = [
    #     "/amax/home/zhangjin/ICF/Exp/Random/ml-100k/1_2022-12-19--14:23:55.481601/.log",
    #     "/amax/home/zhangjin/ICF/Exp/Pop/ml-100k/1_2022-12-19--14:28:40.540583/.log",
    #     "/amax/home/zhangjin/ICF/Exp/ICF/ml-100k/1_2022-12-19--14:51:53.444546/.log", #0
    #     "/amax/home/zhangjin/ICF/Exp/ICF/ml-100k/1_2022-12-19--14:44:28.439034/.log", # 0.1
    #     "/amax/home/zhangjin/ICF/Exp/ICF/ml-100k/1_2022-12-19--14:40:13.300487/.log", #1
    #     "/amax/home/zhangjin/ICF/Exp/ICF/ml-100k/1_2022-12-19--14:44:28.351012/.log", #5
    # ]


    for model,path in zip(model_list,path_list):
        with open(path) as f:
            s = f.read()
            # a5 = p5.findall(s)
            # p20 = float(a5[0].split(":")[1])

            for T in T_list:
                df.loc[model, f"P@{T}"] = float(pt(T).findall(s)[0].split(":")[1]) 
                df.loc[model, f"R@{T}"] = float(pt_r(T).findall(s)[0].split(":")[1]) 


    style = df.style.apply(highlight_max).set_caption("dim 64")
    # dfi.export(style, 'result.png')


# df = pd.DataFrame(np.random.rand(6,4))
# df_styled = df.style.background_gradient()

