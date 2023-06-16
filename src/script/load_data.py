import pandas as pd
from typing import Tuple
import numpy as np
import logging
import itertools
class DataBasic:
    def __init__(self, datan:str) -> None:
        """
        Initialize a LoadData object.

        Args:
        - `datan`: the name of the dataset to be loaded
        - "eval_method: 
            NI20: no interation rating set to 0
            rated: only recommend interacted items in dataset

        Returns:
        - None
        """

        self.datan = datan

        self.eval_method = "NI20"

        if self.datan == "ml-1m":
            self.datapath = "data/ml-1m/ratings.dat"
        elif self.datan == "ml-100k":
            self.datapath = "data/ml-100k/u.data"
        elif self.datan == "KuaiRec":
            self.datapath = "data/KuaiRec_2.0/data/small_matrix.csv"
        elif self.datan == "ADS16":
            self.datapath = "data/ADS16/data.csv"
        elif self.datan == "EachMovie":
            self.datapath = "data/EachMovie/eachmovie_triple"

class LoadData(DataBasic):
    def __init__(self, datan: str, **kwargs) -> None:
        """
        This class is used to load and process data from a given dataset.

        """
        super().__init__(datan)

        if self.datan in ["ml-1m", "ml-100k"]:
            
            if self.datan == "ml-1m":
                df = pd.read_csv(self.datapath, delimiter = '::', names = ['user_id', 'item_id', 'rating', 'time'], engine='python')
            elif self.datan == "ml-100k":
                df = pd.read_csv(self.datapath, delimiter = '\t', names = ['user_id', 'item_id', 'rating', 'time'])

            df["feedback01"] = np.where(df['rating'] >= 4, 1, 0)
            df = df.sort_values('time')

            self.df = df

            grouped_data = df.groupby('user_id')
            self.exposure = {user_id: list(grouped_data.get_group(user_id)['item_id']) for user_id in grouped_data.groups} 

            self.exposure_satisﬁed = {user_id: (grouped_data.get_group(user_id)['rating'] >= 4).sum() for user_id in grouped_data.groups} 


        elif self.datan == "KuaiRec":
            df = pd.read_csv(self.datapath)
            self.seed = kwargs.get("seed", 123)
            # df = df.sample(frac = 1, random_state = seed)

            df['rating'] = np.where(df['watch_ratio'] >= 2, 1, 0)
            df["feedback01"] = df['rating']
            df["item_id"] = df["video_id"]

            df = df[['user_id', 'item_id', 'rating', 'time', "feedback01"]]
            df = df.sort_values('time')

            self.df = df

            grouped_data = df.groupby('user_id')
            
            self.exposure = {user_id: list(grouped_data.get_group(user_id)['item_id']) for user_id in grouped_data.groups} 

            self.exposure_satisﬁed = {user_id: (grouped_data.get_group(user_id)['rating'] >= 1).sum() for user_id in grouped_data.groups} 

        elif self.datan == "EachMovie":
            df = pd.read_csv(self.datapath, sep="   ", engine='python', header=None, names=["item_id", "user_id","rating"], dtype=int)
            self.seed = kwargs.get("seed", 123)
            
            df["feedback01"] = np.where(df['rating'] >= 4, 1, 0)
            df = df[['user_id', 'item_id', 'rating', "feedback01"]]
            
            self.df = df
            grouped_data = df.groupby('user_id')
            
            self.exposure = {user_id: list(grouped_data.get_group(user_id)['item_id']) for user_id in grouped_data.groups} 

            self.exposure_satisﬁed = {user_id: (grouped_data.get_group(user_id)['rating'] >= 1).sum() for user_id in grouped_data.groups} 


        # elif self.datan == "ADS16":
        #     df = pd.read_csv(self.datapath) 
        #     seed = kwargs.get("seed", 123)
        #     df = df.sample(frac = 1, random_state = seed)
        #     df["feedback01"] = np.where(df['rating'] >= 4, 1, 0)

        #     self.df = df

        #     grouped_data = df.groupby('user_id')
        #     self.exposure = {user_id: list(grouped_data.get_group(user_id)['item_id']) for user_id in grouped_data.groups} 

        #     self.exposure_satisﬁed = {user_id: (grouped_data.get_group(user_id)['rating'] >= 4).sum() for user_id in grouped_data.groups} 

                    
        else:
            assert False

        self.grouped_data = grouped_data
        self.item_id_set = set(df["item_id"].drop_duplicates())
        self.user_id_set = set(df["user_id"].drop_duplicates())

             
    
    def divide_data(self, percentage: float, task = "test"):
        """
        Args:
        - `percentage`: the percentage of rows to include in the pre-training part
        """

        num_rows = round(percentage * self.df.shape[0])

        self.pre_training_data = self.df.iloc[:num_rows]
        self.online_data = self.df.iloc[num_rows:]

        self.pre_training_user_id = self.pre_training_data["user_id"].drop_duplicates()

        print(f"task = {task}")

        if self.datan in ["ml-1m", "ml-100k", "ADS16", "EachMovie"]:

            if task == "coldstart": 
                self.online_data = self.online_data[self.online_data["user_id"].apply(lambda x: x not in list(self.pre_training_user_id))]

            counts = self.online_data.groupby("user_id")["rating"].apply(lambda x: (x >= 0).sum())            
            counts = counts.sort_values()
            mask = counts < 120
            mask.iloc[:-200] = True # only test 200 user

            self.online_data = self.online_data[~mask[self.online_data["user_id"]].values] # Use the mask to filter out rows with users that have fewer than 120 rows with rating >= 4

            self.online_id = list(self.online_data["user_id"])
            self.online_id_stat = list(self.online_data["user_id"].drop_duplicates())

            self.online_id = [self.online_id_stat for i in range(200)]
            self.online_id = list(itertools.chain.from_iterable(self.online_id))

            if task == "coldstart":
                self.pre_training_data = self.pre_training_data[self.pre_training_data["user_id"].apply(lambda x: x not in self.online_id_stat)]
            elif task == "warmstart":
                pass
            else:
                assert False 

            
        elif self.datan in ["KuaiRec"]:    
            uids = pd.Series(list(self.user_id_set))
            self.online_id_stat = list(uids.sample(200, random_state = self.seed))
            self.online_id = [self.online_id_stat for i in range(200)]
            self.online_id = list(itertools.chain.from_iterable(self.online_id))

            if task == "coldstart":
                self.pre_training_data = self.pre_training_data[self.pre_training_data["user_id"].apply(lambda x: x not in self.online_id_stat)]
            elif task == "warmstart":
                pass
            else:
                assert False 

        else:
            assert False




    def feedback(self, uid, iid):
        """
            return (rating, 01feedback)
        """
        if self.datan in ["ml-1m", "ml-100k", "ADS16", "EachMovie"]:
            df = self.df

            # s = df.loc[(df['user_id'] == uid) & (df['item_id'] == iid), 'rating']
            user_df = self.grouped_data.get_group(uid)
            s = user_df.loc[(user_df["item_id"] == iid), "rating"]
            if len(s) == 0:
                return 0,0
            else:
                rating = int(s)
                feedback_ = 0 if rating <= 3 else 1
                return rating, feedback_ 
       
        elif self.datan in ["KuaiRec"]:
            df = self.df

            # s = df.loc[(df['user_id'] == uid) & (df['item_id'] == iid), 'rating']
            # s = df.query("user_id == @uid and item_id == @item_id")["rating"]
            user_df = self.grouped_data.get_group(uid)
            s = user_df.loc[(user_df["item_id"] == iid), "rating"]
            if len(s) == 0:
                return 0,0
            else:
                rating = int(s)
                return rating, rating

        else:
            assert False
