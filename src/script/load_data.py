import pandas as pd
from typing import Tuple
import numpy as np
import scipy as sp
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
                df_item = pd.read_csv('data/ml-1m/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
                df_item['Genres'] = df_item['Genres'].str.split('|')
                self.df_item = df_item

                exploded_data = df_item.explode('Genres')
                exploded_data["GenresIndex"], self.all_Genres  = pd.factorize(exploded_data['Genres'])
                self.all_Genres_num = len(self.all_Genres)
                self.df_genres = exploded_data[['MovieID', 'GenresIndex']].rename(columns={"MovieID":"item_id"})

                self.sim_vec_dict2_sim_ma()


            elif self.datan == "ml-100k":
                df = pd.read_csv(self.datapath, delimiter = '\t', names = ['user_id', 'item_id', 'rating', 'time'])

            df["feedback01"] = np.where(df['rating'] >= 4, 1, 0)
            df = df.sort_values('time')

            self.df = df

            grouped_data = df.groupby('user_id')
            self.exposure = {user_id: list(grouped_data.get_group(user_id)['item_id']) for user_id in grouped_data.groups} 

            self.exposure_satisﬁed = {user_id: (grouped_data.get_group(user_id)['rating'] >= 4).sum() for user_id in grouped_data.groups} 


        elif self.datan == "KuaiRec":

            df_item = pd.read_csv("data/KuaiRec_2.0/data/item_categories.csv")
            # df_item = pd.read_json("data/KuaiRec_2.0/data/item_categories.csv", lines=True)
            df_item['Genres'] = df_item['feat'].apply(lambda x :list(map(int,  x[1:-1].split(","))))
            exploded_data = df_item.explode('Genres')
            exploded_data["GenresIndex"], self.all_Genres  = exploded_data["Genres"], list(set(exploded_data["Genres"]))
            self.all_Genres_num = len(self.all_Genres)
            self.df_genres = exploded_data[['video_id', 'GenresIndex']].rename(columns={"video_id":"item_id"})


            self.sim_vec_dict2_sim_ma()


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

            self.exposure_satisﬁed = {user_id: (grouped_data.get_group(user_id)['rating'] >= 4).sum() for user_id in grouped_data.groups} 


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

    def get_genre_vec(self, item_id):
        sim_vec = np.zeros(self.all_Genres_num)
        gindex = list(self.df_genres[self.df_genres["item_id"] == item_id].GenresIndex)
        sim_vec[gindex] = 1
        return sim_vec

    def sim_vec_dict2_sim_ma(self):
        self.item_sim_vec_dict = {}
        for iid in list(self.df_genres.item_id.drop_duplicates()):
            self.item_sim_vec_dict[iid] = self.get_genre_vec(iid)

        sim_d = self.item_sim_vec_dict
        sim_ma = np.zeros([len(sim_d.keys()), self.all_Genres_num])

        self.sim_id2index_dict = {}
        for index,id in enumerate(list(sim_d.keys())):
            self.sim_id2index_dict[id] = index
            sim_ma[index] = sim_d[id]
        self.sim_ma = sim_ma
        


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
        self.task = task

        if self.datan in ["ml-1m", "ml-100k", "ADS16", "EachMovie"]:

            if task == "coldstart": 
                self.online_data = self.online_data[self.online_data["user_id"].apply(lambda x: x not in list(self.pre_training_user_id))]

                counts = self.online_data.groupby("user_id")["rating"].apply(lambda x: (x >= 0).sum())            
                counts = counts.sort_values()
                mask = counts < 120
                mask.iloc[:-200] = True # only test 200 user
                self.online_data = self.online_data[~mask[self.online_data["user_id"]].values] 

                self.online_id = list(self.online_data["user_id"])
                self.online_id_stat = list(self.online_data["user_id"].drop_duplicates())

                self.online_id = [self.online_id_stat for i in range(200)]
                self.online_id = list(itertools.chain.from_iterable(self.online_id))

            
                self.pre_training_data = self.pre_training_data[self.pre_training_data["user_id"].apply(lambda x: x not in self.online_id_stat)]

            elif task == "warmstart":
                self.pre_training_data["item_sim_index"] = self.pre_training_data["item_id"].apply(lambda x:self.sim_id2index_dict[x])
                self.online_data["item_sim_index"] = self.online_data["item_id"].apply(lambda x:self.sim_id2index_dict[x])
                self.df["item_sim_index"] = self.df["item_id"].apply(lambda x:self.sim_id2index_dict[x])


                counts = self.df.groupby("user_id")["rating"].apply(lambda x: (x >= 0).sum())            
                counts = counts.sort_values()
                mask = counts < 120
                # mask.iloc[:-200] = True # only test 200 user
                data_uid_set = set(self.df[~mask[self.df["user_id"]].values]["user_id"])

                # counts = self.pre_training_data.groupby("user_id")["rating"].apply(lambda x: (x >= 0).sum())            
                # counts = counts.sort_values()
                # mask = counts < 30
                # # mask.iloc[:-200] = True # only test 200 user
                # train_data_uid_set = set(self.pre_training_data[~mask[self.pre_training_data["user_id"]].values]["user_id"]) 

                uid_list = list( data_uid_set)

                sim_df = pd.DataFrame(columns=["sim_score", "id"])
                sim_df["id"] = uid_list

                df_ug = self.df.groupby("user_id")
                # pre_ug = self.pre_training_data.groupby("user_id")
                for uid in uid_list:
                    u_data = df_ug.get_group(uid)
                    sim_vec0 = self.sim_ma[list(u_data.item_sim_index)].sum(0)

                    # pre_data = pre_ug.get_group(uid)
                    # sim_vec1 = self.sim_ma[list(pre_data.item_sim_index)].sum(0)

                    l0 = list(u_data.item_sim_index)
                    l1 = l0[:len(l0) // 2]
                    sim_vec1 = self.sim_ma[l1].sum(0)  
                    sim_vec2 = sim_vec0 - sim_vec1
                                        
                    sim_score = sim_vec1 @ sim_vec2 / (np.linalg.norm(sim_vec1) * np.linalg.norm(sim_vec2))
                    sim_df.sim_score.loc[sim_df.query(f"id == {uid}").index] = sim_score

                self.online_id_stat = list(sim_df.sort_values(by="sim_score")["id"])[:200]
                self.online_id = [self.online_id_stat for i in range(200)]
                self.online_id = list(itertools.chain.from_iterable(self.online_id))

                self.warm_start_user_df_set1_dict = {}
                self.warm_start_user_df_set2_dict = {}
                for uid in self.online_id_stat:
                    u_data = df_ug.get_group(uid)
                    num_ = len(u_data)//2
                    self.warm_start_user_df_set1_dict[uid] = u_data.iloc[:num_]
                    self.warm_start_user_df_set2_dict[uid] = u_data.iloc[num_:]

                self.pre_training_data = self.pre_training_data[self.pre_training_data["user_id"].apply(lambda x: x not in self.online_id_stat)]

            else:
                assert False 

            
        elif self.datan in ["KuaiRec"]:    
            
            if task == "coldstart":
                uids = pd.Series(list(self.user_id_set))
                self.online_id_stat = list(uids.sample(200, random_state = self.seed))
                self.online_id = [self.online_id_stat for i in range(200)]
                self.online_id = list(itertools.chain.from_iterable(self.online_id))
                self.pre_training_data = self.pre_training_data[self.pre_training_data["user_id"].apply(lambda x: x not in self.online_id_stat)]
            
            elif task == "warmstart":
                self.pre_training_data["item_sim_index"] = self.pre_training_data["item_id"].apply(lambda x:self.sim_id2index_dict[x])
                self.online_data["item_sim_index"] = self.online_data["item_id"].apply(lambda x:self.sim_id2index_dict[x])
                self.df["item_sim_index"] = self.df["item_id"].apply(lambda x:self.sim_id2index_dict[x])


                counts = self.df.groupby("user_id")["rating"].apply(lambda x: (x >= 0).sum())            
                counts = counts.sort_values()
                mask = counts < 120
                # mask.iloc[:-200] = True # only test 200 user
                data_uid_set = set(self.df[~mask[self.df["user_id"]].values]["user_id"])

                # counts = self.pre_training_data.groupby("user_id")["rating"].apply(lambda x: (x >= 0).sum())            
                # counts = counts.sort_values()
                # mask = counts < 30
                # # mask.iloc[:-200] = True # only test 200 user
                # train_data_uid_set = set(self.pre_training_data[~mask[self.pre_training_data["user_id"]].values]["user_id"]) 

                uid_list = list( data_uid_set)

                sim_df = pd.DataFrame(columns=["sim_score", "id"])
                sim_df["id"] = uid_list

                df_ug = self.df.groupby("user_id")
                # pre_ug = self.pre_training_data.groupby("user_id")
                for uid in uid_list:
                    u_data = df_ug.get_group(uid)
                    sim_vec0 = self.sim_ma[list(u_data.item_sim_index)].sum(0)

                    # pre_data = pre_ug.get_group(uid)
                    # sim_vec1 = self.sim_ma[list(pre_data.item_sim_index)].sum(0)

                    l0 = list(u_data.item_sim_index)
                    l1 = l0[:len(l0) // 2]
                    sim_vec1 = self.sim_ma[l1].sum(0)                    
                    sim_vec2 = sim_vec0 - sim_vec1
                                        
                    sim_score = sim_vec1 @ sim_vec2 / (np.linalg.norm(sim_vec1) * np.linalg.norm(sim_vec2))
                    sim_df.sim_score.loc[sim_df.query(f"id == {uid}").index] = sim_score

                self.online_id_stat = list(sim_df.sort_values(by="sim_score")["id"])[:200]
                self.online_id = [self.online_id_stat for i in range(200)]
                self.online_id = list(itertools.chain.from_iterable(self.online_id))

                self.warm_start_user_df_set1_dict = {}
                self.warm_start_user_df_set2_dict = {}
                for uid in self.online_id_stat:
                    u_data = df_ug.get_group(uid)
                    num_ = len(u_data)//2
                    self.warm_start_user_df_set1_dict[uid] = u_data.iloc[:num_]
                    self.warm_start_user_df_set2_dict[uid] = u_data.iloc[num_:]

                self.pre_training_data = self.pre_training_data[self.pre_training_data["user_id"].apply(lambda x: x not in self.online_id_stat)]
            
            else:
                assert False 

        else:
            assert False




    def feedback(self, user, iid):
        """
            return (rating, 01feedback)
        """
        uid = user.id

        if self.task == "warmstart":
            try:
                if user.online_round <= 60:
                    user_df = self.warm_start_user_df_set1_dict[uid]
                else:
                    user_df = self.warm_start_user_df_set2_dict[uid]
            except:
                user_df = self.grouped_data.get_group(uid)

        elif self.task in ["coldstart"]:
            user_df = self.grouped_data.get_group(uid)
        
        else:
            assert False

        if self.datan in ["ml-1m", "ml-100k", "ADS16", "EachMovie"]:
            # s = df.loc[(df['user_id'] == uid) & (df['item_id'] == iid), 'rating']
            s = user_df.loc[(user_df["item_id"] == iid), "rating"]
            if len(s) == 0:
                return 0,0
            else:
                rating = int(s)
                feedback_ = 0 if rating <= 3 else 1
                return rating, feedback_ 
       
        elif self.datan in ["KuaiRec"]:
            # s = df.loc[(df['user_id'] == uid) & (df['item_id'] == iid), 'rating']
            # s = df.query("user_id == @uid and item_id == @item_id")["rating"]
            
            # user_df = self.grouped_data.get_group(uid)
            
            s = user_df.loc[(user_df["item_id"] == iid), "rating"]
            if len(s) == 0:
                return 0,0
            else:
                rating = int(s)
                return rating, rating

        else:
            assert False
