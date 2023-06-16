import os,sys,json
import logging
import numpy as np
import torch
import pandas as pd
# from ..UI_cls_dir import UI_cls
from ..ExpConfig_dir import ExpConfig

class ModelBase:
    """

    para_dict:
        datan, modelname, exp_id

    IMPLEMENTED:
        self.print

    TO IMPLEMENT in child class:
        self.model_init
        self.model_train
        self.online_init
        self.rec
        self.update_u
    """
    def __init__(self, para_dict, ec) -> None:

        self.para_dict = para_dict
        # self.ec = ExpConfig(para_dict)
        self.ec = ec
        self.func_inherit()

        self.modeln = None
        self.online_round = 0

        self.online_rec_total_num = para_dict.get("online_rec_total_num", 40)                
        self.rec_list_len = para_dict.get("rec_list_len", 1)

    def func_inherit(self):
        self.print = self.ec.print
        self.save_cls = self.ec.save_cls


    def train_online_test(self,ui_cls, data_cls):
        pass

    def pretrain(self, ui_cls, pretrain_data):
        ui_cls.update_data(pretrain_data)
        self.model_init(ui_cls)
        self.model_train(ui_cls)

    def online(self, ui_cls, data_cls):
        """
        TO IMPLEMENT
            self.online_init
            self.rec
            self.update_u
        """
        online_id = data_cls.online_id

        self.print(f"online test user num = {len(data_cls.online_id_stat)}")

        self.online_init(ui_cls, data_cls)
        
        
        # self.save_cls(ui_cls, "ui.class")

        TT = len(online_id)
        for t in range(TT):
            self.online_round += 1
            user_id = online_id[t]                                   
            user = ui_cls.get_user(user_id)
            if user.online_round >= self.online_rec_total_num:
                continue

            rec_item_id, rec_item_id_list = self.rec(user, ui_cls)

            rating, feedback01 = data_cls.feedback(user, rec_item_id)

            feedback_ = [data_cls.feedback(user, iid) for iid in rec_item_id_list] 
            rec_list_ratings = [i[0] for i in feedback_]
            rec_list_feedback01 = [i[1] for i in feedback_]
            
            ex_info = None

            if t % 100 == 0:
                self.print(f"{t=}, {user_id=}, {rec_item_id_list=}, {rec_list_ratings=}")
            
            df = pd.DataFrame([[user_id, iid, r, f01]  for iid, r, f01 \
                in zip(rec_item_id_list, rec_list_ratings, rec_list_feedback01)],\
                    columns = ['user_id', 'item_id', 'rating', 'feedback01'])
            ui_cls.update_data(df, ex_info)

            
            self.online_update_u_info(user, rec_item_id_list, feedback01, rec_list_feedback01)
            
            self.online_update_u(user, ui_cls)            
            self.ex_para_update(user)

            self.online_update_model(ui_cls)

            # if t % 1000 == 0:
            #     ui_cls.evaluate_online(data_cls)
        
        
        # self.ec.save_cls(ui_cls, "ui_end.class")


    

    def kargmax(self, score_vec, k):
        if type(score_vec) == torch.Tensor:
            score_vec = np.array(score_vec.detach().cpu())
        return score_vec.argsort()[-k:][::-1]
    
    def _struct_candidate_item_id_set(self, user, data_cls):
        if data_cls.eval_method == "NI20":
            return data_cls.item_id_set - set(user.interacted_item_id)
        elif data_cls.eval_method == "rated":
            return set(data_cls.exposure[user.id]) - set(user.interacted_item_id)
        else:
            assert False

    def online_init(self, ui_cls, data_cls):
        ui_cls.check_item_id_set_exists(data_cls.item_id_set)
        ui_cls.check_user_id_set_exists(data_cls.user_id_set)
        ui_cls.online_init(data_cls)

    def online_update_u_info(self, user, rec_item_id_list, feedback01, rec_list_feedback01):
        user.online_round += self.rec_list_len
        for rec_item_id in rec_item_id_list:
            user.online_candidate_list.remove(rec_item_id)
        user.online_feedback01.append(feedback01)
        user.online_rec_list_feedback01.append(rec_list_feedback01)

    def model_init(self, ui_cls):
        pass

    def model_train(self, ui_cls):
        pass


    def online_update_u(self, user, ui_cls):
        pass

    def ex_para_update(self, user, **kwargs):
        pass

    def ex_func_mf(self, user, rec_item_id, rating, ui_cls, t):
        pass

    def online_update_model(self, ui_cls):
        pass 

    def multivariate_normal_(self, m, cov, n=1):
        return multivariate_normal(m, cov, n)

class ui_func_base:        
    def __init__(self, para_dict) -> None:
        self.is_tensor = para_dict.get("is_tensor", 0)
        self.modeln = para_dict.get("modelname", -1)

        self.embedding_dim = para_dict.get("embedding_dim", 20)
        self.para_dict = para_dict

        if self.modeln in ["LGCNICF_FixG_VI", "LGCNICF"]:  
            self.mu = para_dict["mu_u"] * torch.ones(self.embedding_dim, requires_grad=True) # para_dict["mu_u"] = 0
            # self.mu += para_dict["sigma_u"] * torch.randn(self.embedding_dim)
            
            self.rho = torch.zeros(self.embedding_dim, requires_grad=True)

        elif self.modeln in ["GCNICF", "GCNICF_LVI", "GCNICF_Meta", "GCNICF_Meta_V2"]:
            
            self.mu = para_dict["mu_u"] * np.ones(self.embedding_dim) # para_dict["mu_u"] = 0
            self.rho = np.zeros(self.embedding_dim)


        elif self.modeln in ["LGCNICF_DynamicG_VI"]: 
            self.mu = para_dict["mu_u"] * torch.ones(self.embedding_dim, requires_grad=True)
            self.mu += para_dict["init"] * torch.randn(self.embedding_dim)
            
            self.rho = torch.ones(self.embedding_dim, requires_grad=True)

        elif self.modeln in ["LGCNICF_FixG_LTS"]:
            self.mu = para_dict["mu_u"] * torch.ones(self.embedding_dim, requires_grad=True) 
            self.mu += para_dict["sigma_u"] * torch.randn(self.embedding_dim)

        elif self.modeln == "ICF":
            self.mu = para_dict["mu_u"] * np.ones(self.embedding_dim)
            self.cov = para_dict["sigma_u"] * np.eye(self.embedding_dim)

        elif self.modeln in ["Pop", "Pos", "Random", "PosPlus"]:
            pass
        
        elif self.modeln in ["MF"]:
            self.mu = para_dict["mu_u"] * np.ones(self.embedding_dim) + para_dict["sigma_u"] * np.random.randn(self.embedding_dim)
            
        else:
            assert False
    
    def get_vec(self, method = "mean"):
        """
            mu : vec
            cov : vec
            method = mean / sampling
        """
        if method == "mean":
            return self.mu
        elif method == "sampling":
            if self.is_tensor:
                cov = torch.diag(torch.log(1 + torch.exp(self.rho))) 
                vecs = multivariate_normal(self.mu, cov) 
            else:
                vecs = multivariate_normal(self.mu, self.cov)
            return vecs
        else: 
            raise ValueError("method not implment")

def multivariate_normal(m, cov, n=1):
    d = cov.shape[0]
    mean = m[:,np.newaxis]
    
    if np.linalg.norm(cov, "fro") == 0:
        A = np.zeros_like(cov)
    else:
        A = np.linalg.cholesky(cov)

    # Sample X from standard normal
    Z = np.random.normal(size=(d, n))

    # Apply the transformation
    X = A.dot(Z) + mean
    X = X.T
    if n == 1:
        return X.flatten()
    else:
        return X