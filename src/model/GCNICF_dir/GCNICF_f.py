from ..ModelBase_dir import *
from ..LGCNICF_FixG_VI_dir import *
from ..UI_cls_dir import *
import numpy as np
import numpy.linalg as LA
from torch import nn
import torch
import torch.optim as optim
from ..UI_cls_dir import UI_cls
import copy,os
import pandas as pd
# from mypackages.decorator import for_all_methods


class GCNICF(ModelBase):

    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)
        self.dim = para_dict["embedding_dim"]

        self.K = para_dict.get("K", 3)
        self.lr = para_dict.get("lr", 0.01) 

        self.lambda_u = para_dict.get("lambda_u", 1)
        self.lambda_i = para_dict.get("lambda_i", 1)

        self.sigma = para_dict.get("sigma", 1) # noisy 
        self.sigma_u = para_dict.get("sigma_u", 1)
        self.sigma_i = para_dict.get("sigma_i", 1)
        self.explore_method = para_dict.get("explore_method", "UCB") 
        self.explore_v = para_dict.get("explore_v", 1) 

        self.test_iters = para_dict.get("test_iters", 1000)
        # self.test_num = para_dict.get("test_num", 100)

        self.max_iter = para_dict["max_iter"]
        self.online_iter = para_dict.get("online_iter", 50)
        self.device = para_dict["device"]
        self.dtype = torch.float32
        
        self.test_para = para_dict.get("test_para", "0")

    def train_online_test(self,ui_cls, data_cls):
        ui_cls.update_data(data_cls.pre_training_data)
        self.model_init(ui_cls)

        df_init = 1

        test_num = int(self.max_iter // self.test_iters)
        for i in range(test_num):

            ## DEBUG
            # self.model_train(ui_cls, max_iter = 100000)

            if i == 0:
                ui_cls_online = copy.deepcopy(ui_cls)                
            else:
                self.model_train(ui_cls, max_iter = self.test_iters)
                
                ui_cls_online = copy.deepcopy(ui_cls)
           
            self.online(ui_cls_online, data_cls)

            
            result = ui_cls_online.evaluate(data_cls)
            if df_init:
                result_df = pd.DataFrame(result, index=[i])
                df_init = 0
            else:
                new_df = pd.DataFrame(result, index=[i])
                result_df = pd.concat([result_df, new_df])
            
            result_df.to_csv( os.path.join(self.ec.save_path, "results_df.csv"))
        
    # def copy_ui_cls(self,ui_cls):
    #     new_ui_cls = UI_cls(self.para_dict, self.ec)
    #     M = ui_cls.user_num
    #     N = ui_cls.item_num

    #     for i in range(M):
            
    #         user = ui_cls.get_user_by_index(i)
    #         newuser = new_ui_cls.get_user(user.id)
    #         newuser = user

    #         user.mu = user.mu.detach()
    #         user.rho = user.rho.detach()
    #     for i in range(N):
    #         item = ui_cls.get_item_by_index(i)
    #         item.mu = item.mu.detach()
    #         item.rho = item.rho.detach()

    def construct_model(self, ui_cls:UI_cls):
        """
            attribute:
                M,N
        """
        ui_cls.get_paramter()
        model = gcn()
        model.U_mean = nn.Parameter(ui_cls.U_mean ) 
        model.U_rho = nn.Parameter(ui_cls.U_rho ) 
        model.I_mean = nn.Parameter(ui_cls.I_mean ) 
        model.I_rho = nn.Parameter(ui_cls.I_rho ) 
        model.to(dtype=self.dtype, device=self.device) 
        model.M, model.d = model.U_mean.shape
        model.N = model.I_mean.shape[0]

        return model

    def get_E0(self, model):
        M,N,d = model.M, model.N, model.d
        
        standard_gaussian_noisy = torch.randn((M,d)).to(device=self.device)
        U_vecs = model.U_mean + standard_gaussian_noisy * self.VI_std(model.U_rho)

        standard_gaussian_noisy = torch.randn((N,d)).to(device=self.device)
        I_vecs = model.I_mean + standard_gaussian_noisy * self.VI_std(model.I_rho)

        E0 = torch.cat((U_vecs, I_vecs), 0)

        return E0

    def model_init(self, ui_cls):
        super().model_init(ui_cls)
        self.gcn = self.construct_model(ui_cls) 


    def get_graph_ma(self, ui_cls):
        if self.test_para == "0":
            return ui_cls.graph_ma(self.K)
        elif self.test_para == "adjust":
            return ui_cls.graph_ma_adjust(self.K)
        elif self.test_para == "simple1":
            return ui_cls.graph_ma_simple1()
        else:
            assert False

    
    def model_train(self, ui_cls:UI_cls, max_iter = None):
        """
        TO IMPLEMENT:
            - [x] self.construct_model
            - [x] self.get_E0
            - [x] self.update_UI_cls
        """

        # model = self.construct_model(ui_cls)
        model = self.gcn
        optimizer = optim.SGD(model.parameters(), lr = self.lr) 
        
        M = model.M
        N = model.N

        self.graph_ma = self.get_graph_ma(ui_cls).to(device=self.device)

        if max_iter == None:
            max_iter = self.max_iter

        for ii in range(max_iter):

            E0 = self.get_E0(model)
            Embedding = self.graph_ma @ E0
            U_embedding = Embedding[:M, :]
            I_embedding = Embedding[M:, :]
            UI_rating = U_embedding @ I_embedding.T
            loss = 0
            for u_index in range(M):
                user = ui_cls.get_user_by_index(u_index)
                item_indices = user.interacted_item_index
                feedback = torch.tensor(user.feedback).to(device=self.device)
                loss += ((UI_rating[u_index, item_indices] - feedback)**2).sum()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()          
            if (ii + 1) % 100 == 0:
                self.print(f"iteration {ii + 1}, loss = {loss.item()}")

        self.update_UI_cls(ui_cls, model)

    def update_UI_cls(self, ui_cls:UI_cls, model):
        I_rho = model.I_rho.detach().cpu().numpy() 
        U_rho = model.U_rho.detach().cpu().numpy() 
        U_mean = model.U_mean.detach().cpu().numpy() 
        I_mean = model.I_mean.detach().cpu().numpy() 

        N,d = I_mean.shape
        M,d = U_mean.shape

        for i in range(M):
            user = ui_cls.get_user_by_index(i)
            user.mu = U_mean[i]
            user.rho = U_rho[i]
        for i in range(N):
            item = ui_cls.get_item_by_index(i)
            item.mu = I_mean[i]
            item.rho = I_rho[i]          

    def online_update_u(self, user, ui_cls):
        if len(user.interacted_item_id) <= 50: 
            self.update_u(user, ui_cls)
        elif 50 < len(user.interacted_item_id) <= 100:
            if len(user.interacted_item_id) % 5 == 0:
                self.update_u(user, ui_cls)
        else:
            if len(user.interacted_item_id) % 10 == 0:
                self.update_u(user, ui_cls)


    def VI_std(self, M):
        return torch.log(1 + torch.exp(M)) 

    def VI_var(self,M):
        return self.VI_std(M) ** 2

    # @property
    # def EE_Mean(self):
    #     """
    #         E[E^T E]
    #     """
    #     E0 = torch.cat((self.U_mean, self.I_mean), 0) # (M+N, d)

    #     E_rho = torch.cat((self.U_rho, self.I_rho), 0) # (M+N, d) 

    #     std = self.VI_std(E_rho)

    #     Mean = E0 @ E0.T + torch.diag((std ** 2).sum(1))        
    #     return Mean.to(device = self.device)

    # @property
    # def EE_Var(self):
    #     """
    #         Var[E^T E]
    #     """
    #     E0 = torch.cat((self.U_mean, self.I_mean), 0) # (M+N, d)

    #     E_rho = torch.cat((self.U_rho, self.I_rho), 0) # (M+N, d) 

    #     var = self.VI_var(E_rho)
        
    #     B = (E0**2) @ var.T

    #     Var = var @ var.T + B + B.T

    #     Var_diag = (E0**4).sum(1) + 3 * (var **2).sum(1)

    #     nn = Var.shape[0]

    #     for i in range(nn):
    #         Var[i,i] = Var_diag[i]

    #     return Var.to(device = self.device)


    def online_init(self, ui_cls: UI_cls, data_cls):
        super().online_init(ui_cls, data_cls)
        ui_cls.get_paramter()

        self.I_rho = ui_cls.I_rho
        self.I_mean = ui_cls.I_mean
        N,d = self.I_mean.shape
        
        self.U_rho = ui_cls.U_rho
        self.U_mean = ui_cls.U_mean

        E0 = torch.cat((self.U_mean, self.I_mean), 0)

        E0_rho = torch.cat((self.U_rho, self.I_rho), 0) 

        self.graph_ma = self.get_graph_ma(ui_cls).to_dense()
         

        Embedding = self.graph_ma @ E0
        Var = (self.graph_ma ** 2) @ self.VI_var(E0_rho)

        self.FI_mu = Embedding[-N:,:].detach().numpy()
        self.FI_var = Var[-N:,:].detach().numpy()


    def rec(self,  user, ui_cls:UI_cls):
        """
            Var(X^T Y) = \mu_x^T \Sigma_y \mu_x + \mu_y^T \Sigma_x \mu_y + Trace(\Sigma_x \Sigma_y)
        """
        candidate_item_id_set = user.online_candidate_list
        
        if self.explore_method == "UCB":
            candidate_index = [ ui_cls.index["item"]["id2index"][id] for id in candidate_item_id_set]

            items_mu = self.FI_mu[candidate_index] # (n,d)
            items_var = self.FI_var[candidate_index] # (n,d)

            if hasattr(user, "online_mu"):
                user_vec = user.online_mu
            else:
                user_vec = np.zeros(self.dim)
            
            if hasattr(user, "online_cov"):
                cov = user.online_cov
            else:
                cov = np.eye(self.dim) 

            var = ((items_mu @ cov ) * items_mu).sum(1)  +  (items_var * user_vec ) @ user_vec + items_var @ np.diag(cov) 
            score_vec = items_mu @ user_vec + self.explore_v * np.sqrt( np.log(user.online_round + 2) * var)

            k = self.rec_list_len
            if len(candidate_item_id_set) < k:
                k = len(candidate_item_id_set) 
            
            indices = self.kargmax(score_vec, k)
            n = indices[0]
            
            candidate_item_id_set = list(candidate_item_id_set)
            rec_item_id = candidate_item_id_set[n]
            rec_item_id_list = [candidate_item_id_set[i] for i in indices]

            return rec_item_id, rec_item_id_list
        else:
            assert False
    
    def update_u(self, user, ui_cls):
        u = user
        index = u.interacted_item_index
        r = np.array(u.feedback)
        Du = self.FI_mu[index] 

        A_inv = LA.inv(Du.T @ Du + self.lambda_u * np.eye(self.dim))
        u.online_mu = A_inv @ ( Du.T @  r)
        u.online_cov = A_inv * (self.sigma ** 2)



class gcn(nn.Module):
    def __init__(self):
        super().__init__()
