        
from ..UI_cls_dir import UI_cls
import numpy as np
import numpy.linalg as LA
from torch import nn
import torch
import torch.optim as optim
from mypackages.decorator import for_all_methods
from ..ModelBase_dir import ModelBase
"""
    适应不同的VI后验

    框架：
        参数在ui类中, 优化时 ui类 -> 模型 -> ui类

    1. 固定 gcn item embedding,  VI/LTS
    2. 在线构图（当前做法）,  VI / LTS

    Fix/Dynamic 影响方法 online

"""

flag_list = ["DynamicG_VI", "FixG_VI", "FixG_LTS", "FixG_TP"]
"""
DynamicG_VI:
    累计一部分数据后完整的更新模型
FixG_VI:
    用户向量,物品向量同时维护分布
FixG_LTS:
    不维护物品向量分布
FixG_TP:(true posteriori)
    与ICF同一实现,user两个参数,mu和cov,item向量不维护分布
"""

class gcn(nn.Module):
    def __init__(self):
        super().__init__()

# @for_all_methods(profile)
class LGCNICF_Base(nn.Module, ModelBase):
    def __init__(self, para_dict, ec) -> None:
        nn.Module.__init__(self)
        ModelBase.__init__(self, para_dict, ec)

        self.embedding_dim = para_dict.get("embedding_dim", 20)        
        self.d = self.embedding_dim
        self.K = para_dict.get("K", 3)
        self.lr = para_dict.get("lr", 0.01) 
        self.sigma = para_dict.get("sigma", 1) # noisy 
        self.sigma_u = para_dict.get("sigma_u", 1)
        self.sigma_i = para_dict.get("sigma_i", 1)
        self.explore_method = para_dict.get("explore_method", "UCB") 
        self.explore_v = para_dict.get("explore_v", 1) 

        self.max_iter = para_dict["max_iter"]
        self.online_iter = para_dict.get("online_iter", 50)
        self.device = para_dict["device"]
        self.dtype = torch.float32

        self.lossfunc = para_dict.get("lossfunc", "reg")

        

    def loss_func_(self, uI_rating, item_indices, feedback):
        if self.lossfunc == "reg":
            return ((uI_rating[item_indices] - feedback)**2).sum()
        elif self.lossfunc == "classify":
            eps = 1e-7
            return -torch.log(torch.sigmoid((2 * feedback - 1) * uI_rating[item_indices]) + eps).sum()
        else:
            assert False

    def loss_func(self, UI_rating, u_index, item_indices, feedback):
        uI_rating = UI_rating[u_index,:]
        return self.loss_func_(uI_rating, item_indices, feedback)


    def model_train(self, ui_cls:UI_cls, pretrain_data):
        """
        TO IMPLEMENT:
            self.construct_model
            self.get_E0_pretrain
            self.optim_pretrain
            self.update_UI_cls
        """

        model = self.construct_model(ui_cls)
        optimizer = optim.SGD(model.parameters(), lr = self.lr) 
        
        M = model.M
        N = model.N

        self.graph_ma = ui_cls.graph_ma(self.K).to(device=self.device)
        for ii in range(self.max_iter):

            E0 = self.get_E0_pretrain(model)
            Embedding = self.graph_ma @ E0
            U_embedding = Embedding[:M, :]
            I_embedding = Embedding[M:, :]

            UI_rating = U_embedding @ I_embedding.T

            loss = 0
            for u_index in range(M):
                user = ui_cls.get_user_by_index(u_index)
                item_indices = user.interacted_item_index
                feedback = torch.tensor(user.feedback).to(device=self.device)
                # loss += ((UI_rating[u_index, item_indices] - feedback)**2).sum()
                loss += self.loss_func(UI_rating, u_index, item_indices, feedback)
            optimizer.zero_grad()
            loss.backward()

            self.optim_pretrain(optimizer)            

            self.print(f"iteration {ii + 1}, loss = {loss.item()}")
        self.update_UI_cls(ui_cls, model)

    # def online(self, ui_cls:UI_cls, data_cls):
    #     """
    #     TO IMPLEMENT
    #         self.online_init
    #         self.rec
    #         self.update_u
    #     """
    #     online_id = data_cls.online_id
    #     self.online_base(ui_cls, data_cls)
    #     self.online_init(ui_cls)

    #     # self.graph_ma = ui_cls.graph_ma(self.K).to(device=self.device).to_dense()

    #     TT = len(online_id)
    #     for t in range(TT):
    #         self.online_round += 1
    #         user_id = online_id[t]                                   
    #         user = ui_cls.get_user(user_id)
    #         if user.online_round >= self.online_rec_total_num:
    #             continue

    #         # candidate_item_id_set = self._struct_candidate_item_id_set(user, data_cls)
    #         candidate_item_id_set = user.online_candidate_list

    #         rec_item_id, rec_item_id_list = self.rec(user, ui_cls, candidate_item_id_set)

    #         rating, feedback01 = data_cls.feedback(user_id, rec_item_id)
    #         ex_info = {"online_rec_list": rec_item_id_list}

    #         self.print(f"{t=}, {user_id=}, {rec_item_id=}, {rating =},{rec_item_id_list=}")
    #         ui_cls.update_data((user_id, rec_item_id, rating), ex_info)
            
            
    #         self.online_update_u_base(user, rec_item_id, feedback01)
    #         self.online_update_u(user, ui_cls)
            
    #         self.ex_para_update(user)

    #         if t % 1000 == 0:
    #             ui_cls.evaluate_online(data_cls)

    def online_update_u(self, user, ui_cls):
        if len(user.interacted_item_id) <= 50: 
            self.update_u(user, ui_cls)
        elif 50 < len(user.interacted_item_id) <= 100:
            if len(user.interacted_item_id) % 5 == 0:
                self.update_u(user, ui_cls)
        else:
            if len(user.interacted_item_id) % 10 == 0:
                self.update_u(user, ui_cls)


## TODO: TO IMPLEMENT
    def construct_model(self):
        pass
    def get_E0_pretrain(self):
        pass
    def optim_pretrain(self, optimizer):
        optimizer.step()
    def update_UI_cls(self, ui_cls, model):
        pass 
    def rec(self):
        pass
    def update_u(self):
        pass
    def online_init(self, ui_cls):
        pass

    def ex_para_update(self, user):
        pass
    
    def ex_backward(self,model):
        pass


## not use
    def eval_loss(self, ui_cls, model):
        
        U_vecs = model.U_mean 
        I_vecs = model.I_mean 
        M,d = U_vecs.shape

        E0 = torch.cat((U_vecs, I_vecs), 0)
        Embedding = ui_cls.graph_ma(self.K).to(device=self.device) @ E0
        U_embedding = Embedding[:M, :]
        I_embedding = Embedding[M:, :]

        R = U_embedding @ I_embedding.T

        s = 0
        for u_index in range(ui_cls.user_num):
            u_id = ui_cls.index["user"]["index2id"][u_index]
            interacted_item_index = ui_cls.data["user"][u_id].interacted_item_index
            ratings = ui_cls.data["user"][u_id].feedback

            for i_index,r in zip(interacted_item_index, ratings):
                s += (r - R[u_index, i_index])**2
        
        return s
