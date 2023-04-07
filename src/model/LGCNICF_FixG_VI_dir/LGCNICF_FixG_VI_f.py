        
from ..LGCNICF_Base_dir import LGCNICF_Base,gcn
from ..UI_cls_dir import UI_cls
import numpy as np
import numpy.linalg as LA
from torch import nn
import torch
import torch.optim as optim
from mypackages.decorator import for_all_methods

"""
    self.construct_model
    self.get_E0_pretrain
    self.optim_pretrain
    self.update_UI_cls
    self.online_init
    self.rec
    self.update_u

"""
# @for_all_methods(profile)
class LGCNICF_FixG_VI(LGCNICF_Base):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)

    def construct_model(self, ui_cls:UI_cls, flag = "pre", user = None):
        """
            attribute:
                M,N
        """
        if flag == "pre":
            ui_cls.get_paramter()
            model = gcn()
            model.U_mean = nn.Parameter(ui_cls.U_mean ) # + torch.randn(ui_cls.U_mean.shape)
            model.U_rho = nn.Parameter(ui_cls.U_rho ) # + torch.randn(ui_cls.U_rho.shape) 
            model.I_mean = nn.Parameter(ui_cls.I_mean ) # + torch.randn(ui_cls.I_mean.shape) 
            model.I_rho = nn.Parameter(ui_cls.I_rho ) # + torch.randn(ui_cls.I_rho.shape) 
            model.to(dtype=self.dtype, device=self.device) 
            model.M, model.d = model.U_mean.shape
            model.N = model.I_mean.shape[0]

        elif flag == "online":
            model = gcn()
            assert user != None
            model.u_mean = nn.Parameter(user.mu.to(dtype=self.dtype, device=self.device))
            model.u_rho = nn.Parameter(user.rho.to(dtype=self.dtype, device=self.device))
            model.to(dtype=self.dtype, device=self.device)
        
        else:
            assert False

        return model

    def get_E0_pretrain(self, model):
        M,N,d = model.M, model.N, model.d
        
        standard_gaussian_noisy = torch.randn((M,d)).to(device=self.device)
        U_vecs = model.U_mean + standard_gaussian_noisy * self.VI_std(model.U_rho)

        standard_gaussian_noisy = torch.randn((N,d)).to(device=self.device)
        I_vecs = model.I_mean + standard_gaussian_noisy * self.VI_std(model.I_rho)

        E0 = torch.cat((U_vecs, I_vecs), 0)

        return E0

    def optim_pretrain(self, optimizer):
        return super().optim_pretrain(optimizer)

    def update_UI_cls(self, ui_cls:UI_cls, model, flag = "pre", user = None):
        
        if flag == "pre":
            I_rho = model.I_rho 
            U_rho = model.U_rho 
            U_mean = model.U_mean 
            I_mean = model.I_mean 

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
        elif flag == "online":
            assert user != None
            user.mu = model.u_mean
            user.rho = model.u_rho  
        else:
            assert False  


    def online_init(self, ui_cls):
        super().online_init(ui_cls)
        ui_cls.get_paramter()

        self.I_rho = ui_cls.I_rho
        self.I_mean = ui_cls.I_mean
        N,d = self.I_mean.shape
        
        self.U_rho = ui_cls.U_rho
        self.U_mean = ui_cls.U_mean


        E0 = torch.cat((self.U_mean, self.I_mean), 0)

        # self.graph_ma = ui_cls.graph_ma(self.K).to(device=self.device).to_dense()
        self.graph_ma = ui_cls.graph_ma(self.K).to_dense()
         

        Embedding = self.graph_ma @ E0

        self.FI = Embedding[-N:,:]


    def online_base(self, ui_cls, data_cls):
        super().online_base(ui_cls, data_cls)
        self.online_init(ui_cls)


    def rec(self, user, ui_cls:UI_cls):
        
        candidate_item_id_set = user.online_candidate_list
        
        if self.explore_method == "UCB":
            candidate_index = [ ui_cls.index["item"]["id2index"][id] for id in candidate_item_id_set]

            u_index = user.index
            uid = user.id

            self.u_g = ui_cls.g_value(user.id, self.K, flag="user")
            self.u_g = torch.tensor(self.u_g).to(device = self.device, dtype=self.dtype)
            
            u_g = self.u_g

            M = self.U_mean.shape[0]
            I_g = self.graph_ma[ M + np.array(candidate_index), :].to(device = self.device, dtype=self.dtype)

            score_mean = u_g @ self.EE_Mean @ I_g.T

            score_std = torch.sqrt( (u_g)**2 @ self.EE_Var @ (I_g.T**2) )

            score_vec = score_mean + self.explore_v * score_std

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
        model = self.construct_model(ui_cls, flag="online", user = user)
        index = user.index

        optimizer = optim.SGD(model.parameters(), lr = self.lr) 
        
        E0 = torch.cat((self.U_mean, self.I_mean), 0).to(device = self.device).detach()
        M = self.U_mean.shape[0]

        item_indices = user.interacted_item_index
        feedback = torch.tensor(user.feedback).to(device = self.device)

        self.print(f"{len(user.interacted_item_index)=}")
        for ii in range(self.online_iter):
            
            standard_gaussian_noisy = torch.randn(model.u_mean.shape).to(device=self.device)            

            vec = model.u_mean + standard_gaussian_noisy * self.VI_std(model.u_rho) 

            # E0[index, :] = vec
            u_g_u = self.u_g[index]

            u_embedding = self.u_g @ E0
            u_embedding = u_embedding - u_g_u * E0[index, :] + u_g_u * vec
            
            # I_embedding = self.graph_ma[M:, :] @ E0
            I_embedding = self.FI.to(device = self.device).detach()

            uI_rating = I_embedding @ u_embedding
            
            loss = self.loss_func_(uI_rating, item_indices, feedback)
        
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            if ii % 10 == 0:
                self.print(f"iteration {ii + 1}, loss = {loss.item()}")

            if loss.item() < 1e-2:
                break
        self.update_UI_cls(ui_cls, model, flag = "online", user=user)

    def VI_std(self, M):
        return torch.log(1 + torch.exp(M)) 

    def VI_var(self,M):
        return self.VI_std(M) ** 2

    @property
    def EE_Mean(self):
        """
            E[E^T E]
        """
        E0 = torch.cat((self.U_mean, self.I_mean), 0) # (M+N, d)

        E_rho = torch.cat((self.U_rho, self.I_rho), 0) # (M+N, d) 

        std = self.VI_std(E_rho)

        Mean = E0 @ E0.T + torch.diag((std ** 2).sum(1))        
        return Mean.to(device = self.device)

    @property
    def EE_Var(self):
        """
            Var[E^T E]
        """
        E0 = torch.cat((self.U_mean, self.I_mean), 0) # (M+N, d)

        E_rho = torch.cat((self.U_rho, self.I_rho), 0) # (M+N, d) 

        var = self.VI_var(E_rho)
        
        B = (E0**2) @ var.T

        Var = var @ var.T + B + B.T

        Var_diag = (E0**4).sum(1) + 3 * (var **2).sum(1)

        nn = Var.shape[0]

        for i in range(nn):
            Var[i,i] = Var_diag[i]

        return Var.to(device = self.device)


    def get_subE0(self, ui_cls:UI_cls, u_set, i_set):
        M,N = len(u_set),len(i_set)

        d = self.d

        subE0 = np.zeros([M+N, d])

        for i in range(M):
            uid = u_set[i]
            user = ui_cls.get_user(uid)
            subE0[i,:] = user.mu
        for j in range(N):
            iid = i_set[j]
            item = ui_cls.get_item(iid)
            subE0[j+M,:] = item.mu
        
        return subE0

    def update_online_U_mean_rho(self, user):
        index = user.index
        self.U_mean[index,:] = user.mu
        self.U_rho[index, :] = user.rho

    def ex_para_update(self, user):
        self.update_online_U_mean_rho(user)