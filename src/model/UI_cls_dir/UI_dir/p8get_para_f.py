from .p7eval_f import eval
import torch
import numpy as np

class get_para(eval):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)


    def get_U(self, method = "mean"):
        self.U = np.zeros((self.user_num, self.embedding_dim))

        for i in range(self.user_num):
            id = self.index["user"]["index2id"][i]
            user = self.data["user"][id]
            self.U[i,:] = user.get_vec(method)
        
        return self.U

    def get_I(self, method = "mean"):
        self.I = np.zeros((self.item_num, self.embedding_dim))

        for i in range(self.item_num):
            id = self.index["item"]["index2id"][i]
            item = self.data["item"][id]
            self.I[i,:] = item.get_vec(method)

        return self.I

    def get_U_mean(self):
        """
            for light gcn
        """
        self.U_mean = torch.zeros((self.user_num, self.embedding_dim))
        for i in range(self.user_num):    
            user = self.get_user_by_index(i)
            self.U_mean[i,:] = torch.tensor(user.mu)
        
        return self.U_mean

    def get_I_mean(self):
        """
            for light gcn
        """
        self.I_mean = torch.zeros((self.item_num, self.embedding_dim))
        for i in range(self.item_num):    
            item = self.get_item_by_index(i)
            self.I_mean[i,:] = torch.tensor(item.mu)
        
        return self.I_mean

    def get_U_rho(self):
        self.U_rho = torch.zeros((self.user_num, self.embedding_dim))
        for i in range(self.user_num):    
            user = self.get_user_by_index(i)
            self.U_rho[i,:] = torch.tensor(user.rho)
        
        return self.U_rho

    def get_I_rho(self):
        """
            for light gcn
        """
        self.I_rho = torch.zeros((self.item_num, self.embedding_dim))
        for i in range(self.item_num):    
            item = self.get_item_by_index(i)
            self.I_rho[i,:] = torch.tensor(item.rho)
        
        return self.I_rho

    def get_paramter(self):
        self.get_I_mean()
        self.get_I_rho()
        self.get_U_mean()
        self.get_U_rho()

