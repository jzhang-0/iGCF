from ..ModelBase_dir import ModelBase
from ..UI_cls_dir import UI_cls
import numpy as np
import numpy.linalg as LA

class ICF(ModelBase):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)

        self.embedding_dim = para_dict.get("embedding_dim", 20)
        self.lambda_u = para_dict.get("lambda_u", 1)
        self.lambda_i = para_dict.get("lambda_i", 1)
        self.sigma = para_dict.get("sigma", 1) # noisy 
        self.explore_method = para_dict.get("explore_method", "UCB") 
        self.explore_v = para_dict.get("explore_v", 1) 

        self.max_iter = para_dict["max_iter"]

    def model_train(self, ui_cls:UI_cls):
        self.I = ui_cls.get_I(method= "sampling")        
        self.U = ui_cls.get_U(method= "sampling")

        for ii in range(self.max_iter):
            
            for uid in ui_cls.data["user"]:
                u = ui_cls.data["user"][uid] 
                self.update_u(u)
            
            loss = ui_cls.eval_loss()
            self.U = ui_cls.get_U(method= "mean")   
            self.print(f"iteration {ii + 1}, u update, {loss=}")        
              
            for iid in ui_cls.data["item"]:
                i = ui_cls.data["item"][iid]  
                self.update_i(i)
            
            loss = ui_cls.eval_loss()

            self.I = ui_cls.get_I(method= "mean")            
            self.print(f"iteration {ii + 1}, i update, {loss=}")

            self.print(f"iteration {ii + 1} complete")     

    def rec(self, user, ui_cls:UI_cls):

        candidate_item_id_set = user.online_candidate_list
        
        candidate_index = [ ui_cls.index["item"]["id2index"][id] for id in candidate_item_id_set]
        item_ma = ui_cls.I[candidate_index] # (n,d)

        if self.explore_method == "UCB":
            user_vec = user.get_vec(method = "mean")
            cov = user.cov
            score_vec = item_ma @ user_vec + self.explore_v * np.sqrt( ((item_ma @ cov ) * item_ma).sum(1) )

        elif self.explore_method == "TS":
            # user_vec = user.get_vec(method = "sampling")
            try:
                user_vec = self.multivariate_normal_(user.mu, user.cov * self.explore_v)
            except:
                self.ec.save_data("cov", user.cov, self.ec.save_path)
                assert False
            score_vec = item_ma @ user_vec
        else:
            assert False

        k = self.rec_list_len
        if len(candidate_item_id_set) < 5:
            k = len(candidate_item_id_set) 
        
        indices = self.kargmax(score_vec, k)
        n = indices[0]
        
        candidate_item_id_set = list(candidate_item_id_set)
        rec_item_id = candidate_item_id_set[n]
        rec_item_id_list = [candidate_item_id_set[i] for i in indices]

        return rec_item_id, rec_item_id_list



    def online_update_u(self, user, ui_cls):
        
        if len(user.interacted_item_id) <= 50: #TODO:improve
            self.update_u(user)
        else:
            if len(user.interacted_item_id) % 10 == 0:
                self.update_u(user)

    def update_u(self, u):
        index = u.interacted_item_index
        r = np.array(u.feedback)
        Du = self.I[index] 

        A_inv = LA.inv(Du.T @ Du + self.lambda_u * np.eye(self.embedding_dim))
        u.mu = A_inv @ ( Du.T @  r)
        u.cov = A_inv * (self.sigma ** 2)

    def update_i(self, i):
        index = i.interacted_user_index
        r = np.array(i.feedback)
        Bi = self.U[index] 

        A_inv = LA.inv(Bi.T @ Bi + self.lambda_i * np.eye(self.embedding_dim))
        i.mu = A_inv @ (Bi.T @ r)
        i.cov = A_inv * (self.sigma ** 2)

    # def MAP_init_U_A(self, ui_cls):
    #     self.I = ui_cls.get_I(method= "mean")        
    #     # for uid in ui_cls.data["user"]:
    #     #     u = ui_cls.data["user"][uid] 
    #     #     index = u.interacted_item_index
    #     #     r = np.array(u.feedback)
    #     #     Du = self.I[index] 

    #     #     u.A = Du.T @ Du + self.lambda_u * np.eye(self.embedding_dim)

    # def online_init(self, ui_cls):
    #     self.MAP_init_U_A(ui_cls)
    
    def online_init(self, ui_cls, data_cls):
        super().online_init(ui_cls, data_cls)
        self.I = ui_cls.get_I(method= "mean") 
        