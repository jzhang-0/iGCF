from ..GCNICF_dir import *



"""
重写函数
    model_train loss改为 classify loss
    update_u 改为LVI update
"""

class GCNICF_LVI(GCNICF):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)

        self.LVI_iters = para_dict.get("LVI_iters", 2)
    

    def model_train(self, ui_cls:UI_cls, max_iter = None):
        """        
        """
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
                feedback = torch.tensor(user.feedback01).to(device=self.device) 
                eps = 1e-7               
                loss += - torch.log(torch.sigmoid((2 * feedback - 1) * UI_rating[u_index, item_indices]) + eps).sum()

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()          
            if (ii + 1) % 100 == 0:
                self.print(f"iteration {ii + 1}, loss = {loss.item()}")

        self.update_UI_cls(ui_cls, model)


    def update_u(self, user, ui_cls):
        u = user

        max_items_num = 200
        if not hasattr(u, "Epsilon_vec"):
            u.Epsilon_vec = np.zeros(max_items_num) 

        if u.interacted_num > max_items_num:
            raise Exception(f"num must be less than max_items_num, plaease add max_items_num")

        for _ in range(self.LVI_iters):
            self.updata_u_E_step(user)
            self.update_u_M_step(user)

    def lambda_epsilon(self, Epsilon_vec):
        L = np.zeros_like(Epsilon_vec)

        loc0 = np.where(Epsilon_vec == 0)[0]
        loc_n0 = np.where(Epsilon_vec != 0)[0] 

        L[loc0] = 1/8

        L[loc_n0] = 1 / (4 * Epsilon_vec[loc_n0]) * np.tanh(Epsilon_vec[loc_n0] / 2)
        # L[np.isnan(L)] = 
        return L

    def updata_u_E_step(self, user):
        u = user
       
        index = u.interacted_item_index
        r = np.array(u.feedback01)
        Du = self.FI_mu[index] 

        num = u.interacted_num
        Epsilon_vec = u.Epsilon_vec[:num]

        Lambda_E = self.lambda_epsilon(Epsilon_vec)

        A_inv = LA.inv(Du.T @ (Du * 2 * Lambda_E[:,np.newaxis]) + self.lambda_u * np.eye(self.dim))
        u.online_mu = A_inv @ ( Du.T @  (r - 1/2) )
        u.online_cov = A_inv * (self.sigma ** 2)

    def update_u_M_step(self, user):
        u = user
        index = u.interacted_item_index
        Du = self.FI_mu[index] 
        num = u.interacted_num
        
        u.Epsilon_vec[:num] = np.sqrt((Du @ (u.online_cov + \
                                              u.online_mu[:, np.newaxis] @ u.online_mu[:, np.newaxis].T ) * Du).sum(1))

