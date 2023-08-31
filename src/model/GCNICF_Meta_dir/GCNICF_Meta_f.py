from ..GCNICF_dir import *


class GCNICF_Meta(GCNICF):
    """
    """
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)
        self.meta_update = para_dict.get("meta_update", "0")

    def model_train(self, ui_cls: UI_cls, max_iter=None):
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

            if M > 500:
                sample_num  = 500
                iter_users_index = np.random.choice(M, size = sample_num, replace=False)
            else:
                iter_users_index = np.arange(M)
            
            total_datapoints_num = 0
            for u_index in iter_users_index:
                user = ui_cls.get_user_by_index(u_index)
                item_indices = user.interacted_item_index
                feedback = torch.tensor(user.feedback).to(device=self.device)

                total_datapoints_num += len(item_indices)
                loss += ((UI_rating[u_index, item_indices] - feedback)**2).sum()

                if len(item_indices) < N / 4:
                    neg_set = set(range(N)) - set(item_indices)
                    neg_index = np.random.choice(list(neg_set), len(item_indices), replace= False) 

                    total_datapoints_num += len(item_indices)
                    loss += ((UI_rating[u_index, neg_index])**2).sum() 

            loss = loss / total_datapoints_num            
            # loss += self.lambda_u * (Embedding[iter_users_index]**2).sum()
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()          
            if (ii + 1) % 100 == 0:
                self.print(f"iteration {ii + 1}, loss = {loss.item()}")

        self.update_UI_cls(ui_cls, model)


    def online_init(self, ui_cls: UI_cls, data_cls):
        super().online_init(ui_cls, data_cls)
        dls  = data_cls
        p_uids = dls.pre_training_data["user_id"].drop_duplicates()
        online_uids = dls.online_id_stat

        n = len(p_uids)
        
        self.p_uindics = [ui_cls.index["user"]["id2index"][user_id] for user_id in p_uids]
        online_uindics = [ui_cls.index["user"]["id2index"][user_id] for user_id in online_uids]

        self.online_mu_meta = self.FU_mu[self.p_uindics].mean(0) 
        
        
        zero_mu = self.FU_mu[self.p_uindics] - self.online_mu_meta
        self.online_cov_meta = ( zero_mu.T @ zero_mu ) / (n-1)
        
         
        try:
            self.online_cov_meta_inv = LA.inv(self.online_cov_meta)
        except:
            print("WARNING: online_cov_meta is singularity, use np.linalg.pinv instead")
            self.online_cov_meta_inv = LA.pinv(self.online_cov_meta)



        for uid in online_uids:
            user = ui_cls.get_user_by_id(uid)
            user.online_mu = self.online_mu_meta
            user.online_cov = self.online_cov_meta ## DEBUG check matrix?



    def rec(self,  user, ui_cls:UI_cls):
        """
            Var(X^T Y) = \mu_x^T \Sigma_y \mu_x + \mu_y^T \Sigma_x \mu_y + Trace(\Sigma_x \Sigma_y)
        """
        candidate_item_id_set = user.online_candidate_list
        
        if self.explore_method == "UCB":
            candidate_index = [ ui_cls.index["item"]["id2index"][id] for id in candidate_item_id_set]

            items_mu = self.FI_mu[candidate_index] # (n,d)
            items_var = self.FI_var[candidate_index] # (n,d)

            user_vec = user.online_mu            
            cov = user.online_cov

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
        if self.meta_update == "0":
            pass
        elif self.meta_update in ["half", "lin", "no_meta"]:
            u = user
            index = u.interacted_item_index
            r = np.array(u.feedback)
            Du = self.FI_mu[index] 

            A_inv = LA.inv(Du.T @ Du + self.lambda_u * np.eye(self.dim))
            online_mu = A_inv @ ( Du.T @  r)
            online_cov = A_inv * (self.sigma ** 2)

            if self.meta_update == "half":
                u.online_mu = (online_mu + self.online_mu_meta)/2
                u.online_cov = ( online_cov + self.online_cov_meta ) / (2**2)

            elif self.meta_update == "lin":
                t = u.online_round
                T = self.online_rec_total_num 

                w_meta = max( -2 / T * t + 1, 0)
                w_onine = 1 - w_meta

                u.online_mu = w_meta * self.online_mu_meta + w_onine * online_mu
                u.online_cov = (w_meta ** 2) * self.online_cov_meta + (w_onine ** 2) * online_cov
            
            elif self.meta_update == "no_meta":
                u.online_mu = online_mu 
                u.online_cov = online_cov

        elif self.meta_update == "meta_prior":
            u = user
            index = u.interacted_item_index
            r = np.array(u.feedback)
            Du = self.FI_mu[index] 

            # meta_cov_inv = np.diag(self.online_cov_meta_diag ** (-1))
            meta_cov_inv = self.online_cov_meta_inv

            try:
                A_inv = LA.inv(Du.T @ Du * (1 / self.sigma ** 2) + meta_cov_inv )
            except:
                A_inv = LA.pinv(Du.T @ Du * (1 / self.sigma ** 2) + meta_cov_inv )

            online_mu = A_inv @ ( Du.T @  r + meta_cov_inv @ self.online_mu_meta)
            online_cov = A_inv

            u.online_mu = online_mu
            u.online_cov = online_cov

        
        else:
            raise Exception(f"not support meta update method {self.meta_update}")

            