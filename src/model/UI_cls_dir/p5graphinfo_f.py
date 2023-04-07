from .p4collectinfo_f import collectinfo
import numpy as np
import torch

class graphinfo(collectinfo):
    def __init__(self, para_dict) -> None:
        super().__init__(para_dict)
        self.agg_ma_pow = {} # key in [0, 1, ..., K]


    @property
    def adjacent(self):
        M = self.user_num
        N = self.item_num
        
        R = np.zeros((M,N))
        for i in range(M):
            user = self.get_user_by_index(i)
            item_indics = user.interacted_item_index
            R[i, item_indics] = 1
        O1 = np.zeros((M,M))
        A1 = np.c_[O1, R]

        O2 = np.zeros((N,N))
        A2 = np.c_[R.T, O2]
        return np.r_[A1, A2]

    @property
    def aggregation_ma(self):
        A = self.adjacent
        d = (A != 0).sum(0)

        loc = (d==0)
        d[loc] += 1

        d_2 = d**(-1/2)
        d_2[loc] -= 1

        ma = (d_2[:,np.newaxis]) * A *  d_2
        return torch.tensor(ma).to_sparse().to(dtype = torch.float32)

    def graph_ma(self, K):
        """
            aggregate K number
        """
        agg_ma = self.aggregation_ma
        ma_dim = agg_ma.shape[0]
        I = np.eye(ma_dim)
        I = torch.tensor(I).to_sparse().to(dtype = torch.float32)

        Graph_ma = torch.zeros_like(agg_ma)
        agg_ma_powk = I

        for _ in range(K):    
            Graph_ma += agg_ma_powk
            agg_ma_powk = torch.sparse.mm(agg_ma_powk, agg_ma)
        Graph_ma += agg_ma_powk

        Graph_ma = 1/(1 + K) * Graph_ma

        return Graph_ma


    @property
    def adjacent_adjust(self):
        """
        record 1 -1 info
        """
        M = self.user_num
        N = self.item_num
        
        R = np.zeros((M,N))
        for i in range(M):
            user = self.get_user_by_index(i)
            item_indics = user.interacted_item_index
            
            R[i, item_indics] = np.array(user.feedback01) * 2 - 1
        O1 = np.zeros((M,M))
        A1 = np.c_[O1, R]

        O2 = np.zeros((N,N))
        A2 = np.c_[R.T, O2]
        return np.r_[A1, A2]

    @property
    def aggregation_ma_adjust(self):
        A = self.adjacent_adjust
        d = (A != 0).sum(0)

        loc = (d==0)
        d[loc] += 1

        d_2 = d**(-1/2)
        d_2[loc] -= 1

        ma = (d_2[:,np.newaxis]) * A *  d_2
        return torch.tensor(ma).to_sparse().to(dtype = torch.float32)

    def graph_ma_adjust(self, K):
        """
            aggregate K number
        """
        agg_ma = self.aggregation_ma_adjust
        ma_dim = agg_ma.shape[0]
        I = np.eye(ma_dim)
        I = torch.tensor(I).to_sparse().to(dtype = torch.float32)

        Graph_ma = torch.zeros_like(agg_ma)
        agg_ma_powk = I
        self.agg_ma_pow[0] = I

        for k in range(K):    
            Graph_ma += agg_ma_powk
            agg_ma_powk = torch.sparse.mm(agg_ma_powk, agg_ma)
            self.agg_ma_pow[k+1] = agg_ma_powk.to_dense()
        Graph_ma += agg_ma_powk

        Graph_ma -= I

        Graph_ma = 1 / K * Graph_ma

        return Graph_ma


    def g_value_k(self, id, k, flag):
        M, N = self.user_num, self.item_num

        g = np.zeros(M+N)
        if flag == "user":
            user = self.get_user(id)
            if k == 0:
                index_ = user.index
                g[index_] = 1
                return g
            elif k >= 1:
                item_id_list = user.interacted_item_id

                for item_id in item_id_list:
                    item = self.get_item(item_id)
                    g += 1 / (np.sqrt(user.interacted_num * item.interacted_num)) * self.g_value_k(item_id, k - 1, flag="item")
                return g 
       
        elif flag == "item":
            item = self.get_item(id)
            if k == 0:
                index_ = item.index
                g[index_ + M] = 1
                return g
            elif k >= 1:
                user_id_list = item.interacted_user_id

                for u_id in user_id_list:
                    user = self.get_user(u_id)
                    g += 1 / (np.sqrt(user.interacted_num * item.interacted_num)) * self.g_value_k(u_id, k - 1, flag="user")
                return g 

        else:
            assert False

    def g_value(self, id, K, flag):
        M, N = self.user_num, self.item_num
        g = np.zeros(M + N)

        for k in range(K + 1):
            g += self.g_value_k(id, k, flag)
        
        return g / (K+1)


    def khop(self, uid, k):
        user = self.get_user(uid)

        user_neighbors_set = set([user.id])
        item_neighbors_set = set([])
        
        i = 0
        while k>0:
            k -= 1
            i += 1
            if i % 2 == 1:
                for uid in user_neighbors_set:
                    user = self.get_user(uid)
                    item_neighbors = set(user.interacted_item_id)
                    item_neighbors_set += item_neighbors
            else:
                for iid in item_neighbors_set:
                    item = self.get_item(iid)
                    user_neighbors = set(item.interacted_user_id)
                    user_neighbors_set += user_neighbors

        return user_neighbors_set, item_neighbors_set


    def sub_adjacent(self, u_set, i_set):
        M = len(u_set)
        N = len(i_set)
        
        R = np.zeros((M,N))
        for i in range(M):
            user = self.get_user(u_set[i])
            item_indics = user.interacted_item_index
            R[i, item_indics] = 1
        O1 = np.zeros((M,M))
        A1 = np.c_[O1, R]

        O2 = np.zeros((N,N))
        A2 = np.c_[R.T, O2]
        return np.r_[A1, A2]

    def sub_aggregation_ma(self, u_set, i_set):
        A = self.sub_adjacent(u_set, i_set)
        d = (A != 0).sum(0)

        d[(d==0)] += 1

        d_2 = d**(-1/2)
        ma = (d_2[:,np.newaxis]) * A *  d_2
        return ma

    def sub_graph_ma(self, K, u_set,i_set):
        """
            aggregate K number
        """
        sub_agg_ma = self.sub_aggregation_ma(u_set, i_set)
        ma_dim = sub_agg_ma.shape[0]
        I = np.eye(ma_dim)
        # I = torch.tensor(I).to_sparse().to(dtype = torch.float32)

        Graph_ma = torch.zeros_like(sub_agg_ma)
        agg_ma_powk = I

        for _ in range(K):    
            Graph_ma += agg_ma_powk
            agg_ma_powk = agg_ma_powk @ sub_agg_ma
        Graph_ma += agg_ma_powk

        Graph_ma = 1/(1 + K) * Graph_ma

        return Graph_ma
    