from .p3onlineinfo_f import onlineinfo
import numpy as np

class usereval(onlineinfo):
    def __init__(self, id, para_dict) -> None:
        super().__init__(id, para_dict)

    def precision(self, T):
        if self.online_round < T:
            return None
                
        return (np.array(self.online_feedback01[:T])).sum()
    
    def recall_(self, T, satisﬁed_num):
        if self.online_round < T:
            return None
        
        return (np.array(self.online_feedback01[:T])).sum() / satisfied_num

    def recall(self, T, dls):
        satisﬁed_num = dls.exposure_satisﬁed[self.id] # - (np.array(self.online_feedback01)).sum()
        return self.recall_(T, satisfied_num)

    def DCG(self, list_rating): 
        """
            len(list_rating) = K
        """
        s = 0
        K = len(list_rating)
        for k in range(K):
            rating = list_rating[k]
            s += (2**rating - 1) / np.log2(k + 2)
        return s

    def IDCG(self, K):
        """
            ideal DCG
        """
        list_rating = np.ones(K)
        return self.DCG(list_rating)

    def nDCG(self, T, data_cls):
        if self.online_round / self.rec_list_len < T:
            return None   

        s = 0 
        K = len(self.online_rec_list_feedback01[0])
        for t in range(T):

            list_rating = self.online_rec_list_feedback01[t]

            if len(self.online_rec_list_feedback01[t]) < K:
                continue
            
            s += self.DCG(list_rating) / self.IDCG(K)
        
        return s
