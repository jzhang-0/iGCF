from ..Pop_dir import Pop
from ...UI_cls_dir import UI_cls
import numpy as np
import numpy.linalg as LA

class Pos(Pop):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)

    def rec(self, user, ui_cls:UI_cls):
        
        candidate_item_id_set = user.online_candidate_list
        
        N = len(candidate_item_id_set)
        pos_score = np.zeros(N)
        for i in range(N):
            iid = candidate_item_id_set[i]
            pos_score[i] = ui_cls.data["item"][iid].pos

        k = self.rec_list_len
        if len(candidate_item_id_set) < 5:
            k = len(candidate_item_id_set) 
        
        indices = self.kargmax(pos_score, k)
        n = indices[0]        

        
        rec_item_id = candidate_item_id_set[n]
        rec_item_id_list = [candidate_item_id_set[i] for i in indices]

        return rec_item_id, rec_item_id_list

