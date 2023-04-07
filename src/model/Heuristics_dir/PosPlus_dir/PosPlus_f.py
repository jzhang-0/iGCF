from ..Pos_dir import *


class PosPlus(Pos):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)
    

    # def online_update_model(self, ui_cls):
        

    def rec(self, user, ui_cls: UI_cls):
        ui_cls.update_UI_similarity_MA()

        candidate_item_id_set = user.online_candidate_list

        u_index = user.index

        similarityMA = ui_cls.UI_similarityMA
        u_scored = similarityMA[u_index]

        weight_vec = similarityMA @ u_scored
        weight_vec = np.where(weight_vec > 0, weight_vec, 0)
        if weight_vec.sum() == 0:
            weight_vec = np.ones_like(weight_vec)
        
        pos_score_all = (weight_vec[:, np.newaxis] * similarityMA).sum(0)

        candidate_item_index = [ui_cls.index["item"]["id2index"][iid] for iid in candidate_item_id_set]

        # N = len(candidate_item_id_set)
        # pos_score = np.zeros(N)
        # for i in range(N):
        #     iid = candidate_item_id_set[i]
        #     pos_score[i] = ui_cls.data["item"][iid].pos

        pos_score = pos_score_all[candidate_item_index]

        k = self.rec_list_len
        if len(candidate_item_id_set) < 5:
            k = len(candidate_item_id_set) 
        
        indices = self.kargmax(pos_score, k)
        n = indices[0]        

        
        rec_item_id = candidate_item_id_set[n]
        rec_item_id_list = [candidate_item_id_set[i] for i in indices]

        return rec_item_id, rec_item_id_list
