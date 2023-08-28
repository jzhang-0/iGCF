from ...ModelBase_dir import ModelBase
from ...UI_cls_dir import UI_cls
import numpy as np
import numpy.linalg as LA
import random
# from mypackages.decorator import for_all_methods
# @for_all_methods(profile)
class Random(ModelBase):
    """
        worst case
    """
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)

    def rec(self, user, ui_cls):
        
        candidate_item_id_set = user.online_candidate_list
        k = self.rec_list_len
        if len(candidate_item_id_set) < 5:
            k = len(candidate_item_id_set) 

        id_set = random.sample(candidate_item_id_set, k)
        rec_item_id = id_set[0]
        rec_item_id_list = list(id_set)

        return rec_item_id, rec_item_id_list

    # def online_update_u(self, user):
    #     user.online_round += 1