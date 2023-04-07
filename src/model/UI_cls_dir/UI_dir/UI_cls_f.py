
from ..User_dir import User
from ..Item_dir import Item
import numpy as np
import pandas as pd
import logging
from .p9ui_similarity import ui_s
# from mypackages.decorator import for_all_methods
import torch
from torch import tensor

# @for_all_methods(profile)
class UI_cls(ui_s):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)

    def eval_loss(self):
        self.get_U()
        self.get_I()

        R = self.U @ self.I.T # M * N

        s = 0
        for u_index in range(self.user_num):
            u_id = self.index["user"]["index2id"][u_index]
            interacted_item_index = self.data["user"][u_id].interacted_item_index
            ratings = self.data["user"][u_id].feedback

            for i_index,r in zip(interacted_item_index, ratings):
                s += (r - R[u_index, i_index])**2
        
        return s