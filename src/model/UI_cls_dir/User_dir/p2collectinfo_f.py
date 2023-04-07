from .p1basicinfo_f import basicinfo
import numpy as np

class collectinfo(basicinfo):
    def __init__(self, id, para_dict) -> None:
        super().__init__(id, para_dict)

        self.interacted_item_id = []
        self.interacted_item_index = []

        self.feedback = []
        self.feedback01 = []

    @property
    def interacted_num(self):
        return len(self.interacted_item_id)


    def update_interaction(self, item_id, item_index, feedback, feedback01):
        self.interacted_item_id.append(item_id)
        self.interacted_item_index.append(item_index)

        self.feedback.append(feedback)
        self.feedback01.append(feedback01)

    @property
    def pos_index(self):
        return np.array(self.interacted_item_index)[np.where(np.array(self.feedback01)==1)[0]]