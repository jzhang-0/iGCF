from .p2collectinfo_f import collectinfo

class posinfo(collectinfo):
    def __init__(self, id, para_dict) -> None:
        super().__init__(id, para_dict)
        self.tem_pos = -1
        self.tem_len = -1

    @property
    def pos(self):
        if len(self.interacted_user_id) == 0:
            return 0
        
        if self.tem_len == len(self.interacted_user_id):
            return self.tem_pos
        else:
            # self.tem_pos = sum(self.feedback) / len(self.interacted_user_id)
            self.tem_pos = sum(self.feedback) 
            self.tem_len = len(self.interacted_user_id)
            return self.tem_pos

