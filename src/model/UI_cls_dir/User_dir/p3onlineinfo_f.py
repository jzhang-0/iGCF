from .p2collectinfo_f import collectinfo

class onlineinfo(collectinfo):
    def __init__(self, id, para_dict) -> None:
        super().__init__(id, para_dict)

        self.online_round = 0
        self.online_candidate_list = []


        self.online_rec_list = []
        self.online_rec_list_feedback = [] 
        self.online_rec_list_feedback01 = []

    @property
    def online_feedback01(self):
        assert self.online_round > 0 
        return self.feedback01[ -self.online_round : ]

    @property
    def online_interacted_item_id(self):
        assert self.online_round > 0
        return self.interacted_item_id[ - self.online_round:]
