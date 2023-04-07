from .p1basicinfo_f import basicinfo

class collectinfo(basicinfo):
    def __init__(self, id, para_dict) -> None:
        super().__init__(id, para_dict)
        self.interacted_user_id = []
        self.interacted_user_index = []
        self.feedback = []
        self.feedback01 = []
        
    @property
    def interacted_num(self):
        return len(self.interacted_user_id)


    @property
    def pop(self):
        return len(self.interacted_user_id)


    def update_interaction(self, user_id, user_index, feedback, feedback01):

        self.interacted_user_id.append(user_id)
        self.interacted_user_index.append(user_index)

        self.feedback.append(feedback)
        self.feedback01.append(feedback01)

        
        