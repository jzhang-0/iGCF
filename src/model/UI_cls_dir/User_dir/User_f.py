from .p4usereval_f import usereval

class User(usereval):
    def __init__(self, id, para_dict) -> None:
        super().__init__(id, para_dict)
        self.rec_list_len = para_dict.get("rec_list_len", 1)