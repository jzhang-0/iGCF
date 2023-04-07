
class base:
    def __init__(self, para_dict) -> None:
        self.para_dict = para_dict
        self.rec_list_len = para_dict.get("rec_list_len", 1)
        

        self.user_num = 0
        self.item_num = 0
        self.embedding_dim = para_dict.get("embedding_dim", 20)

        self.data = {}
        self.data["user"] = {}
        self.data["item"] = {}

        self.index = {}
        self.index["user"] = {}
        self.index["user"]["id2index"] = {}
        self.index["user"]["index2id"] = {}
        
        self.index["item"] = {} 
        self.index["item"]["id2index"] = {}
        self.index["item"]["index2id"] = {}
