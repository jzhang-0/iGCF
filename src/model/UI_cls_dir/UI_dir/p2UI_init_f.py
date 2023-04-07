from .p1base_f import base
from ..User_dir import User
from ..Item_dir import Item

class UI_init(base):
    def __init__(self, para_dict) -> None:
        super().__init__(para_dict)

    def add_new_user(self, user_id):
        user = User(user_id, self.para_dict)        
        self.data["user"][user_id] = user
        user.index = self.user_num
        self.index["user"]["index2id"][self.user_num] = user_id
        self.index["user"]["id2index"][user_id] = self.user_num
        self.user_num += 1

    def add_new_item(self, item_id):
        item = Item(item_id, self.para_dict)
        self.data["item"][item_id]= item
        item.index = self.item_num
        self.index["item"]["index2id"][self.item_num ] = item_id
        self.index["item"]["id2index"][item_id] = self.item_num
        self.item_num += 1


    def check_item_id_set_exists(self, item_id_set):
        for item_id in item_id_set:
            if item_id not in self.data["item"]:
                self.add_new_item(item_id)

    def check_user_id_set_exists(self, user_id_set):
        for user_id in user_id_set:
            if user_id not in self.data["user"]:
                self.add_new_user(user_id)
