from .p2UI_init_f import UI_init

class UI_index(UI_init):
    def __init__(self, para_dict) -> None:
        super().__init__(para_dict)

    def get_user(self, user_id):
        if user_id in self.data["user"]:
            user = self.data["user"][user_id]                
        else:
            self.add_new_user(user_id)
            user = self.data["user"][user_id]
        return user

    def get_item(self, i_id):
        if i_id in self.data["item"]:
            item = self.data["item"][i_id]                
        else:
            self.add_new_item(i_id)
            item = self.data["item"][i_id]
        return item

    def get_user_by_id(self, user_id):
        return self.get_user(user_id)

    def get_item_by_id(self, item_id):
        return self.get_item(item_id)

    def get_user_by_index(self, u_index):
        u_id = self.index["user"]["index2id"][u_index]
        return self.get_user(u_id)

    def get_item_by_index(self, i_index):
        i_id = self.index["item"]["index2id"][i_index]
        return self.get_item(i_id)
