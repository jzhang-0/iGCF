from .p5graphinfo_f import graphinfo

class online(graphinfo):
    def __init__(self, para_dict) -> None:
        super().__init__(para_dict)

    def online_init(self, dls):
        self.online_id_stat = dls.online_id_stat
        
        for uid in self.data["user"]:
            user = self.data["user"][uid]
            user.online_candidate_list = list(set(dls.item_id_set) - set(user.interacted_item_id))

