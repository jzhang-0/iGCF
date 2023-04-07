from .p8get_para_f import *

class ui_s(get_para):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)
        self.UI_similarity_time = 0

        self.UI_similarity_preMA =  np.zeros([10000, 10000])
    

    @property
    def UI_similarityMA(self):
        return self.UI_similarity_preMA[:self.user_num, :self.item_num]
    

    def update_UI_similarity_MA(self):
        if self.UI_similarity_time < self.datanum:
            df = self.df[ self.UI_similarity_time : self.datanum ]

            for index, row in df.iterrows():
                user_id = row.user_id
                item_id = row.item_id
                # rating = row.rating 
                feedback01 = row.feedback01
                
                user = self.get_user(user_id)
                item = self.get_item(item_id)

                self.UI_similarity_preMA[user.index, item.index] = 2 * feedback01 - 1
            
            self.UI_similarity_time = self.datanum
        
