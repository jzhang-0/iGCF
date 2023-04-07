from .p3UI_index_f import UI_index
import pandas as pd
import numpy as np

class collectinfo(UI_index):
    def __init__(self, para_dict) -> None:
        super().__init__(para_dict)
        self.datanum = 0
        self.df = pd.DataFrame(np.zeros([int(1e8), 4], dtype=int), columns = ["user_id", "item_id", "rating", "feedback01"])
        


    def update_data(self, data, ex_info = None):
        if type(data) == pd.DataFrame:
            df = data[["user_id", "item_id", "rating", "feedback01"]]
        elif type(data) == tuple and len(data) == 4:
            df = pd.DataFrame(np.array(data)[np.newaxis,:], columns =["user_id", "item_id", "rating", "feedback01"])
        

        N = len(df)
        self.df[self.datanum:self.datanum + N] = df
        self.datanum += N
        
        for index, row in df.iterrows():
            user_id = row.user_id
            item_id = row.item_id
            rating = row.rating 
            feedback01 = row.feedback01
            
            user = self.get_user(user_id)
            item = self.get_item(item_id)

            user.update_interaction(item_id, item.index, rating, feedback01)
            
            item.update_interaction(user_id, user.index, rating, feedback01)
