from .p6online_f import online
import pandas as pd

class eval(online):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict)
        self.ec = ec
        self.print = self.ec.print
    

    def evaluate(self, data_cls):
        online_user_id_list = data_cls.online_id_stat
        T_prec_rall_list = [5, 10, 20, 40, 60, 80, 100, 120]

        result = {}

        for T in T_prec_rall_list:
            Precsion = pd.Series([self.data["user"][user_id].precision(T) for user_id in online_user_id_list]).mean()

            Recall = pd.Series([self.data["user"][user_id].recall(T, data_cls) for user_id in online_user_id_list]).mean()

            self.print(f"Precsion@{T}:{Precsion}")
            self.print(f"Recall@{T}:{Recall}")

            result[f"P@{T}"] = Precsion
            result[f"R@{T}"] = Recall
        
        
        nDCG_T_list = [10, 20, 40]

        for T in nDCG_T_list:
        
            nDCG = pd.Series([self.data["user"][user_id].nDCG(T, data_cls) for user_id in online_user_id_list]).mean() 
            self.print(f"nDCG_{self.rec_list_len}@{T}:{nDCG}")
            result[f"nDCG_{self.rec_list_len}@{T}"] = nDCG
        
        return result

    def evaluate_online(self, data_cls):
        self.evaluate(data_cls)



