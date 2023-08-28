from ..GCNICF_Meta_dir import *

class GCNICF_Meta_V2(GCNICF_Meta):
    def __init__(self, para_dict, ec) -> None:
        super().__init__(para_dict, ec)
        self.epoch = para_dict.get("epoch", 10)
    
    def train_online_test(self,ui_cls, data_cls):
        ui_cls.update_data(data_cls.pre_training_data)
        self.model_init(ui_cls)

        df_init = 1

        train_data = data_cls.pre_training_data
        user_id2index_dict = ui_cls.index["user"]["id2index"]
        item_id2index_dict = ui_cls.index["item"]["id2index"]
        train_data["u_index"] = train_data["user_id"].apply(lambda x : user_id2index_dict[x])
        train_data["i_index"] = train_data["item_id"].apply(lambda x : item_id2index_dict[x])
        
        for i in range(self.epoch):

            # if i == 0:
            #     ui_cls_online = copy.deepcopy(ui_cls)                
            # else:
            #     self.model_train(ui_cls, train_data)                
            #     ui_cls_online = copy.deepcopy(ui_cls)

            self.model_train(ui_cls, train_data)                
            ui_cls_online = copy.deepcopy(ui_cls)


            self.online(ui_cls_online, data_cls)
            
            result = ui_cls_online.evaluate(data_cls)
            if df_init:
                result_df = pd.DataFrame(result, index=[i])
                df_init = 0
            else:
                new_df = pd.DataFrame(result, index=[i])
                result_df = pd.concat([result_df, new_df])
            
            result_df.to_csv( os.path.join(self.ec.save_path, "results_df.csv"))

    
    def model_train(self, ui_cls: UI_cls, train_data):
        model = self.gcn
        optimizer = optim.SGD(model.parameters(), lr = self.lr) 
        
        M = model.M
        N = model.N

        self.graph_ma = self.get_graph_ma(ui_cls).to_dense().to(device=self.device)
        # self.graph_ma = self.graph_ma
        
        bsize = 200
        iter_ = len(train_data) // bsize

        self.print(f"total iter num = {iter_}")
        for ii in range(iter_ + 1):

            start = ii*bsize
            end = min((ii+1) * bsize, len(train_data))
            data = train_data.iloc[start:end]
            E0 = self.get_E0(model)
            
            loss = 0

            u_index = list(data.u_index)
            i_index = list(data.i_index + M)
            rating = torch.tensor(list(data.rating)).to(device=self.device, dtype = self.dtype)

            u_embed =  self.graph_ma[u_index] @ E0           
            i_embed =  self.graph_ma[i_index] @ E0           

            loss = (((u_embed * i_embed).sum(1) - rating) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
            if (ii + 1) % 10 == 0:
                self.print(f"iteration {ii + 1}, loss = {loss.item()}")

        self.update_UI_cls(ui_cls, model)
    