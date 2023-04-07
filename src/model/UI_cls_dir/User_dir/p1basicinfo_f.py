from ...ModelBase_dir import ui_func_base
class basicinfo(ui_func_base):        
    def __init__(self, id, para_dict) -> None:
        super().__init__(para_dict)
        self.id = id 
        self.index = None

