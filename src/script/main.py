from pack_import import *

parser = argparse.ArgumentParser(description='Initialize Parameters!')
parser = parser_config(parser)
args = parser.parse_args()
args = config_args(args)

para_dict = args.__dict__
modeln = args.modelname
same_seeds(args.seed)
dls = LoadData(args.datan, seed = args.seed)

## ex config
para_dict["argv"] = sys.argv[1:]


## main
def main():
    ec = ExpConfig(para_dict)
    ui_cls = UI_cls(para_dict, ec)
    dls.divide_data(para_dict["percentage"], task = para_dict["task"])
    if modeln == "Random":
        model = Random(para_dict, ec)
    
    elif modeln == "ICF":
        model = ICF(para_dict, ec)
    
    elif modeln == "Pop":
        model = Pop(para_dict, ec)
    
    elif modeln == "GCNICF_Meta_V2":
        model = GCNICF_Meta_V2(para_dict, ec)


    else:
        parser.print_help()
        assert False
    


    if modeln in ["GCNICF", "GCNICF_LVI", "GCNICF_Meta", "GCNICF_Meta_V2"]:
        model.train_online_test(ui_cls, dls)
        if para_dict["save_cls"]:
            ec.save_cls(model, "model.class")
            ec.save_cls(ui_cls, "ui_cls.class")
            ec.save_cls(dls, "dls.class")
            # sys.exit(0)
    
    else:
        model.pretrain(ui_cls, dls.pre_training_data)

        if para_dict["save_cls"]:
            ec.save_cls(model, "model.class")
            ec.save_cls(ui_cls, "ui_cls.class")
            ec.save_cls(dls, "dls.class")
            # sys.exit(0)
        
        model.online(ui_cls, data_cls=dls)
        ui_cls.evaluate(dls)


if __name__ == "__main__":
    main()
        