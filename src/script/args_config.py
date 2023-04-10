
import torch,os

def parse_default_config(parser):
    parser.add_argument('--seed', default='123', type = int, help='random seed')  
    parser.add_argument('--mu_u', default = 0, type = float, help='init mean vec of user embedding')   
    parser.add_argument('--mu_i', default = 0, type = float, help='init mean vec of item embedding')   

    parser.add_argument('--online_rec_total_num', default = 40, type = int, help='online_rec_total_num')   


    parser.add_argument('--sigma_u', default = 1, type = float, help='init var vec of user embedding')   
    parser.add_argument('--sigma_i', default = 1, type = float, help='init var vec of item embedding')    
    
    parser.add_argument('-p', '--percentage', default = 0.5, type=float, help='percentage of pretrain data')

    ### ICF
    parser.add_argument('--lambda_u', default = 1, type = float, help='lambda_u reg')   
    parser.add_argument('--lambda_i', default = 1, type = float, help='lambda_i reg')    
    parser.add_argument('--sigma', default = 1, type = float, help='noise variance')   
    parser.add_argument('--max_iter', default = 10, type = int, help='max_iter in coordinates descent')   
    
    ### LGCNICF
    parser.add_argument('--online_iter', default = 50, type = int, help='max_iter in coordinates descent')   
    parser.add_argument('--init', default = 1, type = int, help='normal init')   

    parser.add_argument('--save_cls', default = 0, type = int, help='save class')   

    parser.add_argument('--test_iters', default = 1000, type = int, help='test_iters')   

    ### LVI
    parser.add_argument('--LVI_iters', default = 2, type = int, help='LVI EM iters')   
        

    return parser

def parser_config(parser):
    parse_default_config(parser)

    parser.add_argument('--rec_list_len', default = 1, type = int, help='rec_list_len')   
    parser.add_argument('--task', default = "test", type = str, help='test / coldstart / ')   
    

    parser.add_argument('--exp_id', default='test', type=str, help='exp id')     
    parser.add_argument('--datan', default='-1', type=str, help='the name of dataset')
    parser.add_argument('-m', '--modelname', default="-1", type=str, help='the name of model')
    
    parser.add_argument('-d', '--embedding_dim', default = 20, type=int, help='embedding dim')    
    parser.add_argument('-E', '--explore_method', default='None', type=str, help='explore method')
    parser.add_argument('-v', '--explore_v', default = 1, type=float, help='explore strength')
    parser.add_argument('--lr', default = 0.01, type=float, help='lr')
    parser.add_argument('--K', default = 3, type=int, help='layers')

    parser.add_argument('--cuda', default = -1, type = int, help='gpu number')
    
    parser.add_argument('--lossfunc', default = "reg", type = str, help='loss function, reg or classify')
    
    ### graph aggreation
    parser.add_argument('--test_para', default = "0", type = str, help='for test constructing graph neightbors matrix.')

    ### Meta
    parser.add_argument('--meta_update', default = "0", type = str, help='0:no update. half: 1/2. lin: (t = 0, 1, 0) (t = total round, 0, 1)')

    return parser

def config_args(args):
    if args.cuda == -1:
        args.device = "cpu"
    else:
        args.device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        
    return args

        