
import torch
import numpy as np
from preprocessing.load_data import *
import random
import os
from glob import glob
import xarray as xr
from utils import quick_helpers as qh
import pandas as pd
from models.bilstm import BiLSTM, BiLSTM_with_Flux
from models.cloudy_bilstm import Cloudy_BiLSTM, Cloudy_BiLSTM_with_Flux
from models.simple_nn import SimpleNN, SimpleNN_with_Flux
from models.flux import Flux
import yaml
from argparse import Namespace

var_to_normvar = { # this mapping was used because sometimes variables have different names depending on their format e.g. .nc or .grb
    "cli":"cli", 
    "clw":"clw", 
    "cl":"cl",
    "pfull":"pfull", 
    "ta":"ta",
    "hur":"hur",
    "extra_3d_cli":"extra_3d_cli", 
    "extra_3d_clw":"extra_3d_clw", 
    "extra_3d_pfull":"extra_3d_pfull", 
    "extra_3d_ta":"extra_3d_ta",
    "extra_3d_hus":"extra_3d_hus",
    "cosmu0":"cosmu0", 
    "extra_2d_albedo":"extra_2d_albedo",
    "extra_2d_prw":"extra_2d_prw", 
    "extra_2d_clivi":"extra_2d_clivi", 
    "extra_2d_cllvi":"extra_2d_cllvi",
    "rsut":"rsut",
    "rlut":"rlut",
    "rsu":"rsu",
    "rlu":"rlu",
    "rsd":"rsd",
    "rld":"rld",
    "ts_rad": "ts_rad",
    "emissivity": "emissivity",
    "rho": "rho",
    "hfls": "hfls",
    "hfss": "hfss",
    "o3": "o3",
    "rsds": "rsds", 
    "rsus": "rsus",
    "rlds": "rlds", 
    "rlus": "rlus",
    "rvds_dir": "rvds_dir",
    "rvds_dif": "rvds_dif", 
    "rpds_dir": "rpds_dir", 
    "rpds_dif": "rpds_dif", 
    "rnds_dir": "rnds_dir", 
    "rnds_dif": "rnds_dif",
    "tend_ta_rsw": "tend_ta_rsw",
    "tend_ta_rlw": "tend_ta_rlw",
    "dz": "dz",
    "extra_3d_in_cli": "extra_3d_cli", 
    "extra_3d_in_clw": "extra_3d_clw",  
    "extra_3d_in_hus": "extra_3d_hus",
    "extra_2d_in_ts_rad": "extra_2d_ts_rad", 
    "extra_3d_in_cl": "cl", 
    "extra_3d_in_o3": "o3", 
    "extra_3d_in_t": "extra_3d_ta", 
    "extra_3d_in_rho": "rho",
    "extra_2d_in_toa_flux": "toa",
}
   

def load_model(args, extra_shape=0):
    model = create_model(args, extra_shape=extra_shape)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    return model

def create_model(args, extra_shape=0, shap=False):
    if "FLUX" in args.model_type:
        if "SW" in args.model_type:
            in_vars = args.variables["in_vars"][:-4]
            e=4
        elif "LW" in args.model_type:
            in_vars = args.variables["in_vars"]
            e=0
        baseline_model = Cloudy_BiLSTM(args.model_type, 
                                       output_features=1, 
                                       extra_shape=0, 
                                       norm_file=args.norm_file, 
                                       in_vars=in_vars, 
                                       out_vars = args.variables["out_vars"][:1])
        baseline_model.load_state_dict(torch.load(args.baseline_path))

        model = Cloudy_BiLSTM_with_Flux(baseline_model, 
                                        args.model_type, 
                                        output_features=args.n_flux_vars, 
                                        in_vars=in_vars, 
                                        out_vars = args.variables["out_vars"],
                                        extra_shape=extra_shape,
                                        lr=args.learning_rate,
                                        weight_decay=args.weight_decay,
                                        shap=shap)
    elif "HR" in args.model_type:
        model = Cloudy_BiLSTM(args.model_type, 1, args.norm_file, args.variables["in_vars"], 
                              out_vars = args.variables["out_vars"], extra_shape=extra_shape, weight_decay=args.weight_decay, 
                              lr=args.learning_rate, hidden_size=args.hidden_size)
    return model
    
def setup_args_and_load_data(train=True, args_via_commandline=True, cache =None, *overwrite_args):
    
    if args_via_commandline:
        config_file = qh.arg_parser()
        with open(config_file.config_file, "r") as file:
            args = yaml.safe_load(file)
    else:
        args={}
    if overwrite_args: # for executing the script in a notebook
        for key, val in overwrite_args[0].items():
            args[key] = val
    args = Namespace(**args)
    # set random seeds for reproducibility
    np.random.seed(args.seed) # random seed for numpy
    random.seed(args.seed) # random seed for python
    torch.manual_seed(args.seed) # random seed for pytorch

    if args.dev:
        args.folder="test"
    args.save_folder = "results/"+ args.folder +"/"
    args.result_folder = "results/"+ args.folder +f"/{args.model_type}_{args.eval_on}/"
    args.model_path=args.save_folder+f"baseline_{args.model_type}/model.pth"
    args.pretrained_path =  f"results/{args.pretrained}/baseline_{args.model_type}/model.pth"
    args.checkpoint_path = args.save_folder+f"baseline_{args.model_type}/my_checkpoint/"

    months = ["november", "january", "april", "july" ] # 
    months = months[:1] if args.dev else months
    s = "SW" if "SW" in args.model_type else "LW"
    if args.train_path == "":
        args.train_path = [f"{args.data}{m}/train_{s}.zarr" for m in months]
    if args.val_path == "":
        args.val_path = [f"{args.data}{m}/val_{s}.zarr" for m in months]
    if args.test_path == "":
        args.test_path = [f"{args.data}{m}/test_{s}.zarr" for m in months]

    os.makedirs(args.result_folder, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)

    args.norm_file = pd.read_pickle(args.norm_file_path)

    #Data
    args.grid = xr.load_dataset( args.grid_path)
    args.vgrid = xr.load_dataset( args.vgrid_path)
    args.vert_int_weights = np.load(args.vert_int_weights_path)

    if train:
        args.coarse_train = Np_Dataset(args.train_path, args.variables, vweights=args.vert_int_weights)    
        args.coarse_val = Np_Dataset(args.val_path, args.variables, vweights=args.vert_int_weights)

        print("Getting shapes right...")
        args.train_steps = args.coarse_train.__len__()
        args.validation_steps = args.coarse_val.__len__()

        item = args.coarse_train.__getitem__(0)
        args.extra_shape = args.coarse_train.extra_shape
    
    else:
        args.result_file = args.save_folder+'results_'+ args.model_type + "_" + args.eval_on +'.pickle' 

        paths = {"train": args.train_path,
                "validation": args.val_path,
                "test": args.test_path}
        eval_path=paths[args.eval_on]
        args.coarse_test = Np_Dataset(eval_path, args.variables, vweights=args.vert_int_weights, testset = True)
        
        item = args.coarse_test.__getitem__(0)
        args.extra_shape = 0

    args.x_shape = item[0].shape[0] # first batch, x, first element in batch
    args.y_shape = item[1].shape[0] # first batch, y, first element in batch
    if args.x_mode == "horizontal":
        args.x_shape = item[0].shape # first batch, x, first element in batch
    if args.y_mode == "horizontal":
        args.y_shape = item[1].shape[-1] # first batch, y, first element in batch

    print("x shape: ", args.x_shape)
    print("y shape: ", args.y_shape)
    print("extra shape: ", args.extra_shape)

    return args


