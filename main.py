
import sys
sys.path.append('/home/dongjiayi/workbench/denoise/ComGRN')

import scanpy as sc
import numpy as np
import torch
import os
import warnings
import tqdm
import pandas as pd
import argparse

from omegaconf import OmegaConf

from ComGRN.utils import *
from ComGRN.model import *
from ComGRN.data_processing import *
from ComGRN.eval import *

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='ComGRN')
# 添加位置参数
parser.add_argument('--opt_file', default=None, type=str, help='opt file')
args = parser.parse_args()

opt = OmegaConf.load(args.opt_file) 

if opt.viz:
    color = [opt.data.batch_name, opt.data.state_name]

# setting seed
seed = opt.reproduc.seed
set_seed(seed)

device = torch.device('cuda:0')

data, domain_list, n_domain = prepare_data(opt.data, color)

train_dataloader, val_dataloader = convert_adata_to_dataloader(data,
                                         domain_list,
                                         val_split=opt.model.val_split,
                                         batch_size=opt.model.batch_size,
                                         is_shuffle=True,
                                         seed=seed,
                                         )

# setting training
input_dim = data.X.shape[1]

mdl = ComGRN(
    input_dim = input_dim,
    num_domains = n_domain,
    hparams = opt.model,
)

epoch = opt.model.epochs
mdl.fit(dataloader=train_dataloader, 
            num_epoch=epoch,
            device=device,
            grad_clip=True,
            )

data = mdl.evaluate(
    dataloader=train_dataloader,
    data=data,
    device=device,
)
    
# save data
out_dir = opt.data.data_dir
denoised_adata = sc.AnnData(X=data.obsm['X_denoised'])
denoised_adata.obs_names = data.obs_names
denoised_adata.var_names = data.var_names
denoised_adata.write_h5ad(os.path.join(out_dir, f'denoise_adata_ComGRN.h5ad'))

# save model
torch.save(mdl.state_dict(), os.path.join(out_dir, f'model.pth'))