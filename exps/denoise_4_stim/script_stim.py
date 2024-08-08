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
parser.add_argument('--opt_file', default=None, type=str, help='opt file')
args = parser.parse_args()

opt = OmegaConf.load(args.opt_file) 

if opt.viz:
    color = [opt.data.batch_name, opt.data.state_name]

seed = opt.reproduc.seed
set_seed(seed)
device = torch.device('cuda:0')

data, domain_list, n_domain = prepare_data(opt.data, color, task=opt.task_name)

train_dataloader, val_dataloader = convert_adata_to_dataloader(data,
                                         domain_list,
                                         val_split=opt.model.val_split,
                                         batch_size=opt.model.batch_size,
                                         is_shuffle=True,
                                         seed=seed,
                                         )

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

print("===================visualization========================")

sc.pp.neighbors(data, use_rep='X_denoised')
sc.tl.umap(data)
sc.pl.umap(data, color=color, wspace=0.4, legend_fontsize=14,
            save=f'_train_denoised.pdf',show=False)  

sc.pp.neighbors(data, use_rep='X_recon')
sc.tl.umap(data)
sc.pl.umap(data, color=color, wspace=0.4, legend_fontsize=14,
        save=f'_train_recon.pdf',show=False) 

sc.pp.neighbors(data, use_rep='X_latent')
sc.tl.umap(data)
sc.pl.umap(data, color=color, wspace=0.4, legend_fontsize=14,
        save=f'_train_latent.pdf',show=False) 


# compare with clean data
print("===================compare with clean data=====================")
    
noisy_data = data
clean_data = sc.read_csv(opt.data.data_dir + "clean_data.csv", first_column_names=True).T
sc.pp.normalize_total(clean_data, target_sum=10000)
sc.pp.log1p(clean_data)

print('X_denoised')
clean_noisy_corr, clean_denoised_corr, corr_zeros = eval_corr(clean_data.X, noisy_data.X, 
                                                            noisy_data.obsm['X_denoised'], True)

