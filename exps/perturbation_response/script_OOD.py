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

def train_model(opt, prediction_type, seed, device):
    train_data, train_domain_list, test_data, test_domain_list, n_domain = \
        prepare_data_for_OOD(opt.data, prediction_type)

    train_dataloader, val_dataloader = convert_adata_to_dataloader(train_data,
                                            train_domain_list,
                                            val_split=opt.model.val_split,
                                            batch_size=opt.model.batch_size,
                                            is_shuffle=True,
                                            seed=seed,
                                        )

    mdl = ComGRN(
        input_dim = train_data.X.shape[1],
        num_domains = n_domain,
        hparams = opt.model,
    )

    mdl.fit(dataloader=train_dataloader, 
                num_epoch=opt.model.epochs,
                device=device,
                grad_clip=True,
            )

    train_data = mdl.data_transform(
        dataloader=train_dataloader,
        data=train_data,
        domain_id=1,
        device=device,
    )

    raw_train_adata = sc.AnnData(X=train_data.X, obs=train_data.obs, var=train_data.var)
    trans_train_adata = sc.AnnData(X=train_data.obsm['X_trans'], obs=train_data.obs, var=train_data.var)
    trans_train_adata.obs['condition'] = [2] * trans_train_adata.shape[0]
    trans_train_adata_1 = trans_train_adata[raw_train_adata.obs['condition'] == 0]
    concat_train_adata = raw_train_adata.concatenate(trans_train_adata_1)       
    sc.pp.neighbors(concat_train_adata)
    sc.tl.umap(concat_train_adata)
    sc.pl.umap(concat_train_adata, color=color, wspace=0.4, legend_fontsize=14, 
             save=f'_ood_tcc_{prediction_type}.pdf',show=False) 

    # zero-shot
    test_dataloader, val_dataloader = convert_adata_to_dataloader(test_data,
                                                test_domain_list,
                                                val_split=opt.model.val_split,
                                                batch_size=opt.model.batch_size,
                                                is_shuffle=True,
                                                seed=seed,
                                            )
    
    test_data = mdl.data_transform(
        dataloader=test_dataloader,
        data=test_data,
        domain_id=1,
        device=device,
    )
    
    raw_test_data = sc.AnnData(X=test_data.X, obs=test_data.obs, var=test_data.var)
    trans_test_data = sc.AnnData(X=test_data.obsm['X_trans'], obs=test_data.obs, var=test_data.var)
    trans_test_data.obs['condition'] = [2] * trans_test_data.shape[0]
    trans_adata_1 = trans_test_data[raw_test_data.obs['condition'] == 0]
    concat_adata = raw_test_data.concatenate(trans_adata_1)   
    
    del mdl 
    
    return concat_adata


parser = argparse.ArgumentParser(description='ComGRN')
# 添加位置参数
parser.add_argument('--opt_file', default=None, type=str, help='opt file')
args = parser.parse_args()

opt = OmegaConf.load(args.opt_file) 
# setting seed
seed = opt.reproduc.seed
set_seed(seed)

device = torch.device('cuda:0')

color = [opt.data.condition_name, opt.data.type_name]

type_list = get_cell_type(opt.data)

concat_adata_list = []
for prediction_type in type_list:
    print(prediction_type, " prediction process")
    concat_adata = train_model(opt, prediction_type, seed, device)
    concat_adata_list.append(concat_adata)

data_name = opt.data.data_dir.split('/')[-2]
concat_adata = sc.concat(concat_adata_list)       
sc.pp.neighbors(concat_adata)
sc.tl.umap(concat_adata)
sc.pl.umap(concat_adata, color=color, wspace=0.4, legend_fontsize=14, 
           save=f'_ood_tcc_{data_name}.pdf',show=False)

# save model and data
out_dir = opt.outf
concat_adata.write_h5ad(os.path.join(out_dir, f'{data_name}_adata_ComGRN.h5ad'))

