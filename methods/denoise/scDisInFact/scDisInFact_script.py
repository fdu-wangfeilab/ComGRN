import sys, os
import torch
import numpy as np 
import pandas as pd
from scDisInFact import scdisinfact, create_scdisinfact_dataset
from scDisInFact import utils

import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.decomposition import PCA
import scipy.sparse as sp
import scanpy as sc
import argparse
import anndata
from scipy.sparse import issparse, csr_matrix

plt.rcParams['text.color'] = 'black'
sc.set_figure_params(dpi=100, facecolor='white')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='scDisInFact')

# 添加位置参数
parser.add_argument('--data_path', type=str, help='dir')
parser.add_argument('--raw_count_flag', action='store_true', help='raw_count_flag')

args = parser.parse_args()
data_path = args.data_path
adata = sc.read_h5ad(data_path+'demo.h5ad')

if isinstance(adata.X, csr_matrix):
        adata.X = adata.X.toarray()

adata.obs['batch'] = adata.obs['batch'].apply(lambda x:int(x))
adata.obs["condition"] = "none"
counts = adata.X
meta_cells = adata.obs

test_idx = []
train_idx = (meta_cells["batch"] != -1) 
data_dict = create_scdisinfact_dataset(counts[train_idx,:], meta_cells.loc[train_idx,:], condition_key = ["condition"], batch_key = "batch",
                                       log_trans=args.raw_count_flag)

# default setting of hyper-parameters
reg_mmd_comm = 1e-4
reg_mmd_diff = 1e-4
reg_kl_comm = 1e-5
reg_kl_diff = 1e-2
reg_class = 1
reg_gl = 1

Ks = [8, 2]

batch_size = 64
nepochs = 100
interval = 10
lr = 5e-4
lambs = [reg_mmd_comm, reg_mmd_diff, reg_kl_comm, reg_kl_diff, reg_class, reg_gl]
model = scdisinfact(data_dict = data_dict, Ks = Ks, batch_size = batch_size, interval = interval, lr = lr, 
                    reg_mmd_comm = reg_mmd_comm, reg_mmd_diff = reg_mmd_diff, reg_gl = reg_gl, reg_class = reg_class, 
                    reg_kl_comm = reg_kl_comm, reg_kl_diff = reg_kl_diff, seed = 0, device = device)
model.train()
losses = model.train_model(nepochs = nepochs, recon_loss = "NB")
_ = model.eval()

counts_test = counts[train_idx,:]
meta_test = meta_cells.loc[train_idx,:]
meta_test['batch'] = meta_test['batch'].astype(int)

counts_test_denoised = model.predict_counts(input_counts = counts_test, meta_cells = meta_test, condition_keys = ["condition"], 
                                            batch_key = "batch", predict_conds = None, predict_batch = 0)

# impute data
denoised_adata = anndata.AnnData(X=counts_test_denoised)
denoised_adata.obs_names = adata.obs_names
denoised_adata.var_names = adata.var_names

denoised_adata.write_h5ad(data_path+'denoise_adata_scDisInFact.h5ad')
