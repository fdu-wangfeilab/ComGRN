import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import scvi
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix

# 忽视warning
import warnings
warnings.filterwarnings('ignore')

import argparse
import os

parser = argparse.ArgumentParser(description='scVI')

# 添加位置参数
# 添加位置参数
parser.add_argument('--data_path', type=str, help='data path')
parser.add_argument('--out_path', type=str, help='out path')

args = parser.parse_args()
data_path = args.data_path 
out_path = args.out_path

adata = sc.read_h5ad(data_path+'demo.h5ad')

noisy_adata = adata

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
vae = scvi.model.SCVI(adata)
vae.to_device('cuda:1')
vae.train()
adata.obsm["X_scVI"] = vae.get_latent_representation()
adata.obsm["X_norm"] = vae.get_normalized_expression()

denoised_adata = anndata.AnnData(X=adata.obsm["X_norm"])
denoised_adata.obs_names = noisy_adata.obs_names
denoised_adata.var_names = noisy_adata.var_names

denoised_adata.write_h5ad(out_path+'denoise_adata_scVI.h5ad')
