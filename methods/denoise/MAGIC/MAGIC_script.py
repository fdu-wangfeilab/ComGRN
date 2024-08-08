import magic
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='MAGIC')

# 添加位置参数
parser.add_argument('--data_path', type=str, help='data path')
parser.add_argument('--out_path', type=str, help='out path')

args = parser.parse_args()
data_path = args.data_path 
out_path = args.out_path

adata = sc.read_h5ad(data_path+'demo.h5ad')

noisy_adata = adata

magic_operator = magic.MAGIC()
X_magic = magic_operator.fit_transform(noisy_adata.X)

denoised_adata = anndata.AnnData(X=X_magic)
denoised_adata.obs_names = noisy_adata.obs_names
denoised_adata.var_names = noisy_adata.var_names

denoised_adata.write_h5ad(out_path+'denoise_adata_MAGIC.h5ad')
