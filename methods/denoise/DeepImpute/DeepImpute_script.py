import pandas as pd
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import numpy as np
import argparse

from deepimpute.multinet import MultiNet
from sklearn.cluster import KMeans
import tensorflow as tf
from scipy.sparse import issparse, csr_matrix

parser = argparse.ArgumentParser(description='DeepImpute')

# 添加位置参数
parser.add_argument('--data_path', type=str, help='data path')
parser.add_argument('--out_path', type=str, help='out path')

args = parser.parse_args()
data_path = args.data_path 
out_path = args.out_path

adata = sc.read_h5ad(data_path+'demo.h5ad')

noisy_adata = adata

model = MultiNet()

# Using custom parameters
NN_params = {
        'learning_rate': 1e-4,
        'batch_size': 64,
        'max_epochs': 200,
        'ncores': 5,
        'sub_outputdim': 512,
        'architecture': [
            {"type": "dense", "activation": "relu", "neurons": 200},
            {"type": "dropout", "activation": "dropout", "rate": 0.3}]
    }

multinet = MultiNet(**NN_params)

# train data construction
temp_data = pd.DataFrame(noisy_adata.X)

# fit and predict
model.fit(temp_data)
imputed = model.predict(temp_data)

# impute data
denoised_adata = anndata.AnnData(X=imputed)
denoised_adata.obs_names = noisy_adata.obs_names
denoised_adata.var_names = noisy_adata.var_names

denoised_adata.write_h5ad(out_path+'denoise_adata_DeepImpute.h5ad')
