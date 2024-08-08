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

data, domain_list, n_domain = prepare_data(opt.data, color, args.batch_num, opt.task_name)

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


# GRN inference
print("===================GRN inference===================")
iden_mat = np.ones((input_dim, input_dim)) - np.identity(input_dim)
W_est = mdl.dag_mlp.adj(clean_flg=True).cpu().detach().numpy() * iden_mat
print(W_est.shape)
pred_path = opt.data.data_dir + 'rankedEdge_idag.csv'

gene_list = data.var_names.to_series().apply(lambda x: x.upper())
res = extractEdgesFromMatrix(W_est, gene_list)
res.to_csv(pred_path)

for net in ['/Non-Spec-network.csv', '/Spec-network.csv', '/STR-network.csv', '/lofgof-network.csv', '/network.csv', '/grn.csv']:
    ref_path = opt.data.data_dir + net
    if not os.path.exists(ref_path):
        continue
    pred_path = pred_path
    EPratio, EP = eval_EP_info(ref_path, pred_path)
    AUPRratio, AUPR, roc = eval_AUPR_info(ref_path, pred_path)
    print(net)
    print("EPratio", np.round(EPratio,4))
    print("EP", np.round(EP,4))
    print("AUPRratio", np.round(AUPRratio,4))
    print("AUPR", np.round(AUPR, 4))
    print("AUROC", np.round(roc, 4))
    