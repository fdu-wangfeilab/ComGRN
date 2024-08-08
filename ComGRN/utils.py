

import scanpy as sc
import torch
import scipy.special
import numpy as np

from torch.distributions import Normal, kl_divergence


def plot_umap(adata, color=['celltype'], use_rep='X', save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3,3)) # type: ignore
    adata = adata.copy()
    print(adata)
    
    # pca make low dimension representation
    # sc.pp.scale(hvg_adata, max_value=10)
    # sc.tl.pca(adata)

    # using low dimension to calculate distance
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=color,wspace=0.4, legend_fontsize=14, show=None, save=save_filename)



def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]

    return optim_cls(params, **kwargs)


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss



def kl_loss(mu, var):
    var = torch.exp(0.5 * var)
    return kl_divergence(Normal(mu, var),
                         Normal(torch.zeros_like(mu),torch.ones_like(var))).sum(dim=1).mean()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)