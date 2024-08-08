import numpy as np
import torch
import scanpy as sc 

from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix

from utils import *

def batch_scale(adata, batch_name='batch'):
    """
    Batch-specific scale data
    
    Parameters
    ----------
    adata: AnnData
        Single cell data.
    batch_name: str
        Batch name in adata. DEFAULT: 'batch'
    
    Returns
    -------
    np.array
        Scaled data.
    """
    for b in adata.obs[batch_name].unique():
        idx = np.where(adata.obs[batch_name]==b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        adata.X[idx] = scaler.transform(adata.X[idx])
    return adata.X

class SingleCellDataset(Dataset):
    """
    Dataloader of single-cell data
    """
    def __init__(self, adata, domain_list, use_layer='X'):
        """
        Create a SingleCellDataset object
            
        Parameters
        ----------
        adata: AnnData
            Single cell data.
        domain_list: list
            Domain list of correspondence data.
        use_layer: str
            Layer to use. DEFAULT: 'X'
        """
        self.data = adata.X.squeeze().astype(float)
        self.domain_list = domain_list
        self.shape = adata.shape
        self.use_layer = use_layer
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns
        -------
        int
            Length of the dataset.
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Parameters
        ----------
        idx: int
            Index of the item.
        
        Returns
        -------
        tuple
            A tuple containing the data, domain ID, and index.
        """
        x = self.data[idx]    
        domain_id = self.domain_list[idx]
        return x, domain_id, idx


# a function convert anndata to dataloader
def convert_adata_to_dataloader(adata, domain_list, val_split=0.1, batch_size=128, is_shuffle=True, seed=0):
    # simple converter
    
    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X
    else:
        adata.X = adata.X.toarray()
    batch = domain_list.values
     
    dataset = SingleCellDataset(adata, batch)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size != 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_dataset = dataset
        val_dataset = dataset
        
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=is_shuffle
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=is_shuffle
    )
    
    return train_dataloader, val_dataloader
 

def read_data(opt):
    data_format = opt.data_name.split('.')[-1]
    
    if data_format == 'h5ad':
        adata = sc.read_h5ad(opt.data_dir + opt.data_name)
        
    print(adata.X)
    
    # 如果不是稀疏矩阵的格式则转换为稀疏矩阵
    # if not isinstance(adata.X, csr_matrix):
    #     tmp = scipy.sparse.csr_matrix(adata.X)
    #     adata.X = None
    #     adata.X = tmp
    
    if opt.is_raw_data:
        sc.pp.filter_cells(adata, min_genes=0)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata) # 减少数据范围
        
        # 保存一份数据在 adata.raw 中
        adata.raw = adata

    dropout_mask = (adata.X != 0).astype(float)
    
    # 此处 n_top_features 是 2000
    if type(opt.n_top_features) == int and opt.n_top_features > 0:
        # 此处是取 2000 个HVG 
        sc.pp.highly_variable_genes(adata, n_top_genes=opt.n_top_features, batch_key=opt.batch_name, inplace=False, subset=True)
    

    if opt.batch_scale:
        print("batch scale start!")
        adata.X = batch_scale(adata, opt.batch_name)
    else:
        print("scale start!")
        scaler = MaxAbsScaler(copy=False).fit(adata.X)
        adata.X = scaler.transform(adata.X)

    
    return adata, dropout_mask


def read_data_for_OOD(opt):
    data_format = opt.data_name.split('.')[-1]
    
    if data_format == 'h5ad':
        adata = sc.read_h5ad(opt.data_dir + opt.data_name)
        
    print(adata.X)
    
    # 如果不是稀疏矩阵的格式则转换为稀疏矩阵
    # if not isinstance(adata.X, csr_matrix):
    #     tmp = scipy.sparse.csr_matrix(adata.X)
    #     adata.X = None
    #     adata.X = tmp
    
    if opt.is_raw_data:
        sc.pp.filter_cells(adata, min_genes=0)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata) # 减少数据范围
        
        # 保存一份数据在 adata.raw 中
        adata.raw = adata
        

    dropout_mask = (adata.X != 0).astype(float)
    
    # 此处 n_top_features 是 2000
    if type(opt.n_top_features) == int and opt.n_top_features > 0:
        # 此处是取 2000 个HVG 
        sc.pp.highly_variable_genes(adata, n_top_genes=opt.n_top_features, batch_key=opt.batch_name, inplace=False, subset=True)
    

    if opt.batch_scale:
        print("batch scale start!")
        adata.X = batch_scale(adata, opt.batch_name)
    else:
        print("scale start!")
        scaler = MaxAbsScaler(copy=False).fit(adata.X)
        adata.X = scaler.transform(adata.X)
        # sc.pp.scale(adata)
    
    print(adata.X)
    
    return adata, dropout_mask
        
def prepare_data(opt, color, batch_num=None):
    
    # read adata
    adata, dropout_mask = read_data(opt)

    # extract the parameter
    n_domain = len(set(adata.obs[opt.batch_name]))
    domain_list = adata.obs[opt.batch_name].apply(lambda x: int(x))
    
    if batch_num != None: 
        adata = adata[adata.obs[opt.batch_name]==batch_num]
        n_domain = len(set(adata.obs[opt.batch_name]))
        domain_list = adata.obs[opt.batch_name].apply(lambda x: 0)
        
    # visualize the raw data
    if opt.vis:
        plot_umap(adata, color, save_filename=f'_raw_umap.pdf')
        
    return adata, domain_list, n_domain


        
def prepare_data_for_zeroshot(opt, color, task=None):
    
    # read adata
    adata, dropout_mask = read_data(opt)

    # extract the parameter
    n_domain = len(set(adata.obs['batch'])) - 1
    train_adata = adata[adata.obs['batch'] != n_domain]
    train_domain_list = train_adata.obs[opt.batch_name].apply(lambda x: int(x))
    
    test_adata = adata[adata.obs['batch'] == n_domain]
    test_domain_list = test_adata.obs[opt.batch_name].apply(lambda x: 0)

    return adata, train_adata, train_domain_list, test_adata, test_domain_list, n_domain

def get_cell_type(opt):
    adata = sc.read_h5ad(opt.data_dir + opt.data_name)
    type_list = list(set(adata.obs[opt.type_name]))
    return type_list

def prepare_data_for_OOD(opt, prediction_type):
    
    # read adata
    adata, dropout_mask = read_data_for_OOD(opt)
    
    n_domain = len(set(adata.obs[opt.condition_name]))
    
    train_adata = adata[adata.obs[opt.type_name] != prediction_type]
    train_domain_list = train_adata.obs[opt.condition_name].apply(lambda x: int(x))

    test_adata = adata[adata.obs[opt.type_name] == prediction_type]
    test_domain_list = test_adata.obs[opt.condition_name].apply(lambda x: int(x))

    return train_adata, train_domain_list, test_adata, test_domain_list, n_domain

def prepare_data_for_GRN(opt, color, task=None):
    
    # read adata
    adata, dropout_mask = read_data(opt, batch_scale_for_gene=True)


    # extract the parameter
    n_domain = len(set(adata.obs[opt.batch_name]))
    domain_list = adata.obs[opt.batch_name].apply(lambda x: int(x))
    
    # visualize the raw data
    if opt.vis:
        plot_umap(adata, color, save_filename=f'_raw_umap.pdf')
        
    return adata, dropout_mask, domain_list, n_domain