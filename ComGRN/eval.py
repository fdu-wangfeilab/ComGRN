import scipy
import numpy as np
import torch
import pandas as pd
import scipy.stats as stats
import igraph as ig
import scanpy as sc

from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score as sil
from scib_metrics.benchmark import Benchmarker
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_recall_curve,roc_curve,auc, average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

try:
    import leidenalg
except ImportError:
    raise ImportError(
        'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
    )

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# related metrics
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

## 1. Gene regulatory network inference
def eval_EP_info(ref_path, pred_path):
    output = pd.read_csv(pred_path, sep=',', header=0, index_col=0)
    output['EdgeWeight'] = abs(output['EdgeWeight'])

    output['Gene1'] = output['Gene1'].astype(str)
    output['Gene2'] = output['Gene2'].astype(str)    
    output = output.sort_values('EdgeWeight',ascending=False)
    
    label = pd.read_csv(ref_path, sep=',', index_col=None, header=0)
    label['Gene1'] = label['Gene1'].astype(str)
    label['Gene2'] = label['Gene2'].astype(str)

    TFs = set(label['Gene1']) 
    Genes = set(label['Gene1']) | set(label['Gene2'])
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]

    label_set = set(label['Gene1']+'|'+label['Gene2'])
    output= output.iloc[:len(label_set)]  # top K predicted edges
    print("k", len(label_set))
    print("possible edge", len(TFs)*len(Genes)-len(TFs))
    EPratio= len(set(output['Gene1']+ '|' +output['Gene2']) & label_set) / (len(label_set)**2/(len(TFs)*len(Genes)-len(TFs)))
    EP = len(set(output['Gene1']+ '|' +output['Gene2']) & label_set) / len(label_set) 
    return EPratio, EP

def eval_AUPR_info(ref_path, pred_path):
    output = pd.read_csv(pred_path, sep=',', header=0, index_col=0)
    output['EdgeWeight'] = abs(output['EdgeWeight'])    
    output['Gene1'] = output['Gene1'].astype(str)
    output['Gene2'] = output['Gene2'].astype(str)
    output = output.sort_values('EdgeWeight',ascending=False)
    
    label = pd.read_csv(ref_path, sep=',', index_col=None, header=0)
    label['Gene1'] = label['Gene1'].astype(str)
    label['Gene2'] = label['Gene2'].astype(str)

    TFs = set(label['Gene1'])
    Genes = set(label['Gene1'])| set(label['Gene2']) - set("AVG")
    # print(len(Genes))
    output = output[output['Gene1'].apply(lambda x: x in TFs)]
    output = output[output['Gene2'].apply(lambda x: x in Genes)]
    label_set = set(label['Gene1']+label['Gene2'])
    # print(len(label_set))

    preds, labels, randoms = [], [], []
    res_d = {}
    l = []
    p= []
    for item in (output.to_dict('records')):
            res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
    for item in (set(label['Gene1'])):
            for item2 in  set(label['Gene1'])| set(label['Gene2']):
                if item == item2: # TF * TG - TF
                    continue
                if item+item2 in label_set:
                    l.append(1) 
                else:
                    l.append(0)  
                if item+ item2 in res_d:
                    p.append(res_d[item+item2])
                else:
                    p.append(-1)
    print("possible edge", len(p))
    AUPRratio = average_precision_score(l,p)/np.mean(l) # 等效于网络密度
    AUPR = average_precision_score(l,p,pos_label=1)
    roc = roc_auc_score(l,p)
    return AUPRratio, AUPR, roc

def extractEdgesFromMatrix(m, geneNames):
    geneNames = np.array(geneNames)
    import copy
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'Gene1': geneNames[idx_send], 'Gene2': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)
    return edges_df


## 2. Correlation
def eval_corr(clean_data, noisy_data, denoised_data, mask=False):
    if mask: # eval the imputation recovery capability
        corr_zeros = []
        for c in range(noisy_data.shape[0]):
            mask_data = noisy_data[c] == 0
            # print(clean_data[c,mask_data].shape)
            sub_clean_data = clean_data[c,mask_data] / np.linalg.norm(clean_data[c,mask_data])
            sub_denoise_data = denoised_data[c,mask_data] / np.linalg.norm(denoised_data[c,mask_data])
            corr_zeros.append(spearmanr(sub_clean_data, sub_denoise_data)[0])
        corr_zeros_mean = np.mean(np.array(corr_zeros)[~np.isnan(corr_zeros)])
        corr_zeros_std = np.std(np.array(corr_zeros)[~np.isnan(corr_zeros)])
        print("clean_denoised_corr(zero): ", corr_zeros_mean, corr_zeros_std)
    else:
        corr_zeros = None
    # eval the gene recovery capability
    clean_noisy_list = [spearmanr(clean_data[c,:], noisy_data[c,:])[0] for c in range(clean_data.shape[0])]
    clean_noisy_corr_mean = np.mean(np.array(clean_noisy_list)[~np.isnan(clean_noisy_list)])
    clean_noisy_corr_std = np.std(np.array(clean_noisy_list)[~np.isnan(clean_noisy_list)])
    clean_denoised_list = [spearmanr(clean_data[c,:], denoised_data[c,:])[0] for c in range(clean_data.shape[0])]
    clean_denoised_corr_mean = np.mean(np.array(clean_denoised_list)[~np.isnan(clean_denoised_list)])
    clean_denoised_corr_std = np.std(np.array(clean_denoised_list)[~np.isnan(clean_denoised_list)])
    print("clean_noisy_corr: ", clean_noisy_corr_mean, clean_noisy_corr_std)
    print("clean_denoised_corr: ", clean_denoised_corr_mean, clean_denoised_corr_std)
    
    return clean_noisy_corr_mean, clean_denoised_corr_mean, corr_zeros_mean


## 3. batch correction
def eval_bc_metrics(adata, label_key, batch_key, layer_name):
    bm = Benchmarker(
    adata,
    batch_key=batch_key,
    label_key=label_key,
    embedding_obsm_keys=layer_name,
    n_jobs=-1,
    )
    bm.benchmark()
    
    df = bm.get_results(min_max_scale=False)
    print(df)
    return df


## 4. DEG
def get_deg_list(adata, cell_type: str = "1", threshold: float = 0.05):
    sc.tl.rank_genes_groups(adata, cell_type, method="wilcoxon")
    deg = {}
    for i in set(adata.obs[cell_type]):
        gene_list = adata.uns["rank_genes_groups"]["names"][str(i)]
        msk = adata.uns["rank_genes_groups"]["pvals"][str(i)] <= threshold
        deg[str(i)] = list(gene_list[msk])
    return deg

def get_topn_deg_list(adata, cell_type: str = "1", n_genes : int=2000):
    sc.tl.rank_genes_groups(adata, cell_type, n_genes=n_genes, method="wilcoxon")
    deg = {}
    for i in set(adata.obs[cell_type]):
        gene_list = adata.uns["rank_genes_groups"]["names"][str(i)]
        deg[str(i)] = list(gene_list)
    print(len(gene_list))
    return deg




#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#
# Benchmark the discovery of key genes
#
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
def compute_auroc(inf, gt):
    """
    Calculate AUROC score
    """
    inf_abs = np.abs(inf)
    gt_abs = np.abs(gt)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(inf_abs, gt_abs)
    return AUROC

def compute_auprc(inf, gt):
    """
    Calculate AUPRC score
    """
    inf_abs = np.abs(inf)
    gt_abs = np.abs(gt)
    _, _, _, _, AUPRC, AUROC, _ = _compute_auc(inf_abs, gt_abs)
    return AUPRC

def _compute_auc(inf, gt):
    
    if np.max(inf) == 0:
        return 0, 0, 0, 0, 0, 0, 0
    else:
        fpr, tpr, thresholds = roc_curve(y_true=gt, y_score=inf, pos_label=1)
        
        if len(set(gt)) == 1:
            prec, recall = np.array([0., 1.]), np.array([1., 0.])
        else:
            prec, recall, thresholds = precision_recall_curve(y_true=gt, probas_pred=inf, pos_label=1)

        return prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), thresholds    

def compute_earlyprec(inf, gt):
    """\
    Description:
    ------------
        Calculate the early precision ratio. 
        Early precision: the fraction of true positives in the top-k edges. 
        Early precision ratio: the ratio of the early precision between estim and random estim.
        directed: the directed estimation or not
    Parameters:
    ------------
        inf: estimated score
        gt: ground truth score
    """
    # take absolute values
    inf = np.abs(inf)
    gt = np.abs(gt)
    # select k value
    num_true = np.sum(gt!=0)
    num_inf = np.sum(inf!=0)
    k = min(num_inf, num_true)
    # find the kth inf weight
    infTopk = inf[np.argsort(inf)[-k]]
    # find the top k index in predict
    topIdx = np.where(inf >= infTopk)[0]
    # find the true edges in ground truth
    trueIdx = np.where(gt!=0)[0]
    intersectionSet = np.intersect1d(topIdx, trueIdx)
    Eprec = len(intersectionSet)/(len(topIdx)+1e-12)
    # Erec = len(intersectionSet)/(len(trueIdx)+1e-12)
    return Eprec

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g

def _compute_connectivities_umap(
    knn_indices,
    knn_dists,
    n_neighbors,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
):
    """\
    This is from umap.fuzzy_simplicial_set [McInnes18]_.
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """



    # place holder since we use precompute matrix
    X = coo_matrix(([], ([], [])), shape=(knn_indices.shape[0], 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]

    return connectivities.tocsr()

def leiden_cluster(
    X = None, 
    knn_indices = None,
    knn_dists = None,
    resolution = 30.0,
    random_state = 0,
    n_iterations: int = -1,
    k_neighs = 30,
    sigma = 1,
    affin = None,
    **partition_kwargs):



    partition_kwargs = dict(partition_kwargs)
    
    if affin is None:
        if (knn_indices is None) or (knn_dists is None):
            # X is needed
            if X is None:
                raise ValueError("`X' and `knn_indices & knn_dists', at least one need to be provided.")

            neighbor = NearestNeighbors(n_neighbors = k_neighs)
            neighbor.fit(X)
            # get test connectivity result 0-1 adj_matrix, mode = 'connectivity' by default
            knn_dists, knn_indices = neighbor.kneighbors(X, n_neighbors = k_neighs, return_distance = True)

        affin = _compute_connectivities_umap(knn_indices = knn_indices, knn_dists = knn_dists, n_neighbors = k_neighs, set_op_mix_ratio=1.0, local_connectivity=1.0)
        affin = affin.todense()
        
    partition_type = leidenalg.RBConfigurationVertexPartition
    g = get_igraph_from_adjacency(affin, directed = True)

    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution

    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    groups = np.array(part.membership)

    return groups

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

def fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False):
    '''
    pvalue correction for false discovery rate.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like, 1d
        Set of p-values of the individual tests.
    alpha : float, optional
        Family-wise error rate. Defaults to ``0.05``.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests). ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).
        Defaults to ``'indep'``.
    is_sorted : bool, optional
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to ``alpha * m/m_0`` where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Both methods exposed via this function (Benjamini/Hochberg, Benjamini/Yekutieli)
    are also available in the function ``multipletests``, as ``method="fdr_bh"`` and
    ``method="fdr_by"``, respectively.

    See also
    --------
    multipletests

    '''
    pvals = np.asarray(pvals)
    assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected

def wilcoxon_rank_sum(counts_x, counts_y, fdr = False):
    # wilcoxon rank sum test
    ngenes = counts_x.shape[1]
    assert ngenes == counts_y.shape[1]
    pvals = []
    for gene_i in range(ngenes):
        _, pval = stats.ranksums(x = counts_x[:, gene_i].squeeze(), y = counts_y[:, gene_i].squeeze())
        pvals.append(pval)
    pvals = np.array(pvals)

    # fdr correction
    if fdr:
        _, pvals = fdrcorrection(pvals)
    return pvals