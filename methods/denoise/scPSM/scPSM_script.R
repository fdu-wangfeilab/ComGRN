
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
data <- args[2]
version <- args[3]
batch_num <- as.integer(args[4])
out_dir <- args[5]

library(reticulate) # load the reticulate package
use_python("/home/djy/.conda/envs/py38/bin/python")

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(BiocNeighbors))
library(BiocParallel)
library(S4Vectors)
library(scPSM)

library(anndata)
data_path <- paste0(data_dir, data, '/', version, '/demo.h5ad') 
ad <- read_h5ad(data_path)
ad$var_names <- paste("gene", ad$var_names, sep = "")

sc <- import("scanpy")
sc$pp$normalize_total(ad, 10000)
sc$pp$log1p(ad)
ad$obs_names_make_unique()

print(ad)

sparse_matrix <- Matrix(t(ad$X), sparse = TRUE)

sd <- CreateSeuratObject(counts = sparse_matrix, meta.data = ad$obs)

sd.list <- SplitObject(object = sd, split.by = "batch")

batches <- list()
for (i in 1:batch_num) {
    var_name <- paste0("b", i)
    batches[[var_name]] <- NormalizeData(object = sd.list[[as.character(i-1)]])[["RNA"]]$data
}

# cell * gene

# 运行高可变性基因分析
sc$pp$highly_variable_genes(ad)
hvg <- ad$var_names[ad$var['highly_variable']$highly_variable]

markers <- unlist(hvg[1:10])

psm.data <- psm_integrate(batches = batches, markers = markers, hvg = hvg, merge.order = 1:batch_num)
dim(psm.data)
denoised_data <- psm.data

# 检查文件夹是否存在
if (!file.exists(out_dir)) {
  dir.create(out_dir)
}

d_f_name <- paste0(out_dir, '/denoise_data_scPSM.csv')

write.table(denoised_data,
    file = d_f_name,
    sep = ",",
    col.names = T,
    row.names = T
)
