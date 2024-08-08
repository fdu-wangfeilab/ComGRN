
library("devtools")
library("scMultiSim")
library("reticulate")
use_python("/home/djy/.conda/envs/py38/bin/python")

pwd <- "/home/djy/dataset/scMultiSim/"
args <- commandArgs(trailingOnly = TRUE)
GRN_num <- as.integer(args[1])
if (GRN_num == 100) {
  data(GRN_params_100)
  GRN_params <- GRN_params_100
} else {
  data(GRN_params_1139)
  GRN_params <- GRN_params_1139
}

cell_num <- as.integer(args[2])

tree_num <- as.integer(args[3])
if (tree_num == 3) {
  tree <- Phyla3()
} else {
  tree <- Phyla5()
}

batch_num <- as.integer(args[4])
alpha_mean <- as.numeric(args[5]) 

randseed <- as.integer(args[6])

save_dir <- paste0("/tree", tree_num, '_cells', cell_num, "_grn", GRN_num,  '_b', batch_num, '_drop', alpha_mean, '_v4/seed', randseed)
dir.create(paste0(pwd, save_dir), recursive=TRUE, showWarnings = FALSE)

par(mfrow=c(1,2))


options <- list(
  rand.seed = 0, 
  GRN = GRN_params, 
  num.cells = cell_num, 
  num.cifs = 50, 
  cif.sigma = 0.5,  
  tree = tree,  
  diff.cif.fraction = 0.8, 
  do.velocity = T 
)

results <- sim_true_counts(options)

names(results)

write.csv(x = results$counts, file = paste0(pwd, save_dir,  "/clean_data.csv"), quote = FALSE)

# we change the step of divide_batches and add_expr_noise
# because we assume the expr noise is related to batch

results$counts_obs <- results$counts
divide_batches(results, nbatch = batch_num, effect = 3, randseed=randseed)

results$counts <- results$counts_with_batches
add_expr_noise(results, 
            alpha_mean=alpha_mean,
            alpha_sd=0.001,
            randseed=randseed)


write.csv(x = results$counts_obs, file = paste0(pwd, save_dir, "/noise_data.csv"), quote = FALSE)


colnames(GRN_params) <- c("Gene2", "Gene1", "EdgeWeight")
write.csv(x = GRN_params, file = paste0(pwd, save_dir, "/grn.csv"), quote = FALSE)

# import anndata
anndata <- import("anndata", convert = FALSE)

# create anndata object
adata <- anndata$AnnData(X = t(results$counts_obs))

# add obs name and var name
adata$obs_names <- results$cell_meta["cell_id"]
adata$var_names <- lapply(1:results$num_genes, function(x) {return(as.character(x))})


meta_cell <- results$cell_meta
colnames(meta_cell) <- c('cell_id', 'cellstate', 'depth', 'batch')
meta_cell[["batch"]] <- sapply(meta_cell[["batch"]], function(x) as.character(as.integer(x) - 1))

# check vector
for (col in names(meta_cell)) {
  if (!is.vector(meta_cell[[col]])) {
    meta_cell[[col]] <- unlist(meta_cell[[col]])
  }
}

adata$obs <- meta_cell

adata$write(paste0(pwd, save_dir, "/demo.h5ad"))

