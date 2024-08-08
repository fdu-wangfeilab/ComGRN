

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
data <- args[2]
version <- args[3]
out_dir <- args[4]

library(SAVERX)
library(reticulate) # load the reticulate package
use_python("/home/djy/.conda/envs/saverx4/bin/python")

algorithm <- "saverx"

weights_file <- ""

f <- './noisy_data.csv' 

print("Running saverx")

pretrain_flag <- if (weights_file == "") F else T

e <- try(saverx(f,
    verbose = T,
    data.species = "Human",
    is.large.data = F,
    clearup.python.session = F,
    use.pretrain = pretrain_flag,
    pretrained.weights.file = weights_file,
    ncores = 8
))


print("Saving results")

tmp <- readRDS(file = e)
tmp <- round(tmp$estimate, digits = 4)

# 检查文件夹是否存在
if (!file.exists(out_dir)) {
  dir.create(out_dir)
}

d_f_name <- paste0(out_dir, '/denoise_data_SAVERX.csv')

write.table(t(tmp),
    file = d_f_name,
    sep = ",",
    col.names = T,
    row.names = T
)

unlink(gsub("denoised.rds", "", e), recursive = T)