# Run BC-tSNE on processed features, returning a joint 2D layout.
# The R package performs truncated SVD and batch-corrected t-SNE after centering/scaling.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript bc_tsne.R <input.h5> <output.h5>")
input_path <- args[1]
output_path <- args[2]

.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(file.path(dirname(normalizePath(.self)), "io.R"))

suppressMessages({
  library(bcTSNE)
  library(stats)
})

input <- read_input(input_path)
perplexity <- as.numeric(input$params$perplexity)
seed <- as.integer(input$params$seed)
set.seed(seed)

# Time bctsne as one fused SVD/layout call; no separate stage boundary is exposed.
algo_start <- elapsed_seconds()

joint <- rbind(input$x0, input$x1)           # (cells x features), c(x0, x1) order
if (any(!is.finite(joint))) stop("BC-tSNE input contains non-finite values")

feature_sd <- apply(joint, 2, sd)
constant_features <- !is.finite(feature_sd) | feature_sd == 0
if (any(constant_features)) {
  message(sprintf("Removing %d constant BC-tSNE input features", sum(constant_features)))
  joint <- joint[, !constant_features, drop = FALSE]
}
if (ncol(joint) == 0L) stop("BC-tSNE has no non-constant input features")

X <- scale(joint)

batch <- factor(input$batch_ids)
Z <- model.matrix(~ -1 + batch)              # N x n_batches design (no intercept)

# k = truncated-SVD rank used inside bctsne; must exceed ncol(Z) and fit the data.
k <- min(50L, ncol(X) - 1L, nrow(X) - 1L)
if (k <= ncol(Z)) stop("BC-tSNE has too few cells or non-constant features for its batch model")

result <- bctsne(X = X, Z = Z, k = k, outDim = 2L, perplexity = perplexity)
embedding <- as.matrix(result$Y)
rownames(embedding) <- row_order(input)
algo_seconds <- elapsed_seconds() - algo_start
write_output(output_path, embedding, timings = list(time_total = algo_seconds))
