# Run LIGER 1 iNMF on raw counts with the pinned rliger 1.x API.
# Normalize, select features, scale, and quantile-align factors; return H.norm rows
# in batch-0/batch-1 order. rliger 2.x uses a separate adapter.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript liger1.R <input.h5> <output.h5>")
input_path <- args[1]
output_path <- args[2]

.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(file.path(dirname(normalizePath(.self)), "io.R"))

suppressMessages(library(rliger))

input <- read_input(input_path)
k <- as.integer(input$params$dim)
seed <- as.integer(input$params$seed)
preselected_features <- isTRUE(as.logical(input$params$preselected_features)[1])
requested_features <- as.integer(input$params$requested_features)[1]
if (is.na(requested_features)) requested_features <- 0L
couple_quantile_seed <- isTRUE(as.logical(input$params$couple_quantile_seed)[1])
set.seed(seed)

# Exclude R startup, imports, and input reading from algorithm timing.
algo_start <- elapsed_seconds()

make_matrix <- function(cells_by_genes, tag) {
  m <- t(cells_by_genes) # genes x cells
  rownames(m) <- paste0("g", seq_len(nrow(m)))
  colnames(m) <- paste0(tag, "_", seq_len(ncol(m)))
  m
}

data_list <- list(b0 = make_matrix(input$x0, "b0"), b1 = make_matrix(input$x1, "b1"))

liger_object <- createLiger(data_list, remove.missing = FALSE)
liger_object <- rliger::normalize(liger_object)
if (preselected_features) {
  liger_object@var.genes <- Reduce(intersect, lapply(liger_object@raw.data, rownames))
} else if (requested_features > 0L) {
  liger_object <- selectGenes(liger_object, num.genes = requested_features)
} else {
  liger_object <- selectGenes(liger_object)
}
liger_object <- scaleNotCenter(liger_object)
liger_object <- optimizeALS(liger_object, k = k, rand.seed = seed)
if (couple_quantile_seed) {
  liger_object <- quantile_norm(liger_object, rand.seed = seed)
} else {
  liger_object <- quantile_norm(liger_object)
}
cat(sprintf("NE_ALIGNED_LIGER_SELECTED_FEATURES=%d\n", length(liger_object@var.genes)))

# H.norm: cells x k, in dataset (b0 then b1) order.
embedding <- tryCatch(liger_object@H.norm, error = function(e) NULL)
if (is.null(embedding)) embedding <- getMatrix(liger_object, "H.norm")
embedding <- as.matrix(embedding)

order_names <- row_order(input)
if (!is.null(rownames(embedding)) && all(order_names %in% rownames(embedding))) {
  embedding <- embedding[order_names, , drop = FALSE]
}
algo_seconds <- elapsed_seconds() - algo_start
write_output(output_path, embedding, timings = list(time_pre_embedding = algo_seconds))
