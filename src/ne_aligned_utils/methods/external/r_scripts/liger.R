# Current LIGER iNMF integration with its default factor alignment.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript liger.R <input.h5> <output.h5>")
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
set.seed(seed)

algo_start <- elapsed_seconds()

make_matrix <- function(cells_by_genes, tag) {
  m <- t(cells_by_genes)
  rownames(m) <- paste0("g", seq_len(nrow(m)))
  colnames(m) <- paste0(tag, "_", seq_len(ncol(m)))
  m
}

data_list <- list(b0 = make_matrix(input$x0, "b0"), b1 = make_matrix(input$x1, "b1"))
liger_object <- createLiger(data_list, verbose = FALSE)
liger_object <- normalize(liger_object, verbose = FALSE)
if (!is.null(input$library_size)) {
  # LIGER's normalize divides each cell by its total over the supplied genes; divide by the
  # supplied library size instead, keeping LIGER's form (no log, no scale factor).
  for (d in names(datasets(liger_object))) {
    ld <- dataset(liger_object, d)
    normData(ld) <- divide_by_library_size(rawData(ld), input$library_size)
    datasets(liger_object)[[d]] <- ld
  }
}

if (preselected_features) {
  varFeatures(liger_object) <- rownames(data_list$b0)
} else if (requested_features > 0L) {
  liger_object <- selectGenes(liger_object, nGenes = requested_features, verbose = FALSE)
} else {
  liger_object <- selectGenes(liger_object, verbose = FALSE)
}

liger_object <- scaleNotCenter(liger_object, verbose = FALSE)
liger_object <- runIntegration(liger_object, k = k, seed = seed, verbose = FALSE)
liger_object <- alignFactors(liger_object, seed = seed)
cat(sprintf("NE_ALIGNED_LIGER_SELECTED_FEATURES=%d\n", length(varFeatures(liger_object))))

embedding <- as.matrix(getMatrix(liger_object, "H.norm"))
order_names <- row_order(input)
if (!is.null(rownames(embedding)) && all(order_names %in% rownames(embedding))) {
  embedding <- embedding[order_names, , drop = FALSE]
}

algo_seconds <- elapsed_seconds() - algo_start
write_output(output_path, embedding, timings = list(time_pre_embedding = algo_seconds))
