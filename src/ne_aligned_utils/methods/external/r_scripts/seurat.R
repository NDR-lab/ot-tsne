# Integrate raw-count batches using Seurat v5 layers or v3-style anchors.
# Return dim coordinates in batch-0/batch-1 order. Use the supplied feature panel
# when marked preselected; otherwise select variable features after normalization.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript seurat.R <input.h5> <output.h5>")
input_path <- args[1]
output_path <- args[2]

.self <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
source(file.path(dirname(normalizePath(.self)), "io.R"))

suppressMessages({
  library(Seurat)
  library(Matrix)
})

input <- read_input(input_path)
k <- as.integer(input$params$dim)
seed <- as.integer(input$params$seed)
preselected_features <- isTRUE(as.logical(input$params$preselected_features)[1])
workflow <- as.character(input$params$workflow)[1]
if (is.na(workflow) || workflow == "") workflow <- "v5_cca"
if (!workflow %in% c("legacy", "v5_cca")) {
  stop("workflow must be 'legacy' or 'v5_cca'")
}
set.seed(seed)

# Start algorithm timing after R startup, package imports, and input reading.
algo_start <- elapsed_seconds()

make_object <- function(cells_by_genes, tag) {
  # Seurat expects features x cells.
  m <- t(cells_by_genes)
  rownames(m) <- paste0("g", seq_len(nrow(m)))
  colnames(m) <- paste0(tag, "_", seq_len(ncol(m)))
  obj <- CreateSeuratObject(counts = as(m, "dgCMatrix"))
  obj$batch <- tag
  obj
}

# With supplied library sizes, LogNormalize divides by them instead of the supplied-gene
# totals: data = log1p(counts / library_size * 1e4), Seurat's default scale factor.
use_library_size <- function(obj, library_size) {
  if (is.null(library_size)) return(obj)
  for (layer in grep("^counts", Layers(obj[["RNA"]]), value = TRUE)) {
    counts <- LayerData(obj, assay = "RNA", layer = layer)
    normalized <- divide_by_library_size(counts, library_size, scale = 1e4)
    normalized@x <- log1p(normalized@x)
    LayerData(obj, assay = "RNA", layer = sub("^counts", "data", layer)) <- normalized
  }
  obj
}

objects <- list(make_object(input$x0, "b0"), make_object(input$x1, "b1"))
min_cells <- min(input$n0, input$n1)

if (workflow == "v5_cca") {
  # Current Seurat v5 workflow: one object whose count layers represent batches.
  integrated <- merge(objects[[1]], y = objects[[2]])
  # Merging the two Seurat objects already creates one count layer per batch.
  # Split only if a future Seurat version merges them into one layer.
  if (sum(grepl("^counts", Layers(integrated[["RNA"]]))) == 1L) {
    integrated[["RNA"]] <- split(integrated[["RNA"]], f = integrated$batch)
  }
  integrated <- NormalizeData(integrated, verbose = FALSE)
  integrated <- use_library_size(integrated, input$library_size)
  if (preselected_features) {
    features <- rownames(integrated)
    VariableFeatures(integrated) <- features
  } else {
    integrated <- FindVariableFeatures(integrated, verbose = FALSE)
    features <- VariableFeatures(integrated)
  }
  integrated <- ScaleData(integrated, features = features, verbose = FALSE)
  # Below Seurat's default 30 integration dimensions, integrate at the requested dimension, so
  # seurat-2d integrates in 2 dimensions like the other 2D variants instead of truncating 30.
  working_dims <- if (k < 30L) k else NULL
  npcs <- max(2L, min(max(k, 30L), length(features), ncol(integrated) - 1L))
  if (!is.null(working_dims)) npcs <- max(2L, min(working_dims, ncol(integrated) - 1L))
  integrated <- RunPCA(
    integrated, features = features, npcs = npcs, seed.use = seed, verbose = FALSE
  )
  integration_args <- list(
    object = integrated,
    method = CCAIntegration,
    orig.reduction = "pca",
    new.reduction = "integrated.cca",
    verbose = FALSE
  )
  if (preselected_features) integration_args$features <- features
  if (!is.null(working_dims)) integration_args$dims <- seq_len(npcs)
  integrated <- do.call(IntegrateLayers, integration_args)
  embedding <- Embeddings(integrated, reduction = "integrated.cca")
} else {
  objects <- lapply(objects, function(obj) {
    obj <- NormalizeData(obj, verbose = FALSE)
    obj <- use_library_size(obj, input$library_size)
    if (!preselected_features) obj <- FindVariableFeatures(obj, verbose = FALSE)
    obj
  })
  if (preselected_features) {
    features <- Reduce(intersect, lapply(objects, rownames))
  } else {
    features <- SelectIntegrationFeatures(object.list = objects)
  }
  kf <- max(2L, min(200L, min_cells - 1L))
  kw <- max(2L, min(100L, min_cells - 1L))
  anchor_dims <- 1:max(2L, min(30L, min_cells - 1L))
  anchors <- FindIntegrationAnchors(
    object.list = objects, anchor.features = features,
    dims = anchor_dims, k.filter = kf, k.score = min(30L, kf), verbose = FALSE
  )
  integrated <- IntegrateData(
    anchorset = anchors, dims = anchor_dims, k.weight = kw, verbose = FALSE
  )
  DefaultAssay(integrated) <- "integrated"
  if (preselected_features) {
    integrated <- ScaleData(integrated, features = features, verbose = FALSE)
  } else {
    integrated <- ScaleData(integrated, verbose = FALSE)
  }
  npcs <- max(2L, min(max(k, 30L), ncol(integrated) - 1L, nrow(integrated) - 1L))
  if (preselected_features) {
    integrated <- RunPCA(
      integrated, features = features, npcs = npcs, seed.use = seed, verbose = FALSE
    )
  } else {
    integrated <- RunPCA(integrated, npcs = npcs, seed.use = seed, verbose = FALSE)
  }
  embedding <- Embeddings(integrated, reduction = "pca")
}

embedding <- embedding[row_order(input), seq_len(min(k, ncol(embedding))), drop = FALSE]
# Report integration time; Python adds backend time for two-stage methods.
algo_seconds <- elapsed_seconds() - algo_start
write_output(output_path, embedding, timings = list(time_pre_embedding = algo_seconds))
