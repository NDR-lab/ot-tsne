# Read batch matrices and parameters from HDF5 and write row-aligned embeddings.
# Inputs are cells x genes; stored dimensions resolve hdf5r orientation differences.
# Output rows follow batch 0 then batch 1.

suppressMessages(library(hdf5r))

read_input <- function(path) {
  f <- H5File$new(path, mode = "r")
  on.exit(f$close_all(), add = TRUE)

  x0 <- f[["x0"]]$read()
  x1 <- f[["x1"]]$read()
  batch_ids <- as.integer(f[["batch_ids"]]$read())
  params <- h5attributes(f[["params"]])
  # Optional per-cell divisor for count normalisation, in c(x0, x1) order.
  library_size <- if (f$exists("library_size")) as.numeric(f[["library_size"]]$read()) else NULL

  n0 <- as.integer(params$n0)
  n1 <- as.integer(params$n1)
  n_features <- as.integer(params$n_features)

  # Ensure (cells x genes) orientation regardless of hdf5r's axis order.
  if (nrow(x0) != n0 && ncol(x0) == n0) x0 <- t(x0)
  if (nrow(x1) != n1 && ncol(x1) == n1) x1 <- t(x1)
  stopifnot(nrow(x0) == n0, nrow(x1) == n1, ncol(x0) == n_features, ncol(x1) == n_features)
  if (!is.null(library_size)) {
    stopifnot(length(library_size) == n0 + n1, all(is.finite(library_size)), all(library_size > 0))
    names(library_size) <- c(paste0("b0_", seq_len(n0)), paste0("b1_", seq_len(n1)))
  }

  list(
    x0 = x0, x1 = x1, batch_ids = batch_ids, params = params,
    n0 = n0, n1 = n1, n_features = n_features, library_size = library_size
  )
}

# Divide each cell's counts (a genes x cells sparse matrix) by its supplied library size,
# matched by cell name. Count methods use this instead of the total over the supplied genes.
divide_by_library_size <- function(counts, library_size, scale = 1) {
  cells <- colnames(counts)
  if (is.null(cells) || !all(cells %in% names(library_size))) {
    stop("library_size does not cover every cell in the count matrix")
  }
  counts <- as(counts, "CsparseMatrix")
  counts@x <- counts@x / rep.int(library_size[cells], diff(counts@p)) * scale
  counts
}

# Row names Python expects back, in c(x0, x1) concatenation order.
row_order <- function(input) {
  c(paste0("b0_", seq_len(input$n0)), paste0("b1_", seq_len(input$n1)))
}

# Optional timings are seconds stored as HDF5 group attributes. Child algorithm
# durations exclude startup, imports, and I/O; the parent records overhead separately.
# Absent timings remain NaN downstream.
write_output <- function(path, embedding, timings = NULL) {
  embedding <- as.matrix(embedding)
  storage.mode(embedding) <- "double"
  if (file.exists(path)) file.remove(path)
  f <- H5File$new(path, mode = "w")
  on.exit(f$close_all(), add = TRUE)
  f[["embedding"]] <- embedding
  if (!is.null(timings) && length(timings) > 0) {
    group <- f$create_group("timings")
    for (key in names(timings)) {
      value <- timings[[key]]
      if (!is.null(value) && is.finite(as.numeric(value))) {
        h5attr(group, key) <- as.numeric(value)
      }
    }
  }
  write_environment(f)
  invisible(NULL)
}

# Record the R side of the environment: R version, the BLAS and LAPACK actually loaded, and
# the version of every package loaded by the script. Rebuilt environments can resolve
# different dependency versions under the same pinned top-level packages.
write_environment <- function(f) {
  session <- sessionInfo()
  group <- f$create_group("environment")
  h5attr(group, "r_version") <- R.version.string
  h5attr(group, "blas") <- if (is.null(session$BLAS)) "" else session$BLAS
  h5attr(group, "lapack") <- La_library()
  h5attr(group, "lapack_version") <- La_version()
  # In a conda environment, R.home() is <prefix>/lib/R; conda-meta names every installed
  # package as name-version-build, including the BLAS and compiler runtimes R links against.
  conda_meta <- file.path(dirname(dirname(R.home())), "conda-meta")
  if (dir.exists(conda_meta)) {
    builds <- sub("\\.json$", "", list.files(conda_meta, pattern = "\\.json$"))
    h5attr(group, "conda_packages") <- paste(sort(builds), collapse = ";")
  }
  packages <- f$create_group("r_packages")
  for (name in sort(loadedNamespaces())) {
    h5attr(packages, name) <- as.character(utils::packageVersion(name))
  }
}

# Use elapsed process time so wall-clock adjustments do not affect durations.
elapsed_seconds <- function() {
  as.numeric(proc.time()[["elapsed"]])
}

# Locate io.R's own directory so method scripts can source it robustly under Rscript.
ne_aligned_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  getwd()
}
