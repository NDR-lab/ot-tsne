#!/usr/bin/env bash
# Install pinned rliger 1.0.1 and bcTSNE 0.11.1 after creating envs/r/environment.yml.
# Both build from source using the environment's C, C++, and Fortran toolchain.
# Set MAMBA_ROOT_PREFIX to the micromamba root before running this script.
set -euo pipefail

R_ROOT="${NE_ALIGNED_R_ROOT:-$HOME/.local/opt/ne-aligned-r}"
MAMBA="${NE_ALIGNED_MICROMAMBA:-$R_ROOT/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$R_ROOT/mmroot}"
ENV_NAME="${NE_ALIGNED_R_ENV:-rverify}"
WORK="${TMPDIR:-/tmp}/ne-aligned-r-extras.$$"

RUN=("$MAMBA" run -n "$ENV_NAME")
[ -x "$MAMBA" ] || { echo "error: no micromamba at $MAMBA" >&2; exit 1; }

# --- rliger and its CRAN-only dependencies ------------------------------------------------------
# Install pinned dependencies before their dependents, retaining upgrade="never".
"${RUN[@]}" Rscript -e '
options(repos = c(CRAN = "https://cloud.r-project.org"))
pins <- c(
  GPArotation = "2026.8-1",
  mnormt      = "2.1.2",
  psych       = "2.6.5",
  iterators   = "1.0.14",
  foreach     = "1.5.2",
  doParallel  = "1.0.17",
  mclust      = "6.1.3",
  rliger      = "1.0.1"
)
for (pkg in names(pins)) {
  want <- pins[[pkg]]
  have <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
  if (!is.na(have) && have == want) { cat("ok      ", pkg, want, "\n"); next }
  cat("install ", pkg, want, "\n")
  remotes::install_version(pkg, version = want, upgrade = "never", quiet = TRUE)
}
'

# --- bcTSNE -------------------------------------------------------------------------------------
# Pin the audited bcTSNE snapshot because upstream has no 0.11.1 tag, then apply the saved patch
# envs/r/bcTSNE-82e8be9.patch (two source edits) and remove the prebuilt objects upstream ships.
BCTSNE_COMMIT="82e8be9ff42f03ace8f7112d36d4a3818485aace"
BCTSNE_PATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/bcTSNE-82e8be9.patch"
if "${RUN[@]}" Rscript -e 'quit(status = if (requireNamespace("bcTSNE", quietly = TRUE)) 0 else 1)'; then
  echo "ok       bcTSNE $("${RUN[@]}" Rscript -e 'cat(as.character(packageVersion("bcTSNE")))')"
else
  echo "install  bcTSNE 0.11.1 (from source)"
  mkdir -p "$WORK"
  trap 'rm -rf "$WORK"' EXIT
  git clone --depth 1 https://github.com/emanuelealiverti/BC_tSNE "$WORK/BC_tSNE"
  cd "$WORK/BC_tSNE"
  git fetch --depth 1 origin "$BCTSNE_COMMIT"
  git checkout --detach "$BCTSNE_COMMIT"

  # The patch removes splatter from Imports (only the vignette uses it; it pulls in Bioconductor)
  # and sets R_forceSymbols(dll, FALSE), because R/bctsne.R calls .C("bctsne") by string, which
  # modern R rejects otherwise.
  git apply "$BCTSNE_PATCH"
  # Upstream ships .o/.so built for the author's platform; delete them so the package compiles here.
  rm -f src/*.o src/*.so

  "${RUN[@]}" R CMD INSTALL --preclean .
  cd - > /dev/null
fi

# --- report -------------------------------------------------------------------------------------
"${RUN[@]}" Rscript -e '
cat("\n", R.version.string, "\n\n", sep = "")
for (p in c("hdf5r", "Seurat", "SeuratObject", "Matrix", "rliger", "bcTSNE", "Rtsne", "RSpectra"))
  cat(sprintf("  %-14s %s\n", p, as.character(packageVersion(p))))
'
