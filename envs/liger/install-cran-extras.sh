#!/usr/bin/env bash
# Install the pinned current rliger API after creating envs/liger/environment.yml.
# RcppPlanc comes from conda-forge; building it from CRAN is less portable because its CMake
# discovery can reject otherwise usable conda HDF5 installations.
set -euo pipefail

R_ROOT="${NE_ALIGNED_R_ROOT:-$HOME/.local/opt/ne-aligned-r}"
MAMBA="${NE_ALIGNED_MICROMAMBA:-$R_ROOT/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$R_ROOT/mmroot}"
ENV_NAME="${NE_ALIGNED_LIGER_ENV:-liger2}"
RUN=("$MAMBA" run -n "$ENV_NAME")

[ -x "$MAMBA" ] || { echo "error: no micromamba at $MAMBA" >&2; exit 1; }

"${RUN[@]}" Rscript -e '
options(repos = c(CRAN = "https://cloud.r-project.org"))
# HighFive is a CRAN-only dependency; pin it first so installing rliger cannot pull a newer one.
pins <- c(HighFive = "3.3.0", rliger = "2.2.1")
for (pkg in names(pins)) {
  want <- pins[[pkg]]
  have <- tryCatch(as.character(packageVersion(pkg)), error = function(e) NA_character_)
  if (is.na(have) || have != want) {
    remotes::install_version(pkg, version = want, upgrade = "never")
  }
  stopifnot(as.character(packageVersion(pkg)) == want)
}
cat(R.version.string, "\n")
for (p in c("hdf5r", "RcppPlanc", "HighFive", "rliger"))
  cat(sprintf("  %-12s %s\n", p, as.character(packageVersion(p))))
'
