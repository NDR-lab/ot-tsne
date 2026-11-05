#!/usr/bin/env bash
# Build rc-tsne from the vendored FIt-SNE fork using the pinned FFTW environment.
# Run this once on each machine; the compiled binary is not distributed.
#
# Usage:
#   bash scripts/setup/build_rctsne.sh
#   export NE_ALIGNED_RCTSNE_BIN=...   # the line this prints at the end
set -euo pipefail

cd "$(dirname "$0")/../.."
VENDOR="src/ne_aligned_utils/methods/external/vendor/rctsne"
MAMBA="${NE_ALIGNED_MICROMAMBA:-$HOME/.local/opt/ne-aligned-r/bin/micromamba}"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/.local/opt/ne-aligned-r/mmroot}"
ENV_NAME=rctsne

# --- FFTW ------------------------------------------------------------------------------------
# Use an explicit FFTW prefix, or the pinned micromamba environment.
PREFIX="${FFTW_PREFIX:-$MAMBA_ROOT/envs/$ENV_NAME}"
if [ ! -f "$PREFIX/include/fftw3.h" ]; then
  if [ -n "${FFTW_PREFIX:-}" ]; then
    echo "error: no fftw3.h under $PREFIX" >&2
    exit 1
  fi
  if [ ! -x "$MAMBA" ]; then
    echo "error: no micromamba at $MAMBA, and no FFTW at $PREFIX." >&2
    echo "       Set NE_ALIGNED_MICROMAMBA, or install fftw and set FFTW_PREFIX." >&2
    exit 1
  fi
  echo "creating micromamba env '$ENV_NAME' with fftw ..."
  MAMBA_ROOT_PREFIX="$MAMBA_ROOT" "$MAMBA" create -y -f envs/rctsne/environment.yml
fi
[ -f "$PREFIX/include/fftw3.h" ] || { echo "error: no fftw3.h under $PREFIX" >&2; exit 1; }

# --- compiler --------------------------------------------------------------------------------
# Prefer $CXX, then use the first available common C++ compiler.
if [ -n "${CXX:-}" ]; then
  :
elif command -v g++ >/dev/null 2>&1; then
  CXX=g++
elif command -v clang++ >/dev/null 2>&1; then
  CXX=clang++
else
  echo "error: no C++ compiler found. Set CXX, or install g++ or clang++." >&2
  exit 1
fi
echo "compiler: $("$CXX" --version | head -1)"
echo "fftw:     $PREFIX"

# --- build -----------------------------------------------------------------------------------
# Embed the FFTW library path so the binary can find it at run time.
mkdir -p "$VENDOR/bin"
LOG="$VENDOR/bin/build.log"
set +e
"$CXX" -std=c++11 -O3 -Wall -Wextra \
  "$VENDOR/src/sptree.cpp" "$VENDOR/src/tsne.cpp" "$VENDOR/src/nbodyfft.cpp" \
  -o "$VENDOR/bin/fast_ctsne" \
  -pthread -lfftw3 -lm -Wno-address-of-packed-member \
  -I"$VENDOR/src" -I"$PREFIX/include" -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" 2> "$LOG"
status=$?
set -e
if [ $status -ne 0 ]; then
  echo "error: compilation failed. First errors:" >&2
  grep -m5 -i 'error' "$LOG" >&2 || tail -20 "$LOG" >&2
  exit $status
fi

# Reject builds that lost the vendor fix for uninitialized `sum_P`.
if grep -i 'uninitialized' "$LOG" | grep -q 'sum_P'; then
  echo "error: an -Wuninitialized warning for sum_P is back. Vendor patch 4 is missing:" >&2
  grep -i 'uninitialized' "$LOG" | grep -m3 'sum_P' >&2
  exit 1
fi

echo "built:    $VENDOR/bin/fast_ctsne  ($(grep -c 'warning:' "$LOG") warnings, none for sum_P)"
echo
echo "export NE_ALIGNED_RCTSNE_BIN=$VENDOR/bin/fast_ctsne"
