# Vendored native code

One method here is a compiled program rather than a package: **`rc-tsne`**, revised conditional t-SNE. Its sources are in `rctsne/`, copied from the authors' repository with a recorded patch series; `rctsne/VENDOR.md` holds the upstream commit, the per-file licences and every patch.

This directory is the third of the isolation patterns in this repository, and the only one that needs a *build*:

| pattern | what has to exist | contract | doc |
|---|---|---|---|
| `r_scripts/` | an R install with Seurat, rliger, bcTSNE | HDF5 in, HDF5 out | `r_scripts/README.md` |
| `py_scripts/` | a second Python interpreter with numpy < 2 | `.npz` in, `.npz` out | `py_scripts/README.md` |
| `vendor/` | FFTW and a C++ compiler, **plus a build step** | their own binary `.dat` format | this file |

## Why a binary and not a reimplementation

The paper's contribution is the affinity matrix: it conditions the high-dimensional similarities and then runs t-SNE unchanged, and says so explicitly. That makes a Python reimplementation on top of `run_tsne_from_graph` entirely feasible, and it was the first plan. Running their binary instead was a deliberate choice: the number this benchmark reports for rc-tsne is then theirs, not our reading of their paper.

It costs one thing, and the cost is real. rc-tsne is the only method here whose optimiser is not openTSNE -- the binary is a FIt-SNE fork running 750 iterations with its own learning rate and initialisation, against the 200 every other t-SNE method uses. A comparison against `joint-tsne` therefore differs in two things rather than one. That is a documented property of the comparison, not hidden.

## Building

`bash scripts/setup_external_envs.sh` builds the binary through `scripts/setup/build_rctsne.sh`, which gets FFTW from the pinned `envs/rctsne` environment, picks a compiler from `$CXX`, then `g++`, then `clang++`, and links with `-rpath` so the binary finds `libfftw3` at run time. `rctsne/bin/` is git-ignored: the binary is built per machine and is never shipped.

## What the build script guards

Upstream reads an uninitialised `double` as the row normaliser in the function that applies the conditioning, on the code path every default run takes. The patched source initialises it, and the build script **fails** if an `-Wuninitialized` warning reappears -- which is what would happen if someone re-vendored the tree and lost the patch. See `rctsne/VENDOR.md` for the measured effect, which is nil on conclusions and not nil on coordinates.

## Tests

`tests/test_rctsne_subprocess.py`. It runs against `tests/mock_rctsne_binary.py` by default, so the suite needs neither a compiler nor FFTW. The mock parses the real `.dat` input rather than ignoring it, which is how the alpha/beta inversion is checked end to end: the test asserts what the binary was *handed*, not what the Python call site said.

Set `NE_ALIGNED_TEST_REAL_RCTSNE=1` to add two tests that need the binary -- determinism per seed, and a check that our patched driver produces byte-identical output to the authors' unpatched one.

## Adding another vendored program

Copy the shape: sources under `vendor/<name>/`, every upstream licence header preserved, a `VENDOR.md` with the commit and the patch diffs, a build script under `scripts/setup/`, a `<name>_subprocess.py` wrapper resolving the binary from one environment variable through `shlex.split`, and a mock under `tests/`. Add the directory to `extend-exclude` in `pyproject.toml` so ruff does not reformat code whose diffs are the record of what we changed.
