# Vendored: revised conditional t-SNE

Upstream: <https://github.com/aida-ugent/revised-conditional-t-SNE>
Commit: `afda44b6d04066077d6fe46da1127de28de6a034` ("add code", 2023-02-17)
Fetched: 2026-08-20

The method is `rc-tsne` in this benchmark. Paper: Heiter, Kang, Seurinck and Lijffijt,
"Revised Conditional t-SNE: Looking Beyond the Nearest Neighbors", IDA 2023,
[arXiv 2302.03493](https://arxiv.org/abs/2302.03493).

Only the `fitsne/` half of that repository is here. Its `ctsne/` directory implements the
*original* conditional t-SNE, which is a different method and is not part of this benchmark.

## What was copied, and what was not

| here | upstream | note |
|---|---|---|
| `src/*.cpp`, `src/*.h` | `fitsne/src/` | the FIt-SNE fork that implements the method |
| `src/progress_bar/` | `fitsne/src/progress_bar/` | copied with its own `LICENSE` |
| `src/winlibs/*.h`, `*.c`, `*.cpp` | `fitsne/src/winlibs/` | 48 KB of text headers, see below |
| `fast_ctsne.py` | `fitsne/fast_ctsne.py` | the driver, patched; see the patch series |
| `fast_ctsne_upstream.py` | `fitsne/fast_ctsne.py` | byte-identical copy, used only by the equality test |
| `UPSTREAM_README.md` | `fitsne/README.md` | their build and usage notes, unedited |
| `LICENSE` | repository root | the MIT notice for the repository as a whole |
| `data/synthetic_dataset.csv` | `experiments/synthetic/` | 1500 x 10, used by `scripts/analyse/verify_rctsne.py` |
| `data/wandb_synthetic_results.csv` | `experiments/synthetic/` | their published scores for that dataset |

**`src/winlibs/` is needed even off Windows.** `sptree.cpp`, `tsne.cpp` and `nbodyfft.cpp` each
open with an unconditional `#include "winlibs/stdafx.h"`. That header's whole body sits inside
`#ifdef _WIN32`, so it compiles to nothing here, but the file has to exist. The other three
`winlibs` includes -- `fftw3.h` in `nbodyfft.h`, `unistd.h` in `tsne.cpp`, `mman.h` in
`annoylib.h` -- are all inside `#ifdef _WIN32`. The small text files are kept so a Windows build
stays possible; `winlibs/fftw/`, 6.6 MB of Windows DLLs, import libraries and Fortran bindings, is
**not** copied.

Their README's dataset path is wrong: it says `experiments/synthetic/synthetic_data.csv`, and the
file in the repository is `synthetic_dataset.csv`.

## Licences -- not simply MIT

The repository root is MIT (Ghent University AIDA group, 2023), but the vendored files carry their
own upstream notices. Every per-file header is preserved verbatim. Audited file by file:

| file | copyright | licence |
|---|---|---|
| `src/tsne.cpp`, `src/tsne.h`, `src/sptree.cpp`, `src/sptree.h`, `src/vptree.h` | Laurens van der Maaten (Delft University of Technology), 2014 | **BSD 4-clause** |
| `src/annoylib.h` | Spotify AB, 2013 | Apache 2.0 |
| `src/progress_bar/ProgressBar.hpp` |  Prakhar Srivastav, 2016 | MIT |
| `src/nbodyfft.cpp`, `src/nbodyfft.h`, `src/kissrandom.h`, `src/parallel_for.h`, `src/time_code.h`, `src/winlibs/*` | no header | covered by the root MIT notice |
| `fast_ctsne.py` | no header | covered by the root MIT notice |

The van der Maaten files are BSD **4**-clause, which includes the advertising clause: "All
advertising materials mentioning features or use of this software must display the following
acknowledgement: This product includes software developed by the Delft University of Technology."
That is a notice obligation on us, not just a comment in a file. Do not collapse this table into a
single MIT statement.

## Patch series

Patch 4 is to the C++; patches 1 to 3 are to `fast_ctsne.py`. `fast_ctsne_upstream.py` is kept
unmodified beside the patched copy so `tests/test_rctsne_subprocess.py` can assert the two agree.

### Patch 4 -- `src/tsne.cpp`, an uninitialised read

```diff
--- a/src/tsne.cpp
+++ b/src/tsne.cpp
@@ -1539 +1539 @@
-    double sum_P;
+    double sum_P = 0.0;
```

**Why.** In `distances2similarities_conditional`, the function that applies the `cond_alpha` /
`cond_beta` reweighting, the row normaliser was declared uninitialised and read on the next line:

```cpp
if (n_dist>=0) P[n_dist] = DBL_MIN;
double sum_P;
for (int m = 0; m < N; m++) sum_P += P[m];
for(int m = 0; m < N; m++) P[m] /= sum_P;
```

Reading an uninitialised `double` is undefined behaviour. This is not a dead path: `fixed_perp` is
`True` by default in `fast_ctsne.py`, and `distances2similarities` routes `fixed_perp == true`
straight to this function (`tsne.cpp:1716`). Every default rc-t-SNE run reaches it.

Apple clang 21.0.0 at `-O3 -Wall -Wextra` reports it, and reports nothing else of the kind --
exactly one `-Wuninitialized` in a build that emits 150 warnings in total:

```
src/tsne.cpp:1540:33: warning: variable 'sum_P' is uninitialized when used here [-Wuninitialized]
 1540 |     for (int m = 0; m < N; m++) sum_P += P[m];
```

The other two `double sum_P;` declarations, at lines 1430 and 1573, are assigned `DBL_MIN` in
every branch before use. They are left alone.

**What it changes, measured.** Both binaries were built from the same sources differing only in
that line, and run on `data/synthetic_dataset.csv` with the paper's synthetic settings --
perplexity 30, theta 0.2, 750 iterations, `fixed_perp=True`, one thread, seed 42 -- twice each in
fresh processes. `alpha` below is the argument name in their code, which is the paper's beta.

| alpha | build | purity(batch) | Rand(celltype) | run 1 vs run 2 | kNN overlap, k=30 |
|---|---|---|---|---|---|
| 1e-20 | pristine | 0.4950 | 0.9980 | identical | -- |
| 1e-20 | patched | 0.4971 | 0.9980 | identical | 0.5826 |
| 1e-4 | pristine | 0.9217 | 1.0000 | identical | -- |
| 1e-4 | patched | 0.9203 | 1.0000 | identical | 0.6904 |
| 1.0 | pristine | 1.0000 | 0.4897 | identical | -- |
| 1.0 | patched | 1.0000 | 0.4730 | identical | 0.6440 |

Three things follow. The coordinates differ between the two builds -- a kNN overlap of 0.58 to
0.69 rather than 1.0 -- which is ordinary t-SNE sensitivity to any perturbation of its input, not
evidence of corruption. No metric moves outside noise, so **no conclusion in this benchmark
depends on the patch**. And each build is bit-identical across repeated fresh processes, so on
this platform the uninitialised slot holds a stable value.

That last point is exactly why the patch ships anyway. "Stable here" is a property of this
compiler, this optimisation level and this stack layout, not of the program. A Linux cluster build
could land a different value, and nothing in the code would say so. One line removes the question.

`alpha = 1.0` is their own definition of ordinary t-SNE ("With alpha = 1 we compute normal
t-SNE"), and it goes through the same buggy line, so the reference arm was affected too.

**The reproduction, incidentally.** Read down the `alpha` column of the patched rows: at
`alpha = 1` the batch label is perfectly recoverable from the layout (purity 1.0000) and the
celltype is not (Rand 0.4730); conditioning reverses both, and at 1e-20 the batch label is near
chance while the three celltype clusters are recovered essentially perfectly. That is the paper's
Figure 1 claim, on their data, with their settings.

### Patch 1 -- temporary files go where the caller says

Upstream writes `data_<timestamp>.dat` and `result_<timestamp>.dat` into `os.getcwd()`
(`fast_ctsne.py` lines 332, 400, 420, 421, 429-432). Two sweep tasks launched from the same
directory would collide, and a sweep would leave debris in the repository root. The patched driver
takes a directory argument.

### Patch 2 -- the binary is named, not derived

Upstream resolves `os.path.dirname(os.path.realpath(__file__)) + "/bin/fast_ctsne"` (line 389).
The patched driver takes a command list, which `rctsne_subprocess.rctsne_command()` builds from
`NE_ALIGNED_RCTSNE_BIN` with `shlex.split`, the same convention as
`r_subprocess.rscript_command` and `multimap_subprocess.multimap_python_command`. This is also
what lets `tests/mock_rctsne_binary.py` stand in for the binary.

### Patch 3 -- a real error on failure

Upstream catches `CalledProcessError`, prints "t-SNE did not execute correctly", and then falls off
the end of the function, so a failed run **returns `None`** and looks like a successful one to the
caller. In a sweep that would land as a `TypeError` somewhere downstream, or worse, as a silently
skipped task. The patched driver raises `FastCtsneError` carrying the captured output;
`rctsne_subprocess` re-raises it as `RcTsneSubprocessError`. The exception type is defined in the
vendored module itself so that file still imports nothing from this package.

Nothing else in `fast_ctsne.py` changes. In particular the binary `.dat` layout is theirs, field
for field: it is the part that would be easy to get subtly wrong, so it is not rewritten.

## The alpha / beta inversion

**Their code names the conditioning parameter the opposite way round from their paper.** Confirmed
in both files:

- `fast_ctsne.py` asserts `alpha > 0 and alpha <= 1.0`, "With alpha = 1 we compute normal t-SNE",
  and derives `beta = (1 - alpha * ratio) / (1 - ratio)` with
  `ratio = sum(n_l * (n_l - 1) / (n * (n - 1)))`.
- `src/tsne.cpp` applies `cond_alpha` when `labels[getIndex(m)] == label` -- the same-label case,
  commented "apply cond_alpha to decrease similarity" -- and `cond_beta` otherwise.

So their `alpha` is the paper's beta: the small number that suppresses same-label affinity. Our
flag is `--rc-beta`, in paper terms, and it is passed to their `alpha=`. Swapping the two at 1e-4
would raise same-batch similarity roughly twenty thousand fold instead of lowering it, and the
output would look like a very bad integration method rather than a bug.

## Building

`scripts/setup/build_rctsne.sh`. It needs FFTW and a C++ compiler; `bin/` is gitignored, so the
binary is built rather than tracked. The build is platform-specific and is not part of the
reproducibility chain the rest of this repository has -- a cluster run needs the build repeated
there.
