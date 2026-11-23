# README illustrations

- `overview.png` is a raster rendering of panels **a–b** from
  `fig1_w_kissing_embd.pdf`: joint t-SNE and OT-tSNE on the lupus dataset, colored by batch
  and cell type. The manuscript's graph illustrations and benchmark ranking are outside
  this crop, and the panel letters are omitted in the README rendering. The source figure
  is assembled in
  [reproduce_paper_figures.ipynb](../notebooks/reproduce_paper_figures.ipynb).
- `method.png` and `method.pdf` show transport, affinity construction, sparsification,
  and t-SNE optimization. Regenerate them with
  [method_figure.ipynb](../notebooks/method_figure.ipynb).
- `toy_embedding.png` is an output of [examples/toy.py](../examples/toy.py).
  Columns compare joint t-SNE before integration with OT-tSNE; rows show batch and
  population identity using consistent colors across methods.
- `ncov_embedding.png` is the OT-tSNE embedding made by
  [examples/ncov.py](../examples/ncov.py) from the included nCoV PCA inputs. Both panels
  show the same coordinates, colored by sample and cell type.

Running the examples writes to `results/examples/`, not here. t-SNE layouts can differ slightly
between machines, so a rerun may place the clusters differently.

The README uses these local images so viewing it does not require benchmark result files.
