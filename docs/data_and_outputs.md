# Data and outputs usage guide

## What the notebook reads
- CIFAR-10, auto-downloaded to `CFG.data_root = "./data"` (see [`data/README.md`](../data/README.md)).
- No Google Drive access, API keys, or user-provided files are required.

## What the notebook writes (relative to the notebook's working directory)
| File | Written by | Notes |
|---|---|---|
| `/content/simclr_nonlinear_head.pt` | Experiment 1 (`pretrain_simclr`) | SimCLR weights. Path is `/content/...` on Colab, else `./simclr_nonlinear_head.pt` |
| `/content/simclr_no_head.pt` | Experiment 2 | Same as above |
| `results_table.csv` | Results cell | Comparison table (observed accuracies) |
| `simclr_results.png` | Plotting cell | 3-panel results figure (dpi 150) |

On Colab the working directory is `/content`, which is **wiped when the runtime is recycled**. Download or copy
anything you want to keep. The supervised model is not saved to disk by the notebook.

## Keeping outputs on Google Drive (optional, not in the original notebook)
Add this in a new cell *before* the experiments if you want persistence:
```python
from google.colab import drive
drive.mount('/content/drive')
```
then copy files, e.g. `!cp /content/*.pt /content/results_table.csv /content/simclr_results.png "/content/drive/MyDrive/simclr_outputs/"`.
The notebook itself was left unmodified, so this is an opt-in step.

## Committing outputs
Checkpoints (`*.pt`) are git-ignored. If you want to share weights, use GitHub Releases, Hugging Face Hub,
or Git LFS instead of normal commits. `outputs/` is a local scratch folder (git-ignored apart from `.gitkeep`);
move generated files there if you like.
