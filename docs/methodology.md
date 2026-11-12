# Methodology

This document describes what `notebooks/SimCLR_CIFAR10_Colab.ipynb` does, in execution order.

1. **Setup & configuration** – imports, `CFG` hyper-parameters, `QUICK_MODE` switch, seeding, device selection (CUDA if available), fp16 autocast flag.
2. **Data** – four views of CIFAR-10 train/test: two-view SimCLR augmentation (train), plain normalised (feature extraction + test), standard crop/flip (supervised). `DataLoader(drop_last=True)` for SimCLR so every step has exactly 2N = 512 views. A cell visualises original / view *i* / view *j*.
3. **Model** – `make_resnet18_cifar()` (3×3 stride-1 stem, no max-pool, `fc → Identity`) and `SimCLR(use_head)` returning `(h, z)`.
4. **Loss** – `nt_xent_loss(z, tau)`: L2-normalise, cosine-similarity matrix / τ, mask the diagonal, cross-entropy against the positive index `(i+N) mod 2N`. Also returns in-batch top-1 retrieval accuracy as a diagnostic. Cell includes assertions comparing it to a naive per-pair implementation (|diff| = 2.4e-07 in the saved output).
5. **Pretraining** – `pretrain_simclr(use_head, epochs, name)`; Adam + cosine decay + AMP; saves `simclr_<name>.pt`. Run for `use_head=True` (Exp. 1) and `use_head=False` (Exp. 2) with the same seed.
6. **Linear evaluation** – freeze encoder, extract `h` (and `z`) once, standardise with *train* statistics, train one `nn.Linear(d,10)` for `linear_epochs`; report the final-epoch test accuracy. Also done for a randomly initialised encoder.
7. **Supervised baseline** – same ResNet-18 backbone + linear classifier, SGD/OneCycle, trained end-to-end.
8. **Results** – comparison table (`results_table.csv`), three-panel figure (`simclr_results.png`), and linear-probe / supervised learning curves.

Cells must be run in order, top to bottom; later cells depend on variables (`CFG`, loaders, models, `results`) defined earlier.

## Notes on the notebook
- Section headings in the notebook's markdown are not consecutively numbered (e.g. there is no "4" heading for the data cell). This is cosmetic and was left untouched.
- Some code comments mention "15–30 recommended" epochs while the configured value is 40; the configured value (and the markdown table) is what produced the saved results.
