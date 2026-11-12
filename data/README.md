# Data

This project uses **CIFAR-10** (50,000 training + 10,000 test images, 32x32 RGB, 10 classes).

**You do not need to download anything manually.** The notebook calls
`torchvision.datasets.CIFAR10(root="./data", download=True)`, which fetches the dataset
on first run and re-uses it afterwards.

| Where you run | Where the data ends up |
|---|---|
| Google Colab | `/content/data/` (temporary; deleted when the runtime resets) |
| Locally, Jupyter started from the **repo root** | `./data/cifar-10-batches-py/` (this folder) |

Dataset files are git-ignored; only this README is tracked.
CIFAR-10 is distributed by its authors (Krizhevsky, 2009: https://www.cs.toronto.edu/~kriz/cifar.html);
please follow their terms and cite them if you use it.
