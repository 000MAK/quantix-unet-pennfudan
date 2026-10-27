# U-Net Pedestrian Segmentation | Penn-Fudan

A compact PyTorch U-Net that segments pedestrians from the Penn-Fudan image and mask dataset. The pipeline downloads and safely extracts the dataset, makes deterministic train/validation/test splits, and writes a held-out test summary.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 1
```

Use `python main.py --help` to see available options. Data is cached under `data/`; checkpoints and metrics go under `outputs/`. Both directories are excluded from Git.

## Data and scope

The dataset is small and intended for a segmentation workflow demonstration, not a production benchmark. Dataset source: [University of Pennsylvania - Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/).

## Model

Two-level U-Net with convolutional blocks, skip connections, and binary mask prediction. Test performance depends on the small dataset and random split; inspect the generated metrics before drawing conclusions.

## Results

A three-epoch local run reached **IoU 0.0** on validation and test: the network predicted background everywhere. This is a known failure, not a result. Likely fixes are a class-balanced or Dice loss and longer training.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
