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
