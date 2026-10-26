# U-Net Person Segmentation

Train a small U-Net on Penn-Fudan pedestrian masks with deterministic train/validation/test split.

## Run

Use Python 3.10+ in a virtual environment; install `pip install -r requirements.txt`, then run `python main.py --help` (or see project-specific command below). Dataset files, model weights and outputs are kept locally and ignored by Git.

Training code uses a fixed random seed/split and writes held-out metrics and model artifacts to `outputs/`.
