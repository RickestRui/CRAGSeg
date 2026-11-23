# CRAGSeg

**Cross-Modal Region Aggregation for Fire and Smoke Segmentation with Evidence-Guided Graphs**

CRAGSeg segments background, smoke, and fire in paired aerial RGB and infrared images. The graph connects corresponding regions and adds reciprocal cross-position neighbors. Image evidence and preliminary class predictions guide the aggregation of neighboring features before multiscale fusion and decoding.

## Installation

The reference experiments used Python 3.12, PyTorch 2.5.1 and torchvision 0.20.1 with CUDA 12.4. For that configuration, run the following commands in a Python environment from the repository root.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Dataset

Download `CRAGSeg_FLAME2_data.zip` from [Baidu Netdisk](https://pan.baidu.com/s/1yDVbnnljm8W9ebgsmAxRhw?pwd=69wv) using the extraction code `69wv`. Extract the archive at the repository root. The archive contains the paired images, segmentation masks, and fixed split used in the experiments.

```text
data/
  FLAME2/images/                 # RGB, infrared, and ground-truth PNG files
  lists/egr_fs_grouped_20261113/  # Split lists and group assignments
  DATA_SOURCES.md
```

The split contains 566 training, 131 validation, and 205 test pairs. The exclusion list records 90 additional frame IDs used as buffers between splits. In the source masks, grayscale values 0, 125, and 255 denote background, smoke, and fire. The loader maps them to class IDs 0, 1, and 2. To store the data elsewhere, pass the enclosing `data` directory with `--data-dir`.

## Training and evaluation

Check the dataset, train from scratch with one seed, and test the checkpoint selected by validation mIoU.

```bash
python main.py audit --data-dir data
python main.py train --data-dir data --seeds 200 --device cuda
python main.py test --data-dir data --seeds 200 --device cuda
```

Run the five reference seeds (200 through 204) with the settings in `config/default.json`.

```bash
python main.py all --data-dir data --device cuda
```

The default input size is 256 × 256 and the effective batch size is 8. Training uses AdamW with an initial warmup and a cosine learning-rate schedule. The best checkpoint is selected independently for each seed using validation mIoU. Use `--resume` with the original training configuration to continue an interrupted run.

## Data attribution

The visible and infrared frames come from the FLAME 2 dataset by Hopkins et al. The segmentation masks come from RoboFireFuseNet by Fotiou, Mygdalis, and Pitas. The data archive includes the attribution and reuse terms. [FLAME 2 dataset](https://doi.org/10.21227/swyw-6j78) · [RoboFireFuseNet paper](https://doi.org/10.1016/j.patrec.2026.04.024) · [Upstream license notice](licenses/RoboFireFuseNet-MIT.txt)
