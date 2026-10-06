# ITEACH-Net

PyTorch implementation of **ITEACH-Net: Inverted Teacher-studEnt seArCH Network for Emotion Recognition in Conversation**.

This repository implements the ITS-NAS model on **four-class IEMOCAP**: a three-layer Transformer teacher learns from complete input, while a four-layer NAS student learns from incomplete input. It supports constant, Random, and Progressive modality-missing strategies.

## Installation

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

Prepare the IEMOCAP labels and pre-extracted audio, text, and visual features:

```text
IEMOCAP/
├── IEMOCAP_features_raw_4way.pkl
└── features/
    ├── wav2vec-large-c-UTT/
    ├── deberta-large-4-UTT/
    └── manet_UTT/
```

See [GCNet's data preparation instructions](https://github.com/zeroQiaoba/GCNet#datasets) and the [IEMOCAP dataset](https://sail.usc.edu/iemocap/). Training uses five folds, each holding out one session. File formats are described in [Implementation details](docs/implementation.md).

## Training

Run one fold with the Random training strategy:

```bash
python -m iteach_iemocap.train \
  --data-root /path/to/IEMOCAP \
  --output-dir runs/random/seed100/fold1 \
  --fold 1 --seed 100 --mask-type random --epochs 100
```

Use `--mask-type progressive` for Progressive training, `constant-0.7` for a fixed missing rate, or `constant-0.0` for complete input. Run folds 1–5 separately with `--fold`, using a new output directory for each run. Add `--no-cuda` for CPU execution.

Each run writes `best.pt`, `history.jsonl`, `summary.json`, `arguments.json`, and `optimizer.json`. The latter records the actual model parameter counts.

## Implementation and attribution

Model settings, masking, losses, and parameter counts are documented in [Implementation details](docs/implementation.md). Data loading and masking derive from [GCNet](https://github.com/zeroQiaoba/GCNet); see [NOTICE.md](NOTICE.md) for provenance and licensing status.
