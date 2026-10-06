"""Single-mask fixed-checkpoint eight-rate UME for the IEMOCAP4 ITS-NAS Student."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .data import get_fold_loaders
from .engine import make_masked_batch
from .model import ITEACHPair

UME_RATES = tuple(step / 10 for step in range(8))


def reset_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_rate(model, loader, device, rate: float, mask_seed: int):
    reset_seed(mask_seed)
    model.eval()
    predictions, labels_all, valid_all = [], [], []
    missing_count = valid_feature_count = 0
    with torch.no_grad():
        for data in loader:
            _complete, incomplete, input_mask, _qmask, umask, labels = make_masked_batch(
                data, rate, random_training=False)
            audio_dim, text_dim = data[0].size(-1), data[1].size(-1)
            incomplete_b = incomplete.transpose(0, 1).contiguous()
            student_features = {
                "audio": incomplete_b[..., :audio_dim],
                "text": incomplete_b[..., audio_dim:audio_dim + text_dim],
                "video": incomplete_b[..., audio_dim + text_dim:],
            }
            valid = umask.to(device=device).bool()
            student_features = {name: value.to(device=device)
                                for name, value in student_features.items()}
            output = model.student(student_features, valid)
            pred = output["logits"].reshape(-1, 4).argmax(dim=1)
            predictions.append(pred.cpu().numpy())
            labels_all.append(labels.reshape(-1).numpy())
            valid_flat = umask.reshape(-1).numpy()
            valid_all.append(valid_flat)
            valid_feature_count += int(umask.sum().item()) * 3
            missing_count += int(((3 - input_mask.sum(dim=-1).transpose(0, 1)) * umask).sum().item())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels_all)
    mask = np.concatenate(valid_all)
    return {
        "mask_seed": int(mask_seed),
        "nominal_missing_rate": float(rate),
        "actual_missing_feature_fraction": float(missing_count / valid_feature_count),
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=mask)),
        "waf": float(f1_score(y_true, y_pred, sample_weight=mask, average="weighted")),
        "valid_utterances": int(mask.sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="this package's own best.pt; no legacy schema adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--audio-feature", default="wav2vec-large-c-UTT")
    parser.add_argument("--text-feature", default="deberta-large-4-UTT")
    parser.add_argument("--video-feature", default="manet_UTT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mask-seed", type=int, default=7000)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.data_root / "IEMOCAP_features_raw_4way.pkl"
    feature_root = args.data_root / "features"
    _train, test_loaders, dims = get_fold_loaders(
        str(label_path), str(feature_root / args.audio_feature),
        str(feature_root / args.text_feature), str(feature_root / args.video_feature),
        args.batch_size)
    loader = test_loaders[args.fold - 1]
    max_tokens = loader.dataset.max_len
    model = ITEACHPair(*dims, max_tokens=max_tokens)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "iteach_iemocap4_v3":
        raise ValueError("checkpoint is not in this package's ITEACH-IEMOCAP4 format")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)

    rows = []
    for rate in UME_RATES:
        row = evaluate_rate(model, loader, device, rate, args.mask_seed)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    with (args.output_dir / "ume.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "format": "iteach_iemocap4_ume_v3",
        "fold": args.fold,
        "training_seed": checkpoint["seed"],
        "checkpoint": str(args.checkpoint),
        "selected_epoch": checkpoint["epoch"],
        "test_used_for_checkpoint_selection": checkpoint["test_used_for_checkpoint_selection"],
        "measurement_note": "This is an optional remeasurement with fresh masks; use the checkpoint training summary for the selected epoch curve.",
        "ume_fixed_checkpoint_no_reselection": True,
        "mask_seed": int(args.mask_seed),
        "rates": rows,
        "eight_rate_mean_waf": float(np.mean([row["waf"] for row in rows])),
        "eight_rate_mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "eight_rate_mean_actual_missing_feature_fraction": float(np.mean(
            [row["actual_missing_feature_fraction"] for row in rows])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
