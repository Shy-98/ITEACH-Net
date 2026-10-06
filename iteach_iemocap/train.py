"""Train the fixed ITEACH ITS-NAS pair on one IEMOCAP4 session fold."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import torch

from .data import get_fold_loaders
from .engine import run_epoch
from .losses import MaskedCELoss, MaskedHiddenDistillationLoss
from .model import ITEACHPair

EVALUATION_RATES = tuple(step / 10 for step in range(8))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="folder containing IEMOCAP_features_raw_4way.pkl and features/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new, empty directory for this fold/seed run")
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--audio-feature", default="wav2vec-large-c-UTT")
    parser.add_argument("--text-feature", default="deberta-large-4-UTT")
    parser.add_argument("--video-feature", default="manet_UTT")
    parser.add_argument("--mask-type", default="constant-0.7",
                        help="constant-R for fixed-rate IME, or random/progressive")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--router-learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    if args.learning_rate <= 0 or args.router_learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rates must be positive and weight decay nonnegative")
    if args.mask_type.startswith("constant-"):
        try:
            rate = float(args.mask_type.split("-", 1)[1])
        except ValueError:
            parser.error("fixed mask type must be constant-R with numeric R")
        if not 0.0 <= rate <= 0.7:
            parser.error("constant missing rate must be in [0,0.7]")
    elif args.mask_type not in ("random", "progressive"):
        parser.error("--mask-type must be constant-R, random, or progressive")
    return args


def seed_process(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_process(args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = args.data_root
    label_path = dataset_root / "IEMOCAP_features_raw_4way.pkl"
    feature_root = dataset_root / "features"
    train_loaders, test_loaders, dims = get_fold_loaders(
        str(label_path), str(feature_root / args.audio_feature),
        str(feature_root / args.text_feature), str(feature_root / args.video_feature),
        args.batch_size)
    fold_index = args.fold - 1
    train_loader, test_loader = train_loaders[fold_index], test_loaders[fold_index]
    max_tokens = train_loader.dataset.max_len
    model = ITEACHPair(*dims, max_tokens=max_tokens)

    all_parameters = list(model.parameters())
    router_parameters = list(model.router_parameters())
    router_ids = {id(parameter) for parameter in router_parameters}
    all_ids = {id(parameter) for parameter in all_parameters}
    non_router_parameters = [p for p in all_parameters if id(p) not in router_ids]
    if (len(router_ids) != len(router_parameters) or
            router_ids.union({id(p) for p in non_router_parameters}) != all_ids):
        raise RuntimeError("Router and non-Router optimizer groups are not a disjoint partition")

    optimizer = torch.optim.AdamW([
        {"params": non_router_parameters, "lr": args.learning_rate},
        {"params": router_parameters, "lr": args.router_learning_rate},
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    cls_loss = MaskedCELoss()
    hidden_loss = MaskedHiddenDistillationLoss()

    optimizer_metadata = {
        "optimizer": "AdamW",
        "learning_rate_non_router": args.learning_rate,
        "learning_rate_router": args.router_learning_rate,
        "weight_decay": args.weight_decay,
        "parameter_values_teacher": sum(p.numel() for p in model.teacher.parameters()),
        "parameter_values_student": sum(p.numel() for p in model.student.parameters()),
        "parameter_values_total": sum(p.numel() for p in model.parameters()),
        "architecture": model.architecture,
    }
    (args.output_dir / "optimizer.json").write_text(
        json.dumps(optimizer_metadata, indent=2) + "\n")
    (args.output_dir / "arguments.json").write_text(
        json.dumps(vars(args) | {"data_root": str(args.data_root),
                                 "output_dir": str(args.output_dir)}, indent=2) + "\n")

    if args.mask_type.startswith("constant-"):
        fixed_rate = float(args.mask_type.split("-", 1)[1])
        strategy = "constant"
    else:
        fixed_rate = 0.7
        strategy = args.mask_type

    evaluation_rates = ([fixed_rate] if strategy == "constant" else list(EVALUATION_RATES))
    selection_metric = ("test_waf_at_training_rate" if strategy == "constant"
                        else "equal_weight_mean_test_waf_over_eight_rates")
    selection_protocol = ("one held-out test evaluation at the fixed training rate per epoch"
                          if strategy == "constant" else
                          "one held-out test evaluation at each of eight rates per epoch; select by their equal-weight mean WAF")
    history_path = args.output_dir / "history.jsonl"
    best_selection_score = -1.0
    best_epoch = None
    best_evaluations = None
    best_mean_waf = None
    best_mean_accuracy = None
    best_path = args.output_dir / "best.pt"
    with history_path.open("w") as history:
        for epoch in range(args.epochs):
            training = run_epoch(model, train_loader, device, cls_loss, hidden_loss,
                                 train=True, optimizer=optimizer, strategy=strategy,
                                 fixed_rate=fixed_rate, epoch_index=epoch)
            evaluations = []
            for rate in evaluation_rates:
                metrics = run_epoch(model, test_loader, device, cls_loss, hidden_loss,
                                    train=False, fixed_rate=rate)
                evaluations.append({"nominal_missing_rate": float(rate), **metrics})
            mean_waf = float(np.mean([row["waf"] for row in evaluations]))
            mean_accuracy = float(np.mean([row["accuracy"] for row in evaluations]))
            selection_score = (evaluations[0]["waf"] if strategy == "constant" else mean_waf)
            row = {
                "epoch": epoch + 1,
                "fold": args.fold,
                "seed": args.seed,
                "training_strategy": strategy,
                "training_fixed_rate": fixed_rate if strategy == "constant" else None,
                "selection_evaluation_rates": evaluation_rates,
                "selection_metric": selection_metric,
                "selection_score": float(selection_score),
                "evaluation_mean_waf": mean_waf,
                "evaluation_mean_accuracy": mean_accuracy,
                "architecture": model.architecture,
                "train": training,
                "evaluation_by_rate": evaluations,
            }
            history.write(json.dumps(row, sort_keys=True) + "\n")
            history.flush()
            if selection_score > best_selection_score:
                best_selection_score = float(selection_score)
                best_epoch = epoch + 1
                best_evaluations = evaluations
                best_mean_waf = mean_waf
                best_mean_accuracy = mean_accuracy
                torch.save({
                    "format": "iteach_iemocap4_v3",
                    "epoch": best_epoch,
                    "seed": args.seed,
                    "fold": args.fold,
                    "arguments": json.loads((args.output_dir / "arguments.json").read_text()),
                    "architecture": model.architecture,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "selection_metric": selection_metric,
                    "selection_score": float(selection_score),
                    "selection_evaluation_rates": evaluation_rates,
                    "evaluation_mean_waf": mean_waf,
                    "evaluation_mean_accuracy": mean_accuracy,
                    "evaluation_metrics_by_rate": evaluations,
                    "test_used_for_checkpoint_selection": True,
                }, best_path)
            print(json.dumps({"epoch": epoch + 1, "train": training,
                              "selection_score": float(selection_score),
                              "selection_metric": selection_metric,
                              "evaluation_by_rate": evaluations}, sort_keys=True), flush=True)

    summary = {
        "format": "iteach_iemocap4_v3",
        "fold": args.fold,
        "seed": args.seed,
        "training_strategy": strategy,
        "training_rate": fixed_rate if strategy == "constant" else None,
        "selection_evaluation_rates": evaluation_rates,
        "selected_epoch": best_epoch,
        "selection_score": best_selection_score,
        "selection_metric": selection_metric,
        "selection_protocol": selection_protocol,
        "evaluation_mean_waf": best_mean_waf,
        "evaluation_mean_accuracy": best_mean_accuracy,
        "per_rate_results": best_evaluations,
        "test_used_for_checkpoint_selection": True,
        "checkpoint": str(best_path),
        "optimizer": optimizer_metadata,
        "architecture": model.architecture,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
