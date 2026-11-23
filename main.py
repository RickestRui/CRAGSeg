"""Train CRAGSeg and evaluate a validation-selected checkpoint on the grouped FLAME2 split."""
import argparse
import json
from pathlib import Path

import torch

from cragseg.data import audit_data
from cragseg.detail_metrics import DetailCounts
from cragseg.engine import convolution_channels_last, dataset, make_loader, setup, train_one, validate
from cragseg.io import write_csv
from cragseg.model import CRAGSeg


def evaluate(data_dir, output, config, seed, device, split, checkpoint):
    run_dir = output / f"seed_{seed}"
    weights = checkpoint or run_dir / "weights" / "best.pth"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint {weights}. Train this seed or pass --checkpoint.")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    training_seed = state.get("run", {}).get("seed")
    if training_seed is not None and training_seed != seed:
        raise ValueError(f"Checkpoint seed {training_seed} differs from the selected seed {seed}.")
    setup(seed, config, device)
    model = CRAGSeg()
    model.load_state_dict(state.get("model", state))
    model.to(device)
    if device.type == "cuda":
        model = convolution_channels_last(model)
    loader, _ = make_loader(dataset(data_dir, config, split, seed), config, device, seed)
    destination = run_dir / split
    counts = DetailCounts()

    def observe(_name, prediction, truth):
        counts.update(prediction, truth)

    summary = validate(model, loader, config, device, destination, observe)
    summary.update(counts.summary())
    summary.update(seed=seed, split=split)
    if "best_epoch" in state:
        summary["best_epoch"] = int(state["best_epoch"]) + 1
    write_csv(destination / "summary.csv", [summary])
    print(f"{split} seed {seed}: mIoU={summary['miou']:.6f} ({summary['samples']} images)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "train", "val", "test", "all"),
                        help="all runs training, validation and testing in that order")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="directory containing FLAME2/ and lists/ (default: data)")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/default.json")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", help="one or more seeds (defaults to the config list)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="resume interrupted training")
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint for one seed with val or test")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.phase == "audit":
        result = audit_data(args.data_dir, args.data_dir / config["split_dir"], config["split_counts"])
        print(json.dumps(result, indent=2))
        return
    seeds = args.seeds or config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("provide distinct nonnegative seeds")
    if args.checkpoint and (args.phase not in ("val", "test") or len(seeds) != 1):
        parser.error("--checkpoint requires val or test and a single seed")
    if config["micro_batch"] < 1 or config["effective_batch"] < 1 or config["effective_batch"] % config["micro_batch"]:
        parser.error("effective_batch must be a positive multiple of micro_batch")
    kind = args.device
    if kind == "auto":
        kind = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
    if kind == "cuda" and config["amp"] and config["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("this GPU does not support bfloat16. Select float16 or disable amp in the config.")
    device = torch.device(kind)
    if args.phase in ("train", "all"):
        for seed in seeds:
            train_one(args.data_dir, args.output_dir, config, seed, device, args.resume)
    for split in ("val", "test"):
        if args.phase in (split, "all"):
            for seed in seeds:
                evaluate(args.data_dir, args.output_dir, config, seed, device, split, args.checkpoint)


if __name__ == "__main__":
    main()
