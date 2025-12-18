#!/usr/bin/env python
"""Train ORIGAMI on a JSONL dataset using the Pipeline API.

This script:
1. Loads data from JSONL file
2. Configures preprocessing (optional discretization or scaling for numerics)
3. Trains an ORIGAMI model using OrigamiPipeline
4. Saves the trained pipeline
5. Evaluates prediction accuracy on train/eval splits
6. Generates sample outputs

Usage:
    # Train from scratch with default settings
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target

    # Train with numeric discretization
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target --discretize

    # Train with continuous numeric scaling (uses MoG head)
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target --scale

    # Load checkpoint and continue training
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target --checkpoint model.pt --epochs 10

    # Load checkpoint and just evaluate (no training)
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target --checkpoint model.pt --epochs 0
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from origami import OrigamiPipeline, PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train ORIGAMI on JSONL dataset")
    parser.add_argument("--data", type=str, default="datasets/car.jsonl", help="Path to JSONL data")
    parser.add_argument("--target-key", type=str, default="target", help="Target field to predict")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/eval split ratio")

    # Numeric preprocessing (mutually exclusive)
    numeric_group = parser.add_mutually_exclusive_group()
    numeric_group.add_argument(
        "--discretize", action="store_true", help="Discretize high-cardinality numeric fields into bins"
    )
    numeric_group.add_argument(
        "--scale",
        action="store_true",
        help="Scale high-cardinality numeric fields (uses continuous MoG head)",
    )
    parser.add_argument(
        "--cat-threshold",
        type=int,
        default=100,
        help="Max unique values for categorical (preprocess fields with more)",
    )
    parser.add_argument("--n-bins", type=int, default=20, help="Number of bins for discretization")
    parser.add_argument(
        "--bin-strategy",
        type=str,
        default="quantile",
        choices=["quantile", "uniform", "kmeans"],
        help="Binning strategy for discretization",
    )

    # Model architecture
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--kvpe-pooling",
        type=str,
        default="sum",
        choices=["sum", "weighted", "rotary", "gru", "transformer"],
        help="KVPE pooling strategy",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--upscale", type=int, default=4, help="Upscale factor for data augmentation"
    )
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--shuffle", action="store_true", help="Enable key-order shuffling during training"
    )

    # Checkpointing
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path", type=str, default="model.pt", help="Path to save trained pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load pipeline from checkpoint (for continuing training or evaluation)",
    )

    # Logging
    parser.add_argument(
        "--print-every", type=int, default=100, help="Print log line every N batches"
    )
    parser.add_argument(
        "--eval-every", type=int, default=1000, help="Compute metrics every N batches"
    )

    return parser.parse_args()


def load_data(path: str) -> list[dict]:
    """Load JSONL data file."""
    data = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            # Remove MongoDB _id field if present
            obj.pop("_id", None)
            data.append(obj)
    return data


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("ORIGAMI Pipeline Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.data}...")
    all_data = load_data(args.data)
    print(f"  Total samples: {len(all_data)}")

    # Check target distribution
    targets = [obj[args.target_key] for obj in all_data]
    target_dist = Counter(targets)
    print(f"  Target distribution: {dict(target_dist)}")
    num_classes = len(target_dist)

    # Split data
    random.shuffle(all_data)
    split_idx = int(len(all_data) * args.train_ratio)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    print(f"  Train size: {len(train_data)}")
    print(f"  Eval size: {len(eval_data)}")

    # Determine numeric mode
    if args.discretize:
        numeric_mode = "discretize"
    elif args.scale:
        numeric_mode = "scale"
    else:
        numeric_mode = "none"

    # Load or create pipeline
    if args.checkpoint:
        print(f"\nLoading pipeline from checkpoint: {args.checkpoint}")
        pipeline = OrigamiPipeline.load(args.checkpoint)
        print(f"  Loaded configuration:")
        print(f"    d_model: {pipeline.config.d_model}")
        print(f"    n_layers: {pipeline.config.n_layers}")
        print(f"    numeric_mode: {pipeline.config.numeric_mode}")
        print(f"    Model parameters: {pipeline._model.get_num_parameters():,}")
    else:
        # Create pipeline config
        print("\nCreating pipeline...")
        config = PipelineConfig(
            # Model architecture
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            kvpe_pooling=args.kvpe_pooling,
            # Numeric preprocessing
            numeric_mode=numeric_mode,
            cat_threshold=args.cat_threshold,
            n_bins=args.n_bins,
            bin_strategy=args.bin_strategy,
            # Training
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            upscale_factor=args.upscale,
            shuffle_keys=args.shuffle,
        )

        print(f"  Configuration:")
        print(f"    d_model: {config.d_model}")
        print(f"    n_heads: {config.n_heads}")
        print(f"    n_layers: {config.n_layers}")
        print(f"    d_ff: {config.d_ff}")
        print(f"    kvpe_pooling: {config.kvpe_pooling}")
        print(f"    numeric_mode: {config.numeric_mode}")
        if numeric_mode == "discretize":
            print(f"    n_bins: {config.n_bins}")
            print(f"    bin_strategy: {config.bin_strategy}")
        print(f"    batch_size: {config.batch_size}")
        print(f"    upscale_factor: {config.upscale_factor}")
        print(f"    shuffle_keys: {config.shuffle_keys}")

        pipeline = OrigamiPipeline(config)

    # Train (skip if epochs is 0)
    if args.epochs > 0:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")

        # Train with verbose progress (ProgressCallback added automatically)
        # Evaluation metrics are configured via TrainingConfig (eval_metrics, eval_strategy)
        pipeline.fit(train_data, eval_data=eval_data, epochs=args.epochs, verbose=True)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"  Model parameters: {pipeline._model.get_num_parameters():,}")
        print("=" * 60)

        # Save the trained pipeline
        print(f"\nSaving pipeline to {args.save_path}...")
        pipeline.save(args.save_path)
        print(f"  Saved ({Path(args.save_path).stat().st_size / 1024:.1f} KB)")
    else:
        print("\n" + "=" * 60)
        print("Skipping training (--epochs 0)")
        print("=" * 60)

    # =========================================================================
    # Evaluation: Compare train vs validation accuracy
    # =========================================================================
    print("\n" + "=" * 60)
    print("Evaluation: Train vs Validation Accuracy")
    print("=" * 60)

    # Sample from train set (same size as eval for fair comparison)
    train_sample_size = min(len(eval_data), len(train_data))
    train_sample = random.sample(train_data, train_sample_size)

    # Prepare objects with target set to None for prediction
    def prepare_for_prediction(obj, target_key):
        obj_copy = obj.copy()
        obj_copy[target_key] = None
        return obj_copy

    # Evaluate on train sample
    train_inputs = [prepare_for_prediction(obj, args.target_key) for obj in train_sample]
    train_predictions = pipeline.predict_batch(train_inputs, target_key=args.target_key)

    train_correct = sum(
        1
        for pred, obj in zip(train_predictions, train_sample, strict=True)
        if pred == obj[args.target_key]
    )
    train_accuracy = train_correct / train_sample_size * 100

    # Evaluate on full eval set
    eval_inputs = [prepare_for_prediction(obj, args.target_key) for obj in eval_data]
    eval_predictions = pipeline.predict_batch(eval_inputs, target_key=args.target_key)

    eval_correct = sum(
        1
        for pred, obj in zip(eval_predictions, eval_data, strict=True)
        if pred == obj[args.target_key]
    )
    eval_accuracy = eval_correct / len(eval_data) * 100

    random_baseline = 100.0 / num_classes

    print(f"\nRandom baseline: {random_baseline:.1f}% (1/{num_classes} classes)")
    print(f"Train accuracy: {train_correct}/{train_sample_size} = {train_accuracy:.1f}%")
    print(f"Eval accuracy:  {eval_correct}/{len(eval_data)} = {eval_accuracy:.1f}%")

    if train_accuracy > random_baseline * 1.5:
        if eval_accuracy > random_baseline * 1.5:
            print("\n[OK] Model is learning and generalizing!")
        else:
            print("\n[WARN] Model may be overfitting (good on train, poor on eval)")
    else:
        print("\n[FAIL] Model is NOT learning (train accuracy near random)")
        print("       This suggests a fundamental issue with training or the model.")

    # Show example predictions
    print("\n" + "=" * 60)
    print("Example Predictions")
    print("=" * 60)

    for i, obj in enumerate(eval_data[:5]):
        actual = obj[args.target_key]
        pred_input = prepare_for_prediction(obj, args.target_key)
        predictions = pipeline.predict_proba(pred_input, target_key=args.target_key, top_k=4)
        print(f"\nExample {i + 1}:")
        print(f"  Actual: {actual}")
        print("  Top predictions:")
        for value, prob in predictions:
            marker = "*" if value == actual else " "
            print(f"    {marker} {value}: {prob:.1%}")

    # =========================================================================
    # Generation: Sample from learned distribution
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generated Samples from Learned Distribution")
    print("=" * 60)

    samples = pipeline.generate(num_samples=5, max_length=256, seed=args.seed)

    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}:")
        print(f"  {json.dumps(sample)}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
