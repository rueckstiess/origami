#!/usr/bin/env python
"""Train ORIGAMI on a JSONL dataset.

This script:
1. Loads data from JSONL file
2. Optionally discretizes high-cardinality numeric fields
3. Trains an ORIGAMI model (or loads existing checkpoint)
4. Evaluates prediction accuracy on train/eval splits

Usage:
    # Train from scratch
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target

    # Load checkpoint and continue training
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target --checkpoint checkpoints/best.pt --epochs 10

    # Load checkpoint and just evaluate (no training)
    python examples/train_jsonl.py --data datasets/car.jsonl --target-key target --checkpoint checkpoints/best.pt --epochs 0
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from origami.inference import OrigamiGenerator, OrigamiPredictor
from origami.model import OrigamiConfig, OrigamiModel, TrainingConfig
from origami.preprocessing import NumericDiscretizer
from origami.tokenizer import JSONTokenizer
from origami.training import OrigamiTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ORIGAMI on JSONL dataset")
    parser.add_argument("--data", type=str, default="datasets/car.jsonl", help="Path to JSONL data")
    parser.add_argument("--target-key", type=str, default="target", help="Target field to predict")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/eval split ratio")

    # Numeric discretization
    parser.add_argument("--discretize", action="store_true", help="Enable numeric discretization")
    parser.add_argument(
        "--cat-threshold",
        type=int,
        default=100,
        help="Max unique values for categorical (discretize fields with more)",
    )
    parser.add_argument("--n-bins", type=int, default=20, help="Number of bins for discretization")
    parser.add_argument(
        "--bin-strategy",
        type=str,
        default="quantile",
        choices=["quantile", "uniform", "kmeans"],
        help="Binning strategy",
    )

    # Model architecture
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Training
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--upscale", type=int, default=4, help="Upscale factor for data augmentation"
    )
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory for saving"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load model from checkpoint (for continuing training or evaluation)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Enable key-order shuffling during training"
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
    print("ORIGAMI JSONL Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.data}...")
    all_data = load_data(args.data)
    print(f"  Total samples: {len(all_data)}")

    # Check target distribution
    from collections import Counter

    targets = [obj[args.target_key] for obj in all_data]
    target_dist = Counter(targets)
    print(f"  Target distribution: {dict(target_dist)}")
    num_classes = len(target_dist)

    # Split data (before discretization to avoid data leakage)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * args.train_ratio)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    print(f"  Train size: {len(train_data)}")
    print(f"  Eval size: {len(eval_data)}")

    # Optional: Discretize high-cardinality numeric fields
    discretizer = None
    if args.discretize:
        print("\nDiscretizing numeric fields...")
        print(f"  cat_threshold: {args.cat_threshold}")
        print(f"  n_bins: {args.n_bins}")
        print(f"  strategy: {args.bin_strategy}")

        discretizer = NumericDiscretizer(
            cat_threshold=args.cat_threshold,
            n_bins=args.n_bins,
            strategy=args.bin_strategy,
        )
        # Fit on train data only, then transform both
        train_data = discretizer.fit_transform(train_data)
        eval_data = discretizer.transform(eval_data)

        print(f"  Discretized fields: {sorted(discretizer.discretized_fields)}")
        print(f"  Pass-through fields: {sorted(discretizer.passthrough_fields)}")

    # Load or create model
    if args.checkpoint:
        # Load model and tokenizer from checkpoint
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        model, tokenizer = OrigamiModel.load(args.checkpoint)
        if tokenizer is None:
            raise ValueError("Checkpoint does not contain tokenizer state. Cannot continue.")
        print("  Loaded model config:")
        print(f"    vocab_size: {model.config.vocab_size}")
        print(f"    d_model: {model.config.d_model}")
        print(f"    n_heads: {model.config.n_heads}")
        print(f"    n_layers: {model.config.n_layers}")
        print(f"    Parameters: {model.get_num_parameters():,}")
    else:
        # Create tokenizer
        print("\nFitting tokenizer...")
        tokenizer = JSONTokenizer()
        tokenizer.fit(all_data)
        print(f"  Vocabulary size: {tokenizer.vocab.size}")

        # Create model
        print("\nCreating model...")
        config = OrigamiConfig(
            vocab_size=tokenizer.vocab.size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_depth=tokenizer.max_depth,
            max_array_position=tokenizer.max_array_index,
            use_grammar_constraints=True,
        )
        model = OrigamiModel(config, vocab=tokenizer.vocab)

        num_params = model.get_num_parameters()
        print(f"  d_model: {args.d_model}")
        print(f"  n_heads: {args.n_heads}")
        print(f"  n_layers: {args.n_layers}")
        print(f"  Parameters: {num_params:,}")

    # Create training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        upscale_factor=args.upscale,
        warmup_steps=args.warmup_steps,
        save_every_n_epochs=10,
    )

    # Create trainer
    shuffle = args.shuffle
    print("\nInitializing trainer...")
    print(f"  Key-order shuffling: {'enabled' if shuffle else 'disabled'}")
    trainer = OrigamiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        config=train_config,
        checkpoint_dir=args.checkpoint_dir,
        shuffle=shuffle,
    )

    # Train (skip if epochs is 0)
    if args.epochs > 0:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")

        state = trainer.train()

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"  Final epoch: {state.epoch + 1}")
        print(f"  Total steps: {state.global_step}")
        print(f"  Best eval loss: {state.best_eval_loss:.4f}")
        print("=" * 60)
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

    predictor = OrigamiPredictor(model, tokenizer)

    # Sample from train set (same size as eval for fair comparison)
    train_sample_size = min(len(eval_data), len(train_data))
    train_sample = random.sample(train_data, train_sample_size)

    # Evaluate on train sample
    train_correct = 0
    for obj in train_sample:
        actual = obj[args.target_key]
        predicted = predictor.predict(obj, target_key=args.target_key)
        if predicted == actual:
            train_correct += 1

    train_accuracy = train_correct / train_sample_size * 100

    # Evaluate on full eval set
    eval_correct = 0
    for obj in eval_data:
        actual = obj[args.target_key]
        predicted = predictor.predict(obj, target_key=args.target_key)
        if predicted == actual:
            eval_correct += 1

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
        predictions = predictor.predict(obj, target_key=args.target_key, top_k=4)
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

    generator = OrigamiGenerator(model, tokenizer)
    samples = generator.generate(num_samples=5, max_length=256, seed=args.seed)

    for i, sample in enumerate(samples):
        print(f"\nSample {i + 1}:")
        print(f"  {json.dumps(sample)}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
