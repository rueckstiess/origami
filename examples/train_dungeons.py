#!/usr/bin/env python
"""Train ORIGAMI on the dungeons synthetic dataset and demonstrate inference.

This script:
1. Trains an ORIGAMI model on the dungeons puzzle dataset
2. Demonstrates prediction (target field accuracy on validation set)
3. Demonstrates generation (sampling from the learned distribution)
4. Demonstrates embeddings (creating document vectors)

Usage:
    python examples/train_dungeons.py
"""

import argparse
import random
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.dungeons import generate_data
from origami.inference import OrigamiEmbedder, OrigamiGenerator, OrigamiPredictor
from origami.model import OrigamiConfig, OrigamiModel, TrainingConfig
from origami.tokenizer import JSONTokenizer
from origami.training import OrigamiTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ORIGAMI on dungeons dataset")
    parser.add_argument("--train-size", type=int, default=500, help="Training set size")
    parser.add_argument("--eval-size", type=int, default=100, help="Evaluation set size")
    parser.add_argument("--num-doors", type=int, nargs=2, default=[3,5], help="Door count range")
    parser.add_argument("--num-colors", type=int, default=3, help="Number of key colors")
    parser.add_argument("--num-treasures", type=int, default=5, help="Number of treasures")
    parser.add_argument("--with-monsters", action="store_true", help="Include monsters")
    parser.add_argument("--shuffle-rooms", action="store_true", help="Shuffle room order")

    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--upscale", type=int, default=10, help="Upscale factor for data augmentation")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable key-order shuffling during training")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("ORIGAMI Dungeons Training")
    print("=" * 60)

    # Generate data
    print(f"\nGenerating data...")
    print(f"  Train size: {args.train_size}")
    print(f"  Eval size: {args.eval_size}")
    print(f"  Doors: {args.num_doors[0]}-{args.num_doors[1]}")
    print(f"  Colors: {args.num_colors}")
    print(f"  Treasures: {args.num_treasures}")
    print(f"  With monsters: {args.with_monsters}")
    print(f"  Shuffle rooms: {args.shuffle_rooms}")

    train_data = generate_data(
        num_instances=args.train_size,
        num_doors_range=tuple(args.num_doors),
        num_colors=args.num_colors,
        num_treasures=args.num_treasures,
        with_monsters=args.with_monsters,
        shuffle_rooms=args.shuffle_rooms,
        seed=args.seed,
    )

    eval_data = generate_data(
        num_instances=args.eval_size,
        num_doors_range=tuple(args.num_doors),
        num_colors=args.num_colors,
        num_treasures=args.num_treasures,
        with_monsters=args.with_monsters,
        shuffle_rooms=args.shuffle_rooms,
        seed=args.seed + 1000,  # Different seed for eval
    )

    # Create tokenizer
    print("\nFitting tokenizer...")
    tokenizer = JSONTokenizer()
    tokenizer.fit(train_data + eval_data)
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
    shuffle = not args.no_shuffle
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

    # Train
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

    # =========================================================================
    # SANITY CHECK: Compare train vs validation accuracy
    # =========================================================================
    print("\n" + "=" * 60)
    print("SANITY CHECK: Train vs Validation Accuracy")
    print("=" * 60)

    predictor = OrigamiPredictor(model, tokenizer)

    # Sample from train set (same size as eval for fair comparison)
    train_sample_size = min(len(eval_data), len(train_data))
    train_sample = random.sample(train_data, train_sample_size)

    # Evaluate on train sample
    train_correct = 0
    for obj in train_sample:
        actual = obj["treasure"]
        predicted = predictor.predict(obj, target_key="treasure")
        if predicted == actual:
            train_correct += 1

    train_accuracy = train_correct / train_sample_size * 100

    # Evaluate on full eval set
    eval_correct = 0
    for obj in eval_data:
        actual = obj["treasure"]
        predicted = predictor.predict(obj, target_key="treasure")
        if predicted == actual:
            eval_correct += 1

    eval_accuracy = eval_correct / len(eval_data) * 100

    random_baseline = 100.0 / args.num_treasures

    print(f"\nRandom baseline: {random_baseline:.1f}% (1/{args.num_treasures} treasures)")
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

    # =========================================================================
    # INFERENCE DEMONSTRATIONS
    # =========================================================================

    # 1. Prediction: Detailed analysis
    print("\n" + "=" * 60)
    print("PREDICTION: Detailed Validation Analysis")
    print("=" * 60)

    total = len(eval_data)
    print(f"\nResults for {total} validation instances:")
    print(f"  Correct: {eval_correct}/{total}")
    print(f"  Accuracy: {eval_accuracy:.1f}%")

    # Show a few example predictions with probabilities
    print("\nExample predictions (top-3):")
    for i, obj in enumerate(eval_data[:3]):
        actual = obj["treasure"]
        predictions = predictor.predict(obj, target_key="treasure", top_k=3)
        print(f"\n  Instance {i + 1}:")
        print(f"    Clues: door={obj['door']}, key_color={obj['key_color']}")
        print(f"    Actual: {actual}")
        print("    Predictions:")
        for value, prob in predictions:
            marker = "✓" if value == actual else " "
            print(f"      {marker} {value}: {prob:.1%}")

    # 2. Generation: Sample from the learned distribution
    print("\n" + "=" * 60)
    print("GENERATION: Sampling from learned distribution")
    print("=" * 60)
    print("\nNote: Generation quality depends heavily on training. Lower")
    print("temperature (e.g., 0.3) produces more consistent outputs.")

    generator = OrigamiGenerator(model, tokenizer)

    # Use lower temperature for more consistent generation
    gen_temperature = 0.3
    print(f"\nGenerating 3 complete dungeon instances (temperature={gen_temperature})...")
    generated = generator.generate(
        num_samples=3, max_length=500, seed=args.seed, temperature=gen_temperature
    )

    for i, obj in enumerate(generated):
        print(f"\n  Generated instance {i + 1}:")
        if obj is None:
            print("    (generation failed - returned None)")
            continue

        if not obj:
            print("    (empty object {})")
            continue

        # Show all keys present
        keys = list(obj.keys())
        print(f"    Keys: {keys}")

        # Show structure summary if keys match expected
        if "door" in obj:
            print(f"    door: {obj['door']}")
        if "key_color" in obj:
            print(f"    key_color: {obj['key_color']}")
        if "corridor" in obj and isinstance(obj["corridor"], list):
            print(f"    corridor: {len(obj['corridor'])} rooms")
        if "treasure" in obj:
            print(f"    treasure: {obj['treasure']}")

    # Generate completions from a partial prefix
    print("\nGenerating completions from partial prefix...")
    prefix = {"door": 2, "key_color": "blue"}
    torch.manual_seed(args.seed + 100)  # Set seed before generation
    completions = generator.generate_from_prefix(
        prefix, num_samples=3, max_length=500, temperature=gen_temperature
    )

    print("  Prefix: door=2, key_color=blue")
    for i, obj in enumerate(completions):
        if obj is None:
            print(f"  Completion {i + 1}: (generation failed)")
            continue
        if not obj:
            print(f"  Completion {i + 1}: (empty object)")
            continue

        # Show all keys
        keys = list(obj.keys())
        corridor_len = len(obj.get("corridor", [])) if isinstance(obj.get("corridor"), list) else "N/A"
        treasure = obj.get("treasure", "N/A")
        print(f"  Completion {i + 1}: keys={keys}, corridor={corridor_len} rooms, treasure={treasure}")

    # 3. Embeddings: Create document vectors
    print("\n" + "=" * 60)
    print("EMBEDDINGS: Creating document vectors")
    print("=" * 60)

    embedder = OrigamiEmbedder(model, tokenizer, pooling="mean")

    print(f"\nEmbedding dimension: {embedder.embedding_dim}")

    # Embed a few documents
    sample_docs = eval_data[:5]
    embeddings = embedder.embed_batch(sample_docs)

    print(f"Embedded {len(sample_docs)} documents -> shape: {tuple(embeddings.shape)}")

    # Compute pairwise similarities
    print("\nPairwise cosine similarities:")
    similarities = torch.mm(embeddings, embeddings.t())

    # Header
    print("       ", end="")
    for i in range(len(sample_docs)):
        print(f"  Doc{i + 1}", end="")
    print()

    for i in range(len(sample_docs)):
        print(f"  Doc{i + 1}", end="")
        for j in range(len(sample_docs)):
            print(f"  {similarities[i, j]:.2f}", end="")
        print()

    # Show document details
    print("\nDocument details:")
    for i, doc in enumerate(sample_docs):
        print(f"  Doc{i + 1}: door={doc['door']}, key={doc['key_color']}, treasure={doc['treasure']}")

    # Target-specific embeddings (embedding at prediction point)
    print("\nTarget-specific embeddings (at 'treasure' key):")
    embedder_target = OrigamiEmbedder(model, tokenizer, pooling="target")
    target_embeddings = embedder_target.embed_batch(sample_docs, target_key="treasure")
    print(f"  Shape: {tuple(target_embeddings.shape)}")

    # Show that same treasure -> similar embeddings
    treasures = [doc["treasure"] for doc in sample_docs]
    print(f"  Treasures: {treasures}")

    target_sims = torch.mm(target_embeddings, target_embeddings.t())
    print("  Target embedding similarities:")
    for i in range(len(sample_docs)):
        for j in range(i + 1, len(sample_docs)):
            same_treasure = "same" if treasures[i] == treasures[j] else "diff"
            print(f"    Doc{i + 1}-Doc{j + 1}: {target_sims[i, j]:.3f} ({same_treasure} treasure)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
