"""Train subcommand for Origami CLI."""

from __future__ import annotations

import random
import sys
from typing import TYPE_CHECKING

import click

from origami.cli.data_loaders import DataFormat, detect_format, load_data

if TYPE_CHECKING:
    pass


def _parse_value(value: str):
    """Parse a string value to its appropriate type.

    Handles: None, True, False, int, float, string.
    """
    if value.lower() == "none":
        return None
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Keep as string
        return value


# Mapping from flat parameter names to nested config paths
_PARAM_SECTION_MAP = {
    # Model params
    "d_model": "model",
    "n_heads": "model",
    "n_layers": "model",
    "d_ff": "model",
    "dropout": "model",
    "max_depth": "model",
    "max_array_position": "model",
    "kvpe_pooling": "model",
    "backbone": "model",
    "num_mixture_components": "model",
    "max_seq_length": "model",
    # Training params
    "batch_size": "training",
    "learning_rate": "training",
    "num_epochs": "training",
    "warmup_steps": "training",
    "weight_decay": "training",
    "dataloader_num_workers": "training",
    "shuffle_keys": "training",
    "eval_strategy": "training",
    "eval_steps": "training",
    "eval_epochs": "training",
    "eval_sample_size": "training",
    "eval_on_train": "training",
    "target_key": "training",
    "constrain_grammar": "training",
    "constrain_schema": "training",
    # Data params
    "numeric_mode": "data",
    "cat_threshold": "data",
    "n_bins": "data",
    "bin_strategy": "data",
    "max_vocab_size": "data",
    "infer_schema": "data",
}


def parse_set_params(set_params: tuple[str, ...]) -> dict:
    """Parse --set KEY=VALUE parameters into nested config dictionaries.

    Supports both flat keys (auto-mapped to sections) and dot notation:
    - --set d_model=256         -> {"model": {"d_model": 256}}
    - --set model.d_model=256   -> {"model": {"d_model": 256}}
    - --set training.batch_size=64 -> {"training": {"batch_size": 64}}

    Args:
        set_params: Tuple of "KEY=VALUE" strings

    Returns:
        Nested dict with "model", "training", "data" keys
    """
    result: dict = {"model": {}, "training": {}, "data": {}}

    for param in set_params:
        if "=" not in param:
            raise click.BadParameter(f"Invalid --set format: '{param}'. Use KEY=VALUE.")

        key, value = param.split("=", 1)
        key = key.strip()
        value_parsed = _parse_value(value.strip())

        if "." in key:
            # Dot notation: section.field
            section, field = key.split(".", 1)
            if section not in result:
                raise click.BadParameter(
                    f"Unknown config section: '{section}'. Valid: model, training, data."
                )
            result[section][field] = value_parsed
        else:
            # Flat key - look up which section it belongs to
            section = _PARAM_SECTION_MAP.get(key)
            if section is None:
                raise click.BadParameter(
                    f"Unknown config parameter: '{key}'. Use --set section.field=value "
                    f"or see documentation for valid parameters."
                )
            result[section][key] = value_parsed

    return result


@click.command()
@click.option(
    "-d",
    "--data",
    required=True,
    help="Training data. Format auto-detected: .csv, .json, .jsonl, or mongodb:// URI.",
)
@click.option(
    "--db",
    default=None,
    help="MongoDB database name (required with mongodb:// URI).",
)
@click.option(
    "-c",
    "--collection",
    default=None,
    help="MongoDB collection name (required with mongodb:// URI).",
)
@click.option(
    "--skip",
    type=int,
    default=0,
    help="Skip N samples at the beginning of training data.",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    help="Limit training data to N samples (0 = unlimited).",
)
@click.option(
    "--project",
    default=None,
    help="MongoDB-style projection. Include: '{\"a\": 1}'. Exclude: '{\"x\": 0}'.",
)
@click.option(
    "--val",
    default=None,
    help="Validation data file. Format auto-detected from extension.",
)
@click.option(
    "--val-collection",
    default=None,
    help="Validation collection name (MongoDB mode). Uses same --db.",
)
@click.option(
    "--train-ratio",
    type=float,
    default=None,
    help="Split training data into train/val with this ratio (e.g., 0.8).",
)
@click.option(
    "-t",
    "--target-key",
    default=None,
    help="Target field to predict. Required for accuracy metrics during training.",
)
@click.option(
    "-e",
    "--epochs",
    type=int,
    default=10,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "-l",
    "--lr",
    type=float,
    default=1e-3,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "-D",
    "--d-model",
    type=int,
    default=128,
    show_default=True,
    help="Model hidden dimension.",
)
@click.option(
    "-H",
    "--n-heads",
    type=int,
    default=4,
    show_default=True,
    help="Number of attention heads.",
)
@click.option(
    "-L",
    "--n-layers",
    type=int,
    default=4,
    show_default=True,
    help="Number of transformer layers.",
)
@click.option(
    "-n",
    "--numeric-mode",
    type=click.Choice(["disabled", "discretize", "scale"]),
    default="disabled",
    show_default=True,
    help="Numeric field handling: disabled, discretize (binning), or scale (continuous).",
)
@click.option(
    "-o",
    "--output",
    required=True,
    help="Output path for trained model (e.g., model.pt).",
)
@click.option(
    "--eval-sample-size",
    type=int,
    default=None,
    help="Sample N examples for progress evaluations (faster). Default: full eval.",
)
@click.option(
    "--set",
    "set_params",
    multiple=True,
    help="Set any PipelineConfig parameter. Example: --set d_ff=1024",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose progress output.",
)
def train(
    data: str,
    db: str | None,
    collection: str | None,
    skip: int,
    limit: int,
    project: str | None,
    val: str | None,
    val_collection: str | None,
    train_ratio: float | None,
    target_key: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    numeric_mode: str,
    output: str,
    eval_sample_size: int | None,
    set_params: tuple[str, ...],
    seed: int,
    verbose: bool,
) -> None:
    """Train an Origami model on data.

    \b
    Examples:
      # Train on JSONL file
      origami train -d data.jsonl -t label -e 20 -o model.pt

      # Train with validation split
      origami train -d data.jsonl -t label --train-ratio 0.8 -o model.pt

      # Train with separate validation file
      origami train -d train.jsonl --val val.jsonl -t label -o model.pt

      # Train with custom architecture
      origami train -d data.jsonl -t label -D 256 -L 6 -H 8 -o model.pt

      # Train with continuous numeric handling
      origami train -d data.jsonl -t label -n scale -o model.pt

      # Train from MongoDB
      origami train -d mongodb://localhost:27017 --db mydb -c train -t label -o model.pt
    """
    import torch

    from origami import OrigamiPipeline
    from origami.config import DataConfig, ModelConfig, OrigamiConfig, TrainingConfig
    from origami.training import TableLogCallback, accuracy

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)

    # Parse escape hatch parameters (returns nested dict)
    extra_params = parse_set_params(set_params)

    # Load training data
    click.echo(f"Loading training data from {data}...")
    train_data = load_data(
        data, db=db, collection=collection, skip=skip, limit=limit, project=project
    )
    click.echo(f"  Loaded {len(train_data)} samples")

    # Load or split validation data
    eval_data = None
    if val:
        click.echo(f"Loading validation data from {val}...")
        eval_data = load_data(val, db=db, collection=None)
        click.echo(f"  Loaded {len(eval_data)} samples")
    elif val_collection:
        if detect_format(data) != DataFormat.MONGODB:
            raise click.BadParameter("--val-collection requires MongoDB data source")
        click.echo(f"Loading validation data from collection {val_collection}...")
        eval_data = load_data(data, db=db, collection=val_collection)
        click.echo(f"  Loaded {len(eval_data)} samples")
    elif train_ratio:
        # Split training data
        random.shuffle(train_data)
        split_idx = int(len(train_data) * train_ratio)
        eval_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        click.echo(f"  Split: {len(train_data)} train, {len(eval_data)} validation")

    # Build nested config from CLI args
    model_kwargs = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        **extra_params.get("model", {}),
    }

    training_kwargs = {
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": epochs,
        **extra_params.get("training", {}),
    }

    data_kwargs = {
        "numeric_mode": numeric_mode,
        **extra_params.get("data", {}),
    }

    # Enable continuous head when using scale mode
    actual_numeric_mode = data_kwargs.get("numeric_mode", "disabled")
    if actual_numeric_mode == "scale":
        model_kwargs.setdefault("use_continuous_head", True)

    # Add evaluation config if target_key is provided
    if target_key:
        training_kwargs["target_key"] = target_key
        training_kwargs.setdefault("eval_strategy", "epoch")  # Default, but respect --set
        training_kwargs.setdefault("eval_metrics", {"accuracy": accuracy})
        if eval_sample_size:
            training_kwargs["eval_sample_size"] = eval_sample_size

    try:
        config = OrigamiConfig(
            model=ModelConfig(**model_kwargs),
            training=TrainingConfig(**training_kwargs),
            data=DataConfig(**data_kwargs),
        )
    except (ValueError, TypeError) as e:
        raise click.BadParameter(f"Invalid configuration: {e}") from e

    if verbose:
        click.echo("\nConfiguration:")
        click.echo(config.to_yaml())

    # Create and train pipeline
    click.echo("\nTraining...")
    pipeline = OrigamiPipeline(config)

    try:
        # Training handles KeyboardInterrupt gracefully - model is always saved
        pipeline.fit(
            train_data,
            eval_data=eval_data,
            epochs=epochs,
            callbacks=[TableLogCallback()],
            verbose=verbose,
        )
    except Exception as e:
        click.echo(f"\nTraining failed: {e}", err=True)
        sys.exit(1)

    # Save model (works whether training completed or was interrupted)
    click.echo(f"\nSaving model to {output}...")
    pipeline.save(output)

    # Report final stats
    if verbose:
        params = pipeline._model.get_num_parameters()
        click.echo(f"Model parameters: {params:,}")

    click.echo("Done!")
