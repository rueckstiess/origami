"""Embed subcommand for Origami CLI."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from origami.cli.data_loaders import load_data


def detect_output_format(path: str) -> str:
    """Detect embedding output format from file extension.

    Args:
        path: Output file path

    Returns:
        Format string: "npy", "csv", or "pt"
    """
    path_lower = path.lower()
    if path_lower.endswith(".npy"):
        return "npy"
    elif path_lower.endswith(".csv"):
        return "csv"
    elif path_lower.endswith(".pt") or path_lower.endswith(".pth"):
        return "pt"
    else:
        raise click.BadParameter(
            f"Cannot detect format from '{path}'. Use .npy, .csv, or .pt extension."
        )


@click.command()
@click.option(
    "-m",
    "--model",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained pipeline (.pt file).",
)
@click.option(
    "-d",
    "--data",
    required=True,
    help="Input data. Format auto-detected: .csv, .json, .jsonl, or mongodb:// URI.",
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
    help="Skip N samples at the beginning.",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    help="Limit to N samples (0 = unlimited).",
)
@click.option(
    "--project",
    default=None,
    help="MongoDB-style projection. Include: '{\"a\": 1}'. Exclude: '{\"x\": 0}'.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(),
    help="Output file. Format from extension: .npy (numpy), .csv, or .pt (torch).",
)
@click.option(
    "-p",
    "--pooling",
    type=click.Choice(["mean", "max", "last", "target"]),
    default="mean",
    show_default=True,
    help="Pooling strategy for embeddings.",
)
@click.option(
    "-t",
    "--target-key",
    default=None,
    help="Target key (required for pooling=target).",
)
@click.option(
    "--no-normalize",
    is_flag=True,
    help="Disable L2 normalization of embeddings.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Display model configuration.",
)
def embed(
    model: str,
    data: str,
    db: str | None,
    collection: str | None,
    skip: int,
    limit: int,
    project: str | None,
    output: str,
    pooling: str,
    target_key: str | None,
    no_normalize: bool,
    verbose: bool,
) -> None:
    """Create embeddings for input data.

    \b
    Examples:
      # Create embeddings as numpy array
      origami embed -m model.pt -d data.jsonl -o embeddings.npy

      # Create embeddings as CSV
      origami embed -m model.pt -d data.jsonl -o embeddings.csv

      # Create embeddings as torch tensor
      origami embed -m model.pt -d data.jsonl -o embeddings.pt

      # Use target-specific pooling
      origami embed -m model.pt -d data.jsonl -o emb.npy -p target -t label

      # Disable normalization
      origami embed -m model.pt -d data.jsonl -o emb.npy --no-normalize
    """

    from origami import OrigamiPipeline

    # Validate pooling options
    if pooling == "target" and not target_key:
        raise click.BadParameter("-t/--target-key is required when pooling=target")

    # Detect output format
    output_format = detect_output_format(output)

    # Load model
    click.echo(f"Loading model from {model}...")
    pipeline = OrigamiPipeline.load(model)

    if verbose:
        click.echo("\nConfiguration:")
        click.echo(pipeline.config.to_yaml())

    # Load input data
    click.echo(f"Loading data from {data}...")
    input_data = load_data(
        data, db=db, collection=collection, skip=skip, limit=limit, project=project
    )
    click.echo(f"  Loaded {len(input_data)} samples")

    # Create embeddings
    click.echo("Creating embeddings...")
    normalize = not no_normalize
    embeddings = pipeline.embed_batch(
        input_data,
        pooling=pooling,
        target_key=target_key,
        normalize=normalize,
    )

    click.echo(f"  Shape: {embeddings.shape}")

    # Save embeddings
    click.echo(f"Saving to {output}...")
    _save_embeddings(embeddings, output, output_format)
    click.echo(f"  Saved ({Path(output).stat().st_size / 1024:.1f} KB)")


def _save_embeddings(embeddings: np.ndarray, path: str, fmt: str) -> None:
    """Save embeddings in the specified format."""
    if fmt == "npy":
        np.save(path, embeddings)

    elif fmt == "csv":
        np.savetxt(path, embeddings, delimiter=",", fmt="%.6f")

    elif fmt == "pt":
        import torch

        tensor = torch.from_numpy(embeddings)
        torch.save(tensor, path)
