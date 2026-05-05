"""Evaluate subcommand for Origami CLI."""

from __future__ import annotations

import click

from origami.cli.data_loaders import load_data
from origami.training.metrics import list_metrics


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
    help="Test data. Format auto-detected: .csv, .json, .jsonl, or mongodb:// URI.",
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
    "-t",
    "--target-key",
    required=True,
    help="Target field to evaluate predictions on.",
)
@click.option(
    "--metrics",
    multiple=True,
    type=click.Choice(list_metrics()),
    default=["accuracy"],
    show_default=True,
    help="Metrics to compute. Can be specified multiple times.",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Evaluate on random sample of N examples.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Display model configuration.",
)
def evaluate(
    model: str,
    data: str,
    db: str | None,
    collection: str | None,
    skip: int,
    limit: int,
    project: str | None,
    target_key: str,
    metrics: tuple[str, ...],
    sample_size: int | None,
    verbose: bool,
) -> None:
    """Evaluate a trained Origami model.

    \b
    Examples:
      # Basic evaluation with accuracy
      origami evaluate -m model.pt -d test.jsonl -t label

      # Multiple metrics
      origami evaluate -m model.pt -d test.jsonl -t label --metrics accuracy --metrics array_f1

      # Evaluate on sample
      origami evaluate -m model.pt -d test.jsonl -t label --sample-size 500
    """
    from origami import OrigamiPipeline

    # Load model
    click.echo(f"Loading model from {model}...")
    pipeline = OrigamiPipeline.load(model)

    if verbose:
        click.echo("\nConfiguration:")
        click.echo(pipeline.config.to_yaml())

    # Load test data
    click.echo(f"Loading test data from {data}...")
    test_data = load_data(
        data, db=db, collection=collection, skip=skip, limit=limit, project=project
    )
    click.echo(f"  Loaded {len(test_data)} samples")

    # Build metrics dict (use metric name as both prefix and value)
    metrics_dict = {name: name for name in metrics}

    # Run evaluation
    click.echo("\nEvaluating...")
    results = pipeline.evaluate(
        test_data,
        target_key=target_key,
        metrics=metrics_dict,
        sample_size=sample_size,
    )

    # Print results
    click.echo("\nResults:")
    click.echo("-" * 30)

    # Print loss first
    if "loss" in results:
        click.echo(f"  loss: {results['loss']:.4f}")

    # Print other metrics
    for name, value in results.items():
        if name != "loss":
            if isinstance(value, float):
                click.echo(f"  {name}: {value:.4f}")
            else:
                click.echo(f"  {name}: {value}")

    click.echo("-" * 30)
