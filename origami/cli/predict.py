"""Predict subcommand for Origami CLI."""

from __future__ import annotations

import json
import sys
from typing import TextIO

import click

from origami.cli.data_loaders import load_data


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
    "-t",
    "--target-key",
    required=True,
    help="Target field to predict.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file. Default: stdout.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["values", "json", "jsonl"]),
    default="values",
    show_default=True,
    help="Output format: values (one per line), json (array), jsonl (with original data).",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for inference.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Display model configuration.",
)
def predict(
    model: str,
    data: str,
    db: str | None,
    collection: str | None,
    skip: int,
    limit: int,
    project: str | None,
    target_key: str,
    output: str | None,
    output_format: str,
    batch_size: int,
    verbose: bool,
) -> None:
    """Predict target values for input data.

    \b
    Examples:
      # Predict to stdout (one value per line)
      origami predict -m model.pt -d input.jsonl -t label

      # Predict to file as JSON array
      origami predict -m model.pt -d input.jsonl -t label -o predictions.json -f json

      # Predict with original data (JSONL format)
      origami predict -m model.pt -d input.jsonl -t label -f jsonl | head

    \b
    Output formats:
      values - One prediction value per line (default)
      json   - JSON array of all predictions
      jsonl  - JSONL with original data + predicted value
    """
    from origami import OrigamiPipeline

    # Load model
    click.echo(f"Loading model from {model}...", err=True)
    pipeline = OrigamiPipeline.load(model)

    if verbose:
        click.echo("\nConfiguration:", err=True)
        click.echo(pipeline.config.to_yaml(), err=True)

    # Load input data
    click.echo(f"Loading data from {data}...", err=True)
    input_data = load_data(
        data, db=db, collection=collection, skip=skip, limit=limit, project=project
    )
    click.echo(f"  Loaded {len(input_data)} samples", err=True)

    # Prepare inputs (set target to None)
    inputs = []
    for obj in input_data:
        obj_copy = obj.copy()
        obj_copy[target_key] = None
        inputs.append(obj_copy)

    # Run predictions
    click.echo("Predicting...", err=True)
    predictions = pipeline.predict_batch(inputs, target_key=target_key, batch_size=batch_size)

    # Determine output destination
    out_file: TextIO
    if output:
        out_file = open(output, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    try:
        _write_predictions(out_file, predictions, input_data, target_key, output_format)
    finally:
        if output:
            out_file.close()

    if output:
        click.echo(f"Wrote {len(predictions)} predictions to {output}", err=True)


def _write_predictions(
    out_file: TextIO,
    predictions: list,
    input_data: list[dict],
    target_key: str,
    output_format: str,
) -> None:
    """Write predictions in the specified format."""
    if output_format == "values":
        for pred in predictions:
            # Convert to string representation
            if isinstance(pred, str):
                out_file.write(f"{pred}\n")
            else:
                out_file.write(f"{json.dumps(pred)}\n")

    elif output_format == "json":
        json.dump(predictions, out_file, indent=2)
        out_file.write("\n")

    elif output_format == "jsonl":
        for obj, pred in zip(input_data, predictions, strict=True):
            # Add prediction to original object
            output_obj = obj.copy()
            output_obj[target_key] = pred
            out_file.write(json.dumps(output_obj) + "\n")
