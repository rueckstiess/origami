"""Generate subcommand for Origami CLI."""

from __future__ import annotations

import json
import sys
from typing import TextIO

import click


@click.command()
@click.option(
    "-m",
    "--model",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained pipeline (.pt file).",
)
@click.option(
    "-n",
    "--count",
    type=int,
    default=10,
    show_default=True,
    help="Number of samples to generate.",
)
@click.option(
    "--temp",
    "temperature",
    type=float,
    default=1.0,
    show_default=True,
    help="Sampling temperature. Higher = more random.",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Top-k sampling (keep top k tokens).",
)
@click.option(
    "--top-p",
    type=float,
    default=None,
    help="Nucleus sampling (keep tokens with cumulative prob >= p).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file. Default: stdout (JSONL format).",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for generation.",
)
@click.option(
    "--max-length",
    type=int,
    default=512,
    show_default=True,
    help="Maximum sequence length for generation.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Display model configuration.",
)
def generate(
    model: str,
    count: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    output: str | None,
    batch_size: int,
    max_length: int,
    verbose: bool,
) -> None:
    """Generate synthetic data from a trained model.

    \b
    Examples:
      # Generate 10 samples to stdout
      origami generate -m model.pt

      # Generate 100 samples with temperature
      origami generate -m model.pt -n 100 --temp 0.8

      # Generate with top-k sampling
      origami generate -m model.pt -n 50 --top-k 50

      # Generate to file
      origami generate -m model.pt -n 1000 -o samples.jsonl
    """
    from origami import OrigamiPipeline

    # Load model
    click.echo(f"Loading model from {model}...", err=True)
    pipeline = OrigamiPipeline.load(model)

    if verbose:
        click.echo("\nConfiguration:", err=True)
        click.echo(pipeline.config.to_yaml(), err=True)

    # Generate samples
    click.echo(f"Generating {count} samples...", err=True)
    samples = pipeline.generate(
        num_samples=count,
        batch_size=batch_size,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    # Determine output destination
    out_file: TextIO
    if output:
        out_file = open(output, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    try:
        for sample in samples:
            out_file.write(json.dumps(sample) + "\n")
    finally:
        if output:
            out_file.close()

    if output:
        click.echo(f"Wrote {len(samples)} samples to {output}", err=True)
