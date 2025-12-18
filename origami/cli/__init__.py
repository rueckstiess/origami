"""Origami CLI - Command-line interface for training and using Origami models.

Usage:
    origami train -d data.jsonl -t label -e 20 -o model.pt
    origami evaluate -m model.pt -d test.jsonl -t label
    origami predict -m model.pt -d input.jsonl -t label
    origami generate -m model.pt -n 100
    origami embed -m model.pt -d data.jsonl -o embeddings.npy
"""

from __future__ import annotations

import click

from origami.cli.embed import embed
from origami.cli.evaluate import evaluate
from origami.cli.generate import generate
from origami.cli.predict import predict
from origami.cli.train import train


@click.group()
@click.version_option(package_name="origami")
def main() -> None:
    """Origami - JSON object modeling with transformers.

    Train models to predict, generate, and embed JSON objects.

    \b
    Examples:
      origami train -d data.jsonl -t label -e 20 -o model.pt
      origami predict -m model.pt -d input.jsonl -t label
      origami generate -m model.pt -n 100
    """
    pass


# Register subcommands
main.add_command(train)
main.add_command(evaluate)
main.add_command(predict)
main.add_command(generate)
main.add_command(embed)


if __name__ == "__main__":
    main()
