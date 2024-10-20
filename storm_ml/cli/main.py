import click

from .generate import generate
from .predict import predict
from .train import train

CONTEXT_SETTINGS = dict(max_content_width=120)


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


main.add_command(train)
main.add_command(generate)
main.add_command(predict)
