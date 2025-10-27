import click

from origami.cli.embed import embed
from origami.cli.predict import predict
from origami.cli.train import train

CONTEXT_SETTINGS = dict(max_content_width=120)


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


main.add_command(train)
main.add_command(predict)
main.add_command(embed)
