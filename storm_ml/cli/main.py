import click
from click_option_group import optgroup
from typing import Optional 

from .train import train

CONTEXT_SETTINGS = dict(max_content_width=120)

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass

@click.command()
def generate():
    pass

@click.command()
def predict():
    pass


main.add_command(train)
main.add_command(generate)
main.add_command(predict)