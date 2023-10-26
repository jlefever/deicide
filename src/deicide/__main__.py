import click


@click.group()
def cli():
    pass


@cli.command()
def split():
    click.echo("Split a file")


cli()