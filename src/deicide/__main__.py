import sqlite3

import click
import pandas as pd
from tabulate import tabulate

import deicide.validation2 as vd
import deicide.dendrogram as dg
from deicide import db, loading
from deicide.deicide import cluster_dataset


@click.group()
def cli():
    pass


def parse_version(db_path: str, version: str) -> int:
    res = loading.try_identify_version(db_path, version)
    if res is None:
        print("Invalid version")
        quit(1)
    return res


@cli.command()
@click.option(
    "db_path",
    "--db",
    required=True,
    help="Project db file created by cochange-tool",
    type=click.Path(),
)
@click.option(
    "version",
    "--version",
    required=True,
    help="Commit as a SHA-1 hash or reference name",
)
@click.option(
    "target",
    "--target",
    required=True,
    help="Filename of file to split (must be inside the db)",
)
@click.option(
    "output",
    "--output",
    default="output.xlsx",
    help="Filename of output Excel spreadsheet"
)
def split(db_path: str, version: str, target: str, output: str):
    commit_id = parse_version(db_path, version)
    print("Loading file...")
    ds = loading.load_dataset(db_path, target, commit_id)
    print("Clustering...")
    entities_df = cluster_dataset(ds)
    my_clustering = vd.to_my_clustering(entities_df).normalize()
    dg.dump_indicators(entities_df, output, my_clx=my_clustering)


@cli.command()
@click.option(
    "db_path",
    "--db",
    required=True,
    help="Project db file created by cochange-tool",
    type=click.Path(),
)
@click.option(
    "version",
    "--version",
    required=True,
    help="Commit as a SHA-1 hash or reference name",
)
@click.option(
    "n",
    "-n",
    default=100,
    help="Max number of results to return",
    type=click.INT,
)
def list_candidates(db_path: str, version: str, n: int):
    commit_id = parse_version(db_path, version)
    with sqlite3.connect(db_path) as con:
        df = db.fetch_candidate_files2(con, commit_id, n)
    print(tabulate(df, headers="keys", tablefmt="psql"))


@cli.command()
@click.option(
    "db_path",
    "--db",
    required=True,
    help="Project db file created by cochange-tool",
    type=click.Path(),
)
def list_versions(db_path: str):
    with sqlite3.connect(db_path) as con:
        df = db.fetch_versions(con)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    print(tabulate(df, headers="keys", tablefmt="psql"))

cli()
