import json
import logging
from pathlib import Path

import click

from deicide.db import DbDriver
from deicide.deicide import deicide

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to SQLite DB from neodepends.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Path to output JSON file. Must not exist.",
)
@click.option("--filename", required=True, type=str, help="Filename in the database.")
@click.option(
    "--commit-hash",
    required=False,
    type=str,
    help="Commit hash (required if DB has multiple versions).",
)
def main(
    input: Path,
    output: Path,
    filename: str,
    commit_hash: str | None,
):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Open database
    db_driver = DbDriver(input)

    # Ensure commit_hash is set
    commit_hashes = db_driver.load_commit_hashes()
    if len(commit_hashes) == 0:
        logger.error("Database contains no commit hashes.")
        quit(-1)
    if commit_hash and commit_hash not in commit_hashes:
        logger.error(
            "Given commit hash ({commit_hash}) does not exist in the database."
        )
        quit(-1)
    if not commit_hash and len(commit_hashes) > 1:
        logger.error(
            f"Database contains multiple commit hashes ({','.join(commit_hashes)}). Please specify using --commit-hash."
        )
        quit(-1)
    if not commit_hash:
        commit_hash = commit_hashes[0]
    db_driver.set_commit_hash(commit_hash)

    # Ensure filename exists
    filenames = db_driver.load_filenames()
    if filename not in filenames:
        logger.error(f"Database does not contain filename ({filename}).")
    parent_id = filenames[filename]

    # Load children
    children = db_driver.load_children(parent_id)
    if len(children) == 1:
        # If there is only one top level item (e.g. a Java class), skip to its children.
        parent_id = children[0].id
        children = db_driver.load_children(parent_id)
    if len(children) == 0:
        logger.error("File contains no entities")
        quit(-1)

    # Load clients
    clients = db_driver.load_clients(parent_id)

    # Load deps
    internal_deps = db_driver.load_internal_deps(parent_id)
    client_deps = db_driver.load_client_deps(parent_id)

    # Run algorithm
    res = deicide(children, clients, internal_deps, client_deps)

    # Write output
    with open(output, "w") as f:
        json.dump(list(res), f)


if __name__ == "__main__":
    main()
