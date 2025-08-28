import json
import logging
# import itertools as it
from pathlib import Path

import click

from deicide.core import Entity
from deicide.db import DbDriver
from deicide.deicide import deicide
from deicide.dv8 import create_dv8_clustering, create_dv8_dependency
from deicide.semantic import KielaClarkSimilarity, LSISimilarity
from deicide.corpus import collect_corpus

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
    help="Path to output text file. Must not exist.",
)
@click.option("--filename", required=True, type=str, help="Filename in the database.")
@click.option(
    "--commit-hash",
    required=False,
    type=str,
    help="Commit hash (required if DB has multiple versions).",
)
@click.option(
    "--dv8-result",
    is_flag=True,
    default=False,
    help="Generate DV8 clustering output (.dv8-clustering.json) and DV8 dependency \
        output",
)
@click.option(
    "--semantic-model",
    type=click.Choice(["kiela_clark", "lsi"], case_sensitive=False),
    default="lsi",
    help="Semantic similarity model to use.",
)
def main(
    input: Path, output: Path, filename: str, commit_hash: str | None, dv8_result: bool, semantic_model: str
) -> None:
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
            f"Database contains multiple commit hashes ({','.join(commit_hashes)}). \
                Please specify using --commit-hash."
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

    # Load all contents and entities
    file_content = db_driver.load_file_content(parent_id)
    all_contents = db_driver.load_all_contents()
    all_entities = db_driver.load_all_entities()
    # Take 500 contents from all_contents that are not file_content and merge that with file_content
    file_content_key = next(iter(file_content.keys()))
    num_extra_contents = min(500, len(all_contents) - 1)
    additional_contents = [
        (content_key, content_value) for content_key, content_value in all_contents.items()
        if content_key != file_content_key
    ][:num_extra_contents]
    training_contents = {**file_content}
    # Create semantic similarity model
    if semantic_model == "kiela_clark":
        semantic = KielaClarkSimilarity()
    else:
        semantic = LSISimilarity()
        training_contents = {**file_content, **dict(additional_contents)}
    corpus: dict[str, str] = collect_corpus(all_entities, training_contents)
    semantic.fit(corpus)

    # Run algorithm
    clustering = deicide(children, clients, internal_deps + client_deps, semantic)

    # Create hex_id to entity mapping (file entities + clients)
    id_to_entity = {entity.id: entity for entity in children}

    # Add clients to the mapping with modified names
    for client in clients:
        modified_client = Entity(
            id=client.id,
            name=f"(Client) {client.name}",
            parent_id=client.parent_id,
            kind=client.kind,
            start_byte=client.start_byte,
            end_byte=client.end_byte,
            comment_start_byte=client.comment_start_byte,
            comment_end_byte=client.comment_end_byte,
            content_id=client.content_id,
        )
        id_to_entity[modified_client.id] = modified_client

    # Write output
    with open(output, "w") as f:
        for id, cluster in clustering:
            name = id_to_entity[id].name
            f.write(f"{name} : {cluster}\n")

    output_name = output.stem

    # Generate optional output based on flags
    if dv8_result:
        dv8_clustering = create_dv8_clustering(clustering, id_to_entity, output_name)
        dv8_output = output.with_suffix(".dv8-clustering.json")
        with open(dv8_output, "w") as f:
            json.dump(dv8_clustering.to_dict(), f, indent=2)
        dsm_dependencies = create_dv8_dependency(
            id_to_entity, internal_deps, client_deps, output_name
        )
        dsm_output = output.with_suffix(".dv8-dependency.json")
        with open(dsm_output, "w") as f:
            json.dump(dsm_dependencies, f, indent=2)


if __name__ == "__main__":
    main()
