from deicide.core import Entity

def extract_entity_content(file_content: bytes, entity: Entity) -> str:
    """
    Extracts the content of the given entity from the file content.
    """
    # take the subarray from entity.start to entity.end
    full_content = file_content[entity.start:entity.end]

    # remove the doc comment from full_content
    if entity.cmt_start is not None and entity.cmt_end is not None:
        content_without_doc_comment = full_content[:entity.cmt_start - entity.start] + full_content[entity.cmt_end - entity.start + 1:]
    else:
        content_without_doc_comment = full_content

    return content_without_doc_comment