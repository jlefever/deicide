from deicide.core import Entity

USE_DOC_COMMENTS = True


def extract_entity_content(file_content: bytes, entity: Entity) -> str:
    """
    Extracts the content of the given entity from the file content.
    """
    if USE_DOC_COMMENTS:
        start_byte = min(entity.start, entity.cmt_start)
        end_byte = max(entity.end, entity.cmt_end)
    else:
        start_byte = entity.start
        end_byte = entity.end
    return file_content[start_byte:end_byte].decode()
