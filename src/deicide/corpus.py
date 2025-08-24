from deicide.core import Entity

USE_DOC_COMMENTS = True


def extract_entity_content(file_content: bytes, entity: Entity) -> str:
    """
    Extracts the content of the given entity from the file content.
    """
    if USE_DOC_COMMENTS and (
        entity.comment_start_byte is not None and entity.comment_end_byte is not None
    ):
        start_byte = min(entity.start_byte, entity.comment_start_byte)
        end_byte = max(entity.end_byte, entity.comment_end_byte)
    else:
        start_byte = entity.start_byte
        end_byte = entity.end_byte
    return file_content[start_byte:end_byte].decode()
