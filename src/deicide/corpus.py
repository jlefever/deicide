import json
from deicide.core import Entity
from pathlib import Path

USE_DOC_COMMENTS = True

# Load stop_words.json and flatten all words into a single list
with open(Path(__file__).parent / "stop_words.json", encoding="utf-8") as f:
    stop_words_dict: dict[str, list[str]] = json.load(f)
stop_words: set[str] = {word for words in stop_words_dict.values() for word in words}


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

def filter_code_members(entities: list[Entity]) -> list[Entity]:
    """Return only entities of kind 'Method', 'Constructor', or 'Field'."""
    return [e for e in entities if e.kind in {"Method", "Constructor", "Field"}]


def tokenize_document(content: str) -> list[str]:
    # Split the content into tokens based on whitespace and punctuation
    tokens: list[str] = []
    current_token: list[str] = []
    for char in content:
        if char.isspace() or char in {".", ",", ";", "(", ")", "[", "]", "{", "}"}:
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
        else:
            current_token.append(char)
    if current_token:
        tokens.append("".join(current_token))
    return tokens

def remove_stop_words(words: list[str]) -> list[str]:
    return [w for w in words if w not in stop_words]

