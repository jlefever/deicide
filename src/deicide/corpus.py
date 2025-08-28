from itertools import chain, pairwise
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


def _filter_code_members(entities: list[Entity]) -> list[Entity]:
    """Return only entities of kind 'Method', 'Constructor', or 'Field'."""
    return [e for e in entities if e.kind in {"Method", "Constructor", "Field"}]

def _split_space_and_punctuation(content: str) -> list[str]:
    """
    Split the content into list of words, not including space and punctuations
    """
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


def _remove_stop_words(words: list[str]) -> list[str]:
    return [w for w in words if w not in stop_words]


def _join_singles(terms: list[str]) -> list[str]:
    ret: list[str] = []
    joined_term: list[str] = []
    for t in terms:
        if len(t) == 1:
            joined_term.append(t[0])
        elif len(t) > 1:
            if len(joined_term) > 0:
                ret.append("".join(joined_term))
                joined_term = []
            ret.append(t)
    if len(joined_term) > 0:
        ret.append("".join(joined_term))
    return ret


def _split_camel(name: str) -> list[str]:
    if name.isupper():
        return [name.lower()]
    indices = [i for i, x in enumerate(name) if x.isupper() or x.isnumeric()]
    indices = [0] + indices + [len(name)]
    return _join_singles([name[a:b].lower() for a, b in pairwise(indices)])


def _split_identifier(name: str) -> list[str]:
    by_spaces = name.split(" ")
    by_underscores = chain(*(z.split("_") for z in by_spaces))
    return list(chain(*(_split_camel(z) for z in by_underscores)))


def _tokenize(doc: str) -> list[str]:
    words: list[str] = _split_space_and_punctuation(doc)
    identifiers: list[str] = _remove_stop_words(words)
    corpus: list[str] = list(chain(*(_split_identifier(z) for z in identifiers)))
    return corpus


def collect_corpus(
    entities: list[Entity], contents: dict[str, bytes]
) -> dict[str, str]:
    """
    For each entity in the list, filter out if they're not code members
    and extract their content
    """
    entities = _filter_code_members(entities)
    corpus: dict[str, str] = {}
    for entity in entities:
        if entity.content_id not in contents:
            # corpus[entity.id] = ""
            continue
        content = extract_entity_content(contents[entity.content_id], entity)
        corpus[entity.id] = content
    return corpus

def tokenize_corpus(corpus: dict[str, str]) -> dict[str, list[str]]:
    """
    Tokenizes the content of each entity in the corpus.
    """
    return {entity_id: _tokenize(content) for entity_id, content in corpus.items()}
