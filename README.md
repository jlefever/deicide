# Deicide

To get started, run:

```
uv run deicide --help
```

This project is managed by [uv](https://docs.astral.sh/uv/).

## Example

Deicide takes input from [Neodepends](https://github.com/jlefever/neodepends). Below is an example.

```bash
git clone https://github.com/apache/doris
neodepends --input=doris/ --output=doris.db -ljava -D
uv run deicide --input=doris.db --output=doris_clustering.json --filename=fe/fe-core/src/main/java/org/apache/doris/dictionary/DictionaryManager.java
```
