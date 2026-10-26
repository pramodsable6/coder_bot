# Coder Bot

A small Python prototype for semantic code search. It splits source files into
language-aware chunks, creates embeddings with Microsoft's UniXcoder model, and
stores them in a persistent Chroma vector database.

## Requirements

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

Install the project dependencies:

```bash
uv sync
```

The embedding model (`microsoft/unixcoder-base`) is downloaded by
`sentence-transformers` the first time it is used. No API key is needed for the
current Hugging Face embedding workflow.

## Build the code index

Before indexing, set `CODEBASE_PATH` in `create_embeddings.py` to the source
directory you want to search. Its current value, `../Repos/Python`, is relative
to the directory from which you run the command.

Run the indexer from the project root:

```bash
uv run python create_embeddings.py
```

Supported file types are Python, JavaScript, TypeScript, Go, C++, Java, and
Markdown. The generated vector database is stored in `./chroma_db`.

## Search the index

Install and start [Ollama](https://ollama.com/), then download a coding model:

```bash
ollama pull qwen2.5-coder:7b
```

After building the index and ensuring the Ollama service is running, start the
interactive query loop:

```bash
uv run python query_embeddings.py
```

Enter coding questions about the indexed codebase; type `exit` or `quit` to
stop. Each answer uses the four most relevant code chunks retrieved from Chroma
and includes source paths for context.

The default model is `qwen2.5-coder:7b`. Override it or the Ollama server URL
with environment variables (these can also be set in `.env`):

```bash
OLLAMA_MODEL=your-model
OLLAMA_BASE_URL=http://localhost:11434
```

## Inspect the database

`pandas_db_viewer.py` loads the Chroma collection into a pandas DataFrame and
prints a preview and its shape. It also enters the Python debugger at the end
of execution.
