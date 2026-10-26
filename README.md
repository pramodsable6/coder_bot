# Coder Bot

A small Python RAG prototype for coding questions about a source codebase. It
splits source files into language-aware chunks, creates embeddings with
Microsoft's UniXcoder model, and stores them in a persistent Chroma vector
database. For each question, it retrieves relevant code and passes that context
to a local Ollama language model, which generates a natural-language answer.

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

## Ask coding questions

Install and start [Ollama](https://ollama.com/), then download a coding model:

```bash
ollama pull qwen2.5-coder:7b
```

If Ollama is not already running as a service, start it in another terminal:

```bash
ollama serve
```

After building the index and ensuring the Ollama service is running, start the
interactive query loop:

```bash
uv run python query_embeddings.py
```

Enter a coding question about the indexed codebase; type `exit` or `quit` to
stop. The app embeds the question, retrieves the four most relevant chunks from
Chroma, and sends them with the question to Ollama. The LLM uses this retrieved
context to generate an answer and is instructed not to invent codebase details;
it should say when the retrieved context is insufficient. Retrieved context,
including source paths, is printed before the generated answer.

The default answer-generation model is `qwen2.5-coder:7b`. Override the model
or Ollama server URL with environment variables (these can also be set in
`.env`):

```bash
OLLAMA_MODEL=your-model
OLLAMA_BASE_URL=http://localhost:11434
```

## Inspect the database

`pandas_db_viewer.py` loads the Chroma collection into a pandas DataFrame and
prints a preview and its shape. It also enters the Python debugger at the end
of execution.
