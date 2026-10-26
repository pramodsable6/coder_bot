import os

import httpx
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="microsoft/unixcoder-base")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    base_url=ollama_base_url,
    temperature=0,
)

db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a coding assistant. Use the retrieved code context to answer "
            "questions about this codebase. Do not invent codebase details. If "
            "the context does not answer the question, say so and distinguish "
            "any general guidance from facts about the codebase. Cite source "
            "paths when referring to retrieved code.",
        ),
        ("human", "Question:\n{question}\n\nRetrieved code:\n{context}"),
    ]
)
answer_chain = prompt | llm | StrOutputParser()


def answer_question(question: str) -> None:
    results = db.similarity_search(question, k=4)
    if not results:
        print("No code was retrieved. Build the index first or try another question.")
        return

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in results
    )
    print("CONTEXT:\n" + context)
    
    try:
        answer = answer_chain.invoke({"question": question, "context": context})
    except httpx.ConnectError:
        print(
            f"\nCannot connect to Ollama at {ollama_base_url}. "
            "Install and start Ollama, then retry. On Linux, run "
            "`ollama serve` in another terminal."
        )
        return
    print(f"\n{answer}\n")


def main() -> None:
    print("Ask a coding question about the indexed codebase. Type 'exit' to quit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer_question(question)


if __name__ == "__main__":
    main()
