from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.generate import GEMINI_API_KEY_MISSING_MESSAGE, generate_answer
from src.retrieve import retrieve_chunks
from src.vector_store import load_document_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_FOLDER = PROJECT_ROOT / "faiss_index"
ENV_FILE = PROJECT_ROOT / ".env"
INDEX_NOT_CREATED_MESSAGE = (
    "The document index has not been created. Please run the ingestion step using:\n"
    "python -m src.ingest\n"
    "and restart the application."
)
GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}


def has_gemini_api_key() -> bool:
    """Check whether the Gemini API key is available."""
    load_dotenv(dotenv_path=ENV_FILE)
    return bool(os.getenv("GEMINI_API_KEY"))


def get_greeting_answer(question: str) -> dict | None:
    """Return a simple greeting without using retrieval."""
    cleaned_question = " ".join(question.lower().strip().split())

    if cleaned_question in GREETING_WORDS:
        return {
            "answer": "Hello! I am DocuQuery AI. Ask me a question about your uploaded documents and I will answer from the knowledge base with citations.",
            "citations": [],
        }

    return None


def print_document_info() -> None:
    """Show which documents are available in the saved index."""
    documents = load_document_summary(INDEX_FOLDER)

    if not documents:
        print(INDEX_NOT_CREATED_MESSAGE)
        return

    print("\nIndexed documents:")
    for document in documents:
        requirement_status = "Yes" if document.get("meets_assignment_requirement") else "No"
        print(
            f'- {document["filename"]} | pages: {document["pages"]} | words: {document.get("words", 0)} '
            f'| chunks: {document["chunks"]} | meets assignment rule: {requirement_status}'
        )


def print_retrieved_chunks(retrieved_chunks: list[dict]) -> None:
    """Print the retrieved chunks and metadata."""
    if not retrieved_chunks:
        return

    print("\nRetrieved chunks:")
    for chunk in retrieved_chunks:
        print("-" * 80)
        print(f'File: {chunk["filename"]}')
        print(f'Page: {chunk["page_number"]}')
        print(f'Chunk: {chunk["chunk_number"]}')
        print(f'Score: {chunk["score"]:.4f}')
        print(chunk["text"])


def run_chat_loop() -> None:
    """Start a simple CLI loop for asking questions."""
    print("DocuQuery AI")
    print("Ask questions about the indexed documents.")
    print("Type 'exit' to close the bot.")
    print_document_info()

    if not has_gemini_api_key():
        print(f"\n{GEMINI_API_KEY_MISSING_MESSAGE}")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.")
            continue

        greeting_answer = get_greeting_answer(question)
        if greeting_answer is not None:
            print("\nFinal Answer:")
            print(greeting_answer["answer"])
            print("\nSources:")
            print("- None")
            continue

        try:
            retrieved_chunks = retrieve_chunks(question=question, index_folder=INDEX_FOLDER, k=5)
        except FileNotFoundError:
            print(INDEX_NOT_CREATED_MESSAGE)
            continue

        try:
            answer = generate_answer(question=question, retrieved_chunks=retrieved_chunks)
        except ValueError as error:
            print(str(error))
            continue
        except RuntimeError as error:
            print(str(error))
            continue

        print_retrieved_chunks(retrieved_chunks)

        print("\nFinal Answer:")
        print(answer["answer"])

        print("\nSources:")
        if answer["citations"]:
            for citation in answer["citations"]:
                print(f"- {citation}")
        else:
            print("- None")


if __name__ == "__main__":
    run_chat_loop()
