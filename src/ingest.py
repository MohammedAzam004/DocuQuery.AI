from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from src.chunk import create_chunks
from src.embed import embed_texts
from src.vector_store import create_faiss_index, save_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"
INDEX_FOLDER = PROJECT_ROOT / "faiss_index"


def clean_text(text: str) -> str:
    """Clean extracted text so retrieval works better."""
    if not text:
        return ""

    text = text.replace("-\n", "")
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if re.fullmatch(r"page\s+\d+", line.lower()):
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


def load_pdf(file_path: Path) -> list[dict]:
    """Load one PDF and return one record per page."""
    reader = PdfReader(str(file_path))
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        pages.append(
            {
                "filename": file_path.name,
                "page_number": page_number,
                "text": text,
            }
        )

    return pages


def load_txt(file_path: Path) -> list[dict]:
    """Load one text file as a single page-like record."""
    text = clean_text(file_path.read_text(encoding="utf-8"))
    if not text:
        return []

    return [
        {
            "filename": file_path.name,
            "page_number": 1,
            "text": text,
        }
    ]


def load_docx(file_path: Path) -> list[dict]:
    """Load one DOCX file as a single page-like record."""
    document = Document(str(file_path))
    text = clean_text("\n".join(paragraph.text for paragraph in document.paragraphs))

    if not text:
        return []

    return [
        {
            "filename": file_path.name,
            "page_number": 1,
            "text": text,
        }
    ]


def load_documents(data_folder: Path = DATA_FOLDER) -> list[dict]:
    """Load all supported files from the data folder."""
    # We keep the file discovery simple so it is easy to explain.
    supported_files = sorted(
        [
            file_path
            for file_path in data_folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt", ".docx"}
        ]
    )

    all_documents: list[dict] = []

    for file_path in supported_files:
        # Pick the right loader based on the file type.
        if file_path.suffix.lower() == ".pdf":
            all_documents.extend(load_pdf(file_path))
        elif file_path.suffix.lower() == ".txt":
            all_documents.extend(load_txt(file_path))
        elif file_path.suffix.lower() == ".docx":
            all_documents.extend(load_docx(file_path))

    return all_documents


def build_document_summary(documents: list[dict], chunks: list[dict]) -> list[dict]:
    """Create a small summary used by the CLI and Streamlit UI."""
    summary_map: dict[str, dict] = {}

    for document in documents:
        filename = document["filename"]
        summary_map.setdefault(
            filename,
            {
                "filename": filename,
                "pages": 0,
                "words": 0,
                "chunks": 0,
                "meets_assignment_requirement": False,
            },
        )
        summary_map[filename]["pages"] += 1
        summary_map[filename]["words"] += len(document["text"].split())

    for chunk in chunks:
        summary_map[chunk["filename"]]["chunks"] += 1

    for document_summary in summary_map.values():
        document_summary["meets_assignment_requirement"] = (
            document_summary["pages"] >= 2 or document_summary["words"] >= 500
        )

    return list(summary_map.values())


def run_indexing() -> None:
    """Run the full indexing pipeline once."""
    print("Loading documents from data folder...")
    documents = load_documents()

    if not documents:
        raise ValueError("No supported documents were found in the data folder.")

    print(f"Loaded {len(documents)} page-level records.")

    print("Creating chunks...")
    # The assignment asks for 500-character chunks with 100-character overlap.
    chunks = create_chunks(documents=documents, chunk_size=500, overlap=100)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks in batches...")
    # We send the full chunk list to the embedding model in batches, not one by one.
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(chunk_texts, batch_size=32)

    print("Building FAISS index...")
    index = create_faiss_index(embeddings)

    print("Saving index to disk...")
    # Save both the FAISS file and the metadata files so querying stays separate.
    document_summary = build_document_summary(documents=documents, chunks=chunks)
    save_index(index=index, chunks=chunks, documents=document_summary, index_folder=INDEX_FOLDER)

    print("Indexing complete.")
    print(f"Documents indexed: {len(document_summary)}")
    print(f"Index saved in: {INDEX_FOLDER}")


if __name__ == "__main__":
    run_indexing()
