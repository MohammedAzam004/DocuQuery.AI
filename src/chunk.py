from __future__ import annotations

    
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split one text string into fixed-size chunks with overlap."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        if start > 0 and text[start - 1] != " " and text[start] != " ":
            while start < len(text) and text[start] != " ":
                start += 1

        end = start + chunk_size
        if end < len(text): 
            adjusted_end = end
            while adjusted_end > start and text[adjusted_end - 1] != " ":
                adjusted_end -= 1

            if adjusted_end - start > chunk_size * 0.7:
                end = adjusted_end

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(
    documents: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Create chunks and keep file/page metadata on every chunk."""
    all_chunks: list[dict] = []

    for document in documents:
        text_chunks = split_text_into_chunks(
            text=document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk_text in enumerate(text_chunks, start=1):
            all_chunks.append(
                {
                    "chunk_id": f'{document["filename"]}_page_{document["page_number"]}_chunk_{chunk_number}',
                    "filename": document["filename"],
                    "page_number": document["page_number"],
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )

    return all_chunks
