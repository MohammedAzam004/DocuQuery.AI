from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

NO_RELEVANT_INFO_MESSAGE = (
    "No relevant information found in the knowledge base. Please ensure documents are properly ingested."
)
GEMINI_API_KEY_MISSING_MESSAGE = (
    f"GEMINI_API_KEY is missing. Add it to {ENV_FILE} and restart the application."
)
GEMINI_REQUEST_FAILED_MESSAGE = (
    "Could not generate an answer right now. Please check your Gemini API key and internet connection, then try again."
)
GEMINI_HIGH_DEMAND_MESSAGE = (
    "Gemini is currently under high demand. Please wait a few seconds and try again."
)
GEMINI_RESPONSE_INVALID_MESSAGE = (
    "Gemini returned an unreadable response. Please try the question again."
)


def build_valid_citations(retrieved_chunks: list[dict]) -> list[str]:
    """Build the list of citation strings allowed in the final answer."""
    citations: list[str] = []

    for chunk in retrieved_chunks:
        citation = f'{chunk["filename"]} | page {chunk["page_number"]} | chunk {chunk["chunk_number"]}'
        if citation not in citations:
            citations.append(citation)

    return citations


def build_context(retrieved_chunks: list[dict]) -> str:
    """Turn retrieved chunks into a prompt-ready context block."""
    context_parts: list[str] = []

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        # Each chunk includes its source file and page number.
        context_parts.append(
            "\n".join(
                [
                    f"Source ID: S{source_number}",
                    f'File: {chunk["filename"]}',
                    f'Page: {chunk["page_number"]}',
                    f'Chunk: {chunk["chunk_number"]}',
                    f'Text: {chunk["text"]}',
                ]
            )
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """Create a grounded prompt for Gemini."""
    return f"""
You are DocuQuery AI, an intelligent document assistant.

Rules:
1. Answer only from the context provided below.
2. Do not use outside knowledge, assumptions, or training data.
3. If the answer is not clearly supported by the context, return exactly "{NO_RELEVANT_INFO_MESSAGE}".
4. If the user asks something outside the documents, politely refuse by returning exactly "{NO_RELEVANT_INFO_MESSAGE}".
5. Keep the answer clear, concise, and grounded in the retrieved content.
6. Return source citations only from the provided context.
7. Every citation must use the exact format "filename | page X | chunk Y".

Return valid JSON with this shape:
{{
  "answer": "your answer here",
  "citations": ["filename | page X | chunk Y"]
}}

If the answer is not found, return:
{{
  "answer": "{NO_RELEVANT_INFO_MESSAGE}",
  "citations": []
}}

Question:
{question}

Context:
{context}
""".strip()


def generate_answer(question: str, retrieved_chunks: list[dict]) -> dict:
    """Generate a grounded answer with Gemini 2.5 Flash."""
    if not retrieved_chunks:
        return {
            "answer": NO_RELEVANT_INFO_MESSAGE,
            "citations": [],
        }

    load_dotenv(dotenv_path=ENV_FILE)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(GEMINI_API_KEY_MISSING_MESSAGE)

    context = build_context(retrieved_chunks)
    prompt = build_prompt(question=question, context=context)

    client = genai.Client(api_key=api_key)
    response = None

    # Ask Gemini for JSON so the UI can display the answer and citations cleanly.
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            )
            break
        except Exception as error:
            error_text = str(error)

            if "503" in error_text or "UNAVAILABLE" in error_text:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(GEMINI_HIGH_DEMAND_MESSAGE) from error

            raise RuntimeError(GEMINI_REQUEST_FAILED_MESSAGE) from error

    # Convert the JSON text from Gemini into a normal Python dictionary.
    try:
        response_text = (response.text or "").strip()
        if response_text.startswith("```json"):
            response_text = response_text.removeprefix("```json").removesuffix("```").strip()
        elif response_text.startswith("```"):
            response_text = response_text.removeprefix("```").removesuffix("```").strip()

        result = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(GEMINI_RESPONSE_INVALID_MESSAGE) from error
    answer_text = str(result.get("answer", "")).strip()
    valid_citations = build_valid_citations(retrieved_chunks)

    if not answer_text or answer_text == "Not found in documents":
        answer_text = NO_RELEVANT_INFO_MESSAGE

    if answer_text == NO_RELEVANT_INFO_MESSAGE:
        return {
            "answer": NO_RELEVANT_INFO_MESSAGE,
            "citations": [],
        }

    filtered_citations: list[str] = []
    for citation in result.get("citations", []):
        if citation in valid_citations and citation not in filtered_citations:
            filtered_citations.append(citation)

    if not filtered_citations:
        filtered_citations = valid_citations[:2]

    return {
        "answer": answer_text,
        "citations": filtered_citations,
    }
