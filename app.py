from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.generate import (
    GEMINI_API_KEY_MISSING_MESSAGE,
    NO_RELEVANT_INFO_MESSAGE,
    generate_answer,
)
from src.retrieve import retrieve_chunks
from src.vector_store import load_document_summary


PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_FOLDER = PROJECT_ROOT / "faiss_index"
ENV_FILE = PROJECT_ROOT / ".env"
INDEX_NOT_CREATED_MESSAGE = (
    "The document index has not been created. Please run the ingestion step using:\n"
    "python -m src.ingest\n"
    "and restart the application."
)
GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}


st.set_page_config(page_title="DocuQuery AI", page_icon=":page_facing_up:", layout="wide")


def has_gemini_api_key() -> bool:
    """Check whether the Gemini API key is available."""
    load_dotenv(dotenv_path=ENV_FILE)
    return bool(os.getenv("GEMINI_API_KEY"))


def get_greeting_answer(question: str) -> dict | None:
    """Return a small greeting without calling Gemini."""
    cleaned_question = " ".join(question.lower().strip().split())

    if cleaned_question in GREETING_WORDS:
        return {
            "answer": "Hello! I am DocuQuery AI. Ask me a question about the uploaded documents and I will answer from the knowledge base with citations.",
            "citations": [],
        }

    return None


def add_custom_css() -> None:
    """Add simple styling to make the UI cleaner."""
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
                color: #102033;
            }
            .hero-card, .info-card, .answer-card, .source-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #dde5f0;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                animation: fadeUp 0.5s ease;
            }
            .hero-title {
                font-size: 2.1rem;
                font-weight: 700;
                color: #15304f;
                margin-bottom: 0.4rem;
            }
            .hero-text {
                color: #45556d;
                font-size: 1rem;
                line-height: 1.6;
            }
            .metric-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #15304f;
                margin-bottom: 0.2rem;
            }
            .metric-label {
                color: #607089;
                font-size: 0.95rem;
            }
            .section-title {
                font-size: 1.2rem;
                font-weight: 700;
                color: #15304f;
                margin-bottom: 0.7rem;
            }
            .chunk-box {
                border-left: 4px solid #4c8bf5;
                padding-left: 14px;
                margin-bottom: 14px;
            }
            @keyframes fadeUp {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header() -> None:
    """Show the page title and short project description."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">DocuQuery AI</div>
            <div class="hero-text">
                Ask questions about the documents in the knowledge base and get grounded answers with source citations.
                The bot retrieves the top matching chunks from FAISS and answers only from those documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_document_info(documents: list[dict]) -> None:
    """Display file count, page count, and chunk count."""
    total_documents = len(documents)
    total_pages = sum(document["pages"] for document in documents)
    total_chunks = sum(document["chunks"] for document in documents)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="metric-value">{total_documents}</div>
                <div class="metric-label">Documents Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="metric-value">{total_pages}</div>
                <div class="metric-label">Pages Loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="metric-value">{total_chunks}</div>
                <div class="metric-label">Chunks Available</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-title">Document Library</div>', unsafe_allow_html=True)
    st.caption("Assignment rule: each document should be at least 2 pages or 500 words.")

    for document in documents:
        requirement_status = "Yes" if document.get("meets_assignment_requirement") else "No"
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom: 12px;">
                <strong>{document["filename"]}</strong><br>
                Pages: {document["pages"]} &nbsp;&nbsp;|&nbsp;&nbsp; Words: {document.get("words", 0)}
                &nbsp;&nbsp;|&nbsp;&nbsp; Chunks: {document["chunks"]}
                &nbsp;&nbsp;|&nbsp;&nbsp; Meets Assignment Rule: {requirement_status}
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_setup_status(api_key_ready: bool) -> None:
    """Show simple setup status for the user."""
    st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

    if api_key_ready:
        st.success("Gemini API key detected. The app is ready to answer questions.")
    else:
        st.warning(GEMINI_API_KEY_MISSING_MESSAGE)
        st.caption(f"Create a file at {ENV_FILE} and add your Gemini key there.")


def show_answer(answer: dict) -> None:
    """Display the final answer section."""
    st.markdown('<div class="section-title">Final Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="answer-card">
            {answer["answer"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sources(answer: dict) -> None:
    """Display the source citations clearly."""
    st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)

    if answer["citations"]:
        for citation in answer["citations"]:
            st.markdown(
                f"""
                <div class="source-card" style="margin-bottom: 10px;">
                    {citation}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="source-card">
                None
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_retrieved_chunks(retrieved_chunks: list[dict]) -> None:
    """Show retrieved chunks inside an expander."""
    with st.expander("View Retrieved Chunks", expanded=False):
        for index, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(
                f"""
                <div class="chunk-box">
                    <strong>Chunk {index}</strong><br>
                    File: {chunk["filename"]}<br>
                    Page: {chunk["page_number"]}<br>
                    Chunk: {chunk["chunk_number"]}<br>
                    Score: {chunk["score"]:.4f}<br><br>
                    {chunk["text"]}
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    """Run the Streamlit app."""
    add_custom_css()
    show_header()

    if "answer" not in st.session_state:
        st.session_state.answer = None
    if "retrieved_chunks" not in st.session_state:
        st.session_state.retrieved_chunks = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    # Load the saved document summary created during indexing.
    documents = load_document_summary(INDEX_FOLDER)

    if not documents:
        st.error(INDEX_NOT_CREATED_MESSAGE)
        return

    api_key_ready = has_gemini_api_key()

    left_column, right_column = st.columns([1, 1.3], gap="large")

    with left_column:
        show_document_info(documents)
        st.markdown("")
        show_setup_status(api_key_ready)

    with right_column:
        st.markdown('<div class="section-title">Ask a Question</div>', unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=False):
            question = st.text_input(
                "Question",
                placeholder="Example: What are the strongest defenses recommended for a small team?",
                label_visibility="collapsed",
                value=st.session_state.last_question,
            )
            submitted = st.form_submit_button("Get Answer", type="primary", use_container_width=True)

        if submitted:
            st.session_state.last_question = question

            if not question.strip():
                st.warning("Please enter a question first.")
            else:
                greeting_answer = get_greeting_answer(question)

                if greeting_answer is not None:
                    st.session_state.answer = greeting_answer
                    st.session_state.retrieved_chunks = []
                else:
                    try:
                        with st.spinner("Searching the documents and writing a grounded answer..."):
                            # Step 1: retrieve the best matching chunks from FAISS.
                            retrieved_chunks = retrieve_chunks(question=question, index_folder=INDEX_FOLDER, k=5)
                            # Step 2: send only those chunks to Gemini for a grounded answer.
                            answer = generate_answer(question=question, retrieved_chunks=retrieved_chunks)
                    except FileNotFoundError:
                        st.error(INDEX_NOT_CREATED_MESSAGE)
                        return
                    except ValueError as error:
                        st.warning(str(error))
                        return
                    except RuntimeError as error:
                        st.warning(str(error))
                        return
                    except Exception as error:
                        st.error(f"Unexpected error: {error}")
                        return

                    st.session_state.answer = answer
                    st.session_state.retrieved_chunks = retrieved_chunks

        if st.session_state.answer is not None:
            show_answer(st.session_state.answer)
            st.markdown("")
            show_sources(st.session_state.answer)
            st.markdown("")
            if (
                st.session_state.answer["answer"] != NO_RELEVANT_INFO_MESSAGE
                and st.session_state.retrieved_chunks
            ):
                show_retrieved_chunks(st.session_state.retrieved_chunks)


if __name__ == "__main__":
    main()
