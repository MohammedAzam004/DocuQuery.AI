# DocuQuery AI

DocuQuery AI is a Retrieval-Augmented Generation (RAG) document question-answering bot built with Python. It ingests local documents, stores their embeddings in FAISS, and answers user questions only from retrieved document context with source citations.

The project is intentionally simple and beginner-friendly. Indexing and querying are separated so the document index is created once and reused across runs.

Note: the virtual environment is created locally during setup and is ignored by Git. It should not be committed to the repository.

## Features

- Document ingestion from a local `data/` folder
- Support for PDF, TXT, and DOCX files
- Fixed-size chunking with overlap
- Semantic search using FAISS
- Batched embeddings with `all-MiniLM-L6-v2`
- Grounded answer generation using Gemini 2.5 Flash
- Source citations in every answer
- Streamlit frontend and CLI entry point

## Tech Stack

- Python 3.11 recommended for the cleanest setup
- Streamlit 1.56.0
- Sentence Transformers 5.4.0
- Transformers 5.5.4
- PyTorch 2.11.0
- FAISS CPU 1.13.2
- PyPDF 6.10.0
- python-docx 1.2.0
- Google Gen AI SDK (`google-genai`) 1.72.0
- NumPy 2.4.4
- python-dotenv 1.2.2

## Project Structure

```text
DocuQuery.AI/
|-- data/                # knowledge base documents
|-- faiss_index/         # generated locally after running ingestion
|-- src/
|   |-- ingest.py
|   |-- chunk.py
|   |-- embed.py
|   |-- vector_store.py
|   |-- retrieve.py
|   |-- generate.py
|   `-- main.py
|-- app.py               # Streamlit frontend
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- README.md
```

## Architecture

DocuQuery AI follows a standard RAG pipeline:

1. Load documents from `data/`
2. Extract and clean text
3. Split text into chunks
4. Embed chunks in batches
5. Store vectors in FAISS
6. Embed the user question
7. Retrieve the top 5 matching chunks
8. Send only the retrieved context to Gemini
9. Return a grounded answer with citations

## Chunking Strategy

The project uses fixed-size chunking with overlap:

- `chunk_size = 500`
- `overlap = 100`

Why this approach:

- It matches the assignment requirement directly.
- It is simple to understand and easy to explain in an interview.
- The overlap helps preserve context between neighboring chunks.
- Each chunk keeps metadata such as filename, page number, and chunk number.

## Embedding Model

The embedding model is `all-MiniLM-L6-v2`.

Why this model:

- It is lightweight and fast.
- It gives strong semantic search quality for a beginner RAG project.
- It works well on CPU and keeps the setup simple.

Important note:

- Embeddings are created in batches, not one chunk at a time.

## Vector Database

The project uses FAISS for vector storage.

Why FAISS:

- It is fast for local similarity search.
- It is easy to save and load from disk.
- It keeps indexing separate from querying.

The `faiss_index/` folder is generated locally after running the ingestion step.

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd DocuQuery.AI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run indexing

```bash
python -m src.ingest
```

### 6. Run the Streamlit app

```bash
python -m streamlit run app.py
```

### 7. Optional: run the CLI app

```bash
python -m src.main
```

## Environment Variables

Required:

- `GEMINI_API_KEY` - used for Gemini answer generation

## Example Queries

1. Why is multi-factor authentication important for small teams?
   Expected theme: stronger account protection, reduced breach risk, and layered security for small organizations.
2. What kinds of crops are usually a poor fit for urban farming in dense cities?
   Expected theme: crops with deep root systems, large land needs, or low yield per square foot.
3. Why should remote teams separate written updates from live discussions?
   Expected theme: clearer async communication, fewer meeting interruptions, and better documentation.
4. Why is transmission infrastructure a bottleneck in renewable energy projects?
   Expected theme: grid limitations, connection delays, and the challenge of moving power from generation sites to demand centers.
5. What does the remote work handbook recommend for protecting work-life boundaries?
   Expected theme: clear working hours, response expectations, and deliberate routines to separate work from personal time.

## Limitations

- Retrieval quality depends on the top 5 chunks returned by FAISS.
- TXT and DOCX files do not have reliable page boundaries, so they are treated as one page-like record during ingestion.
- Some PDFs with complex layouts may need stronger text cleaning.
- If Gemini is temporarily under high demand, answer generation may need a retry.

## Troubleshooting

### Keras / TensorFlow compatibility error

If you see an error like:

```text
ValueError: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.
```

that usually means you are running the project in an environment that has TensorFlow or standalone Keras installed.

This project does not need TensorFlow. It uses Sentence Transformers with PyTorch only.

Use a clean virtual environment and reinstall only the project requirements:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip uninstall -y tensorflow keras tf-keras
python -m pip install -r requirements.txt
```

The project code also disables optional TensorFlow loading during embedding setup, which helps prevent this error in mixed environments.

### Quick environment check

Run this command after installation:

```powershell
python -c "from src.embed import embed_texts; print(embed_texts(['DocuQuery test']).shape)"
```

Expected result:

```text
(1, 384)
```

## Final Notes

- Run `python -m src.ingest` before starting the app.
- The repository keeps the code simple and beginner-friendly on purpose.
- Answers are designed to stay grounded in the uploaded documents and include citations.
