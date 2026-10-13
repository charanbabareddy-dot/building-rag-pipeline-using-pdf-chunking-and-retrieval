# RAG Pipeline Using PDF Chunking and Retrieval

A Python project that retrieves relevant information from PDF documents and uses Groq to generate answers grounded in the retrieved text.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) workflow for a small collection of e-commerce policy documents. It loads PDF pages, splits their text into overlapping chunks, generates local vector embeddings, and stores those embeddings in a FAISS vector store. For a question, semantic search retrieves relevant chunks and supplies them as context to a language model hosted by Groq.

The project demonstrates PDF processing, document chunking, semantic retrieval, and context-based answer generation.

## Architecture

```text
PDF documents
    │
    ▼
PyPDFLoader — extract page text
    │
    ▼
RecursiveCharacterTextSplitter — create overlapping chunks
    │
    ▼
Hugging Face all-MiniLM-L6-v2 — generate local embeddings
    │
    ▼
FAISS vector store ◄── User question → Query embedding
    │
    ▼
Relevant document chunks + User question
    │
    ▼
Groq: openai/gpt-oss-20b
    │
    ▼
Final answer
```

## Technologies

| Technology | Purpose |
| --- | --- |
| Python | Pipeline implementation |
| LangChain | Document loading, splitting, and retrieval integrations |
| PyPDF / PyPDFLoader | Extract text from PDF pages |
| RecursiveCharacterTextSplitter | Split text into overlapping chunks |
| Hugging Face / Sentence Transformers | Generate local semantic embeddings |
| FAISS | Store embeddings and search for similar chunks |
| Groq | Host the model used for answer generation |
| python-dotenv | Load API credentials from a local `.env` file |
| VS Code | Development environment |

## Embedding Model and Technology Substitution

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The original project requirement specified OpenAI or Azure OpenAI embeddings. Those embedding credentials were unavailable during development, so a local Hugging Face model was used as a practical substitute. This preserves the embedding-and-retrieval workflow while changing the embedding provider; it is a documented deviation from the original requirement.

After the model is downloaded, embeddings are computed locally, so PDF text does not need to be sent to an external embedding API. The initial download requires internet access unless the model is already cached.

## Language Model

Final answers are generated through Groq using:

```text
openai/gpt-oss-20b
```

Groq receives the question and retrieved document context. Local embeddings therefore do not make the entire pipeline offline: answer generation requires a Groq API key and network access.

## Dataset

The recorded test dataset contains three PDFs covering sample e-commerce policies:

| Document | Topics |
| --- | --- |
| `Return_Refund_Warranty.pdf` | Returns, refunds, and warranties |
| `Shipping_Delivery_Guide.pdf` | Shipping and delivery |
| `Payments_Customer_Support.pdf` | Payment methods and customer support |

## Chunking Configuration

PDF text is split using `RecursiveCharacterTextSplitter` with the following configuration:

```python
chunk_size=500
chunk_overlap=100
```

With character-based length measurement, the target maximum chunk size is 500 characters, with up to 100 characters of overlap between adjacent chunks. Overlap helps preserve context around split boundaries. Actual chunk lengths vary with the document text and split points.

## Recorded Project Metrics

| Metric | Recorded value |
| --- | ---: |
| PDF files | 3 |
| Pages loaded | 9 |
| Chunks created | 26 |
| Average chunk length | 406.15 characters |

These values come from the development run recorded in the project conversation. They describe that dataset and configuration; they are not retrieval-accuracy measurements and may change when the documents or splitter settings change.

## Sample Queries and Output

Example questions used during development:

- How long do customers have to return an item?
- How long does international delivery usually take?
- Which payment methods are accepted?

The following sample is recorded in the project conversation, rather than produced by a new execution for this README:

```text
User query:
How long do customers have to return an item?

Retrieved source:
Return_Refund_Warranty.pdf

Final answer:
Customers have 30 calendar days from the delivery date to return an item.
```

Generated wording may vary between runs. Check answers against the retrieved source text, especially when the question involves exceptions or multiple policies.

## Project Structure

Suggested organization based on the development conversation:

```text
rag-pdf-retrieval/
├── data/
│   ├── Return_Refund_Warranty.pdf
│   ├── Shipping_Delivery_Guide.pdf
│   └── Payments_Customer_Support.pdf
├── outputs/                    # Optional saved sample results
├── faiss_index/                # Optional persisted vector index
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── .env                        # Local API key; never commit
└── .gitignore
```

Match the PDF directory to the path configured in `rag_pipeline.py`. The `outputs/` and `faiss_index/` folders are needed only if the script saves results or persists the index.

## Setup and Usage

Run the following commands from the project folder containing `rag_pipeline.py`.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

If your system uses `python3`, substitute `python3` in this command.

### 2. Activate the environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install project dependencies

Use the project's dependency file:

```bash
python -m pip install -r requirements.txt
```

The dependencies should cover the LangChain integrations used by the script, PyPDF, Sentence Transformers, FAISS, Groq, and python-dotenv. Exact packages and versions must match the imports in `rag_pipeline.py`; keep the versions from the working environment for reproducibility.

### 4. Configure Groq credentials

Create a `.env` file in the project folder:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own key. Ensure the script loads this file before creating the Groq client.

### 5. Add the PDFs

Place the three dataset PDFs in the input directory configured by the script, such as `data/`. Use PDFs containing extractable text; scanned image-only PDFs may require OCR before ingestion.

### 6. Run the pipeline

```bash
python rag_pipeline.py
```

The workflow loads pages, creates chunks, computes embeddings, builds or uses a FAISS index, retrieves relevant text, and generates an answer through Groq. The first run may take longer while the embedding model downloads. Use the sample questions in the script's query section if queries are configured there.

## Security and Data Handling

- Keep the Groq API key in `.env` or an environment variable. Never include it in source code, screenshots, output files, or commits.
- Add `.env` to `.gitignore` before committing. Ignore local indexes and outputs when they contain private document content.
- Local embedding computation keeps PDF text out of an external embedding service. Retrieved chunks and questions are still sent to Groq for answer generation; use documents you are authorized to share with that service.
- Treat persisted FAISS indexes as potentially sensitive. If the script loads a serialized index with pickle-based deserialization, load only files you created and trust.
- Treat document text as data. For stronger grounding, instruct the model to answer only from retrieved context and acknowledge when the context is insufficient.

Suggested `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
faiss_index/
outputs/
```

## Limitations and Future Improvements

This is a demonstration pipeline with a small dataset. The recorded metrics do not establish answer accuracy or production readiness. Possible improvements include source and page citations, retrieval evaluation, tests for empty documents and empty search results, configurable query and retrieval settings, structured logging, OCR support, and a user interface.

## Conclusion

The project demonstrates an end-to-end PDF RAG workflow: load documents, create overlapping chunks, generate local Hugging Face embeddings, retrieve relevant text with FAISS, and generate answers using Groq. The documented embedding substitution makes the project usable without OpenAI or Azure OpenAI embedding credentials while clearly distinguishing local retrieval processing from hosted answer generation.
