# RAG pipeline for loading, chunking, embedding, and retrieving information from PDF files.
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# Folder containing the PDF documents
data_folder = Path("data")

# Find all PDF files inside the data folder
pdf_files = list(data_folder.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("No PDF files were found in the data folder.")
all_documents = []

for pdf_file in pdf_files:
    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    all_documents.extend(documents)

print("\nPDF loading completed.")
print(f"Total PDF files found: {len(pdf_files)}")
print(f"Total pages loaded: {len(all_documents)}")

# Split the loaded PDF pages into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(all_documents)
if not chunks:
    raise ValueError("No text chunks were created from the PDF documents.")
print("\nChunking completed.")
print(f"Total chunks created: {len(chunks)}")
total_characters = sum(len(chunk.page_content) for chunk in chunks)

average_chunk_size = (
    total_characters / len(chunks)
    if chunks
    else 0
)

print(f"Average chunk size: {average_chunk_size:.2f} characters")
print("\nSample Chunks:")

for index, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk.page_content)
    print(f"Source: {chunk.metadata.get('source')}")
    print(f"Page: {chunk.metadata.get('page')}")
    
# Local embedding model used as a substitute for OpenAI or Azure OpenAI
# because API credentials are not available.
# The model runs locally after download and is suitable for semantic search.

print("\nLoading local Hugging Face embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Hugging Face embedding model loaded successfully.")
print("\nCreating FAISS vector store...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created successfully.")
vectorstore.save_local("faiss_index")

print("FAISS index saved locally.")

print("\nRunning semantic search...")

query = "What is the return policy for customers?"

results = vectorstore.similarity_search(
    query,
    k=1
)
if not results:
    raise ValueError("No relevant document chunks were retrieved.")
print("\nUser Query:")
print(query)

print("\nTop Matching Result:")
print(results[0].page_content)
print("\nSource Information:")
print(f"Source: {results[0].metadata.get('source')}")
print(f"Page: {results[0].metadata.get('page')}")
print("\nTesting multiple semantic queries...")

sample_queries = [
    "How long do customers have to return an item?",
    "How long does international delivery usually take?",
    "Which payment methods are accepted?"
]

for query in sample_queries:
    results = vectorstore.similarity_search(
        query,
        k=1
    )
    if not results:
        raise ValueError(f"No relevant document chunks were retrieved for query: {query}")
    top_result = results[0]

    print("\n" + "=" * 60)
    print("Query:")
    print(query)

    print("\nTop Matching Chunk:")
    print(top_result.page_content)

    print("\nSource:")
    print(top_result.metadata.get("source"))

    print("Page:")
    print(top_result.metadata.get("page"))
    
# Groq is used for final answer generation.
# Hugging Face is used for local embeddings.
# FAISS is used for semantic retrieval.

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

print("\nGroq model initialized successfully.")
user_query = "How long do customers have to return an item?"

retrieved_docs = vectorstore.similarity_search(
    user_query,
    k=2
)
if not retrieved_docs:
    raise ValueError("No relevant documents were retrieved for the final RAG answer.")

context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

prompt = f"""
You are a document assistant.

Answer the question using only the context below.
If the answer is not available in the context, say:
"The information is not available in the provided documents."

Context:
{context}

Question:
{user_query}

Answer:
"""

response = llm.invoke(prompt)

print("\nFinal RAG Answer:")
print(response.content)