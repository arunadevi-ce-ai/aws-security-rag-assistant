"""
Build Vector Store — AWS Security RAG Assistant
-------------------------------------------------
This script is the "one-time setup" step of the RAG pipeline:
  1. Load a PDF document (AWS Well-Architected Security Pillar whitepaper)
  2. Split it into small overlapping chunks
  3. Convert each chunk into a vector embedding (a numeric representation
     of its meaning) using OpenAI's embedding model
  4. Save those vectors into a local Chroma vector database on disk

Run this ONCE (or again only if you change/replace the source PDF).
Once ./chroma_db exists, use ask_rag.py to ask questions — that script
just reconnects to this saved database, it does NOT re-embed anything.
"""

import time
from dotenv import load_dotenv
import os
# Load OPENAI_API_KEY (and any other secrets) from .env into the environment.
# Must run before any OpenAI client/embeddings object is created.
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Path to the source PDF. Change this if you use a different document.
pdf_path=os.environ.get("DOCUMENT")

# ---------------------------------------------------------------------------
# STEP 1: Load the PDF
# ---------------------------------------------------------------------------
# PyPDFLoader extracts text page-by-page and returns a list of Document
# objects, each with .page_content (the text) and .metadata (e.g. page number).
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Loaded {len(pages)} pages from {pdf_path}")

# ---------------------------------------------------------------------------
# STEP 2: Split into smaller chunks
# ---------------------------------------------------------------------------
# Why chunk at all? A single page can cover multiple unrelated topics, which
# makes embeddings less precise. Smaller, focused chunks retrieve more
# accurately when we later search for a specific question's answer.
#
# chunk_size=1000      -> roughly 1000 characters per chunk
# chunk_overlap=150    -> chunks overlap slightly so we don't accidentally
#                         cut a sentence or idea in half between two chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# ---------------------------------------------------------------------------
# STEP 3: Set up the embeddings model
# ---------------------------------------------------------------------------
# This model converts text into a vector (a list of numbers) that captures
# its meaning. Similar meanings end up with similar vectors, which is what
# lets us later search "by meaning" instead of by exact keyword match.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# STEP 4: Create (or open) the local vector database
# ---------------------------------------------------------------------------
# Chroma is a lightweight, local vector database. persist_directory tells it
# where to save to disk, so the embeddings survive after this script ends —
# you don't need to regenerate them every time you want to ask a question.
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# ---------------------------------------------------------------------------
# STEP 5: Embed and store the chunks IN BATCHES
# ---------------------------------------------------------------------------
# Why batch instead of sending all chunks in one call?
# OpenAI enforces a tokens-per-minute (TPM) limit on embedding requests
# (e.g. 40,000 TPM on some account tiers). Sending all ~665 chunks at once
# can easily exceed that limit and trigger a 429 rate-limit error.
#
# Sending smaller batches, with a short pause between them, keeps each
# request comfortably under the per-minute token limit.
batch_size = 50  # number of chunks per embedding request

total_batches = (len(chunks) // batch_size) + 1

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)  # embeds this batch and saves it to disk
    print(f"Embedded batch {i // batch_size + 1} of {total_batches}")
    time.sleep(20)  # pause so consecutive batches don't exceed the TPM limit

print("Done — vector store saved to ./chroma_db")
