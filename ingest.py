# ingest.py — Ingests PDFs from data/docs/ into a FAISS vector index.
# Run ONCE after placing PDFs in data/docs/.
# Run: venv/bin/python ingest.py

import os
import sys
import pickle
import re

sys.path.insert(0, os.path.dirname(__file__))


def clean_text(text: str) -> str:
    """Remove noise: excessive whitespace, page numbers, headers/footers."""
    # Collapse multiple newlines/spaces
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def chunk_text(text: str, source: str, page: int, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split a page's text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        # Skip chunks that are too short — likely noise
        if len(chunk) > 100:
            chunks.append({"text": chunk, "source": source, "page": page})
        start += chunk_size - overlap
    return chunks


def ingest():
    docs_dir = "data/docs"
    pdfs = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]

    if not pdfs:
        print("No PDFs found in data/docs/. Exiting.")
        return

    print(f"Found {len(pdfs)} PDFs: {pdfs}")

    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []

    for pdf_file in pdfs:
        path = os.path.join(docs_dir, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        try:
            reader = PdfReader(path)
            total_pages = len(reader.pages)
            print(f"  Pages: {total_pages}")
            file_chunks = 0

            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                except Exception:
                    continue

                if not text or len(text.strip()) < 100:
                    # Skip near-empty pages (covers, TOC entries, etc.)
                    continue

                text = clean_text(text)
                chunks = chunk_text(text, source=pdf_file, page=page_num)
                all_chunks.extend(chunks)
                file_chunks += len(chunks)

            print(f"  Chunks extracted: {file_chunks}")

        except Exception as e:
            print(f"  ERROR reading {pdf_file}: {e}")

    if not all_chunks:
        print("No chunks extracted. Check your PDFs.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Building embeddings (this takes a few minutes)...")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/faiss_index.bin")
    with open("data/chunks_metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nDone.")
    print(f"  FAISS index  → data/faiss_index.bin ({index.ntotal} vectors)")
    print(f"  Metadata     → data/chunks_metadata.pkl ({len(all_chunks)} chunks)")
    print("\nVerifying retrieval quality...")
    _verify(model, index, all_chunks)


def _verify(model, index, chunks):
    """Run 5 test queries and print top result — manual quality check."""
    import numpy as np
    test_queries = [
        "operating margin improvement reasons",
        "revenue growth digital services",
        "headcount employees workforce",
        "strategic priorities AI cloud",
        "management commentary FY24 outlook",
    ]
    for q in test_queries:
        vec = model.encode([q]).astype("float32")
        _, indices = index.search(vec, k=1)
        top = chunks[indices[0][0]]
        print(f"\n  Query: '{q}'")
        print(f"  Top chunk: [{top['source']} p.{top['page']}]")
        print(f"  Text: {top['text'][:120]}...")


if __name__ == "__main__":
    ingest()
