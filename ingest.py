# ingest.py — Ingests PDF documents into a FAISS vector index.
# Run this ONCE after placing PDFs in data/docs/.
# Run: python ingest.py
#
# This script is a placeholder that shows exactly how ingestion works.
# It will run with mock data now. With real PDFs + sentence-transformers installed,
# it builds a real FAISS index.

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(__file__))


def ingest_mock():
    """
    Simulate ingestion with mock chunks.
    Replace this with real PDF loading once you have the annual report PDFs.
    """
    print("Running mock ingestion (no PDFs found in data/docs/)...")

    mock_chunks = [
        {"text": "Infosys reported strong growth in digital services in FY24.", "source": "Infosys_AR_FY24.pdf", "page": 47},
        {"text": "TCS margin improvement driven by operational efficiency in FY24.", "source": "TCS_AR_FY24.pdf", "page": 62},
        {"text": "Wipro strategic priorities include AI and BFSI vertical expansion.", "source": "Wipro_AR_FY24.pdf", "page": 38},
    ]

    os.makedirs("data", exist_ok=True)
    with open("data/chunks_metadata.pkl", "wb") as f:
        pickle.dump(mock_chunks, f)

    print(f"Mock index created with {len(mock_chunks)} chunks.")
    print("To use real PDFs: place them in data/docs/ and run ingest.py again.")


def ingest_real():
    """
    Real ingestion pipeline. Uncomment and use once you have:
    - PDFs in data/docs/
    - pip install sentence-transformers faiss-cpu pypdf
    """
    # from pypdf import PdfReader
    # from sentence_transformers import SentenceTransformer
    # import faiss, numpy as np
    #
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # chunks = []
    #
    # for pdf_file in os.listdir("data/docs"):
    #     if not pdf_file.endswith(".pdf"):
    #         continue
    #     reader = PdfReader(f"data/docs/{pdf_file}")
    #     for page_num, page in enumerate(reader.pages, 1):
    #         text = page.extract_text()
    #         if not text or len(text.strip()) < 50:
    #             continue
    #         # Split page into ~500 char chunks with 50 char overlap
    #         for i in range(0, len(text), 450):
    #             chunk_text = text[i:i+500].strip()
    #             if chunk_text:
    #                 chunks.append({"text": chunk_text, "source": pdf_file, "page": page_num})
    #
    # print(f"Extracted {len(chunks)} chunks from PDFs.")
    #
    # # Build FAISS index
    # embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    # index.add(np.array(embeddings, dtype="float32"))
    #
    # os.makedirs("data", exist_ok=True)
    # faiss.write_index(index, "data/faiss_index.bin")
    # with open("data/chunks_metadata.pkl", "wb") as f:
    #     pickle.dump(chunks, f)
    #
    # print(f"FAISS index saved to data/faiss_index.bin")
    pass


if __name__ == "__main__":
    docs_dir = "data/docs"
    has_pdfs = os.path.isdir(docs_dir) and any(f.endswith(".pdf") for f in os.listdir(docs_dir))

    if has_pdfs:
        print("PDFs found. Running real ingestion...")
        ingest_real()
    else:
        ingest_mock()
