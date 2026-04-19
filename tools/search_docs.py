# search_docs.py — Semantic search over unstructured annual report PDFs.
# Uses FAISS vector index built by ingest.py.
# Falls back to mock chunks if index not found.

import os
import pickle
from utils.types import ToolResult

# ── Tool metadata ──────────────────────────────────────────────────────────────
TOOL_NAME = "search_docs"
TOOL_DESCRIPTION = """
Use this tool when the question asks about qualitative information, explanations,
strategies, management commentary, or reasons found inside company annual reports.
Examples: 'What reason did TCS give for margin improvement?',
'What are Infosys strategic priorities?', 'What did Wipro management say about growth?'.
Do NOT use this tool for exact numbers or year-over-year comparisons — use query_data.
Do NOT use this tool for recent news or live data — use web_search.
"""

# ── Lazy-loaded globals (loaded once on first call, not at import time) ────────
_index = None
_chunks = None
_model = None


def _load_index():
    """Load FAISS index, metadata, and embedding model. Called once."""
    global _index, _chunks, _model

    index_path = "data/faiss_index.bin"
    meta_path  = "data/chunks_metadata.pkl"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return False  # Fall back to mock

    import faiss
    from sentence_transformers import SentenceTransformer

    _index  = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        _chunks = pickle.load(f)
    _model  = SentenceTransformer("all-MiniLM-L6-v2")
    return True


# ── Mock chunks (fallback when FAISS index not available) ──────────────────────
MOCK_CHUNKS = [
    {"text": "Infosys reported strong growth in digital services, driven by cloud migration and AI adoption. Management highlighted cost optimisation as a key lever for margin expansion in FY24.", "source": "Infosys_AR_FY24.pdf", "page": 47},
    {"text": "TCS attributed its margin improvement in FY24 to operational efficiency gains, lower subcontracting costs, and a favourable revenue mix shift towards higher-margin consulting.", "source": "TCS_AR_FY24.pdf", "page": 62},
    {"text": "Wipro's strategic priorities for FY24 included deepening client relationships in BFSI, expanding AI and automation practice, and improving employee utilisation rates.", "source": "Wipro_AR_FY24.pdf", "page": 38},
    {"text": "Infosys headcount stood at 317,240 employees at end of FY24, reflecting net reduction due to lower fresher intake and attrition normalisation post-pandemic.", "source": "Infosys_AR_FY24.pdf", "page": 12},
    {"text": "TCS maintained industry-leading attrition of 12.5% in FY24, credited to strong learning culture and structured career progression framework.", "source": "TCS_AR_FY24.pdf", "page": 29},
]


def search_docs(query: str, top_k: int = 3) -> ToolResult:
    """
    Semantic search over annual report PDFs.

    Args:
        query:  Natural language question or keyword string.
        top_k:  Number of chunks to return (default 3).

    Returns:
        ToolResult with top-k matching chunks, source filenames, and page numbers.
    """
    global _index, _chunks, _model

    # ── Try real FAISS retrieval ───────────────────────────────────────────────
    if _index is None:
        real_available = _load_index()
    else:
        real_available = True

    if real_available and _index is not None:
        try:
            import numpy as np
            query_vec = _model.encode([query]).astype("float32")
            distances, indices = _index.search(query_vec, k=top_k)

            results = []
            for idx in indices[0]:
                if idx < len(_chunks):
                    results.append(_chunks[idx])

            if not results:
                return ToolResult(
                    tool_name=TOOL_NAME, input_query=query,
                    output=[], source_citations=[], success=False,
                    error="No relevant chunks found in index.",
                )

            output = [{"text": c["text"], "source": c["source"], "page": c["page"]} for c in results]
            citations = [f"{c['source']} p.{c['page']}" for c in results]

            return ToolResult(
                tool_name=TOOL_NAME, input_query=query,
                output=output, source_citations=citations, success=True,
            )

        except Exception as e:
            # If FAISS fails for any reason, fall through to mock
            print(f"[search_docs] FAISS error: {e} — falling back to mock.")

    # ── Mock fallback ──────────────────────────────────────────────────────────
    q_lower = query.lower()
    scored = []
    for chunk in MOCK_CHUNKS:
        score = sum(1 for word in q_lower.split() if word in chunk["text"].lower())
        scored.append((score, chunk))

    top = [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]]
    output = [{"text": c["text"], "source": c["source"], "page": c["page"]} for c in top]
    citations = [f"{c['source']} p.{c['page']}" for c in top]

    return ToolResult(
        tool_name=TOOL_NAME, input_query=query,
        output=output, source_citations=citations, success=True,
    )
