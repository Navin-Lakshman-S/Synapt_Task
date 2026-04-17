# search_docs.py — Semantic search over unstructured documents (PDFs).
# Right now it uses mock data. When you have real PDFs, swap in the FAISS retriever below.

from utils.types import ToolResult

# ── Tool metadata ──────────────────────────────────────────────────────────────
# This description is written FOR the LLM decision engine.
# It tells the model WHEN to use this tool and — critically — WHEN NOT TO.
TOOL_NAME = "search_docs"
TOOL_DESCRIPTION = """
Use this tool when the question asks about qualitative information, explanations, 
strategies, management commentary, or reasons found inside company annual reports or 
PDF documents. Examples: 'What reason did TCS give for margin improvement?', 
'What are Infosys strategic priorities?'.
Do NOT use this tool for exact numbers, statistics, or year-over-year comparisons — 
use query_data for those. Do NOT use this tool for recent news or live data — 
use web_search for those.
"""

# ── Mock document chunks ───────────────────────────────────────────────────────
# These simulate what a real FAISS vector store would return.
# Each chunk has: text, source filename, page number.
# Replace this list with real FAISS retrieval once PDFs are ingested.
MOCK_CHUNKS = [
    {
        "text": "Infosys reported strong growth in digital services, driven by cloud migration and AI adoption among enterprise clients. The management highlighted cost optimisation as a key lever for margin expansion in FY24.",
        "source": "Infosys_AR_FY24.pdf",
        "page": 47,
    },
    {
        "text": "TCS attributed its margin improvement in FY24 to operational efficiency gains, lower subcontracting costs, and a favourable revenue mix shift towards higher-margin consulting engagements.",
        "source": "TCS_AR_FY24.pdf",
        "page": 62,
    },
    {
        "text": "Wipro's strategic priorities for FY24 included deepening client relationships in the BFSI vertical, expanding its AI and automation practice, and improving employee utilisation rates.",
        "source": "Wipro_AR_FY24.pdf",
        "page": 38,
    },
    {
        "text": "Infosys headcount stood at 317,240 employees at the end of FY24, reflecting a net reduction due to lower fresher intake and attrition normalisation post-pandemic.",
        "source": "Infosys_AR_FY24.pdf",
        "page": 12,
    },
    {
        "text": "TCS maintained its industry-leading attrition rate of 12.5% in FY24, which the management credited to its strong learning and development culture and structured career progression framework.",
        "source": "TCS_AR_FY24.pdf",
        "page": 29,
    },
]


def search_docs(query: str) -> ToolResult:
    """
    Search unstructured documents for text relevant to the query.

    Args:
        query: Natural language question or keyword string.

    Returns:
        ToolResult with top-3 matching chunks, source filenames, and page numbers.
    """
    # ── REAL IMPLEMENTATION (plug in when ready) ───────────────────────────────
    # Uncomment and replace the mock below with this once FAISS index is built:
    #
    # import faiss, pickle, numpy as np
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # index = faiss.read_index("data/faiss_index.bin")
    # with open("data/chunks_metadata.pkl", "rb") as f:
    #     metadata = pickle.load(f)
    #
    # query_vec = model.encode([query])
    # distances, indices = index.search(np.array(query_vec, dtype="float32"), k=3)
    # results = [metadata[i] for i in indices[0]]
    # ──────────────────────────────────────────────────────────────────────────

    # Mock: simple keyword matching to simulate semantic relevance
    query_lower = query.lower()
    scored = []
    for chunk in MOCK_CHUNKS:
        # Score = number of query words found in the chunk text (naive but explainable)
        score = sum(1 for word in query_lower.split() if word in chunk["text"].lower())
        scored.append((score, chunk))

    # Sort by score descending, take top 3
    top3 = [chunk for _, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]

    if not top3:
        return ToolResult(
            tool_name=TOOL_NAME,
            input_query=query,
            output=[],
            source_citations=[],
            success=False,
            error="No relevant chunks found.",
        )

    # Format output as a list of dicts — consistent shape the agent can always parse
    output = [
        {"text": c["text"], "source": c["source"], "page": c["page"]}
        for c in top3
    ]
    citations = [f"{c['source']} p.{c['page']}" for c in top3]

    return ToolResult(
        tool_name=TOOL_NAME,
        input_query=query,
        output=output,
        source_citations=citations,
        success=True,
    )
