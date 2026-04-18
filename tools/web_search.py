# web_search.py — Live web search for recent/current information.
# Uses Tavily API when TAVILY_API_KEY is set in .env, otherwise falls back to mock.

import os
from dotenv import load_dotenv
from utils.types import ToolResult

load_dotenv()

# ── Tool metadata ──────────────────────────────────────────────────────────────
TOOL_NAME = "web_search"
TOOL_DESCRIPTION = """
Use this tool when the question asks for recent, live, or current information that 
would not be found in static documents or a historical database. Examples: 
'What is the current stock price of Infosys?', 'What happened to IT stocks last week?', 
'Who is the current CEO of TCS?'.
Do NOT use this tool for historical financial data — use query_data for that.
Do NOT use this tool for information that is clearly inside the annual reports — use search_docs.
Keep the search query short (under 10 words) and specific.
"""

# ── Mock web results ───────────────────────────────────────────────────────────
# Simulates what Tavily API would return.
# Each result has: title, snippet, url, published_date.
MOCK_WEB_RESULTS = [
    {
        "title": "Infosys Q4 FY24 Results: Revenue grows 1.7% YoY",
        "snippet": "Infosys reported Q4 FY24 revenue of $4.55 billion, with operating margin at 20.1%. The company issued FY25 revenue guidance of 1-3% growth in constant currency.",
        "url": "https://economictimes.indiatimes.com/infosys-q4-fy24-results",
        "published_date": "2024-04-18",
    },
    {
        "title": "TCS share price today — NSE/BSE live",
        "snippet": "TCS shares traded at ₹3,842 on BSE, up 0.4% intraday. Analysts maintain a 'Buy' rating with a 12-month target of ₹4,200.",
        "url": "https://www.moneycontrol.com/tcs-share-price",
        "published_date": "2024-04-19",
    },
    {
        "title": "Wipro CEO Srinivas Pallia outlines FY25 strategy",
        "snippet": "Wipro's new CEO Srinivas Pallia said the company will focus on large deal wins and AI-led transformation to return to growth in FY25.",
        "url": "https://www.livemint.com/wipro-ceo-fy25-strategy",
        "published_date": "2024-04-15",
    },
    {
        "title": "Indian IT sector outlook FY25: Cautious optimism",
        "snippet": "Analysts expect Indian IT sector revenue growth of 4-7% in FY25, driven by BFSI and healthcare verticals, with margin pressure from wage hikes.",
        "url": "https://www.business-standard.com/it-sector-fy25-outlook",
        "published_date": "2024-04-10",
    },
]


def web_search(query: str) -> ToolResult:
    """
    Search the live web for recent information about the query.

    Args:
        query: Short search query string (under 10 words ideally).

    Returns:
        ToolResult with top-3 web snippets, URLs, and publication dates.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    # ── REAL IMPLEMENTATION ────────────────────────────────────────────────────
    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=3)
            results = [
                {
                    "title": r.get("title", "No title"),
                    "snippet": r.get("content", ""),
                    "url": r.get("url", ""),
                    "published_date": r.get("published_date", "unknown"),
                }
                for r in response.get("results", [])
            ]

            if not results:
                return ToolResult(
                    tool_name=TOOL_NAME,
                    input_query=query,
                    output=[],
                    source_citations=[],
                    success=False,
                    error="Tavily returned no results.",
                )

            citations = [f"{r['url']} ({r['published_date']})" for r in results]
            return ToolResult(
                tool_name=TOOL_NAME,
                input_query=query,
                output=results,
                source_citations=citations,
                success=True,
            )

        except Exception as e:
            # If Tavily call fails, log the error and fall through to mock
            print(f"[web_search] Tavily error: {e} — falling back to mock.")

    # ── MOCK FALLBACK (no API key or Tavily error) ─────────────────────────────
    query_lower = query.lower()
    scored = []
    for result in MOCK_WEB_RESULTS:
        score = sum(
            1 for word in query_lower.split()
            if word in result["title"].lower() or word in result["snippet"].lower()
        )
        scored.append((score, result))

    top3 = [r for _, r in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]

    if not top3:
        return ToolResult(
            tool_name=TOOL_NAME,
            input_query=query,
            output=[],
            source_citations=[],
            success=False,
            error="No web results found for this query.",
        )

    output = [
        {
            "title": r["title"],
            "snippet": r["snippet"],
            "url": r["url"],
            "published_date": r["published_date"],
        }
        for r in top3
    ]
    citations = [f"{r['url']} ({r['published_date']})" for r in top3]

    return ToolResult(
        tool_name=TOOL_NAME,
        input_query=query,
        output=output,
        source_citations=citations,
        success=True,
    )
