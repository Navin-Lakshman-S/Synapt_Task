# query_data.py — Query structured financial data from SQLite.
# Uses data/financials.db built by data/build_db.py (real published numbers).
# Falls back to seeding a minimal in-memory DB if the file doesn't exist.

import sqlite3
import os
import re
from utils.types import ToolResult

# ── Tool metadata ──────────────────────────────────────────────────────────────
TOOL_NAME = "query_data"
TOOL_DESCRIPTION = """
Use this tool when the question asks for specific numbers, statistics, comparisons,
or trends from structured financial data. Examples: 'What was Infosys revenue in FY24?',
'Compare operating margins of TCS and Wipro over 4 years', 'Which company had the
highest EPS in FY23?', 'Show quarterly profit for TCS in FY24'.
Do NOT use this tool for qualitative explanations or reasons — use search_docs for those.
Do NOT use this tool for live or recent data not in the database — use web_search for those.
Input should be a natural language question about the data.
"""


def _get_connection() -> sqlite3.Connection:
    """Use real DB if available, otherwise seed a minimal in-memory fallback."""
    db_path = "data/financials.db"
    if os.path.exists(db_path):
        return sqlite3.connect(db_path)
    # Fallback — run data/build_db.py to generate the real DB
    raise FileNotFoundError(
        "data/financials.db not found. Run: venv/bin/python data/build_db.py"
    )


_SCHEMA_DESCRIPTION = """Table: financials
Columns:
  id INTEGER PRIMARY KEY AUTOINCREMENT
  company TEXT  -- values: 'Infosys', 'TCS', 'Wipro'
  fiscal_year TEXT  -- annual: 'FY15'–'FY24'; quarterly: 'Q1FY22'–'Q4FY24'
  type TEXT  -- 'annual' or 'quarterly'
  revenue_cr REAL
  expenses_cr REAL
  operating_profit_cr REAL
  opm_pct REAL  -- operating margin percentage
  other_income_cr REAL
  depreciation_cr REAL
  interest_cr REAL
  net_profit_cr REAL
  eps REAL
  headcount INTEGER"""


def _llm_nl_to_sql(query: str) -> str:
    """
    Use the LLM to translate a natural language question into a SQL SELECT statement.

    Raises:
        ValueError: if the response does not start with SELECT.
    """
    from agent.llm import call_llm

    prompt = (
        f"You are a SQL expert. Given the following table schema:\n\n"
        f"{_SCHEMA_DESCRIPTION}\n\n"
        f"Write a valid SQLite SELECT statement that answers this question:\n"
        f"{query}\n\n"
        f"Return ONLY the SQL statement. No markdown, no explanation, no code fences."
    )

    raw = call_llm(prompt, temperature=0.1)

    # Strip markdown code fences
    raw = re.sub(r"^```sql\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    if not raw.upper().startswith("SELECT"):
        raise ValueError(f"LLM returned a non-SELECT response: {raw!r}")

    return raw


def _rule_based_nl_to_sql(query: str) -> str:
    """
    Translate natural language to SQL against the financials table.
    Rule-based fallback — used when GEMINI_API_KEY is not set or Gemini fails.

    Table schema:
        company, fiscal_year, type (annual/quarterly),
        revenue_cr, expenses_cr, operating_profit_cr, opm_pct,
        other_income_cr, depreciation_cr, interest_cr,
        net_profit_cr, eps, headcount
    """
    q = query.lower()

    # ── Company filter ─────────────────────────────────────────────────────────
    companies = []
    if "infosys" in q:
        companies.append("Infosys")
    if "tcs" in q:
        companies.append("TCS")
    if "wipro" in q:
        companies.append("Wipro")

    if len(companies) == 1:
        company_filter = f"company = '{companies[0]}'"
    elif len(companies) > 1:
        names = ", ".join(f"'{c}'" for c in companies)
        company_filter = f"company IN ({names})"
    else:
        company_filter = ""

    # ── Year / quarter filter ──────────────────────────────────────────────────
    year_filter = ""

    # Check for specific quarter label first e.g. "Q1FY24"
    for qtr in ["q1","q2","q3","q4"]:
        if qtr in q:
            for yr in ["fy21","fy22","fy23","fy24"]:
                if yr in q:
                    label = f"{qtr.upper()}{yr.upper()}"
                    year_filter = f"fiscal_year = '{label}'"
                    break

    # If asking for all quarters of a year e.g. "quarterly FY24" → match Q*FY24
    if not year_filter:
        for yr in ["FY15","FY16","FY17","FY18","FY19","FY20","FY21","FY22","FY23","FY24"]:
            if yr.lower() in q:
                if "quarter" in q or any(f"q{n}" in q for n in [1,2,3,4]):
                    # Match all quarters of that year
                    year_filter = f"fiscal_year LIKE '%{yr}'"
                else:
                    year_filter = f"fiscal_year = '{yr}'"
                break

    # ── Data type filter ───────────────────────────────────────────────────────
    if "quarterly" in q or "quarter" in q or any(f"q{n}" in q for n in [1,2,3,4]):
        type_filter = "type = 'quarterly'"
    elif "annual" in q or "yearly" in q:
        type_filter = "type = 'annual'"
    else:
        type_filter = "type = 'annual'"  # default to annual for cleaner answers

    # ── Combine WHERE clauses ──────────────────────────────────────────────────
    filters = [f for f in [company_filter, year_filter, type_filter] if f]
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    # ── Select the right columns ───────────────────────────────────────────────
    if "revenue" in q or "sales" in q or "turnover" in q:
        cols = "company, fiscal_year, type, revenue_cr"
    elif "operating margin" in q or "opm" in q or "op margin" in q:
        cols = "company, fiscal_year, type, operating_profit_cr, opm_pct"
    elif "operating profit" in q:
        cols = "company, fiscal_year, type, operating_profit_cr, opm_pct"
    elif "net profit" in q or "profit" in q or "earnings" in q:
        cols = "company, fiscal_year, type, net_profit_cr"
    elif "eps" in q or "earnings per share" in q:
        cols = "company, fiscal_year, type, eps"
    elif "headcount" in q or "employees" in q or "workforce" in q or "staff" in q:
        cols = "company, fiscal_year, type, headcount"
    elif "expense" in q or "cost" in q:
        cols = "company, fiscal_year, type, expenses_cr"
    elif "depreciation" in q:
        cols = "company, fiscal_year, type, depreciation_cr"
    elif "interest" in q:
        cols = "company, fiscal_year, type, interest_cr"
    elif "compare" in q or "all" in q or "overview" in q or "summary" in q:
        cols = "company, fiscal_year, type, revenue_cr, opm_pct, net_profit_cr, eps"
    else:
        cols = "company, fiscal_year, type, revenue_cr, opm_pct, net_profit_cr, eps"

    return f"SELECT {cols} FROM financials {where} ORDER BY company, fiscal_year"


def _nl_to_sql(query: str) -> str:
    """
    Translate natural language to SQL.
    Tries Gemini if GEMINI_API_KEY is set; falls back to rule-based on any exception.
    """
    if os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY"):
        try:
            return _llm_nl_to_sql(query)
        except Exception:
            pass
    return _rule_based_nl_to_sql(query)


def query_data(query: str) -> ToolResult:
    """
    Query the structured financials database using a natural language question.

    Args:
        query: Natural language question about financial data.

    Returns:
        ToolResult with rows, column names, and source citation.
    """
    try:
        sql = _nl_to_sql(query)
        conn = _get_connection()
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return ToolResult(
                tool_name=TOOL_NAME,
                input_query=query,
                output={"sql": sql, "rows": [], "columns": columns},
                source_citations=[],
                success=False,
                error="Query returned no results. The data may not exist in the database.",
            )

        output = {
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
        }
        citations = [f"financials.db | SQL: {sql}"]

        return ToolResult(
            tool_name=TOOL_NAME,
            input_query=query,
            output=output,
            source_citations=citations,
            success=True,
        )

    except sqlite3.Error as e:
        return ToolResult(
            tool_name=TOOL_NAME,
            input_query=query,
            output={},
            source_citations=[],
            success=False,
            error=str(e),
        )
    except Exception as e:
        return ToolResult(
            tool_name=TOOL_NAME,
            input_query=query,
            output={},
            source_citations=[],
            success=False,
            error=str(e),
        )
