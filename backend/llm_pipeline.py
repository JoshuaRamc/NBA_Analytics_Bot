import os
import glob
import re
from typing import List, Optional, Dict, Any, Tuple, Union

import pandas as pd
from dotenv import load_dotenv

# Heuristic charting (no LLM)
from chart_agent import get_chart_spec

# LangChain (Gemini) — used ONLY for text insights & planning, not for charts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser  # ✅ pydantic v2–friendly

# ✅ modern pydantic v2
from pydantic import BaseModel as PydBaseModel, Field, ValidationError

# -----------------------------
# Env & Model
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Create backend/.env with GEMINI_API_KEY=...")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

# -----------------------------
# Data discovery & load
# -----------------------------
BACKEND_DIR = os.path.dirname(__file__)

def _discover_dataset_path() -> str:
    for var in ("DATA_FILE", "DATA_CSV_PATH"):
        env_path = os.getenv(var)
        if env_path and os.path.exists(env_path):
            return env_path

    preferred = os.path.join(BACKEND_DIR, "nba_2024-2025_player_stats.xls")
    if os.path.exists(preferred):
        return preferred

    candidates = [
        "nba_2024-2025_player_stats.xlsx",
        "nba_2024_25_per_game.csv",
        "nba_2024_25_per_game.tsv",
        "nba.csv",
        "nba.tsv",
    ]
    for name in candidates:
        p = os.path.join(BACKEND_DIR, name)
        if os.path.exists(p):
            return p

    for pattern in ("*.csv", "*.tsv", "*.xls", "*.xlsx", "*.html", "*.htm"):
        files = sorted(glob.glob(os.path.join(BACKEND_DIR, pattern)))
        if files:
            return files[0]

    raise RuntimeError(
        "Could not find a dataset file. Place your NBA CSV/TSV/XLS/XLSX/HTML "
        "in the backend folder (same dir as main.py) or set DATA_FILE in .env."
    )

DATA_FILE = _discover_dataset_path()

def _maybe_is_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(2048).lower()
        return b"<html" in head or b"<table" in head
    except Exception:
        return False

def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if _maybe_is_html(path):
        tables = pd.read_html(path, flavor="lxml")
        if not tables:
            raise RuntimeError("HTML file detected but no tables found.")
        return tables[0]

    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if ext in [".tsv", ".csv"]:
        sep = "\t" if ext == ".tsv" else None
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")

    try:
        return pd.read_csv(path, engine="python", encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"Unable to load dataset from {path}: {e}")

df_full = _load_table(DATA_FILE)

def _normalize_cols(cols: List[str]) -> List[str]:
    normalized = []
    for c in cols:
        # Add debug logging
        print(f"Processing column: {c}, type: {type(c)}")
        
        if isinstance(c, str):
            normalized.append(c.strip().replace("%", "Pct").replace("/", "_per_").replace(" ", "_"))
        else:
            # Convert non-strings to strings first
            str_col = str(c)
            normalized.append(str_col.strip().replace("%", "Pct").replace("/", "_per_").replace(" ", "_"))
    return normalized

# Add debug print to see what we're working with
print("Original columns:", df_full.columns.tolist())
print("Column types:", [type(col) for col in df_full.columns.tolist()])

df_full.columns = _normalize_cols(df_full.columns.tolist())

# Identify columns
PLAYER_COL_CANDIDATES = [c for c in df_full.columns if str(c).lower() in ["player", "player_name"]]
TEAM_COL_CANDIDATES   = [c for c in df_full.columns if str(c).lower() in ["team", "tm", "team_abbr", "team_id"]]
PLAYER_COL = PLAYER_COL_CANDIDATES[0] if PLAYER_COL_CANDIDATES else df_full.columns[0]
TEAM_COL   = TEAM_COL_CANDIDATES[0] if TEAM_COL_CANDIDATES else None

ALL_PLAYERS: List[str]  = sorted(df_full[PLAYER_COL].dropna().astype(str).unique().tolist())
ALL_COLUMNS: List[str]  = df_full.columns.tolist()

# Metric whitelist
NUMERIC_METRICS_WHITELIST = [
    "PTS", "TRB", "AST", "STL", "BLK", "MP", "FG", "FGA", "FGPct", "3P", "3PA",
    "3PPct", "2P", "2PA", "2PPct", "eFGPct", "FT", "FTA", "FTPct", "ORB", "DRB", "TOV"
]
NUMERIC_METRICS_WHITELIST = [m for m in NUMERIC_METRICS_WHITELIST if m in ALL_COLUMNS]

# Friendly aliases → dataset columns
METRIC_ALIASES = {
    # points
    "points": "PTS", "ppg": "PTS", "scoring": "PTS",
    # rebounds
    "rebounds": "TRB", "rpg": "TRB", "boards": "TRB",
    # assists
    "assists": "AST", "apg": "AST", "dimes": "AST",
    # steals/blocks/turnovers
    "steals": "STL", "spg": "STL", "blocks": "BLK", "bpg": "BLK", "turnovers": "TOV",
    # 3pt %
    "3pt%": "3PPct", "3p%": "3PPct", "3pt percentage": "3PPct", "three point %": "3PPct",
    "three-point %": "3PPct", "three point percentage": "3PPct", "three-point percentage": "3PPct",
    # fg%
    "fg%": "FGPct", "field goal %": "FGPct", "field-goal %": "FGPct", "field goal percentage": "FGPct",
    # ft%
    "ft%": "FTPct", "free throw %": "FTPct", "free-throw %": "FTPct", "free throw percentage": "FTPct",
    # efg
    "efg%": "eFGPct", "effective field goal %": "eFGPct",
    # 3P attempts/makes
    "3pa": "3PA", "3p attempts": "3PA", "3pm": "3P", "3p makes": "3P",
}

# -----------------------------
# Planning schema (pydantic) - FIXED
# -----------------------------
class Filter(PydBaseModel):
    column: str = Field(..., description="Column to filter on (must exist).")
    op: str = Field(..., description="Filter operator: eq, gte, lte, gt, lt, contains")  # Simplified from Literal
    value: Union[str, int, float] = Field(..., description="Value to compare against.")

class QueryPlan(PydBaseModel):
    operation: str = Field(..., description="Plan type: player_lookup, top_n, filter")  # Simplified from Literal
    player_names: List[str] = Field(default_factory=list, description="Player names to include when relevant.")
    metrics: List[str] = Field(default_factory=list, description="Metrics/columns relevant to the question.")
    sort_by: Optional[str] = Field(default=None, description="Metric to sort by for 'top_n'.")
    top_n: Optional[int] = Field(default=10, description="How many rows to return for 'top_n'.")
    filters: List[Filter] = Field(default_factory=list, description="Filters to apply for 'filter'.")
    notes: Optional[str] = Field(default=None, description="Any extra hints for the executor.")

# ✅ Use a JSON parser, then validate with pydantic v2
plan_parser = JsonOutputParser()

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a planner that converts a user's NBA stats question into a simple JSON plan. "
     "The data is per-game for the 2024-25 NBA season. "
     "Use ONLY columns that exist. Player names must be exact strings from the list provided.\n"
     "If the user asks about a specific player, choose operation='player_lookup' and add that name to player_names.\n"
     "If the user asks for leaders/rankings/top, choose operation='top_n' with a numeric sort_by and top_n.\n"
     "Otherwise choose operation='filter' with filters.\n\n"
     "Valid numeric metrics: {numeric_metrics}\n"
     "All columns: {all_columns}\n"
     "Available player names (subset): {player_list}\n\n"
     "{format_instructions}"
    ),
    ("human", "{question}")
])

plan_chain = PLAN_PROMPT | llm | plan_parser

# -----------------------------
# QuerySpec (safe df.query-style) - FIXED
# -----------------------------
class QuerySpec(PydBaseModel):
    """Constrained spec that we compile into pandas operations."""
    where: Optional[str] = Field(default=None, description="A df.query() compatible boolean expression using backticked column names. No @vars.")
    columns: List[str] = Field(default_factory=list, description="Columns to include in the output.")
    sort_by: Optional[str] = Field(default=None, description="Column to sort by.")
    descending: bool = Field(default=True, description="Sort descending?")
    limit: int = Field(default=10, ge=1, le=100, description="Max rows to return (1..100).")

# ✅ Use JSON parser + v2 validation
queryspec_parser = JsonOutputParser()

QUERYSPEC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You generate a strict JSON spec for selecting NBA rows from a pandas DataFrame.\n"
     "The DataFrame columns are: {all_columns}\n\n"
     "Rules:\n"
     "- Use ONLY those columns.\n"
     "- If you filter, put a df.query()-compatible expression in `where`.\n"
     "- Always backtick columns with spaces or symbols (e.g., `3PPct`).\n"
     "- Use only comparison ops: ==, !=, >, >=, <, <=, and, or, not, &, |, ~, parentheses.\n"
     "- DO NOT use @variables, function calls, imports, Python names, or attributes.\n"
     "- If the user asks for top/leader/highest on a metric, set sort_by to that column and limit appropriately.\n"
     "- If the user requests a player, include it in `where` like `Player == \"LeBron James\"` (with correct name).\n"
     "- Output JSON must match the schema.\n"
     "{format_instructions}"
    ),
    ("human", "{question}")
])

queryspec_chain = QUERYSPEC_PROMPT | llm | queryspec_parser

SAFE_QUERY_PATTERN = re.compile(r'^[\w\s`"'"'"'\.\(\)\&\|\~\<\>\=\!\-\+\,\%:/]+$')


def _sanitize_where(expr: Optional[str]) -> Optional[str]:
    if not expr:
        return None
    e = expr.strip()
    # deny obvious unsafe constructs
    forbidden = ("@", "__", "import", "os.", "pd.", "pandas.", "eval", "open(", "read_", "to_")
    if any(tok in e.lower() for tok in forbidden):
        raise ValueError("Disallowed tokens in where expression.")
    # whitelist allowed characters
    if not SAFE_QUERY_PATTERN.match(e):
        raise ValueError("Where expression contains unsupported characters.")
    return e


def _compile_and_run_queryspec(spec: QuerySpec, df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Where
    expr = _sanitize_where(spec.where)
    if expr:
        work = work.query(expr, engine="python")

    # Column projection
    cols = [c for c in spec.columns if c in work.columns]
    if cols:
        work = work[cols]

    # Sorting
    if spec.sort_by and spec.sort_by in work.columns:
        work[spec.sort_by] = pd.to_numeric(work[spec.sort_by], errors="ignore")
        work = work.sort_values(by=spec.sort_by, ascending=not spec.descending)

    # Limit
    work = work.head(max(1, min(100, spec.limit or 10)))
    return work

# -----------------------------
# Insights chain (Markdown)
# -----------------------------
INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are NBA AnalyiXpert. Given a question and a small slice of tabular results (CSV-like text), "
     "write a concise, insightful Markdown response. Focus on the question, highlight key numbers, "
     "and avoid speculation. If nothing matches, say so."),
    ("human",
     "Question:\n{question}\n\nData (up to a few rows):\n{table_text}\n\n"
     "Write a helpful, concise Markdown answer. Use bullet points or a short table if helpful.")
])

insights_chain = INSIGHTS_PROMPT | llm

# -----------------------------
# Heuristics: natural language → simple plan (cheap)
# -----------------------------
TOP_WORDS = ("top", "leader", "leaders", "lead", "led", "highest", "most", "best")


def _resolve_metric(question_lower: str) -> Optional[str]:
    for col in NUMERIC_METRICS_WHITELIST:
        if re.search(rf"\b{re.escape(col.lower())}\b", question_lower):
            return col
    for key, col in METRIC_ALIASES.items():
        if key in question_lower and col in ALL_COLUMNS:
            return col
    if "points" in question_lower or "scor" in question_lower:
        return "PTS" if "PTS" in ALL_COLUMNS else None
    if "rebound" in question_lower:
        return "TRB" if "TRB" in ALL_COLUMNS else None
    if "assist" in question_lower:
        return "AST" if "AST" in ALL_COLUMNS else None
    if ("3pt" in question_lower or "three" in question_lower) and "%" in question_lower:
        return "3PPct" if "3PPct" in ALL_COLUMNS else None
    return None


def _extract_top_n(question_lower: str) -> int:
    m = re.search(r"\btop\s*(\d{1,2})\b", question_lower)
    if m:
        n = int(m.group(1))
        return max(1, min(50, n))
    if any(w in question_lower for w in ("who led", "leader", "highest", "most", "best")):
        return 1
    return 10


def _heuristic_plan(question: str) -> Optional[QueryPlan]:
    ql = question.lower()
    if any(w in ql for w in TOP_WORDS):
        metric = _resolve_metric(ql)
        if metric and metric in NUMERIC_METRICS_WHITELIST:
            n = _extract_top_n(ql)
            return QueryPlan(
                operation="top_n",
                player_names=[],
                metrics=[metric],
                sort_by=metric,
                top_n=n,
                filters=[],
                notes="heuristic top_n"
            )
    return None


def _detect_players_from_text(question: str) -> List[str]:
    ql = question.lower()
    detected = []
    for name in ALL_PLAYERS:
        nm = str(name)
        if nm and nm.lower() in ql:
            detected.append(nm)
    seen = set()
    uniq = []
    for n in detected:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq[:5]

# -----------------------------
# Executors & helpers
# -----------------------------


def _execute_plan(plan: QueryPlan, df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if plan.operation == "player_lookup":
        if plan.player_names:
            names_lower = {p.lower() for p in plan.player_names}
            work = work[work[PLAYER_COL].astype(str).str.lower().isin(names_lower)]
        else:
            work = work.iloc[0:0]

    elif plan.operation == "top_n":
        sort_col = plan.sort_by if plan.sort_by in work.columns else None
        n = plan.top_n or 10
        if sort_col and sort_col in NUMERIC_METRICS_WHITELIST:
            work[sort_col] = pd.to_numeric(work[sort_col], errors="coerce")
            work = work.sort_values(by=sort_col, ascending=False).head(n)
        else:
            work = work.head(n)

    elif plan.operation == "filter":
        for f in plan.filters:
            col = f.column
            if col not in work.columns:
                continue
            if f.op == "eq":
                work = work[work[col].astype(str).str.lower() == str(f.value).lower()]
            elif f.op == "contains":
                work = work[work[col].astype(str).str.contains(str(f.value), case=False, na=False)]
            else:
                series = pd.to_numeric(work[col], errors="coerce")
                val = pd.to_numeric(pd.Series([f.value]), errors="coerce").iloc[0]
                if pd.isna(val):
                    continue
                if f.op == "gte":
                    work = work[series >= val]
                elif f.op == "lte":
                    work = work[series <= val]
                elif f.op == "gt":
                    work = work[series > val]
                elif f.op == "lt":
                    work = work[series < val]

    cols_to_keep = set()
    if plan.metrics:
        for m in plan.metrics:
            if m in work.columns:
                cols_to_keep.add(m)
    for ident in filter(None, [PLAYER_COL, TEAM_COL, "Pos", "Age", "G", "GS"]):
        if ident in work.columns:
            cols_to_keep.add(ident)
    if cols_to_keep:
        work = work[[c for c in work.columns if c in cols_to_keep]]

    return work



def _df_to_llm_text(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "(no rows)"
    trimmed = df.head(max_rows)
    max_cols = min(len(trimmed.columns), 10)
    trimmed = trimmed.iloc[:, :max_cols]
    return trimmed.to_csv(index=False)

# -----------------------------
# Public API used by main.py
# -----------------------------


def ask_nba(question: str) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Natural language → (heuristic plan OR QuerySpec OR LLM plan) → pandas → Markdown insights.
    Returns:
      - markdown insights (str)
      - rows (List[Dict[str, Any]])
      - optional chart_spec (Dict) if the question requested a chart (built heuristically, no LLM)
    """
    # Add input validation and debugging
    print(f"Question type: {type(question)}")
    print(f"Question value: {repr(question)}")
    
    q = (question or "").strip()
    if not isinstance(q, str):
        return ("Invalid question format", [], None)
    
    if not q:
        return ("Please enter a question.", [], None)

    # 0) Try cheap heuristic for leaders/top-N
    plan = _heuristic_plan(q)
    result_df = None

    # 1) If no heuristic plan, try detecting player names directly
    if plan is None:
        detected_players = _detect_players_from_text(q)
        if detected_players:
            plan = QueryPlan(
                operation="player_lookup",
                player_names=detected_players,
                metrics=[], sort_by=None, top_n=None,
                filters=[], notes="heuristic player_lookup"
            )

    # 2) If still no plan, ask the LLM for a *safe* QuerySpec (df.query-style)
    if plan is None:
        try:
            qs_inputs = {
                "question": q,
                "all_columns": ", ".join(ALL_COLUMNS),
                "format_instructions": queryspec_parser.get_format_instructions(),
            }
            print("\n[QUERYSPEC_PROMPT] INPUT:\n", qs_inputs)
            spec_dict: Dict[str, Any] = queryspec_chain.invoke(qs_inputs)
            print("[QUERYSPEC_PROMPT] OUTPUT (raw):\n", spec_dict)
            spec: QuerySpec = QuerySpec.model_validate(spec_dict)
            result_df = _compile_and_run_queryspec(spec, df_full)
        except (ValidationError, ValueError) as e:
            print(f"QuerySpec failed: {e}")
            # 3) If QuerySpec invalid or unsafe, fall back to original planner
            try:
                plan_inputs = {
                    "question": q,
                    "numeric_metrics": ", ".join(NUMERIC_METRICS_WHITELIST),
                    "all_columns": ", ".join(ALL_COLUMNS[:40]) + (" ..." if len(ALL_COLUMNS) > 40 else ""),
                    "player_list": ", ".join(ALL_PLAYERS[:100]) + (" ..." if len(ALL_PLAYERS) > 100 else ""),
                    "format_instructions": plan_parser.get_format_instructions(),
                }
                print("\n[PLAN_PROMPT] INPUT:\n", plan_inputs)
                plan_dict: Dict[str, Any] = plan_chain.invoke(plan_inputs)
                print("[PLAN_PROMPT] OUTPUT (raw):\n", plan_dict)
                plan = QueryPlan.model_validate(plan_dict)
            except Exception as e2:
                print(f"Plan chain failed: {e2}")
                return (f"Error processing question: {str(e2)}", [], None)

    # If we got a DataFrame via QuerySpec path, use it; otherwise execute the plan
    if result_df is None and plan is not None:
        result_df = _execute_plan(plan, df_full)
    elif result_df is None:
        result_df = pd.DataFrame()

    # Fallback: if nothing returned and question looked like leaders/top-N, infer metric
    if result_df.empty and any(w in q.lower() for w in TOP_WORDS):
        inferred_metric = _resolve_metric(q.lower())
        if inferred_metric and inferred_metric in NUMERIC_METRICS_WHITELIST:
            tmp_plan = QueryPlan(
                operation="top_n",
                player_names=[],
                metrics=[inferred_metric],
                sort_by=inferred_metric,
                top_n=_extract_top_n(q.lower()),
                filters=[],
                notes="fallback top_n"
            )
            result_df = _execute_plan(tmp_plan, df_full)

    # Build LLM insights over a compact text view of the results
    table_text = _df_to_llm_text(result_df, max_rows=12)
    try:
        insights_inputs = {"question": q, "table_text": table_text}
        print("\n[INSIGHTS_PROMPT] INPUT:\n", insights_inputs)
        insights = insights_chain.invoke(insights_inputs)
        md = insights.content if hasattr(insights, "content") else str(insights)
        print("[INSIGHTS_PROMPT] OUTPUT (markdown):\n", md)
    except Exception as e:
        md = f"Found {len(result_df)} results but couldn't generate insights: {str(e)}"
        print("[INSIGHTS_PROMPT] ERROR:\n", md)
    
    rows_out = result_df.head(50).to_dict(orient="records")

    # ✅ Build chart spec with NO LLM — heuristic only
    chart_spec = None
    try:
        chart_spec = get_chart_spec(q, result_df)
    except Exception as e:
        print(f"Chart generation failed: {e}")
        pass  # Chart generation is optional

    return md, rows_out, chart_spec



def get_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "rows": int(len(df_full)),
        "players": int(len(ALL_PLAYERS)),
        "model": LLM_MODEL,
        "data_file": os.path.basename(DATA_FILE),
        "player_col": PLAYER_COL,
        "team_col": TEAM_COL,
    }
