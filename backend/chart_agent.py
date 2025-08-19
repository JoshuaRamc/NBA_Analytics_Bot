import re
from typing import List, Optional, Dict, Any
import pandas as pd

# --- Keyword detection --------------------------------------------------------

BAR_WORDS = [
    r"\bbar\s*chart\b", r"\bbar\s*graph\b", r"\bhistogram\b",
]
LINE_WORDS = [
    r"\bline\s*chart\b", r"\bline\s*graph\b", r"\bline\s*plot\b",
]
PIE_WORDS = [
    r"\bpie\s*chart\b",
]
ANY_CHART_WORDS = [
    r"\bchart\b", r"\bgraph\b", r"\bplot\b", r"\bvisual(ization)?\b", r"\bdiagram\b", r"\bfigure\b",
]

TOP_N_REGEX = re.compile(r"\btop\s*(\d{1,2})\b", flags=re.I)

# Preferred axes/metrics (order matters)
PREFERRED_X = ["Player", "Player_Name", "Name", "Team", "Pos"]
PREFERRED_METRICS = [
    "PTS", "TRB", "AST", "STL", "BLK", "3PPct", "FGPct", "eFGPct", "FTPct",
    "3PA", "3P", "FGA", "FG", "TOV", "ORB", "DRB", "MP"
]

# lightweight alias map for metric mentions
ALIAS_TO_COL = {
    "points": "PTS", "ppg": "PTS", "scoring": "PTS",
    "rebounds": "TRB", "rpg": "TRB", "boards": "TRB",
    "assists": "AST", "apg": "AST", "dimes": "AST",
    "steals": "STL", "spg": "STL",
    "blocks": "BLK", "bpg": "BLK",
    "turnovers": "TOV",
    "3pt%": "3PPct", "3p%": "3PPct", "three point %": "3PPct", "three-point %": "3PPct",
    "fg%": "FGPct", "field goal %": "FGPct",
    "ft%": "FTPct", "free throw %": "FTPct",
    "efg%": "eFGPct", "effective field goal %": "eFGPct",
    "3pa": "3PA", "3p attempts": "3PA", "3pm": "3P", "3p makes": "3P",
}

def _has_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)

def _first_match(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        if re.search(p, text, flags=re.I):
            return p
    return None

def _detect_chart_type(question: str) -> Optional[str]:
    q = (question or "").lower()
    if _has_any(BAR_WORDS, q):
        return "bar"
    if _has_any(LINE_WORDS, q):
        return "line"
    if _has_any(PIE_WORDS, q):
        return "pie"
    # default to bar if any generic chart keyword
    if _has_any(ANY_CHART_WORDS, q):
        return "bar"
    return None

def _detect_top_n(question: str, default: int = 10) -> int:
    m = TOP_N_REGEX.search(question or "")
    if m:
        n = int(m.group(1))
        return max(1, min(50, n))
    # common phrasing: leaders/highest/most/best → top 5 by default
    if re.search(r"\b(leader|leaders|highest|most|best)\b", (question or ""), re.I):
        return 5
    return default

def _pick_x_key(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    for c in PREFERRED_X:
        if c in cols:
            return c
    # fallback: first non-numeric column, else first column
    non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    return non_num[0] if non_num else cols[0]

def _pick_series_from_question(question: str, df: pd.DataFrame) -> List[str]:
    q = (question or "").lower()
    cols = set(df.columns.tolist())
    picks: List[str] = []
    # try aliases mentioned in question, keep order
    for alias, col in ALIAS_TO_COL.items():
        if alias in q and col in cols:
            if col not in picks:
                picks.append(col)
        if len(picks) >= 2:
            break
    if picks:
        return picks[:2]
    # else pick from preferred metrics that exist & numeric
    for m in PREFERRED_METRICS:
        if m in cols and pd.api.types.is_numeric_dtype(df[m]):
            picks.append(m)
        if len(picks) >= 2:
            break
    # final safety: any numeric columns (excluding xKey if numeric)
    if not picks:
        numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numerics:
            picks.append(numerics[0])
            if len(numerics) > 1:
                picks.append(numerics[1])
    return picks[:2] if picks else []

# --- Chart creators -----------------------------------------------------------

def create_bar_chart(
    df_small: pd.DataFrame,
    x_key: str,
    series_keys: List[str],
    question: str,
    default_limit: int = 10,
) -> Dict[str, Any]:
    limit = _detect_top_n(question, default=default_limit)
    # Prefer sorting by first metric, desc
    sort_by = series_keys[0] if series_keys else None
    out = df_small.copy()
    if sort_by and sort_by in out.columns and pd.api.types.is_numeric_dtype(out[sort_by]):
        out = out.sort_values(by=sort_by, ascending=False)
    out = out.head(limit)

    return {
        "chartType": "bar",
        "xKey": x_key,
        "series": [{"dataKey": k, "label": k} for k in series_keys[:2]],
        "title": None,
        "subtitle": None,
        "xLabel": x_key,
        "yLabel": series_keys[0] if series_keys else None,
        "sortBy": sort_by,
        "descending": True,
        "limit": limit,
        "data": out.to_dict(orient="records"),
    }

def create_line_chart(
    df_small: pd.DataFrame,
    x_key: str,
    series_keys: List[str],
    question: str,
    default_limit: int = 10,
) -> Dict[str, Any]:
    """
    NOTE: With a one-season, per-player dataset, line charts are usually not ideal.
    We still support them (line across players) in case the user explicitly asks.
    """
    limit = _detect_top_n(question, default=default_limit)
    sort_by = series_keys[0] if series_keys else None
    out = df_small.copy()
    if sort_by and sort_by in out.columns and pd.api.types.is_numeric_dtype(out[sort_by]):
        out = out.sort_values(by=sort_by, ascending=False)
    out = out.head(limit)

    return {
        "chartType": "line",
        "xKey": x_key,
        "series": [{"dataKey": k, "label": k} for k in series_keys[:2]],
        "title": None,
        "subtitle": None,
        "xLabel": x_key,
        "yLabel": series_keys[0] if series_keys else None,
        "sortBy": sort_by,
        "descending": True,
        "limit": limit,
        "data": out.to_dict(orient="records"),
    }

def create_pie_chart(
    df_small: pd.DataFrame,
    x_key: str,
    series_key: str,
    question: str,
    default_limit: int = 10,
) -> Dict[str, Any]:
    """
    Frontend NOTE: you'll need a PieChart renderer to use this.
    We include it for completeness, even if your UI currently renders bar/line only.
    """
    limit = _detect_top_n(question, default=default_limit)
    out = df_small.copy()
    if series_key in out.columns and pd.api.types.is_numeric_dtype(out[series_key]):
        out = out.sort_values(by=series_key, ascending=False)
    out = out.head(limit)

    # For pie charts, we still return the same generic spec so the frontend can decide.
    return {
        "chartType": "pie",
        "xKey": x_key,                   # label key for slices
        "series": [{"dataKey": series_key, "label": series_key}],
        "title": None,
        "subtitle": None,
        "xLabel": x_key,
        "yLabel": series_key,
        "sortBy": series_key,
        "descending": True,
        "limit": limit,
        "data": out.to_dict(orient="records"),
    }

# --- Public entry -------------------------------------------------------------

def get_chart_spec(question: str, df_small: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Pure heuristic chart builder—no LLM.
    Returns a Recharts-friendly spec dict or None if no chart keywords detected.
    """
    if df_small is None or df_small.empty:
        return None

    chart_type = _detect_chart_type(question)
    if chart_type is None:
        return None

    x_key = _pick_x_key(df_small)
    series_keys = _pick_series_from_question(question, df_small)
    if not series_keys:
        return None

    if chart_type == "bar":
        return create_bar_chart(df_small, x_key, series_keys, question)
    if chart_type == "line":
        return create_line_chart(df_small, x_key, series_keys, question)
    if chart_type == "pie":
        # use only first metric for pie
        return create_pie_chart(df_small, x_key, series_keys[0], question)

    # default fallback: bar
    return create_bar_chart(df_small, x_key, series_keys, question)
