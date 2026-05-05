"""Load and clean the Berkeley Lab Queued Up 2025 dataset.

The Queued Up Excel file has multiple sheets; the project-level data sheet is the
one with one row per interconnection request. Column names vary slightly across
editions, so we normalize them to a canonical set.

Run as a script for a quick health check on the loaded data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


CANONICAL_COLUMNS = {
    "project_name": ["project name", "project"],
    "queue_id": ["queue id", "queue identifier", "q#", "queue #"],
    "rto": ["iso/rto", "rto", "iso", "region"],
    "poi": ["poi", "point of interconnection", "substation", "poi name"],
    "state": ["state", "state(s)"],
    "county": ["county"],
    "resource_type": ["type", "resource type", "fuel", "technology"],
    "mw": ["mw", "capacity (mw)", "summer capacity (mw)", "nameplate (mw)"],
    "queue_date": ["queue date", "date entered queue", "ia request date"],
    "ia_signed": ["ia signed", "interconnection agreement signed"],
    "withdrawn_date": ["withdrawn date", "date withdrawn"],
    "operational_date": ["actual operating date", "commercial operation date", "cod"],
    "status": ["status", "queue status", "current status"],
}


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower()).replace("_", " ")


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {_normalize(c): c for c in df.columns}
    rename = {}
    for canonical, candidates in CANONICAL_COLUMNS.items():
        for cand in candidates:
            if cand in lookup:
                rename[lookup[cand]] = canonical
                break
    return df.rename(columns=rename)


def find_data_file() -> Path:
    """Return the first .xlsx in data/. Errors if none or multiple."""
    candidates = sorted(DATA_DIR.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx found in {DATA_DIR}. Download Queued Up from "
            "https://emp.lbl.gov/queues and place the Excel file in data/."
        )
    return candidates[0]


def pick_main_sheet(xlsx_path: Path) -> str:
    """Pick the sheet most likely to contain project-level rows.

    Heuristic: largest sheet that has a queue_date or queue_id-shaped column.
    """
    xls = pd.ExcelFile(xlsx_path)
    best_name, best_score = None, -1
    for sheet in xls.sheet_names:
        sample = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=5)
        if sample.empty:
            continue
        normalized = {_normalize(c) for c in sample.columns}
        score = len(sample.columns)
        if any(c in normalized for c in ("queue id", "q#", "queue #")):
            score += 100
        if any("date" in c and "queue" in c for c in normalized):
            score += 50
        if score > best_score:
            best_name, best_score = sheet, score
    if best_name is None:
        raise ValueError(f"Could not identify a project-level sheet in {xlsx_path}")
    return best_name


def load_queued_up(xlsx_path: Path | None = None) -> pd.DataFrame:
    """Load and clean the Queued Up dataset.

    Returns a DataFrame with normalized column names, parsed dates, a derived
    `withdrawn` boolean target, and a `queue_age_years` feature.
    """
    path = xlsx_path or find_data_file()
    sheet = pick_main_sheet(path)
    df = pd.read_excel(path, sheet_name=sheet)
    df = _rename_columns(df)

    for date_col in ("queue_date", "withdrawn_date", "operational_date", "ia_signed"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if "mw" in df.columns:
        df["mw"] = pd.to_numeric(df["mw"], errors="coerce")

    df["withdrawn"] = _derive_withdrawn(df)
    df["operational"] = _derive_operational(df)

    if "queue_date" in df.columns:
        ref = pd.Timestamp.today().normalize()
        df["queue_age_years"] = (
            (ref - df["queue_date"]).dt.days / 365.25
        ).round(2)

    return df


def _derive_withdrawn(df: pd.DataFrame) -> pd.Series:
    if "withdrawn_date" in df.columns:
        from_date = df["withdrawn_date"].notna()
    else:
        from_date = pd.Series(False, index=df.index)
    if "status" in df.columns:
        from_status = df["status"].astype(str).str.contains(
            "withdraw", case=False, na=False
        )
    else:
        from_status = pd.Series(False, index=df.index)
    return (from_date | from_status).astype(int)


def _derive_operational(df: pd.DataFrame) -> pd.Series:
    if "operational_date" in df.columns:
        from_date = df["operational_date"].notna()
    else:
        from_date = pd.Series(False, index=df.index)
    if "status" in df.columns:
        from_status = df["status"].astype(str).str.contains(
            "operating|in service|operational|commercial", case=False, na=False
        )
    else:
        from_status = pd.Series(False, index=df.index)
    return (from_date | from_status).astype(int)


def summarize(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    if "rto" in df.columns:
        print("\nProjects by RTO:")
        print(df["rto"].value_counts().head(15).to_string())
    print(f"\nWithdrawn: {df['withdrawn'].sum():,} "
          f"({df['withdrawn'].mean():.1%})")
    print(f"Operational: {df['operational'].sum():,} "
          f"({df['operational'].mean():.1%})")
    if "mw" in df.columns:
        print(f"\nTotal MW in queue: {df['mw'].sum():,.0f}")


if __name__ == "__main__":
    df = load_queued_up()
    summarize(df)
