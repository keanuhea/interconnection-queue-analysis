"""Load and clean the Berkeley Lab Queued Up 2025 dataset.

The Queued Up Excel is a report-style workbook with one project-level sheet
(`03. Complete Queue Data`) whose header row sits underneath a "RETURN TO
CONTENTS" banner. We read that sheet with header=1 and rename its columns to
a canonical set the rest of the pipeline expects.

Run as a script for a quick health check on the loaded data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

QUEUE_SHEET = "03. Complete Queue Data"
HEADER_ROW = 1

COLUMN_MAP = {
    "q_id": "queue_id",
    "q_status": "status",
    "q_date": "queue_date",
    "prop_date": "proposed_date",
    "on_date": "operational_date",
    "wd_date": "withdrawn_date",
    "ia_date": "ia_signed",
    "region": "rto",
    "poi_name": "poi",
    "project_name": "project_name",
    "type_clean": "resource_type",
    "mw_1": "mw",
    "county": "county",
    "state": "state",
    "service": "service",
    "cluster": "cluster",
    "utility": "utility",
    "developer": "developer",
}

DATE_COLUMNS = ("queue_date", "proposed_date", "operational_date", "withdrawn_date", "ia_signed")
EXCEL_EPOCH = "1899-12-30"


def find_data_file() -> Path:
    """Return the first .xlsx in data/. Errors if none."""
    candidates = sorted(DATA_DIR.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx found in {DATA_DIR}. Download Queued Up from "
            "https://emp.lbl.gov/queues and place the Excel file in data/."
        )
    return candidates[0]


def _parse_date_column(s: pd.Series) -> pd.Series:
    """Parse a column that may be either Excel-serial floats or already-datetime.

    openpyxl auto-parses Excel cells with date formatting into datetime64; older
    or text-formatted exports come through as numeric serials counted from
    1899-12-30. Handle both.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin=EXCEL_EPOCH, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def load_queued_up(xlsx_path: Path | None = None) -> pd.DataFrame:
    """Load and clean the Queued Up project-level data.

    Returns a DataFrame with canonical column names, parsed dates, derived
    `withdrawn` / `operational` boolean targets, and a `queue_age_years` feature.
    """
    path = xlsx_path or find_data_file()
    df = pd.read_excel(path, sheet_name=QUEUE_SHEET, header=HEADER_ROW)
    df = df.rename(columns={src: dst for src, dst in COLUMN_MAP.items() if src in df.columns})

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = _parse_date_column(df[col])

    if "mw" in df.columns:
        df["mw"] = pd.to_numeric(df["mw"], errors="coerce")

    df["withdrawn"] = _derive_withdrawn(df)
    df["operational"] = _derive_operational(df)

    if "queue_date" in df.columns:
        ref = pd.Timestamp("2024-12-31")
        df["queue_age_years"] = (
            (ref - df["queue_date"]).dt.days / 365.25
        ).round(2)

    return df


def _derive_withdrawn(df: pd.DataFrame) -> pd.Series:
    from_date = df["withdrawn_date"].notna() if "withdrawn_date" in df.columns else False
    from_status = (
        df["status"].astype(str).str.contains("withdraw", case=False, na=False)
        if "status" in df.columns
        else False
    )
    return (from_date | from_status).astype(int)


def _derive_operational(df: pd.DataFrame) -> pd.Series:
    from_date = df["operational_date"].notna() if "operational_date" in df.columns else False
    from_status = (
        df["status"].astype(str).str.contains(
            "operating|in.?service|operational|commercial", case=False, na=False, regex=True
        )
        if "status" in df.columns
        else False
    )
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
    if "queue_date" in df.columns:
        valid = df["queue_date"].dropna()
        if not valid.empty:
            print(f"Queue date range: {valid.min().date()} to {valid.max().date()}")


if __name__ == "__main__":
    df = load_queued_up()
    summarize(df)
