"""Fetch and snapshot PJM's live interconnection queue.

PJM exposes the queue as an Excel export through a planning API. We pull it,
parse it into a DataFrame, and write a parquet snapshot per fetch so we can
diff status / milestone changes week-over-week.

The endpoint and subscription key are mined from PJM's public JS bundle
(see gridstatus PJM module for reference). They've been stable for years but
either could change without notice.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

PJM_QUEUE_URL = "https://services.pjm.com/PJMPlanningApi/api/Queue/ExportToXls"
PJM_SUBSCRIPTION_KEY = "E29477D0-70E0-4825-89B0-43F460BF9AB4"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_DIR = DATA_DIR / "pjm_snapshots"

DATE_COLUMNS = (
    "Submitted Date",
    "Withdrawal Date",
    "Revised In Service Date",
    "Actual In Service Date",
    "Initial Study",
    "Feasibility Study",
    "System Impact Study",
    "Facilities Study",
    "Backfeed Date",
    "Test Energy Date",
)

MILESTONE_GATES = [
    ("Feasibility Study", "Feasibility Study Status"),
    ("System Impact Study", "System Impact Study Status"),
    ("Facilities Study", "Facilities Study Status"),
    ("Interim/Interconnection Service/Generation Interconnection Agreement",
     "Interim/Interconnection Service/Generation Interconnection Agreement Status"),
    ("Construction Service Agreement", "Construction Service Agreement Status"),
]


def fetch_queue(timeout: int = 60) -> pd.DataFrame:
    """Pull the current PJM queue Excel and return a parsed DataFrame."""
    response = requests.post(
        PJM_QUEUE_URL,
        headers={
            "api-subscription-key": PJM_SUBSCRIPTION_KEY,
            "Host": "services.pjm.com",
            "Origin": "https://www.pjm.com",
            "Referer": "https://www.pjm.com/",
        },
        timeout=timeout,
    )
    response.raise_for_status()

    df = pd.read_excel(BytesIO(response.content))

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for mw_col in ("MFO", "MW Energy", "MW Capacity", "MW In Service"):
        if mw_col in df.columns:
            df[mw_col] = pd.to_numeric(df[mw_col], errors="coerce")

    return df


def save_snapshot(df: pd.DataFrame, snapshot_date: date | None = None) -> Path:
    """Write a parquet snapshot stamped with the given date (default: today)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_date = snapshot_date or date.today()
    path = SNAPSHOT_DIR / f"{snapshot_date.isoformat()}.parquet"
    df.to_parquet(path, index=False)
    return path


def list_snapshots() -> list[Path]:
    """Return all snapshots, oldest first."""
    if not SNAPSHOT_DIR.exists():
        return []
    return sorted(SNAPSHOT_DIR.glob("*.parquet"))


def load_snapshot(path: Path | None = None) -> pd.DataFrame:
    """Load a snapshot; defaults to the most recent."""
    snapshots = list_snapshots()
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshots in {SNAPSHOT_DIR}. Run `python -m src.pjm_queue` to pull one."
        )
    target = path or snapshots[-1]
    return pd.read_parquet(target)


def summarize(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()
    if "Status" in df.columns:
        print("Status breakdown:")
        print(df["Status"].value_counts().to_string())
        print()
    if "Submitted Date" in df.columns:
        valid = df["Submitted Date"].dropna()
        if not valid.empty:
            print(f"Submitted-date range: {valid.min().date()} → {valid.max().date()}")
    if "Fuel" in df.columns:
        active = df[df["Status"] == "Active"] if "Status" in df.columns else df
        print(f"\nActive projects by fuel ({len(active):,} total):")
        print(active["Fuel"].value_counts().head(10).to_string())


if __name__ == "__main__":
    df = fetch_queue()
    path = save_snapshot(df)
    print(f"Saved snapshot → {path.relative_to(DATA_DIR.parent)}\n")
    summarize(df)
