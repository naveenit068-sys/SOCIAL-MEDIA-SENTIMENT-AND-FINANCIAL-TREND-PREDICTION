from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def _find_column(df, candidates, description):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    raise ValueError(
        f"Could not identify the {description} column. "
        f"Available columns: {list(df.columns)}"
    )

def load_social_posts(path=DATA_DIR / "social_media_posts.csv"):
    df = pd.read_csv(path)
    date_col = _find_column(df, ["date", "datetime", "created_at", "timestamp"], "date")
    company_col = _find_column(df, ["company", "ticker", "symbol"], "company")
    text_col = _find_column(df, ["post", "text", "content", "message", "tweet"], "post text")

    df = df.rename(columns={
        date_col: "Date",
        company_col: "Company",
        text_col: "Post"
    })
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Company"] = df["Company"].astype(str).str.strip().str.upper()
    df["Post"] = df["Post"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["Date"])
    df = df[df["Post"] != ""].drop_duplicates()
    if df.empty:
        raise ValueError("No valid social media posts remain after preprocessing.")
    return df.sort_values("Date").reset_index(drop=True)

def load_stock_data(path=DATA_DIR / "stock_data.csv"):
    df = pd.read_csv(path)
    date_col = _find_column(df, ["date", "datetime"], "stock date")
    df = df.rename(columns={date_col: "Date"})

    # Handle common Yahoo Finance column naming variations.
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ["open", "high", "low", "close", "volume"]:
            rename[col] = key.capitalize()
        elif key in ["adj_close", "adjclose"]:
            rename[col] = "Adj Close"
    df = df.rename(columns=rename)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stock CSV is missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid stock records remain after preprocessing.")
    return df
