"""
============================================
📂 data_loader.py — Data Collection & Loading
============================================
PURPOSE:
    Load the SMS Spam Collection Dataset, clean it,
    and provide exploration utilities.

DATASET INFO:
    - Source: UCI Machine Learning Repository / Kaggle
    - Name: SMS Spam Collection Dataset
    - Size: 5,574 messages
    - Columns:
        • label  → 'ham' (not spam) or 'spam'
        • message → the actual text of the SMS/email
    - Class distribution: ~87% ham, ~13% spam (imbalanced)

WHY THIS DATASET?
    - Clean and well-labeled
    - Small enough for fast training
    - Good for beginners
    - Realistic text classification problem
============================================
"""

import pandas as pd
import os


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset from a CSV file.

    The dataset uses tab-separated values with columns:
    - v1: label ('ham' or 'spam')
    - v2: message text

    Args:
        filepath: Path to the CSV file. If None, uses default path.

    Returns:
        pd.DataFrame with columns ['label', 'message']
    """
    # Default path — relative to project root
    if filepath is None:
        # Look in the data/ folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "spam.csv")

    # ──────────────────────────────────────────────
    # Load dataset — try tab-separated first (TSV),
    # then fall back to comma-separated (CSV).
    # The dataset uses 'latin-1' encoding because it
    # contains some special characters.
    # ──────────────────────────────────────────────
    try:
        # Try TSV format first (common for this dataset)
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1",
            on_bad_lines="skip",   # skip malformed rows
        )
    except Exception:
        # Fall back to CSV with auto-detection
        df = pd.read_csv(
            filepath,
            encoding="latin-1",
            on_bad_lines="skip",
        )
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    # ──────────────────────────────────────────────
    # Basic cleaning:
    # 1. Drop any rows with missing values
    # 2. Drop exact duplicate messages
    # 3. Reset the index so it's clean (0, 1, 2, ...)
    # ──────────────────────────────────────────────
    df.dropna(subset=["label", "message"], inplace=True)
    df.drop_duplicates(subset=["message"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text labels to numbers:
        'ham'  → 0
        'spam' → 1

    This is needed because ML models work with numbers, not text.

    Args:
        df: DataFrame with a 'label' column

    Returns:
        DataFrame with an additional 'label_encoded' column
    """
    df = df.copy()
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for exploration.

    Returns a dictionary with:
        - total_samples: total number of messages
        - ham_count / spam_count: count of each class
        - ham_pct / spam_pct: percentage of each class
        - avg_message_length: average character count
        - sample_ham / sample_spam: example messages
    """
    summary = {
        "total_samples": len(df),
        "ham_count": len(df[df["label"] == "ham"]),
        "spam_count": len(df[df["label"] == "spam"]),
        "ham_pct": round(len(df[df["label"] == "ham"]) / len(df) * 100, 1),
        "spam_pct": round(len(df[df["label"] == "spam"]) / len(df) * 100, 1),
        "avg_message_length": round(df["message"].str.len().mean(), 1),
        "avg_ham_length": round(
            df[df["label"] == "ham"]["message"].str.len().mean(), 1
        ),
        "avg_spam_length": round(
            df[df["label"] == "spam"]["message"].str.len().mean(), 1
        ),
        "sample_ham": df[df["label"] == "ham"]["message"].head(3).tolist(),
        "sample_spam": df[df["label"] == "spam"]["message"].head(3).tolist(),
    }
    return summary


# ──────────────────────────────────────────────
# Quick test — run this file directly to verify
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("📂 Loading SMS Spam Dataset...")
    print("=" * 50)

    df = load_data()
    df = encode_labels(df)
    summary = get_data_summary(df)

    print(f"\n📊 Total samples:  {summary['total_samples']}")
    print(f"   Ham messages:   {summary['ham_count']} ({summary['ham_pct']}%)")
    print(f"   Spam messages:  {summary['spam_count']} ({summary['spam_pct']}%)")
    print(f"\n📏 Average message length: {summary['avg_message_length']} chars")
    print(f"   Ham avg length:  {summary['avg_ham_length']} chars")
    print(f"   Spam avg length: {summary['avg_spam_length']} chars")
    print(f"\n📝 Sample Ham:  {summary['sample_ham'][0][:80]}...")
    print(f"📝 Sample Spam: {summary['sample_spam'][0][:80]}...")
    print(f"\n✅ Dataset loaded successfully!")
    print(f"\nFirst 5 rows:")
    print(df.head())
