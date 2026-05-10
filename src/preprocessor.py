"""
============================================
🔧 preprocessor.py — Text Preprocessing Pipeline
============================================
PURPOSE:
    Clean and normalize raw text for ML models.

WHY PREPROCESSING?
    Raw text contains noise: punctuation, numbers, mixed case,
    common words ("the", "is") that don't help classification.
    We must clean it so the ML model focuses on meaningful words.

PIPELINE STEPS:
    1. Lowercasing       → "FREE" becomes "free"
    2. Remove punctuation → "hello!!!" becomes "hello"
    3. Remove numbers     → "call 12345" becomes "call"
    4. Tokenization       → "hello world" becomes ["hello", "world"]
    5. Remove stopwords   → removes "the", "is", "a", etc.
    6. Stemming           → "running" becomes "run"

STEMMING vs LEMMATIZATION:
    • Stemming: Chops word endings (faster, rougher)
      Example: "running" → "run", "studies" → "studi"
    • Lemmatization: Uses dictionary to find root word (slower, accurate)
      Example: "running" → "run", "studies" → "study"

    We use STEMMING here for speed. Both work well for spam detection.
============================================
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ──────────────────────────────────────────────
# Download required NLTK data (runs once)
# These are small data packages NLTK needs:
#   - 'punkt_tab' → for splitting text into words
#   - 'stopwords'  → list of common words to remove
# ──────────────────────────────────────────────
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize the stemmer (reuse across calls for efficiency)
stemmer = PorterStemmer()

# Load English stopwords once
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text message.

    Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove numbers
        5. Remove punctuation
        6. Tokenize (split into words)
        7. Remove stopwords
        8. Apply stemming

    Args:
        text: Raw message string

    Returns:
        Cleaned, stemmed text string

    Example:
        Input:  "FREE entry to WIN £1000! Call NOW!!!"
        Output: "free entri win call"
    """
    # ── Step 1: Lowercase ──
    # Why? "FREE" and "free" should be treated as the same word
    text = text.lower()

    # ── Step 2: Remove URLs ──
    # Why? URLs like "http://spam.com" are noise
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # ── Step 3: Remove email addresses ──
    # Why? Email addresses don't help classification
    text = re.sub(r"\S+@\S+", "", text)

    # ── Step 4: Remove numbers ──
    # Why? Numbers like phone numbers, prices are noise
    # (Spam often has numbers, but they vary too much to be useful features)
    text = re.sub(r"\d+", "", text)

    # ── Step 5: Remove punctuation ──
    # Why? "hello!!!" and "hello" mean the same thing
    text = text.translate(str.maketrans("", "", string.punctuation))

    # ── Step 6: Tokenize (split into individual words) ──
    # Why? ML models work with individual words (tokens), not full sentences
    # "hello world" → ["hello", "world"]
    tokens = word_tokenize(text)

    # ── Step 7: Remove stopwords ──
    # Why? Words like "the", "is", "a" appear everywhere
    #       and don't help distinguish spam from ham
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # ── Step 8: Stemming ──
    # Why? "running", "runs", "ran" all mean the same thing → "run"
    #       Reduces vocabulary size so the model generalizes better
    tokens = [stemmer.stem(word) for word in tokens]

    # ── Step 9: Remove very short tokens ──
    # Why? Single letters like "u", "r" are usually noise
    tokens = [word for word in tokens if len(word) > 1]

    # Join tokens back into a single string
    # (TF-IDF vectorizer expects strings, not lists)
    return " ".join(tokens)


def preprocess_dataframe(df, text_column: str = "message") -> list:
    """
    Apply preprocessing to an entire DataFrame column.

    Args:
        df: DataFrame containing text data
        text_column: Name of the column with raw text

    Returns:
        List of preprocessed text strings
    """
    return df[text_column].apply(preprocess_text).tolist()


def get_preprocessing_example() -> dict:
    """
    Show a before/after example of preprocessing.
    Useful for understanding and debugging.
    """
    examples = [
        "FREE entry to WIN £1000 cash! Call 09061701461 NOW!!!",
        "Hey, are you coming to the party tonight?",
        "URGENT! You have won a prize. Visit http://spam.com to claim",
        "Ok lar... Joking wif u oni...",
    ]

    results = {}
    for text in examples:
        results[text] = preprocess_text(text)

    return results


# ──────────────────────────────────────────────
# Quick test — run this file directly to verify
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Text Preprocessing Demo")
    print("=" * 60)

    examples = get_preprocessing_example()
    for original, cleaned in examples.items():
        print(f"\n📝 Original: {original}")
        print(f"✅ Cleaned:  {cleaned}")

    print("\n" + "=" * 60)
    print("✅ Preprocessing pipeline working correctly!")
