"""
============================================
📐 feature_engineer.py — Feature Engineering
============================================
PURPOSE:
    Convert preprocessed text into numerical features
    that ML models can understand.

WHY FEATURE ENGINEERING?
    ML models can't read text — they need numbers.
    We convert text → numbers using vectorization.

TWO APPROACHES EXPLAINED:

┌─────────────────────────────────────────────────────┐
│ 1. BAG OF WORDS (BoW)                               │
│                                                      │
│ Counts how many times each word appears.             │
│                                                      │
│ Example:                                             │
│   Doc 1: "I love cats"                               │
│   Doc 2: "I love dogs"                               │
│                                                      │
│        I  love  cats  dogs                           │
│ Doc1 [ 1,   1,    1,    0 ]                          │
│ Doc2 [ 1,   1,    0,    1 ]                          │
│                                                      │
│ Problem: Common words get high counts but             │
│          aren't useful (like "the", "is")            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 2. TF-IDF (Term Frequency × Inverse Document Freq)  │
│                                                      │
│ Weighs words by importance:                          │
│ • Words appearing in MANY docs → low weight          │
│   (not useful for classification)                    │
│ • Words appearing in FEW docs → high weight          │
│   (more useful for classification)                   │
│                                                      │
│ Formula:                                             │
│   TF  = (count of word in doc) / (total words in doc)│
│   IDF = log(total docs / docs containing word)       │
│   TF-IDF = TF × IDF                                 │
│                                                      │
│ Example:                                             │
│   "free" appears in many spam messages → high TF     │
│   "free" appears in few ham messages → high IDF      │
│   → High TF-IDF score in spam docs!                  │
│                                                      │
│ WHY TF-IDF > BoW?                                    │
│   TF-IDF gives MORE weight to discriminating words   │
│   and LESS weight to common words.                   │
│   This usually leads to better classification.       │
└─────────────────────────────────────────────────────┘

WE USE TF-IDF because it performs better for spam detection.
============================================
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib
import os


def create_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer.

    Args:
        max_features: Maximum number of unique words to keep.
                      We keep the top 5000 most important words.
                      This prevents the feature matrix from being
                      too large (memory + speed).

    Returns:
        Configured TfidfVectorizer (not yet fitted)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,   # Keep top N words only
        ngram_range=(1, 2),          # Use single words AND word pairs
                                      # e.g., "free" and "free call"
        min_df=2,                    # Ignore words appearing in < 2 docs
                                      # (too rare to be useful)
        max_df=0.95,                 # Ignore words appearing in > 95% of docs
                                      # (too common to be useful)
    )
    return vectorizer


def create_bow_vectorizer(max_features: int = 5000) -> CountVectorizer:
    """
    Create a Bag of Words vectorizer (for comparison).

    Args:
        max_features: Maximum vocabulary size

    Returns:
        Configured CountVectorizer (not yet fitted)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    return vectorizer


def fit_vectorizer(vectorizer, texts: list):
    """
    Fit the vectorizer on training text and transform it.

    "Fitting" = learning the vocabulary (which words exist)
    "Transforming" = converting text to numbers using that vocabulary

    Args:
        vectorizer: TfidfVectorizer or CountVectorizer
        texts: List of preprocessed text strings

    Returns:
        Sparse matrix of shape (n_documents, n_features)
    """
    features = vectorizer.fit_transform(texts)
    return features


def transform_text(vectorizer, texts: list):
    """
    Transform new text using an already-fitted vectorizer.

    IMPORTANT: We only TRANSFORM here (not fit_transform).
    The vocabulary was learned during training — we reuse it.
    This ensures training and test data use the same features.

    Args:
        vectorizer: Already fitted vectorizer
        texts: List of preprocessed text strings

    Returns:
        Sparse matrix of features
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, filepath: str):
    """
    Save the fitted vectorizer to disk using joblib.

    Why save? When we deploy the app, we need the SAME vectorizer
    that was used during training. Otherwise, the features won't match.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(vectorizer, filepath)
    print(f"✅ Vectorizer saved to: {filepath}")


def load_vectorizer(filepath: str):
    """
    Load a saved vectorizer from disk.
    """
    vectorizer = joblib.load(filepath)
    print(f"✅ Vectorizer loaded from: {filepath}")
    return vectorizer


def get_top_features(vectorizer, n: int = 20) -> list:
    """
    Get the top N most important features (words) from the vectorizer.

    Useful for understanding what words the model sees.

    Args:
        vectorizer: Fitted vectorizer
        n: Number of top features to return

    Returns:
        List of top feature names
    """
    feature_names = vectorizer.get_feature_names_out()
    return list(feature_names[:n])


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("📐 Feature Engineering Demo")
    print("=" * 60)

    # Sample preprocessed texts
    sample_texts = [
        "free call win prize",
        "hey coming party tonight",
        "urgent claim prize free",
        "ok see tomorrow lunch",
        "congratul win free ticket call",
    ]

    # TF-IDF
    tfidf = create_tfidf_vectorizer(max_features=100)
    tfidf_features = fit_vectorizer(tfidf, sample_texts)

    print(f"\n📊 TF-IDF Features:")
    print(f"   Shape: {tfidf_features.shape}")
    print(f"   (5 documents, {tfidf_features.shape[1]} unique features)")
    print(f"   Top features: {get_top_features(tfidf, 10)}")

    # BoW
    bow = create_bow_vectorizer(max_features=100)
    bow_features = fit_vectorizer(bow, sample_texts)

    print(f"\n📊 Bag of Words Features:")
    print(f"   Shape: {bow_features.shape}")
    print(f"   Top features: {get_top_features(bow, 10)}")

    print("\n✅ Feature engineering working correctly!")
