"""
============================================
🤖 model_trainer.py — Model Building & Training
============================================
PURPOSE:
    Train multiple ML models for spam classification
    and compare their performance.

MODELS USED:
    1. Naive Bayes (MultinomialNB)
       → Best for text classification, fast, works great with TF-IDF
       → Based on Bayes' theorem: P(spam|words) ∝ P(words|spam) × P(spam)
       → Assumes words are independent (hence "naive")

    2. Logistic Regression
       → Linear model, good baseline for text
       → Learns a decision boundary between spam and ham
       → Outputs probabilities (useful for confidence scores)

    3. Support Vector Machine (SVM)
       → Finds the best separating hyperplane between classes
       → Very effective for high-dimensional data (like text)
       → Often gives the best accuracy

    4. Decision Tree
       → Tree-based model, easy to interpret
       → Creates rules like "if word 'free' > 0.5 → spam"
       → Can overfit, but useful for comparison

WHY MULTIPLE MODELS?
    Different models have different strengths.
    We train all 4 and compare to find the best one.
    This is standard practice in industry.
============================================
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import joblib
import os
import time


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.

    WHY SPLIT?
        We train on 80% of data and test on 20%.
        Testing on unseen data tells us how the model
        will perform in the real world.

    WHY random_state=42?
        Makes the split reproducible — you'll get the
        same split every time you run the code.
        (42 is just a convention, any number works)

    Args:
        X: Feature matrix (TF-IDF values)
        y: Labels (0=ham, 1=spam)
        test_size: Fraction of data for testing (0.2 = 20%)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensures same spam/ham ratio in train and test
    )

    print(f"📊 Data Split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def get_models() -> dict:
    """
    Create and return all ML models to train.

    Returns:
        Dictionary of {model_name: model_object}
    """
    models = {
        # ── Naive Bayes ──
        # alpha=0.1 → smoothing parameter (prevents zero probabilities)
        "Naive Bayes": MultinomialNB(alpha=0.1),

        # ── Logistic Regression ──
        # C=1.0 → regularization strength (prevents overfitting)
        # max_iter=1000 → max optimization steps
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),

        # ── SVM (Support Vector Machine) ──
        # We wrap LinearSVC in CalibratedClassifierCV so it can
        # output probabilities (LinearSVC doesn't by default)
        "SVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42, class_weight='balanced'),
            cv=3
        ),

        # ── Decision Tree ──
        # max_depth=20 → limits tree depth to prevent overfitting
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            random_state=42,
            class_weight='balanced'
        ),
    }
    return models


def train_model(model, X_train, y_train, model_name: str = "Model"):
    """
    Train a single model and measure training time.

    Args:
        model: Scikit-learn model object
        X_train: Training features
        y_train: Training labels
        model_name: Name for display

    Returns:
        Trained model, training time in seconds
    """
    print(f"\n🔄 Training {model_name}...")
    start_time = time.time()

    # .fit() is where the model actually LEARNS from the data
    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"✅ {model_name} trained in {train_time:.2f}s")

    return model, train_time


def train_all_models(X_train, y_train) -> dict:
    """
    Train all models and return results.

    Returns:
        Dictionary of {model_name: {"model": trained_model, "time": seconds}}
    """
    models = get_models()
    results = {}

    print("=" * 50)
    print("🚀 Training All Models")
    print("=" * 50)

    for name, model in models.items():
        trained_model, train_time = train_model(
            model, X_train, y_train, name
        )
        results[name] = {
            "model": trained_model,
            "train_time": train_time,
        }

    print("\n" + "=" * 50)
    print("✅ All models trained successfully!")
    return results


def save_model(model, filepath: str):
    """
    Save a trained model to disk using joblib.

    Why joblib over pickle?
        joblib is optimized for large numpy arrays
        (which sklearn models contain internally).
        It's faster and more efficient than pickle.

    Args:
        model: Trained sklearn model
        filepath: Where to save (e.g., "models/naive_bayes.pkl")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Load a saved model from disk.
    """
    model = joblib.load(filepath)
    print(f"✅ Model loaded from: {filepath}")
    return model


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import numpy as np

    print("=" * 50)
    print("🤖 Model Training Demo (synthetic data)")
    print("=" * 50)

    # Create dummy data for testing
    X, y = make_classification(
        n_samples=1000, n_features=100,
        n_classes=2, random_state=42
    )
    # Make non-negative for Naive Bayes
    X = np.abs(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    results = train_all_models(X_train, y_train)

    print(f"\n📊 Models trained: {list(results.keys())}")
    for name, res in results.items():
        acc = res["model"].score(X_test, y_test)
        print(f"   {name}: accuracy={acc:.4f}, time={res['train_time']:.3f}s")
