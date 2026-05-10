"""
============================================
📊 evaluator.py — Model Evaluation & Comparison
============================================
PURPOSE:
    Evaluate trained models using multiple metrics
    and compare them to find the best one.

METRICS EXPLAINED:

    ┌──────────────────────────────────────────────────┐
    │ ACCURACY                                          │
    │ = (correct predictions) / (total predictions)     │
    │                                                    │
    │ Problem: If 90% of emails are ham, a model that   │
    │ ALWAYS predicts "ham" gets 90% accuracy!           │
    │ That's why we need more metrics.                   │
    └──────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │ PRECISION (for spam class)                        │
    │ = (correctly predicted spam) / (all predicted spam)│
    │                                                    │
    │ "Of all emails I flagged as spam,                  │
    │  how many were ACTUALLY spam?"                     │
    │                                                    │
    │ High precision = few false alarms                  │
    │ (important: we don't want real emails in spam!)    │
    └──────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │ RECALL (for spam class)                           │
    │ = (correctly predicted spam) / (all actual spam)   │
    │                                                    │
    │ "Of all actual spam emails,                        │
    │  how many did I catch?"                            │
    │                                                    │
    │ High recall = catches most spam                    │
    └──────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │ F1-SCORE                                          │
    │ = 2 × (precision × recall) / (precision + recall) │
    │                                                    │
    │ The balance between precision and recall.          │
    │ Best single metric for imbalanced data.            │
    └──────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │ CONFUSION MATRIX                                  │
    │                                                    │
    │                 Predicted                          │
    │              Ham     Spam                          │
    │ Actual Ham [ TN       FP ]  ← False Positive      │
    │ Actual Spam[ FN       TP ]  ← False Negative      │
    │                                                    │
    │ TN = True Negative  (ham correctly as ham)         │
    │ FP = False Positive (ham incorrectly as spam) ⚠    │
    │ FN = False Negative (spam missed as ham) ⚠         │
    │ TP = True Positive  (spam correctly as spam)       │
    └──────────────────────────────────────────────────┘
============================================
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a single model on test data.

    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: True test labels
        model_name: Name for display

    Returns:
        Dictionary with all metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Get classification report as string
    report = classification_report(
        y_test, y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    )

    results = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "true_negatives": int(cm[0][0]),   # Ham correctly classified
        "false_positives": int(cm[0][1]),   # Ham wrongly called spam
        "false_negatives": int(cm[1][0]),   # Spam missed
        "true_positives": int(cm[1][1]),    # Spam correctly caught
    }

    return results


def evaluate_all_models(trained_models: dict, X_test, y_test) -> dict:
    """
    Evaluate all trained models and collect results.

    Args:
        trained_models: Dict from train_all_models()
        X_test: Test features
        y_test: True test labels

    Returns:
        Dictionary of {model_name: evaluation_results}
    """
    all_results = {}

    print("=" * 60)
    print("📊 Model Evaluation Results")
    print("=" * 60)

    for name, model_info in trained_models.items():
        model = model_info["model"]
        results = evaluate_model(model, X_test, y_test, name)
        results["train_time"] = model_info["train_time"]
        all_results[name] = results

        # Print summary
        print(f"\n{'─' * 40}")
        print(f"📌 {name}")
        print(f"{'─' * 40}")
        print(f"   Accuracy:  {results['accuracy']:.4f}  "
              f"({results['accuracy']*100:.1f}%)")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall:    {results['recall']:.4f}")
        print(f"   F1-Score:  {results['f1_score']:.4f}")
        print(f"   Train time: {results['train_time']:.3f}s")

    return all_results


def get_comparison_dataframe(all_results: dict) -> pd.DataFrame:
    """
    Create a comparison DataFrame of all models.

    This makes it easy to see which model is best
    across all metrics at a glance.

    Returns:
        DataFrame with models as rows, metrics as columns
    """
    comparison = []

    for name, results in all_results.items():
        comparison.append({
            "Model": name,
            "Accuracy": results["accuracy"],
            "Precision": results["precision"],
            "Recall": results["recall"],
            "F1-Score": results["f1_score"],
            "Train Time (s)": round(results["train_time"], 3),
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    return df


def get_best_model(all_results: dict) -> tuple:
    """
    Find the best model based on F1-Score.

    WHY F1-SCORE?
        For spam detection, both precision and recall matter:
        - High precision → don't flag real emails as spam
        - High recall → catch as much spam as possible
        F1-Score balances both, making it the best metric
        for choosing the final model.

    Returns:
        (best_model_name, best_model_results)
    """
    best_name = max(all_results, key=lambda x: all_results[x]["f1_score"])
    return best_name, all_results[best_name]


def print_best_model_analysis(all_results: dict):
    """
    Print a detailed analysis of the best model and why it won.
    """
    best_name, best_results = get_best_model(all_results)
    comparison_df = get_comparison_dataframe(all_results)

    print("\n" + "=" * 60)
    print("🏆 BEST MODEL ANALYSIS")
    print("=" * 60)

    print(f"\n🥇 Best Model: {best_name}")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"   Precision: {best_results['precision']:.4f}")
    print(f"   Recall: {best_results['recall']:.4f}")

    print(f"\n📊 Full Comparison (sorted by F1-Score):")
    print(comparison_df.to_string(index=False))

    print(f"\n📋 Detailed Classification Report ({best_name}):")
    print(best_results["classification_report"])

    # Confusion matrix analysis
    print(f"\n🔍 Confusion Matrix Analysis ({best_name}):")
    print(f"   ✅ True Negatives (Ham→Ham):     {best_results['true_negatives']}")
    print(f"   ⚠️  False Positives (Ham→Spam):   {best_results['false_positives']}")
    print(f"   ⚠️  False Negatives (Spam→Ham):   {best_results['false_negatives']}")
    print(f"   ✅ True Positives (Spam→Spam):    {best_results['true_positives']}")

    return best_name


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("📊 Evaluator module loaded successfully!")
    print("   Use evaluate_all_models() after training.")
