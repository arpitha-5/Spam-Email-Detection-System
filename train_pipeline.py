"""
============================================
🚀 train_pipeline.py — End-to-End Training Pipeline
============================================
PURPOSE:
    Run the complete ML pipeline from data loading
    to model saving in one script.

    This is the main script you run to train all models.

USAGE:
    python train_pipeline.py

WHAT IT DOES:
    1. Load dataset
    2. Preprocess all messages
    3. Create TF-IDF features
    4. Split into train/test
    5. Train 4 models
    6. Evaluate all models
    7. Find the best model
    8. Save best model + vectorizer
============================================
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path so imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, encode_labels, get_data_summary
from src.preprocessor import preprocess_text, preprocess_dataframe
from src.feature_engineer import (
    create_tfidf_vectorizer,
    fit_vectorizer,
    transform_text,
    save_vectorizer,
    get_top_features,
)
from src.model_trainer import split_data, train_all_models, save_model
from src.evaluator import (
    evaluate_all_models,
    get_comparison_dataframe,
    get_best_model,
    print_best_model_analysis,
)


def run_pipeline():
    """
    Execute the complete training pipeline.
    """
    print("\n" + "=" * 70)
    print("📧  SPAM EMAIL DETECTION — TRAINING PIPELINE")
    print("=" * 70)

    # ──────────────────────────────────────────
    # STEP 1: Load Dataset
    # ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("📂 STEP 1: Loading Dataset")
    print("─" * 70)

    data_path = os.path.join(PROJECT_ROOT, "data", "spam.csv")
    df = load_data(data_path)
    df = encode_labels(df)
    summary = get_data_summary(df)

    print(f"\n   ✅ Dataset loaded: {summary['total_samples']} messages")
    print(f"   📊 Ham:  {summary['ham_count']} ({summary['ham_pct']}%)")
    print(f"   📊 Spam: {summary['spam_count']} ({summary['spam_pct']}%)")
    print(f"\n   📝 Sample Ham message:")
    print(f"      \"{summary['sample_ham'][0][:80]}...\"")
    print(f"   📝 Sample Spam message:")
    print(f"      \"{summary['sample_spam'][0][:80]}...\"")

    # ──────────────────────────────────────────
    # STEP 2: Preprocess Text
    # ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🔧 STEP 2: Preprocessing Text")
    print("─" * 70)

    # Show before/after for one example
    sample_text = df["message"].iloc[0]
    sample_cleaned = preprocess_text(sample_text)
    print(f"\n   Before: \"{sample_text[:60]}...\"")
    print(f"   After:  \"{sample_cleaned[:60]}\"")

    # Preprocess all messages
    print("\n   🔄 Preprocessing all messages...")
    df["cleaned"] = preprocess_dataframe(df, "message")
    print(f"   ✅ Done! {len(df)} messages preprocessed.")

    print("\n" + "─" * 70)
    print("✂️  STEP 3 & 4: Train-Test Split & Feature Engineering")
    print("─" * 70)

    # Split text data FIRST to prevent data leakage
    from sklearn.model_selection import train_test_split
    X_text = df["cleaned"].tolist()
    y = df["label_encoded"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = create_tfidf_vectorizer(max_features=5000)
    X_train = fit_vectorizer(vectorizer, X_train_text)
    X_test = transform_text(vectorizer, X_test_text)

    print(f"\n   ✅ TF-IDF matrix created")
    print(f"   📊 Train Shape: {X_train.shape} (documents × features)")
    print(f"   📊 Test Shape: {X_test.shape} (documents × features)")
    print(f"   📝 Top 15 features: {get_top_features(vectorizer, 15)}")

    # ──────────────────────────────────────────
    # STEP 5: Train Models
    # ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🤖 STEP 5: Training Models")
    print("─" * 70)

    trained_models = train_all_models(X_train, y_train)

    # ──────────────────────────────────────────
    # STEP 6: Evaluate Models
    # ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("📊 STEP 6: Evaluating Models")
    print("─" * 70)

    all_results = evaluate_all_models(trained_models, X_test, y_test)

    # ──────────────────────────────────────────
    # STEP 7: Compare Models & Find Best
    # ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("🏆 STEP 7: Model Comparison")
    print("─" * 70)

    best_name = print_best_model_analysis(all_results)
    comparison_df = get_comparison_dataframe(all_results)

    # ──────────────────────────────────────────
    # STEP 8: Save Best Model + Vectorizer
    # ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("💾 STEP 8: Saving Best Model")
    print("─" * 70)

    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save the best model
    best_model = trained_models[best_name]["model"]
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    save_model(best_model, best_model_path)

    # Save the vectorizer
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    save_vectorizer(vectorizer, vectorizer_path)

    # Also save all individual models (for comparison in UI)
    for name, model_info in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        save_model(model_info["model"], os.path.join(models_dir, filename))

    # Save the comparison results
    comparison_path = os.path.join(models_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"💾 Comparison saved to: {comparison_path}")

    # ──────────────────────────────────────────
    # DONE!
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n   🏆 Best Model: {best_name}")
    print(f"   📁 Saved to: {models_dir}/")
    print(f"   📊 F1-Score: {all_results[best_name]['f1_score']:.4f}")
    print(f"   📊 Accuracy: {all_results[best_name]['accuracy']:.4f}")
    print(f"\n   Next steps:")
    print(f"   1. Run the Streamlit UI:  streamlit run app.py")
    print(f"   2. Or test the predictor: python src/predictor.py")

    return best_name, all_results, comparison_df


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
