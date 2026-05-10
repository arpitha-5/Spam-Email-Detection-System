# 🛡️ SpamShield AI — Spam Email Detection System

A production-ready Machine Learning system that classifies emails/SMS messages as **Spam** or **Ham (Not Spam)** using NLP and multiple ML models, with a premium Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.1-orange.svg)

---

## 🌟 Features
- **Real-Time Prediction:** Classifies messages instantly with a beautiful gauge chart showing spam probability.
- **Multi-Model Comparison:** Trains and compares Naive Bayes, Logistic Regression, SVM, and Decision Trees.
- **Premium UI:** Custom white-theme Streamlit interface with animations and result cards.
- **Spam Word Highlighting:** Identifies and highlights specific spam keywords found in the message.
- **Prediction History:** Keeps a log of recently checked messages during the session.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
You must train the models before running the app for the first time.
```bash
python train_pipeline.py
```

### 3. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```text
├── app.py                  # Streamlit frontend
├── train_pipeline.py       # End-to-end training script
├── api.py                  # FastAPI backend
├── data/                   # Dataset folder
├── models/                 # Saved .pkl models
├── src/                    # Core modules (preprocessor, predictor, etc.)
└── requirements.txt        # Python dependencies
```

---

## 📊 Models Evaluated
- **Naive Bayes (MultinomialNB):** Excellent for text classification with TF-IDF.
- **Support Vector Machine (SVM):** High accuracy in high-dimensional spaces.
- **Logistic Regression:** Solid baseline model.
- **Decision Tree:** Interpretable rules.

*(All models are compared during the training pipeline, and the best one is saved as `best_model.pkl`)*

---

*This project demonstrates a complete end-to-end Machine Learning pipeline.*
