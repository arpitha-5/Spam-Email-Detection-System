"""
============================================
🔮 predictor.py — Advanced Real-time Prediction
============================================
Includes Spam Detection, Phishing URL Detection,
and Email Category Classification.
============================================
"""

import joblib
import os
import re
import numpy as np
from src.preprocessor import preprocess_text

class SpamPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        
        # Advanced keyword detection
        self.spam_keywords = {
            "free", "win", "winner", "won", "prize", "cash", "money",
            "claim", "urgent", "call", "now", "offer", "click",
            "subscribe", "congratulations", "award", "reward", "bonus",
            "discount", "deal", "limited", "exclusive", "guaranteed",
            "credit", "loan", "buy", "order", "purchase", "cheap",
            "investment", "stock", "viagra", "pills", "pharmacy",
            "medication", "weight", "lose", "diet", "miracle", "act",
            "apply", "clearance", "trial", "sample", "membership",
            "unsubscrib", "opt", "remove", "cancel",
        }
        
        self.phishing_keywords = {
            "verify", "suspend", "compromised", "password", "login",
            "update account", "security alert", "unauthorized", "confirm",
            "identity", "validate"
        }

    def load(self, model_path: str, vectorizer_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.is_loaded = True

    def detect_urgency(self, text: str) -> list:
        """Detect urgency-based scam language."""
        urgency_phrases = [
            "act now", "urgent", "limited time", "immediate action required", 
            "your account will be blocked", "response needed", "do it now",
            "hurry", "expires soon", "final notice", "immediate response"
        ]
        found_phrases = []
        text_lower = text.lower()
        for phrase in urgency_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        return found_phrases

    def calculate_safety_score(self, text: str, spam_prob: float, urls: list, spam_words: list, urgency_phrases: list) -> dict:
        """Generate a safety score from 0-100."""
        score = 100
        
        # Deductions
        score -= int(spam_prob * 40)
        score -= len(urls) * 15
        score -= len(spam_words) * 5
        score -= len(urgency_phrases) * 10
        
        # Text analysis deductions
        if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.3:
            score -= 10 # Excessive caps
            
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / max(len(text), 1) > 0.1:
            score -= 10 # Excessive special chars
            
        score = max(0, min(100, score))
        
        if score >= 70:
            risk = "Safe"
        elif score >= 40:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"
            
        return {"score": score, "risk": risk}

    def detect_phishing(self, text: str) -> dict:
        """Detect phishing URLs and phishing-specific keywords."""
        # Simple URL regex
        urls = re.findall(r'(https?://\S+|www\.\S+)', text)
        suspicious_urls = [url for url in urls if '-' in url or len(url) > 50 or re.search(r'\d', url)]
        
        text_lower = text.lower()
        found_phishing_words = [word for word in self.phishing_keywords if word in text_lower]
        
        is_phishing = len(suspicious_urls) > 0 or len(found_phishing_words) >= 2
        return {
            "is_phishing": is_phishing,
            "urls_found": urls,
            "suspicious_urls": suspicious_urls,
            "phishing_keywords": found_phishing_words
        }
        
    def classify_category(self, text: str, is_spam: bool) -> str:
        """Classify email into categories."""
        if is_spam:
            return "Spam"
            
        text_lower = text.lower()
        if any(word in text_lower for word in ["offer", "discount", "sale", "shop", "coupon", "deal"]):
            return "Promotion"
        elif any(word in text_lower for word in ["friend", "following", "liked", "commented", "connect", "network"]):
            return "Social"
        elif any(word in text_lower for word in ["urgent", "important", "deadline", "meeting", "schedule", "boss"]):
            return "Important"
        else:
            return "Updates/General"

    def predict(self, text: str) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        cleaned_text = preprocess_text(text)
        text_vector = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vector)[0]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(text_vector)[0]
            ham_prob = float(probabilities[0])
            spam_prob = float(probabilities[1])
        else:
            ham_prob = 1.0 - prediction
            spam_prob = float(prediction)

        confidence = max(ham_prob, spam_prob)
        spam_words = self._find_spam_words(text)
        risk_level = self._get_risk_level(spam_prob)
        
        # Advanced Features
        phishing_info = self.detect_phishing(text)
        category = self.classify_category(text, prediction == 1)
        
        # New Features: Urgency & Safety Score
        urgency_phrases = self.detect_urgency(text)
        safety_info = self.calculate_safety_score(
            text, spam_prob, phishing_info["urls_found"], spam_words, urgency_phrases
        )
        
        if phishing_info["is_phishing"]:
            risk_level = "Critical (Phishing)"
            
        # ── Override Logic for Obvious Spam ──
        # If the ML model misses a short text but it contains obvious spam keywords
        # or has a very low safety score, we override the prediction.
        is_spam_final = bool(prediction == 1)
        if phishing_info["is_phishing"] or len(spam_words) >= 2 or safety_info["score"] < 50:
            is_spam_final = True
            if spam_prob < 0.5:
                # Adjust probability so UI reflects the override
                spam_prob = 0.85
                ham_prob = 0.15
                confidence = 0.85

        return {
            "prediction": "Spam" if is_spam_final else "Not Spam",
            "is_spam": is_spam_final,
            "confidence": round(confidence, 4),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4),
            "preprocessed_text": cleaned_text,
            "spam_words_found": spam_words,
            "risk_level": risk_level,
            "category": category,
            "phishing_info": phishing_info,
            "urgency_phrases": urgency_phrases,
            "safety_info": safety_info,
            "original_text": text,
        }

    def _find_spam_words(self, text: str) -> list:
        text_lower = text.lower()
        words = text_lower.split()
        found = []
        for word in words:
            clean_word = "".join(c for c in word if c.isalpha())
            if clean_word in self.spam_keywords:
                found.append(word)
        return list(set(found))

    def _get_risk_level(self, spam_prob: float) -> str:
        if spam_prob >= 0.8:
            return "High"
        elif spam_prob >= 0.5:
            return "Medium"
        else:
            return "Low"

    def batch_predict(self, texts: list) -> list:
        return [self.predict(text) for text in texts]
