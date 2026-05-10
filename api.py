import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="SpamShield AI API",
    description="REST API for predicting whether a message is Spam or Ham.",
    version="1.0.0"
)

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
vectorizer_path = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")

# Check if model exists
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    model = None
    vectorizer = None
    print(f"Warning: Model or vectorizer could not be loaded. Please run train_pipeline.py first. Details: {e}")

# Import preprocessor
from src.preprocessor import preprocess_text
spam_keywords = {"win", "free", "urgent", "prize", "cash", "guaranteed", "click", "claim", "offer"}

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    original_text: str
    preprocessed_text: str
    prediction: str
    spam_probability: float
    ham_probability: float
    confidence: float
    risk_level: str
    spam_words_found: list[str]

@app.get("/")
def home():
    return {"message": "Welcome to the SpamShield AI API. Use POST /predict to classify messages."}

@app.post("/predict", response_model=PredictResponse)
def predict_spam(request: PredictRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Models are not loaded. Train the models first.")
        
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Feature extraction
    features = vectorizer.transform([cleaned_text])
    
    # Prediction
    prediction_val = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    spam_prob = probabilities[1]
    ham_prob = probabilities[0]
    
    # Keyword highlight
    words = set(cleaned_text.split())
    found_spam_words = list(words.intersection(spam_keywords))
    
    risk_level = "High" if spam_prob > 0.7 else "Medium" if spam_prob > 0.4 else "Low"
    label = "Spam" if prediction_val == 1 else "Ham"
    
    return PredictResponse(
        original_text=text,
        preprocessed_text=cleaned_text,
        prediction=label,
        spam_probability=float(spam_prob),
        ham_probability=float(ham_prob),
        confidence=float(max(spam_prob, ham_prob)),
        risk_level=risk_level,
        spam_words_found=found_spam_words
    )

if __name__ == "__main__":
    print("Starting API Server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
