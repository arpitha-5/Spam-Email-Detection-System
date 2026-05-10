import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'spam_history.db')

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            original_text TEXT,
            prediction TEXT,
            confidence REAL,
            spam_probability REAL,
            phishing_detected BOOLEAN,
            category TEXT,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(data: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, original_text, prediction, confidence, spam_probability, phishing_detected, category, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(),
        data.get("original_text", ""),
        data.get("prediction", "Unknown"),
        data.get("confidence", 0.0),
        data.get("spam_probability", 0.0),
        data.get("phishing_detected", False),
        data.get("category", "Uncategorized"),
        data.get("risk_level", "Low")
    ))
    conn.commit()
    conn.close()

def get_history_df() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def clear_history():
    conn = get_connection()
    c = conn.cursor()
    c.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()

# Initialize on load
init_db()
