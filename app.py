"""
============================================
🌐 app.py — Streamlit UI for Spam Detection
============================================
Premium UI with real-time prediction, confidence scores,
spam word highlighting, and prediction history.
============================================
"""

import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.predictor import SpamPredictor

# ── Page Config ──
st.set_page_config(
    page_title="SpamShield AI — Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium White Theme ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
    }

    /* Force white background everywhere */
    .main .block-container { background-color: #ffffff; }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fb !important;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] * { color: #1f2937 !important; }

    .main-header {
        background: linear-gradient(135deg, #4f46e5, #6366f1, #818cf8);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(79, 70, 229, 0.18);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-spam {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #ef4444;
    }
    .result-spam h2 { color: #dc2626; }
    .result-spam .label { color: #991b1b; }
    .result-ham {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
    }
    .result-ham h2 { color: #16a34a; }
    .result-ham .label { color: #166534; }
    .result-card h2 { font-size: 2rem; margin: 0; font-weight: 700; }
    .result-card .label { font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem; }

    .metric-card {
        background: #f8f9fb;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .metric-card h3 { color: #6b7280; font-size: 0.8rem; margin: 0; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card p { color: #1f2937; font-size: 1.8rem; margin: 0.3rem 0 0 0; font-weight: 700; }

    .spam-word-tag {
        display: inline-block;
        background: #fef2f2;
        color: #dc2626;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85rem;
        border: 1px solid #fecaca;
        font-weight: 500;
    }

    .history-item {
        background: #f8f9fb;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        border: 1px solid #e5e7eb;
        color: #1f2937;
    }

    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 2.5rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #5b21b6) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* General text color override for white bg */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #1f2937 !important;
    }
    .stMarkdown hr { border-color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ──
@st.cache_resource
def load_predictor():
    """Load model and vectorizer (cached so it loads only once)."""
    predictor = SpamPredictor()
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    vectorizer_path = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        predictor.load(model_path, vectorizer_path)
        return predictor
    return None


# ── Initialize Session State ──
if "history" not in st.session_state:
    st.session_state.history = []


def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ SpamShield AI</h1>
        <p>Intelligent Email & SMS Spam Detection powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    predictor = load_predictor()

    if predictor is None:
        st.error("⚠️ Models not found! Run `python train_pipeline.py` first to train the models.")
        st.code("python train_pipeline.py", language="bash")
        return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        st.markdown("---")

        total = len(st.session_state.history)
        spam_count = sum(1 for h in st.session_state.history if h["is_spam"])
        ham_count = total - spam_count

        st.metric("Total Checked", total)
        st.metric("Spam Detected", spam_count)
        st.metric("Safe Messages", ham_count)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "SpamShield AI uses **TF-IDF** vectorization and "
            "**Machine Learning** to classify messages as spam or ham "
            "with high accuracy."
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    # ── Main Input ──
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### ✉️ Enter Message to Analyze")
        user_input = st.text_area(
            "Paste your email or SMS text below:",
            height=150,
            placeholder="",
            label_visibility="collapsed",
        )

        analyze_clicked = st.button("🔍 Analyze Message", use_container_width=True)

    with col2:
        st.markdown("### 📝 Quick Test Samples")
        sample_spam = [
            "WINNER!! You've been selected for a $1000 prize. Call 1-800-FREE now!",
            "Urgent: Your bank account will be suspended. Click here to verify.",
            "FREE entry to our weekly competition. Text WIN to 80085",
        ]
        sample_ham = [
            "Hey, are you coming to the meeting at 3pm?",
            "Mom said dinner will be ready by 7. Don't be late!",
            "Can you pick up some groceries on your way home?",
        ]

        st.markdown("**Spam samples:**")
        for s in sample_spam:
            if st.button(f"🔴 {s[:45]}...", key=f"spam_{s[:20]}"):
                st.session_state["sample_text"] = s
                st.rerun()

        st.markdown("**Ham samples:**")
        for s in sample_ham:
            if st.button(f"🟢 {s[:45]}...", key=f"ham_{s[:20]}"):
                st.session_state["sample_text"] = s
                st.rerun()

    # Feature: Bulk File Upload
    st.markdown("---")
    with st.expander("📁 Bulk Email Scanner (Upload TXT/CSV)"):
        uploaded_file = st.file_uploader("Upload CSV (must have 'message' column) or TXT", type=["csv", "txt"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df_bulk = pd.read_csv(uploaded_file)
                text_col = next((col for col in ['message', 'text', 'content'] if col in df_bulk.columns.str.lower()), None)
                if text_col:
                    texts = df_bulk[text_col].dropna().astype(str).tolist()
                else:
                    st.error("No valid text column found in CSV.")
                    texts = []
            else:
                texts = [line.strip() for line in uploaded_file.read().decode("utf-8").split('\n') if line.strip()]
            
            if texts:
                st.info(f"Scanning {len(texts)} messages...")
                bulk_results = predictor.batch_predict(texts)
                df_res = pd.DataFrame(bulk_results)[["original_text", "prediction", "confidence", "risk_level"]]
                st.dataframe(df_res)

    # Handle sample text selection
    if "sample_text" in st.session_state and not user_input:
        user_input = st.session_state.pop("sample_text")

    # ── Prediction ──
    if analyze_clicked and user_input.strip():
        result = predictor.predict(user_input.strip())

        # Save to history
        st.session_state.history.insert(0, result)

        # ── Result Display ──
        st.markdown("---")

        if result["is_spam"]:
            # Feature: Real-Time Alert Messages
            st.toast("🚨 Dangerous Spam Detected!", icon="⚠️")
            
            # Feature: Phishing URL Detection Warning
            if result.get("phishing_info", {}).get("is_phishing"):
                st.error("🛑 PHISHING ALERT: Suspicious URLs or intents detected in this message. Do not click any links!")
                
            st.markdown(f"""
            <div class="result-card result-spam">
                <h2>🚨 SPAM DETECTED</h2>
                <div class="label">This message has been classified as spam</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-ham">
                <h2>✅ SAFE MESSAGE</h2>
                <div class="label">This message appears to be legitimate</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Metrics Row ──
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <p>{result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Spam Probability</h3>
                <p>{result['spam_probability']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Ham Probability</h3>
                <p>{result['ham_probability']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            risk_colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
            color = risk_colors.get(result["risk_level"], "#fff")
            st.markdown(f"""
            <div class="metric-card">
                <h3>Risk Level</h3>
                <p style="color: {color}">{result['risk_level']}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability Gauge ──
        st.markdown("#### 📊 Spam Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result["spam_probability"] * 100,
            title={"text": "Spam Score", "font": {"size": 18, "color": "#4f46e5"}},
            number={"suffix": "%", "font": {"color": "#1f2937", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                "bar": {"color": "#4f46e5"},
                "bgcolor": "#f3f4f6",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                    {"range": [30, 70], "color": "rgba(245,158,11,0.15)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(
            height=280,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#1f2937"},
            margin=dict(t=60, b=20, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Spam Words Highlight ──
        if result.get("spam_words_found"):
            st.markdown("#### 🔍 Detected Spam Keywords")
            words_html = " ".join(
                f'<span class="spam-word-tag">{w}</span>'
                for w in result["spam_words_found"]
            )
            st.markdown(words_html, unsafe_allow_html=True)

        # ── Feature 1: Fake Urgency Detector ──
        if result.get("urgency_phrases"):
            st.warning("⚠️ **Fake Urgency Detected**")
            st.write("Detected phrases:")
            for phrase in result["urgency_phrases"]:
                st.markdown(f"- `{phrase}`")

        # ── Feature 2: Email Safety Score ──
        safety = result.get("safety_info", {})
        if safety:
            st.markdown("---")
            st.markdown("### 🛡️ Email Safety Score")
            score_color = "#22c55e" if safety['risk'] == "Safe" else ("#f59e0b" if safety['risk'] == "Moderate Risk" else "#ef4444")
            st.markdown(f"**Safety Score:** <span style='color:{score_color}; font-size:1.2rem'>**{safety['score']}/100**</span>", unsafe_allow_html=True)
            st.markdown(f"**Risk Level:** <span style='color:{score_color}; font-size:1.1rem'>{safety['risk']}</span>", unsafe_allow_html=True)

        # ── Details Expander ──
        with st.expander("🔬 Technical Details"):
            st.markdown(f"**Preprocessed Text:** `{result['preprocessed_text']}`")
            st.json({
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "spam_probability": result["spam_probability"],
                "ham_probability": result["ham_probability"],
                "risk_level": result["risk_level"],
                "spam_words_found": result["spam_words_found"],
            })

    elif analyze_clicked:
        st.warning("⚠️ Please enter a message to analyze.")

    # ── Prediction History ──
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Prediction History")

        for i, h in enumerate(st.session_state.history[:10]):
            icon = "🚨" if h["is_spam"] else "✅"
            label = "SPAM" if h["is_spam"] else "SAFE"
            conf = f"{h['confidence']:.0%}"
            msg_preview = h["original_text"][:80] + ("..." if len(h["original_text"]) > 80 else "")

            st.markdown(f"""
            <div class="history-item">
                <strong>{icon} {label}</strong> &nbsp;|&nbsp;
                Confidence: <strong>{conf}</strong> &nbsp;|&nbsp;
                <span style="opacity:0.7">{msg_preview}</span>
            </div>
            """, unsafe_allow_html=True)

        # Feature: Export Results & Word Cloud
        if len(st.session_state.history) > 0:
            colA, colB = st.columns(2)
            
            with colA:
                st.markdown("<br>", unsafe_allow_html=True)
                df_hist = pd.DataFrame(st.session_state.history)
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export History as CSV", data=csv, file_name="prediction_history.csv", mime="text/csv", use_container_width=True)
            
            with colB:
                spam_texts = [h["original_text"] for h in st.session_state.history if h["is_spam"]]
                if spam_texts:
                    with st.expander("☁️ Show Spam Word Cloud"):
                        wordcloud = WordCloud(width=500, height=250, background_color="white", colormap="Reds").generate(" ".join(spam_texts))
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)

    # ── Model Comparison Section ──
    comparison_path = os.path.join(PROJECT_ROOT, "models", "model_comparison.csv")
    if os.path.exists(comparison_path):
        st.markdown("---")
        st.markdown("### 🏆 Model Performance Comparison")

        comp_df = pd.read_csv(comparison_path)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            comp_df,
            x="Model",
            y=["Accuracy", "Precision", "Recall", "F1-Score"],
            barmode="group",
            color_discrete_sequence=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"],
        )
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#1f2937"},
            xaxis={"gridcolor": "#e5e7eb"},
            yaxis={"gridcolor": "#e5e7eb", "range": [0.8, 1.0]},
            legend={"orientation": "h", "y": -0.2},
            height=400,
            margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
