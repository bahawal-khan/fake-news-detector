"""
Fake News Detector — Streamlit version.

An all-in-one deployment: no separate FastAPI backend + React frontend
needed. Same trained model (vectorizer.pkl, scaler.pkl, model.pkl) as the
main project, just wrapped in a single-file Streamlit UI.
"""

import sys
from pathlib import Path

import streamlit as st

# Make the shared "app" package (predict.py, preprocess.py, features.py)
# importable from backend/
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.predict import get_model  # noqa: E402
from app.preprocess import has_sufficient_text_content  # noqa: E402

st.set_page_config(
    page_title="The Verification Desk · Fake News Detector",
    page_icon="🕵️",
    layout="centered",
)

st.title("🕵️ The Verification Desk")
st.caption(
    "Classic ML (TF-IDF + handcrafted linguistic features + Linear SVM) — "
    "no deep learning. Paste a headline and article body to get a verdict."
)

with st.form("article_form"):
    title = st.text_input("Headline (optional)", placeholder="e.g. City council approves annual budget")
    text = st.text_area(
        "Article body",
        placeholder="Paste the full article text here…",
        height=220,
    )
    submitted = st.form_submit_button("Investigate", use_container_width=True)

if submitted:
    if not text or len(text.strip()) < 10:
        st.error("Paste at least a sentence or two of article body text.")
    elif not has_sufficient_text_content(text):
        st.error(
            "This doesn't look like readable article text (too few real words "
            "after removing numbers/symbols). Please paste actual sentences."
        )
    else:
        with st.spinner("Cross-referencing linguistic evidence…"):
            model = get_model()
            result = model.predict(title, text)

        if result["is_fake"]:
            st.error(f"### 🛑 Verdict: FAKE  \nConfidence: **{result['confidence']*100:.1f}%**")
        else:
            st.success(f"### ✅ Verdict: REAL  \nConfidence: **{result['confidence']*100:.1f}%**")

        col1, col2 = st.columns(2)
        col1.metric("Fake probability", f"{result['fake_probability']*100:.1f}%")
        col2.metric("Real probability", f"{result['real_probability']*100:.1f}%")

        st.caption(f"Model: {result['model_name']}")

st.divider()
st.caption(
    "⚠️ Trained on 2016–2017 US political/world news (Kaggle Fake and Real "
    "News Dataset). Judges writing style only — not a fact-checker, and may "
    "not generalize well to other topics, sources, or time periods."
)