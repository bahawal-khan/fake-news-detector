# 🕵️ The Verification Desk — Fake News Detector

🔗 **Live Demo:** [fake-news-detector-3wnfv6xfig3gk4ujww4mcq.streamlit.app](https://fake-news-detector-3wnfv6xfig3gk4ujww4mcq.streamlit.app)
*(Free tier — the app sleeps after 12h of inactivity. If you see a "wake up" screen, just click it and wait ~30–60s.)*

An end-to-end **classic Machine Learning** (no deep learning) fake news classifier:
EDA → text preprocessing → feature engineering → TF-IDF vectorization → model
comparison → deployable backend + frontend.

Trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(Kaggle, ~44K articles). Best model: **Linear SVM** on TF-IDF (1–2 grams) +
handcrafted linguistic features — **99.3% F1** on held-out test data.

---

## 📁 Project Structure

```
fake-news-detector/
├── streamlit_app.py         # ⭐ all-in-one deployment (currently live — see demo link above)
├── requirements.txt         # dependencies for the Streamlit deployment
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app + routes (/predict, /health)
│   │   ├── predict.py             # loads artifacts, runs inference
│   │   ├── preprocess.py          # text cleaning + input-quality guard (shared everywhere)
│   │   ├── features.py            # handcrafted feature engineering (shared everywhere)
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── model/                 # saved vectorizer.pkl, scaler.pkl, model.pkl
│   ├── pythonanywhere_app.py      # Flask/WSGI version, for PythonAnywhere's free tier
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # main UI — "Verification Desk" (talks to the FastAPI backend)
│   │   ├── api.js               # axios client for the FastAPI backend
│   │   └── components/
│   │       ├── VerdictStamp.jsx
│   │       └── ConfidenceMeter.jsx
│   └── package.json
├── notebooks/
│   ├── 01_eda.ipynb                              # exploratory data analysis
│   ├── 02_feature_engineering_and_modeling.ipynb # features, TF-IDF, training, eval
│   └── train_pipeline.py                         # same pipeline as a plain script
├── data/
│   ├── Fake.csv
│   └── True.csv
└── README.md
```

**Three ways to run this project**, all sharing the exact same trained model and
preprocessing/feature-engineering code:

1. **Streamlit (`streamlit_app.py`)** — one file, one deployment, no separate
   frontend/backend. This is what's currently live (link above).
2. **FastAPI + React (`backend/` + `frontend/`)** — the "real" production-style
   split architecture, for local development or platforms that support Docker
   (Render, Koyeb, Railway, etc.).
3. **Flask (`backend/pythonanywhere_app.py`)** — a WSGI version of the same
   API, specifically for PythonAnywhere's free tier (which doesn't support
   FastAPI's ASGI).

---

## 🧠 What's inside the ML pipeline

| Stage | Details |
|---|---|
| **EDA** | Class balance, text-length distributions, punctuation/caps patterns, word clouds. Found and documented two data-leakage sources (`subject` column, `"(Reuters) -"` byline) — both excluded from training. |
| **Preprocessing** | Lowercasing, URL/HTML stripping, stopword removal, lemmatization (NLTK). |
| **Feature engineering** | `char_count`, `word_count`, `avg_word_length`, `exclamation_count`, `question_count`, `uppercase_word_count`, `punctuation_count`, `sentiment_polarity`, `sentiment_subjectivity` (TextBlob), `title_word_count`. |
| **Vectorization** | TF-IDF, unigrams + bigrams, 15,000 max features. |
| **Models compared** | Multinomial Naive Bayes, Logistic Regression, Linear SVM (calibrated for probabilities). |
| **Best model** | Linear SVM — Accuracy 0.993, Precision 0.993, Recall 0.994, F1 0.993, ROC-AUC 0.9996. |
| **No deep learning** | Pure scikit-learn — no transformers, no neural nets, anywhere in the pipeline. |

---

## 🚀 Running locally

### Option A — Streamlit (simplest, one command)

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
streamlit run streamlit_app.py
```

Opens automatically at http://localhost:8501

### Option B — FastAPI backend + React frontend

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

uvicorn app.main:app --reload --port 8000
```
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

**Frontend:**
```bash
cd frontend
npm install
cp .env.example .env      # set VITE_API_URL if backend isn't on localhost:8000
npm run dev
```
Open http://localhost:5173

> The `backend/app/model/` folder already contains trained artifacts
> (`vectorizer.pkl`, `scaler.pkl`, `model.pkl`, `metadata.pkl`), so both
> options above work immediately — no retraining needed.

### Retraining the model (optional)

Either run the notebooks in order (`01_eda.ipynb` → `02_feature_engineering_and_modeling.ipynb`),
or run the equivalent script directly:

```bash
cd notebooks
pip install -r ../backend/requirements.txt jupyter nbconvert matplotlib seaborn wordcloud
python train_pipeline.py
```

Both regenerate the exact same artifacts consumed by every deployment option above.

---

## ☁️ Deployment

This project is set up for **free, no-credit-card deployment** — here's what
was actually used and what else works:

| Component | Platform used | Why |
|---|---|---|
| **Currently live** | **Streamlit Community Cloud** (`streamlit_app.py`) | Free, no card required, single-file deploy — GitHub repo → share.streamlit.io → point at `streamlit_app.py` → done. |
| Backend alternative | **PythonAnywhere** (`backend/pythonanywhere_app.py`) | Free, no card required. Free tier only supports WSGI, so this is a Flask version of the same API — same model, same logic. |
| Backend + Frontend (split architecture) | Render / Koyeb (backend, via `backend/Dockerfile`) + Vercel / Netlify (frontend) | These support the full FastAPI + React split, but most now require card verification even on free tiers — use only if you have a card, or if a currently-card-free provider is available when you deploy. |

**Streamlit Cloud auto-redeploys** on every `git push` to `main` — no manual
redeploy step needed once it's connected.

---

## 📊 API Reference (FastAPI / Flask versions)

### `POST /predict`

```json
{
  "title": "City council approves annual infrastructure budget",
  "text": "Members of the city council voted 6 to 1 on Tuesday..."
}
```

**Response:**

```json
{
  "label": "Real",
  "is_fake": false,
  "confidence": 0.9992,
  "fake_probability": 0.0008,
  "real_probability": 0.9992,
  "model_name": "LinearSVC"
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "model_name": "LinearSVC" }
```

---

## 🛡️ Input validation

The API guards against bad input at two levels (all three deployment
versions — Streamlit, FastAPI, Flask — share this logic):

1. **Wrong data type** (e.g. sending `"text": 123456789` as a raw JSON number
   instead of a string) → rejected before it ever reaches the model.
2. **Valid string, but no real content** (e.g. pasting only digits, symbols,
   or gibberish — technically a string, since form/textarea inputs are
   always strings) → after cleaning, a check confirms at least 5 real
   alphabetic words remain. If not, a clear error is returned:
   `"This doesn't look like readable article text..."` instead of silently
   returning a misleading high-confidence prediction on empty/garbage input.

## ⚠️ Known limitations

- Trained on 2016–2017 US political/world news — may not generalize well to
  other domains, languages, sources, or more recent misinformation styles.
  Real-world articles that differ in topic or source from the training data
  can occasionally be misclassified — this is an expected generalization
  limit, not a bug.
- `subject` and publication metadata were deliberately excluded (they caused
  leakage), so the model judges **writing style only**, not source credibility
  or fact-checking against real-world evidence.
- Not a substitute for actual fact-checking — treat predictions as a
  linguistic-style signal, not a ground truth verdict.

---

## 🛠️ Tech Stack

**ML:** scikit-learn, pandas, numpy, NLTK, TextBlob
**Deployed as:** Streamlit (live demo)
**Also included:** FastAPI + React (split architecture), Flask (PythonAnywhere-ready)
**Deployment platforms used/supported:** Streamlit Community Cloud, PythonAnywhere, Docker, Render/Koyeb, Vercel/Netlify
