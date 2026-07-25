# 🕵️ The Verification Desk — Fake News Detector

An end-to-end **classic Machine Learning** (no deep learning) fake news classifier:
EDA → text preprocessing → feature engineering → TF-IDF vectorization → model
comparison → FastAPI backend → React frontend.

Trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(Kaggle, ~44K articles). Best model: **Linear SVM** on TF-IDF (1–2 grams) +
handcrafted linguistic features — **99.3% F1** on held-out test data.

---

## 📁 Project Structure

```
fake-news-detector/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app + routes (/predict, /health)
│   │   ├── predict.py       # loads artifacts, runs inference
│   │   ├── preprocess.py    # text cleaning (shared with notebooks)
│   │   ├── features.py      # handcrafted feature engineering (shared)
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── model/           # saved vectorizer.pkl, scaler.pkl, model.pkl
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # main UI — "Verification Desk"
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

### 1. Backend (FastAPI)

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

> The `backend/app/model/` folder already contains trained artifacts
> (`vectorizer.pkl`, `scaler.pkl`, `model.pkl`, `metadata.pkl`), so the API
> works immediately — no retraining needed.

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
cp .env.example .env      # set VITE_API_URL if backend isn't on localhost:8000
npm run dev
```

Open http://localhost:5173

### 3. Retraining the model (optional)

Either run the notebooks in order (`01_eda.ipynb` → `02_feature_engineering_and_modeling.ipynb`),
or run the equivalent script directly:

```bash
cd notebooks
pip install -r ../backend/requirements.txt jupyter nbconvert matplotlib seaborn wordcloud
python train_pipeline.py
```

Both regenerate the exact same artifacts consumed by the backend.

---

## 🐳 Docker (backend)

```bash
cd backend
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api
```

---

## ☁️ Deployment

| Component | Suggested platform | Notes |
|---|---|---|
| Backend | Render / Railway | Point it at `backend/`, it will pick up the `Dockerfile` automatically. Set `PORT=8000`. |
| Frontend | Vercel / Netlify | Point it at `frontend/`, build command `npm run build`, output dir `dist`. Set `VITE_API_URL` env var to your deployed backend URL. |

---

## 📊 API Reference

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

The API guards against bad input at two levels:

1. **Wrong data type** (e.g. sending `"text": 123456789` as a raw JSON number
   instead of a string) → Pydantic rejects it with a `422` error before it
   ever reaches the model: `"Input should be a valid string"`.
2. **Valid string, but no real content** (e.g. pasting only digits, symbols,
   or gibberish — technically a string, since form/textarea inputs are
   always strings) → after cleaning, the backend checks that at least 5 real
   alphabetic words remain. If not, it returns a `400` error:
   `"This doesn't look like readable article text..."` instead of silently
   returning a misleading high-confidence prediction on empty/garbage input.

## ⚠️ Known limitations

- Trained on 2016–2017 US political/world news — may not generalize well to
  other domains, languages, or more recent misinformation styles.
- `subject` and publication metadata were deliberately excluded (they caused
  leakage), so the model judges **writing style only**, not source credibility
  or fact-checking against real-world evidence.
- Not a substitute for actual fact-checking — treat predictions as a
  linguistic-style signal, not a ground truth verdict.

---

## 🛠️ Tech Stack

**ML:** scikit-learn, pandas, numpy, NLTK, TextBlob
**Backend:** FastAPI, uvicorn, Pydantic
**Frontend:** React 19, Vite, axios
**Deployment:** Docker, Render/Railway (API), Vercel/Netlify (frontend)
