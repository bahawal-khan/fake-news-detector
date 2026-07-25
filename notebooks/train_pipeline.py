"""
End-to-end training pipeline for the Fake News Detector.

This script is the source of truth for how the model is trained. The two
notebooks (01_eda.ipynb and 02_feature_engineering_and_modeling.ipynb) walk
through this same logic interactively, cell by cell, with explanations,
plots, and printed metrics.

Run:
    python train_pipeline.py
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "backend" / "app" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "backend"))
from app.preprocess import clean_text  # noqa: E402
from app.features import build_feature_frame, FEATURE_COLUMNS  # noqa: E402

RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def load_and_label():
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"] = 0  # 0 = fake
    true["label"] = 1  # 1 = real
    df = pd.concat([fake, true], axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    df = df[(df["text"].str.strip() != "")].reset_index(drop=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def main():
    log("Loading data...")
    df = load_and_label()
    log(f"Total rows after dedup: {len(df)}")
    log(f"Label distribution:\n{df['label'].value_counts()}")

    log("Cleaning text (this takes a minute)...")
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_combined"] = df["clean_title"] + " " + df["clean_text"]

    log("Building handcrafted features...")
    feat_df = build_feature_frame(df["title"], df["text"])

    log("Splitting train/test (80/20, stratified)...")
    X_text = df["clean_combined"]
    X_feats = feat_df.values
    y = df["label"].values

    X_text_train, X_text_test, X_feats_train, X_feats_test, y_train, y_test = train_test_split(
        X_text, X_feats, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    log("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)
    log(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

    log("Scaling handcrafted features...")
    scaler = StandardScaler()
    X_feats_train_scaled = scaler.fit_transform(X_feats_train)
    X_feats_test_scaled = scaler.transform(X_feats_test)

    log("Combining TF-IDF + handcrafted features...")
    X_train_combined = hstack([X_tfidf_train, csr_matrix(X_feats_train_scaled)]).tocsr()
    X_test_combined = hstack([X_tfidf_test, csr_matrix(X_feats_test_scaled)]).tocsr()

    candidates = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=5, random_state=RANDOM_STATE
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, C=0.5), cv=3
        ),
    }

    results = {}
    log("Training candidate models...")
    for name, clf in candidates.items():
        t0 = time.time()
        # MultinomialNB needs non-negative features; TF-IDF + scaled feats
        # can go slightly negative after StandardScaler, so give NB its
        # own unscaled combined matrix.
        if name == "MultinomialNB":
            X_tr = hstack([X_tfidf_train, csr_matrix(X_feats_train)]).tocsr()
            X_te = hstack([X_tfidf_test, csr_matrix(X_feats_test)]).tocsr()
            X_tr.data[X_tr.data < 0] = 0
        else:
            X_tr, X_te = X_train_combined, X_test_combined

        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else preds

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)
        log(
            f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
            f"f1={f1:.4f} auc={auc:.4f} ({time.time()-t0:.1f}s)"
        )

    best_name = max(results, key=lambda k: results[k]["f1"])
    log(f"Best model by F1: {best_name}")

    best_clf = candidates[best_name]
    if best_name == "MultinomialNB":
        X_te_final = hstack([X_tfidf_test, csr_matrix(X_feats_test)]).tocsr()
        X_te_final.data[X_te_final.data < 0] = 0
    else:
        X_te_final = X_test_combined

    final_preds = best_clf.predict(X_te_final)
    log("\nClassification report (best model):\n" + classification_report(
        y_test, final_preds, target_names=["Fake", "Real"]
    ))
    log("Confusion matrix:\n" + str(confusion_matrix(y_test, final_preds)))

    log("Saving artifacts...")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(best_clf, MODEL_DIR / "model.pkl")
    joblib.dump(
        {"model_name": best_name, "feature_columns": FEATURE_COLUMNS, "metrics": results[best_name]},
        MODEL_DIR / "metadata.pkl",
    )

    pd.DataFrame(results).T.to_csv(MODEL_DIR / "model_comparison.csv")
    log(f"Artifacts saved to {MODEL_DIR}")
    log("Done.")


if __name__ == "__main__":
    main()
