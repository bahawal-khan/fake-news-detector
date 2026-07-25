"""
Handcrafted linguistic feature engineering for the Fake News Detector.
These features are computed on the RAW (uncleaned) text, then combined
with TF-IDF vectors of the CLEANED text before being fed to the model.
"""

import numpy as np
import pandas as pd
from textblob import TextBlob

from .preprocess import (
    avg_word_length,
    count_exclamations,
    count_punctuation,
    count_question_marks,
    count_uppercase_words,
    word_count,
)

FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "question_count",
    "uppercase_word_count",
    "punctuation_count",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "title_word_count",
]


def build_feature_frame(titles: pd.Series, texts: pd.Series) -> pd.DataFrame:
    """Compute handcrafted features for a batch of (title, text) pairs."""
    texts = texts.fillna("")
    titles = titles.fillna("")

    feats = pd.DataFrame(index=texts.index)
    feats["char_count"] = texts.str.len()
    feats["word_count"] = texts.apply(word_count)
    feats["avg_word_length"] = texts.apply(avg_word_length)
    feats["exclamation_count"] = texts.apply(count_exclamations)
    feats["question_count"] = texts.apply(count_question_marks)
    feats["uppercase_word_count"] = texts.apply(count_uppercase_words)
    feats["punctuation_count"] = texts.apply(count_punctuation)

    polarity, subjectivity = [], []
    for t in texts:
        try:
            blob = TextBlob(t[:2000])  # cap length for speed
            polarity.append(blob.sentiment.polarity)
            subjectivity.append(blob.sentiment.subjectivity)
        except Exception:
            polarity.append(0.0)
            subjectivity.append(0.0)
    feats["sentiment_polarity"] = polarity
    feats["sentiment_subjectivity"] = subjectivity

    feats["title_word_count"] = titles.apply(word_count)

    return feats[FEATURE_COLUMNS].astype(float)


def build_feature_row(title: str, text: str) -> np.ndarray:
    """Convenience wrapper for a single (title, text) prediction request."""
    frame = build_feature_frame(pd.Series([title]), pd.Series([text]))
    return frame.values[0]
