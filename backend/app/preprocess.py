"""
Text preprocessing utilities for the Fake News Detector.
Same functions are used at training time (notebooks) and inference time (FastAPI)
so that train/serve skew never happens.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure required NLTK corpora are available (no-op if already downloaded)
for _pkg in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(
            f"corpora/{_pkg}" if _pkg not in ("punkt", "punkt_tab") else f"tokenizers/{_pkg}"
        )
    except LookupError:
        nltk.download(_pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# The True.csv articles in this dataset almost always start with a
# "CITY (Reuters) -" byline. If left in, a model can learn to predict
# "real" just by spotting the word "Reuters" - that's leakage, not
# genuine fake-news detection. We strip it out during cleaning.
REUTERS_BYLINE_PATTERN = re.compile(r"^.*?\(reuters\)\s*-\s*", flags=re.IGNORECASE)


def strip_reuters_byline(text: str) -> str:
    """Remove leading 'CITY (Reuters) -' bylines that leak the source label."""
    return REUTERS_BYLINE_PATTERN.sub("", text, count=1)


def clean_text(text: str) -> str:
    """Lowercase, strip noise, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""

    text = strip_reuters_byline(text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    tokens = text.split()
    tokens = [
        LEMMATIZER.lemmatize(tok) for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def count_exclamations(text: str) -> int:
    return text.count("!")


def count_question_marks(text: str) -> int:
    return text.count("?")


def count_uppercase_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for w in text.split() if len(w) > 1 and w.isupper())


def count_punctuation(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for ch in text if ch in string.punctuation)


def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def avg_word_length(text: str) -> float:
    if not isinstance(text, str) or len(text.split()) == 0:
        return 0.0
    words = text.split()
    return sum(len(w) for w in words) / len(words)


def has_sufficient_text_content(raw_text: str, min_alpha_words: int = 5) -> bool:
    """
    Guards against inputs that are technically valid strings but carry no
    real linguistic signal for the model to work with - e.g. pure numbers,
    symbols, a single repeated character, or random keyboard mashing.

    We check the CLEANED text (after stripping non-alphabetic characters,
    stopwords, etc.) still has a minimum number of real words left.
    """
    cleaned = clean_text(raw_text)
    alpha_words = [w for w in cleaned.split() if w.isalpha()]
    return len(alpha_words) >= min_alpha_words
