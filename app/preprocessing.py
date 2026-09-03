"""Text preprocessing for the fake news detector.

The preprocessing steps mirror exactly the steps used to train the model
(see ``fake_news.ipynb``): strip non-alphabetic characters, lowercase,
remove stopwords (keeping ``not`` so negations are preserved) and stem.
"""

from __future__ import annotations

import re
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_NON_ALPHA = re.compile(r"[^a-zA-Z]")
_STOPWORDS_LOCK: stopwords | None = None


def _get_stopwords() -> set[str]:
    """Return the cached set of stopwords with ``not`` removed."""
    global _STOPWORDS_LOCK
    if _STOPWORDS_LOCK is not None:
        return _STOPWORDS_LOCK
    try:
        words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))
    words.discard("not")
    _STOPWORDS_LOCK = words
    return words


def clean_single_text(text: str) -> str:
    """Preprocess a single raw string into a cleaned, space-joined corpus row.

    Returns the cleaned text string ready to be fed to the vectorizer.
    """
    cleaned = _NON_ALPHA.sub(" ", text)
    cleaned = cleaned.lower()
    tokens = cleaned.split()

    if not tokens:
        return ""

    stopwords_set = _get_stopwords()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords_set]
    return " ".join(tokens)


def clean_corpus(texts: Iterable[str]) -> list[str]:
    """Preprocess an iterable of raw texts into a cleaned corpus list."""
    return [clean_single_text(text) for text in texts]


def tokenize(text: str) -> list[str]:
    """Split a raw string into its (unstemmed) lowercased tokens."""
    cleaned = _NON_ALPHA.sub(" ", text)
    return cleaned.lower().split()
