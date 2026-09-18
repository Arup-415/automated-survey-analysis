"""
modules/keyword_extraction.py
Salient keyword and n-gram extraction using TF-IDF ranking.
"""

from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SURVEY_STOPWORDS = {
    "survey", "question", "response", "feedback", "please", "would",
    "get", "also", "could", "even", "really", "much", "many", "make",
    "na", "none", "nothing", "email", "phone", "url"
}

def extract_top_keywords(texts: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """
    Extracts top unigrams and bigrams ranked by mean TF-IDF scores across responses.
    """
    clean_corpus = texts.dropna().astype(str).tolist()
    clean_corpus = [t for t in clean_corpus if len(t.strip()) > 3]

    n_docs = len(clean_corpus)
    if n_docs < 2:
        return pd.DataFrame(columns=["Keyword / Phrase", "Relevance Score"])

    # Adapt max_df dynamically so small corpora aren't completely pruned
    max_df_val = 1.0 if n_docs < 10 else 0.85

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_df=max_df_val,
        min_df=1,
        max_features=500
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(clean_corpus)
    except ValueError:
        return pd.DataFrame(columns=["Keyword / Phrase", "Relevance Score"])

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1

    keyword_scores = []
    for word, score in zip(feature_names, mean_scores):
        tokens = word.split()
        if not any(token in SURVEY_STOPWORDS for token in tokens):
            keyword_scores.append({"Keyword / Phrase": word, "Relevance Score": round(float(score) * 100, 2)})

    df_keywords = pd.DataFrame(keyword_scores)
    if df_keywords.empty:
        return pd.DataFrame(columns=["Keyword / Phrase", "Relevance Score"])

    return df_keywords.sort_values(by="Relevance Score", ascending=False).head(top_n).reset_index(drop=True)
