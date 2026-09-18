"""
modules/topic_modeling.py
Unsupervised thematic clustering using Non-Negative Matrix Factorization (NMF).
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def discover_topics(texts: pd.Series, sentiments: pd.Series, n_topics: int = 4) -> List[Dict[str, Any]]:
    """
    Discovers major topics, keyword descriptors, response prevalence, and topic sentiment.
    """
    valid_mask = texts.notna() & (texts.str.strip().str.len() > 5)
    valid_texts = texts[valid_mask].tolist()
    valid_sentiments = sentiments[valid_mask].tolist() if sentiments is not None else [0.0] * len(valid_texts)

    if len(valid_texts) < n_topics * 3:
        return []

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.85,
        min_df=2,
        max_features=400
    )

    try:
        X = tfidf.fit_transform(valid_texts)
    except ValueError:
        return []

    actual_topics = min(n_topics, X.shape[1] - 1)
    if actual_topics < 2:
        return []

    nmf = NMF(n_components=actual_topics, random_state=42, init="nndsvda")
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = tfidf.get_feature_names_out()

    assigned_topics = np.argmax(W, axis=1)
    total_valid = len(valid_texts)

    topic_summaries = []
    for topic_idx in range(actual_topics):
        # Top 4 terms for this topic
        top_term_indices = H[topic_idx].argsort()[:-5:-1]
        top_words = [feature_names[i] for i in top_term_indices]

        # Calculate responses assigned to this topic
        topic_count = int(np.sum(assigned_topics == topic_idx))
        pct_share = round((topic_count / total_valid) * 100, 1)

        # Average sentiment for responses in this topic
        topic_sent_scores = [valid_sentiments[i] for i in range(total_valid) if assigned_topics[i] == topic_idx]
        mean_sentiment = round(float(np.mean(topic_sent_scores)), 2) if topic_sent_scores else 0.0

        topic_summaries.append({
            "Topic ID": topic_idx + 1,
            "Topic Name": " / ".join([w.title() for w in top_words[:2]]),
            "Top Keywords": ", ".join(top_words),
            "Responses": topic_count,
            "Share %": pct_share,
            "Mean Sentiment": mean_sentiment
        })

    return sorted(topic_summaries, key=lambda x: x["Responses"], reverse=True)
