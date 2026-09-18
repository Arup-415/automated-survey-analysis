"""
modules/sentiment.py
Sentiment classification and lightweight Aspect-Based Sentiment Analysis (ABSA).
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

# Initialize shared resources
_vader = SentimentIntensityAnalyzer()
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

def analyze_document_sentiment(texts: pd.Series) -> pd.DataFrame:
    """
    Computes sentiment polarities and labels for each response in a text series.
    Labels: Positive (>= 0.05), Neutral (-0.05 < x < 0.05), Negative (<= -0.05).
    """
    results = []
    for text in texts:
        if pd.isna(text) or str(text).strip() == "":
            results.append({
                "sentiment": "Neutral",
                "compound_score": 0.0,
                "pos_score": 0.0,
                "neu_score": 1.0,
                "neg_score": 0.0
            })
            continue

        scores = _vader.polarity_scores(str(text))
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "sentiment": label,
            "compound_score": round(compound, 3),
            "pos_score": round(scores["pos"], 3),
            "neu_score": round(scores["neu"], 3),
            "neg_score": round(scores["neg"], 3)
        })

    return pd.DataFrame(results)

def extract_aspect_sentiments(texts: List[str], min_aspect_freq: int = 3) -> pd.DataFrame:
    """
    Extracts aspect-opinion pairs using dependency parsing:
    Noun/Aspect -> Adjectival modifier/Sentiment.
    """
    if _nlp is None:
        return pd.DataFrame(columns=["Aspect", "Mentions", "Average Sentiment", "Dominant Polarity"])

    aspect_records = []
    # Sample if large corpus to ensure responsive local CPU processing
    sample_texts = texts[:min(len(texts), 1500)]

    for doc in _nlp.pipe([str(t) for t in sample_texts if pd.notna(t) and len(str(t).strip()) > 3], batch_size=50):
        for token in doc:
            # Check for noun aspects modified by an adjective
            if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2:
                aspect = token.lemma_.lower()
                
                # Look for direct adjectival modifiers (amod) or attribute adjectives
                modifiers = [child for child in token.children if child.dep_ in ("amod", "acomp")]
                
                # Check for linking verbs: e.g. "The UI is great" (token -> head verb -> child adj)
                if token.dep_ == "nsubj" and token.head.pos_ == "AUX":
                    modifiers.extend([child for child in token.head.children if child.dep_ in ("acomp", "attr") and child.pos_ == "ADJ"])

                for mod in modifiers:
                    mod_text = mod.text.lower()
                    score = _vader.polarity_scores(mod_text)["compound"]
                    aspect_records.append({
                        "aspect": aspect,
                        "modifier": mod_text,
                        "sentiment_score": score
                    })

    if not aspect_records:
        return pd.DataFrame(columns=["Aspect", "Mentions", "Average Sentiment", "Dominant Polarity"])

    df_aspects = pd.DataFrame(aspect_records)
    grouped = df_aspects.groupby("aspect").agg(
        Mentions=("sentiment_score", "count"),
        Avg_Score=("sentiment_score", "mean")
    ).reset_index()

    grouped = grouped[grouped["Mentions"] >= min_aspect_freq].sort_values(by="Mentions", ascending=False)

    grouped["Average Sentiment"] = grouped["Avg_Score"].round(2)
    grouped["Dominant Polarity"] = grouped["Avg_Score"].apply(
        lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
    )
    grouped.rename(columns={"aspect": "Aspect"}, inplace=True)
    return grouped[["Aspect", "Mentions", "Average Sentiment", "Dominant Polarity"]].head(15)
