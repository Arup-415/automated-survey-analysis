import pandas as pd
from modules.sentiment import analyze_document_sentiment, extract_aspect_sentiments
from modules.keyword_extraction import extract_top_keywords
from modules.topic_modeling import discover_topics

def test_sentiment_classification():
    texts = pd.Series([
        "This software is absolutely fantastic and works smoothly!",
        "Terrible experience, complete billing nightmare.",
        "The application is okay, nothing special."
    ])
    df_sent = analyze_document_sentiment(texts)
    assert df_sent["sentiment"].iloc[0] == "Positive"
    assert df_sent["sentiment"].iloc[1] == "Negative"
    assert df_sent["compound_score"].iloc[0] > 0.4
    assert df_sent["compound_score"].iloc[1] < -0.4

def test_extract_top_keywords():
    corpus = pd.Series([
        "Billing system invoice refund issues.",
        "Billing payment and charge problem with invoice.",
        "Invoice calculation errors in billing."
    ])
    keywords = extract_top_keywords(corpus, top_n=5)
    assert not keywords.empty
    found_words = " ".join(keywords["Keyword / Phrase"].tolist())
    assert "billing" in found_words or "invoice" in found_words

def test_discover_topics():
    corpus = pd.Series([
        "Billing is terrible, refund missing, overcharged payment.",
        "Payment invoice charge problems and billing errors.",
        "Customer service support agent response time is fast.",
        "Helpful customer support desk and great service."
    ] * 5)
    sentiments = pd.Series([-0.8, -0.7, 0.8, 0.9] * 5)
    topics = discover_topics(corpus, sentiments, n_topics=2)
    assert len(topics) == 2
    assert topics[0]["Responses"] > 0
    assert "Share %" in topics[0]
