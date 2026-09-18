"""
modules/insights.py
Rule-based automated insight synthesis translating quantitative and qualitative findings.
"""

from typing import List, Dict, Any

def generate_automated_insights(
    nps_data: Dict[str, Any],
    csat_data: Dict[str, Any],
    sentiment_counts: Dict[str, int],
    anova_results: List[Dict[str, Any]],
    chi2_results: List[Dict[str, Any]],
    topics: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Generates human-readable, evidence-backed findings.
    """
    insights = []

    # 1. NPS Insight
    if nps_data.get("status") == "success":
        nps = nps_data["nps_score"]
        pct_det = nps_data["pct_detractors"]
        if nps >= 50:
            insights.append({
                "Category": "Customer Loyalty (NPS)",
                "Severity": "Positive",
                "Text": f"Excellent customer loyalty observed with an NPS of +{nps}. Promoters ({nps_data['pct_promoters']}%) heavily outweigh detractors."
            })
        elif nps <= 0:
            insights.append({
                "Category": "Customer Loyalty (NPS)",
                "Severity": "Alert",
                "Text": f"Critical loyalty risk: NPS stands at {nps} with detractors comprising {pct_det}% of all respondents."
            })
        else:
            insights.append({
                "Category": "Customer Loyalty (NPS)",
                "Severity": "Neutral",
                "Text": f"Moderate Net Promoter Score of +{nps}. Detractor proportion is currently {pct_det}%."
            })

    # 2. CSAT Insight
    if csat_data.get("status") == "success":
        csat = csat_data["csat_score"]
        if csat >= 75.0:
            insights.append({
                "Category": "Satisfaction (CSAT)",
                "Severity": "Positive",
                "Text": f"High customer satisfaction: {csat}% of respondents rated their experience in the top-two boxes (4 or 5)."
            })
        elif csat < 50.0:
            insights.append({
                "Category": "Satisfaction (CSAT)",
                "Severity": "Alert",
                "Text": f"Low satisfaction alert: Only {csat}% satisfied responses recorded (mean rating {csat_data.get('mean_rating', 'N/A')}/5)."
            })

    # 3. Sentiment Insight
    total_sent = sum(sentiment_counts.values()) if sentiment_counts else 0
    if total_sent > 0:
        neg_pct = round((sentiment_counts.get("Negative", 0) / total_sent) * 100, 1)
        pos_pct = round((sentiment_counts.get("Positive", 0) / total_sent) * 100, 1)
        if neg_pct > 30.0:
            insights.append({
                "Category": "Text Sentiment",
                "Severity": "Alert",
                "Text": f"Elevated negative sentiment: {neg_pct}% of free-text responses contain predominantly negative tone."
            })
        elif pos_pct >= 50.0:
            insights.append({
                "Category": "Text Sentiment",
                "Severity": "Positive",
                "Text": f"Strong positive feedback: {pos_pct}% of unstructured comments express positive sentiment."
            })

    # 4. Inferential Statistical Differences (ANOVA)
    for anova in anova_results:
        if anova.get("status") == "success" and anova.get("is_significant"):
            insights.append({
                "Category": "Variance (ANOVA)",
                "Severity": "Neutral",
                "Text": anova["interpretation"]
            })

    # 5. Categorical Associations (Chi-Square)
    for chi in chi2_results:
        if chi.get("status") == "success" and chi.get("is_significant"):
            insights.append({
                "Category": "Association (Chi-Square)",
                "Severity": "Neutral",
                "Text": chi["interpretation"]
            })

    # 6. Unsupervised Topic Drivers
    for topic in topics[:2]:
        if topic.get("Mean Sentiment", 0) <= -0.15:
            insights.append({
                "Category": "Theme Alert",
                "Severity": "Alert",
                "Text": f"Topic '{topic['Topic Name']}' ({topic['Share %']}% share) exhibits a negative average sentiment score ({topic['Mean Sentiment']})."
            })

    if not insights:
        insights.append({
            "Category": "General Overview",
            "Severity": "Neutral",
            "Text": "Baseline data ingested. No statistically significant deviations or critical thresholds triggered."
        })

    return insights
