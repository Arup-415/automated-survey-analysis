"""
modules/metrics.py
Mathematical computation of primary survey metrics: NPS, CSAT, CES.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

def calculate_nps(series: pd.Series) -> Dict[str, Any]:
    """
    Computes Net Promoter Score (NPS).
    Requires integer scale 0 to 10.
    Formula: % Promoters (9-10) - % Detractors (0-6)
    """
    valid_scores = pd.to_numeric(series, errors="coerce").dropna()
    total = len(valid_scores)
    
    if total == 0:
        return {"status": "error", "message": "No valid numeric responses."}
        
    promoters = int((valid_scores >= 9).sum())
    passives = int(((valid_scores >= 7) & (valid_scores <= 8)).sum())
    detractors = int((valid_scores <= 6).sum())
    
    pct_promoters = (promoters / total) * 100.0
    pct_passives = (passives / total) * 100.0
    pct_detractors = (detractors / total) * 100.0
    
    nps_score = round(pct_promoters - pct_detractors, 1)
    
    return {
        "status": "success",
        "nps_score": nps_score,
        "total_responses": total,
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "pct_promoters": round(pct_promoters, 1),
        "pct_passives": round(pct_passives, 1),
        "pct_detractors": round(pct_detractors, 1)
    }

def calculate_csat(series: pd.Series) -> Dict[str, Any]:
    """
    Computes Customer Satisfaction (CSAT) Top-2-Box score.
    Requires scale 1 to 5.
    Formula: % rating 4 or 5.
    """
    valid_scores = pd.to_numeric(series, errors="coerce").dropna()
    total = len(valid_scores)
    
    if total == 0:
        return {"status": "error", "message": "No valid numeric responses."}
        
    satisfied_count = int((valid_scores >= 4).sum())
    csat_score = round((satisfied_count / total) * 100.0, 1)
    mean_rating = round(float(valid_scores.mean()), 2)
    
    # Rating frequency breakdown
    dist = valid_scores.value_counts(normalize=True).to_dict()
    dist_pct = {int(k): round(v * 100.0, 1) for k, v in dist.items()}
    
    return {
        "status": "success",
        "csat_score": csat_score,
        "mean_rating": mean_rating,
        "total_responses": total,
        "distribution_pct": dist_pct
    }

def calculate_ces(series: pd.Series) -> Dict[str, Any]:
    """
    Computes Customer Effort Score (CES).
    Evaluates mean effort and low-effort percentage (ratings <= 2 on a 1-7 low=easy scale,
    or >= 5 on a 1-7 high=easy scale).
    """
    valid_scores = pd.to_numeric(series, errors="coerce").dropna()
    total = len(valid_scores)
    
    if total == 0:
        return {"status": "error", "message": "No valid numeric responses."}
        
    mean_effort = round(float(valid_scores.mean()), 2)
    median_effort = round(float(valid_scores.median()), 2)
    
    return {
        "status": "success",
        "mean_score": mean_effort,
        "median_score": median_effort,
        "total_responses": total
    }
