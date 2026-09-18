import pandas as pd
import numpy as np
import pytest
from modules.statistical_analysis import (
    run_kruskal_wallis, run_mann_whitney, run_tukey_hsd,
    run_multivariate_regression, compute_key_drivers,
    compute_cronbach_alpha, compute_pca_projection, run_respondent_segmentation,
    run_anova, run_chi_square
)

def test_kruskal_wallis_and_mann_whitney():
    df = pd.DataFrame({
        "tier": ["Basic"] * 25 + ["Pro"] * 25,
        "score": [1, 2, 1, 2, 3] * 5 + [4, 5, 5, 4, 5] * 5
    })
    kw = run_kruskal_wallis(df, "tier", "score")
    assert kw["status"] == "success"
    assert kw["is_significant"] is True

    mw = run_mann_whitney(df, "tier", "score", "Basic", "Pro")
    assert mw["status"] == "success"
    assert mw["is_significant"] is True

def test_cronbach_alpha_reliability():
    # Highly correlated scale items
    base = np.random.randint(1, 6, size=50)
    df = pd.DataFrame({
        "q1": base,
        "q2": np.clip(base + np.random.choice([0, 1], size=50), 1, 5),
        "q3": np.clip(base + np.random.choice([0, -1], size=50), 1, 5)
    })
    res = compute_cronbach_alpha(df, ["q1", "q2", "q3"])
    assert res["status"] == "success"
    assert res["cronbach_alpha"] > 0.65

def test_regression_and_key_drivers():
    x1 = np.linspace(1, 10, 40)
    x2 = np.random.normal(0, 1, 40)
    y = 2.0 * x1 + 0.5 * x2 + np.random.normal(0, 0.2, 40)
    df = pd.DataFrame({"satisfaction": y, "feature_speed": x1, "support": x2})
    
    reg = run_multivariate_regression(df, "satisfaction", ["feature_speed", "support"])
    assert reg["status"] == "success"
    assert reg["r_squared"] > 0.85
    
    drivers = compute_key_drivers(df, "satisfaction", ["feature_speed", "support"])
    assert not drivers.empty
    assert drivers.iloc[0]["Driver / Survey Item"] == "feature_speed"

def test_pca_and_clustering():
    df = pd.DataFrame({
        "f1": np.random.normal(0, 1, 30),
        "f2": np.random.normal(5, 1, 30),
        "f3": np.random.normal(10, 2, 30)
    })
    proj, var = compute_pca_projection(df, ["f1", "f2", "f3"])
    assert proj.shape == (30, 2)
    assert len(var) == 2

    clustered, profile = run_respondent_segmentation(df, ["f1", "f2", "f3"], n_clusters=2)
    assert "Persona Cluster" in clustered.columns
    assert len(profile) == 2
