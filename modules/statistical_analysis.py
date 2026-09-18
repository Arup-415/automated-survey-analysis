"""
modules/statistical_analysis.py
Advanced Statistical Engine:
- Descriptive summaries & Correlation
- Normality testing & Automated Test Recommenders
- Parametric & Non-parametric tests with Effect Sizes (Eta-sq, Epsilon-sq, Rank-biserial, Cramer's V)
- Tukey HSD & Chi-Square
- Multivariate Regression & Key Driver Analysis
- Psychometric Reliability (Cronbach's Alpha)
- Latent Dimensionality (PCA) & Unsupervised Respondent Clustering (k-Means)
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_descriptives(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    records = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        records.append({
            "Variable": col,
            "Count": int(len(s)),
            "Mean": round(float(s.mean()), 2),
            "Std Dev": round(float(s.std()), 2),
            "Median": round(float(s.median()), 2),
            "IQR": round(float(q75 - q25), 2),
            "Skewness": round(float(s.skew()), 2),
            "Min": round(float(s.min()), 2),
            "Max": round(float(s.max()), 2)
        })
    return pd.DataFrame(records)

def check_normality_and_homogeneity(df: pd.DataFrame, group_col: str, numeric_col: str) -> Dict[str, Any]:
    clean_df = df[[group_col, numeric_col]].dropna()
    groups = [group[numeric_col].values for _, group in clean_df.groupby(group_col) if len(group) >= 3]
    if len(groups) < 2:
        return {"status": "error", "message": "Need at least 2 cohorts with >= 3 items for test checks."}
    
    all_vals = clean_df[numeric_col].values
    if len(all_vals) >= 8:
        _, norm_p = stats.normaltest(all_vals)
    else:
        norm_p = 1.0
    is_normal = bool(norm_p > 0.05)
    
    _, lev_p = stats.levene(*groups)
    is_homo = bool(lev_p > 0.05)
    
    recommended = "One-Way ANOVA (Parametric)" if (is_normal and is_homo) else "Kruskal-Wallis (Non-Parametric)"
    
    return {
        "status": "success",
        "is_normal": is_normal,
        "normality_p": round(float(norm_p), 4),
        "is_homoscedastic": is_homo,
        "levene_p": round(float(lev_p), 4),
        "recommended_test": recommended,
        "rationale": "Assumptions met for Parametric testing." if (is_normal and is_homo) else "Normality or equal-variance assumption violated. Non-parametric test recommended."
    }

def run_anova(df: pd.DataFrame, group_col: str, numeric_col: str) -> Dict[str, Any]:
    clean_df = df[[group_col, numeric_col]].dropna()
    groups = [group[numeric_col].values for _, group in clean_df.groupby(group_col) if len(group) >= 2]
    if len(groups) < 2:
        return {"status": "error", "message": "Requires at least 2 cohorts with >= 2 items."}
    
    f_stat, p_val = stats.f_oneway(*groups)
    levene_stat, levene_p = stats.levene(*groups)
    
    grand_mean = clean_df[numeric_col].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for x in clean_df[numeric_col].values)
    eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0
    
    group_means = clean_df.groupby(group_col)[numeric_col].agg(["count", "mean", "std"]).round(2).to_dict("index")
    
    return {
        "status": "success",
        "f_statistic": round(float(f_stat), 3),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "eta_squared": round(float(eta_sq), 3),
        "effect_size_label": "Small" if eta_sq < 0.06 else ("Medium" if eta_sq < 0.14 else "Large"),
        "equal_variance_assumption_met": bool(levene_p > 0.05),
        "levene_p_value": round(float(levene_p), 4),
        "group_summary": group_means,
        "interpretation": (
            f"Statistically significant difference across '{group_col}' (F = {f_stat:.2f}, p = {p_val:.4g}, Eta² = {eta_sq:.3f})."
            if p_val < 0.05 else f"No significant difference across '{group_col}' (p = {p_val:.3f})."
        )
    }

def run_kruskal_wallis(df: pd.DataFrame, group_col: str, numeric_col: str) -> Dict[str, Any]:
    clean_df = df[[group_col, numeric_col]].dropna()
    groups = [group[numeric_col].values for _, group in clean_df.groupby(group_col) if len(group) >= 2]
    if len(groups) < 2:
        return {"status": "error", "message": "Requires at least 2 cohorts with >= 2 items."}
    
    h_stat, p_val = stats.kruskal(*groups)
    N = len(clean_df)
    k = len(groups)
    eps_sq = (h_stat - k + 1) / (N - k) if (N - k) > 0 else 0.0
    eps_sq = max(0.0, min(1.0, float(eps_sq)))
    
    group_medians = clean_df.groupby(group_col)[numeric_col].agg(["count", "median"]).round(2).to_dict("index")
    
    return {
        "status": "success",
        "h_statistic": round(float(h_stat), 3),
        "p_value": float(p_val),
        "is_significant": bool(p_val < 0.05),
        "epsilon_squared": round(float(eps_sq), 3),
        "effect_size_label": "Small" if eps_sq < 0.06 else ("Medium" if eps_sq < 0.14 else "Large"),
        "group_summary": group_medians,
        "interpretation": (
            f"Kruskal-Wallis rank test is significant across '{group_col}' (H = {h_stat:.2f}, p = {p_val:.4g}, Epsilon² = {eps_sq:.3f})."
            if p_val < 0.05 else f"No significant rank difference across '{group_col}' (p = {p_val:.3f})."
        )
    }

def run_mann_whitney(df: pd.DataFrame, group_col: str, numeric_col: str, group_a: str, group_b: str) -> Dict[str, Any]:
    s_a = df[df[group_col] == group_a][numeric_col].dropna()
    s_b = df[df[group_col] == group_b][numeric_col].dropna()
    if len(s_a) < 2 or len(s_b) < 2:
        return {"status": "error", "message": "Each cohort must contain at least 2 items."}
    
    u_stat, p_val = stats.mannwhitneyu(s_a, s_b, alternative="two-sided")
    r_biserial = 1.0 - (2.0 * u_stat / (len(s_a) * len(s_b)))
    
    return {
        "status": "success",
        "u_statistic": round(float(u_stat), 2),
        "p_value": float(p_val),
        "rank_biserial_r": round(float(r_biserial), 3),
        "is_significant": bool(p_val < 0.05),
        "median_a": round(float(s_a.median()), 2),
        "median_b": round(float(s_b.median()), 2),
        "interpretation": (
            f"Divergence between '{group_a}' (Med: {s_a.median():.2f}) and '{group_b}' (Med: {s_b.median():.2f}) (U = {u_stat:.1f}, p = {p_val:.4g}, r = {r_biserial:.2f})."
            if p_val < 0.05 else f"No significant divergence between '{group_a}' and '{group_b}' (p = {p_val:.3f})."
        )
    }

def run_tukey_hsd(df: pd.DataFrame, group_col: str, numeric_col: str) -> pd.DataFrame:
    clean_df = df[[group_col, numeric_col]].dropna()
    if clean_df[group_col].nunique() < 2:
        return pd.DataFrame()
    tukey = pairwise_tukeyhsd(endog=clean_df[numeric_col], groups=clean_df[group_col], alpha=0.05)
    return pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

def run_chi_square(df: pd.DataFrame, cat_col1: str, cat_col2: str) -> Dict[str, Any]:
    clean_df = df[[cat_col1, cat_col2]].dropna()
    contingency = pd.crosstab(clean_df[cat_col1], clean_df[cat_col2])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"status": "error", "message": "Contingency table must be at least 2x2."}
    
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
    assumption_warning = bool(((expected < 5).sum() / expected.size) > 0.20)
    
    return {
        "status": "success",
        "chi2_statistic": round(float(chi2), 3),
        "p_value": float(p_val),
        "cramers_v": round(float(cramers_v), 3),
        "degrees_of_freedom": int(dof),
        "is_significant": bool(p_val < 0.05),
        "low_expected_cells_warning": assumption_warning,
        "contingency_table": contingency,
        "interpretation": (
            f"Significant association between '{cat_col1}' and '{cat_col2}' (Chi² = {chi2:.2f}, p = {p_val:.4g}, Cramér's V = {cramers_v:.3f})."
            if p_val < 0.05 else f"No significant association (p = {p_val:.3f})."
        )
    }

def compute_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], method: str = "pearson") -> pd.DataFrame:
    sub_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub_df.shape[1] < 2:
        return pd.DataFrame()
    return sub_df.corr(method=method).round(3)

def run_multivariate_regression(df: pd.DataFrame, target_col: str, predictor_cols: List[str]) -> Dict[str, Any]:
    clean_df = df[[target_col] + predictor_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean_df) < len(predictor_cols) + 5:
        return {"status": "error", "message": "Sample size too small for specified predictors."}
    
    Y = clean_df[target_col]
    X = clean_df[predictor_cols]
    X_const = sm.add_constant(X)
    
    model = sm.OLS(Y, X_const).fit()
    
    vif_records = []
    for i, col in enumerate(predictor_cols):
        v = variance_inflation_factor(X.values, i) if X.shape[1] > 1 else 1.0
        vif_records.append({"Predictor": col, "VIF": round(float(v), 2), "Multicollinearity": "High" if v > 5.0 else "Acceptable"})
    
    coef_df = pd.DataFrame({
        "Predictor": ["Intercept"] + predictor_cols,
        "Coefficient": model.params.round(3).values,
        "Std Error": model.bse.round(3).values,
        "t-statistic": model.tvalues.round(2).values,
        "p-value": model.pvalues.round(4).values,
        "Significant": (model.pvalues < 0.05).values
    })
    
    return {
        "status": "success",
        "r_squared": round(float(model.rsquared), 3),
        "adj_r_squared": round(float(model.rsquared_adj), 3),
        "f_stat": round(float(model.fvalue), 2),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": coef_df,
        "raw_model": model,
        "vif_summary": pd.DataFrame(vif_records)
    }

def compute_key_drivers(df: pd.DataFrame, target_col: str, predictor_cols: List[str]) -> pd.DataFrame:
    clean_df = df[[target_col] + predictor_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean_df) < 10 or len(predictor_cols) < 2:
        return pd.DataFrame()
    
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(clean_df)
    scaled_df = pd.DataFrame(scaled_vals, columns=[target_col] + predictor_cols)
    
    X = sm.add_constant(scaled_df[predictor_cols])
    Y = scaled_df[target_col]
    model = sm.OLS(Y, X).fit()
    
    std_betas = model.params.drop("const").abs()
    total_impact = std_betas.sum()
    if total_impact == 0:
        return pd.DataFrame()
    
    relative_pct = (std_betas / total_impact) * 100.0
    return pd.DataFrame({
        "Driver / Survey Item": predictor_cols,
        "Standardized Beta": std_betas.round(3).values,
        "Relative Importance %": relative_pct.round(1).values
    }).sort_values(by="Relative Importance %", ascending=False).reset_index(drop=True)

def compute_cronbach_alpha(df: pd.DataFrame, scale_cols: List[str]) -> Dict[str, Any]:
    sub_df = df[scale_cols].apply(pd.to_numeric, errors="coerce").dropna()
    k = sub_df.shape[1]
    if k < 2 or len(sub_df) < 5:
        return {"status": "error", "message": "Requires at least 2 scale items with complete records."}
    
    item_variances = sub_df.var(axis=0, ddof=1).sum()
    total_score = sub_df.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    
    if total_variance == 0:
        return {"status": "error", "message": "Total scale variance is zero."}
        
    alpha = (k / (k - 1)) * (1.0 - (item_variances / total_variance))
    alpha_score = round(float(alpha), 3)
    
    if alpha_score >= 0.90:
        interp = "Excellent internal consistency"
    elif alpha_score >= 0.80:
        interp = "Good scale reliability"
    elif alpha_score >= 0.70:
        interp = "Acceptable reliability"
    elif alpha_score >= 0.60:
        interp = "Questionable / Poor reliability"
    else:
        interp = "Unacceptable internal consistency (items measure disparate concepts)"
        
    return {
        "status": "success",
        "cronbach_alpha": alpha_score,
        "items_evaluated": k,
        "observations": len(sub_df),
        "interpretation": interp
    }

def compute_pca_projection(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, List[float]]:
    sub_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub_df.shape[1] < 2 or len(sub_df) < 5:
        return pd.DataFrame(), []
    
    scaled = StandardScaler().fit_transform(sub_df)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    var_explained = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
    
    proj_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=sub_df.index)
    return proj_df, var_explained

def run_respondent_segmentation(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub_df) < n_clusters * 2:
        return pd.DataFrame(), pd.DataFrame()
    
    scaled = StandardScaler().fit_transform(sub_df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    
    clustered_df = sub_df.copy()
    clustered_df["Persona Cluster"] = [f"Segment {c+1}" for c in clusters]
    profile_df = clustered_df.groupby("Persona Cluster").mean().round(2).reset_index()
    return clustered_df, profile_df
