"""
app.py - Automated Survey Analysis Tool
Enterprise Dashboard featuring:
- Global Cohort Demographic Slicer
- Auto-Recommender Inference with Effect Sizes
- What-If Prescriptive Simulator
- Standalone Executive HTML & Workbook Exporter
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

from modules.data_loader import load_dataset, compute_dataset_profile
from modules.schema_detector import detect_column_schema, get_role_mapping
from modules.preprocessing import clean_survey_data
from modules.metrics import calculate_nps, calculate_csat, calculate_ces
from modules.statistical_analysis import (
    compute_descriptives, check_normality_and_homogeneity, run_anova, run_kruskal_wallis,
    run_mann_whitney, run_tukey_hsd, run_chi_square, compute_correlation_matrix,
    run_multivariate_regression, compute_key_drivers, compute_cronbach_alpha,
    compute_pca_projection, run_respondent_segmentation
)
from modules.sentiment import analyze_document_sentiment, extract_aspect_sentiments
from modules.keyword_extraction import extract_top_keywords
from modules.topic_modeling import discover_topics
from modules.insights import generate_automated_insights
from modules.report_generator import generate_json_report, generate_excel_report, generate_executive_html_report

st.set_page_config(
    page_title="Automated Survey Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        background: 
            linear-gradient(rgba(11, 17, 32, 0.94), rgba(11, 17, 32, 0.97)),
            url('https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=2000&auto=format&fit=crop') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
    }
    .metric-card h2 {
        margin: 6px 0 2px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-card span {
        font-size: 0.75rem;
        font-weight: 500;
    }

    .badge-blue { color: #38bdf8; }
    .badge-green { color: #4ade80; }
    .badge-amber { color: #fbbf24; }
    .badge-red { color: #f87171; }

    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 14px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.5);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 8px 16px;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

PRO_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)")
)

# Sidebar
with st.sidebar:
    st.markdown("### Survey Analytics Engine")
    st.caption("Automated Statistical & NLP Suite")
    st.markdown("---")
    st.markdown("**Dataset Ingestion**")
    uploaded_file = st.file_uploader("Upload Survey File", type=["csv", "xlsx", "xls"])
    load_sample = st.button("Load Benchmark Dataset", use_container_width=True)

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "clean_drop_dups" not in st.session_state:
    st.session_state.clean_drop_dups = True
if "clean_impute_num" not in st.session_state:
    st.session_state.clean_impute_num = "median"
if "clean_impute_cat_mode" not in st.session_state:
    st.session_state.clean_impute_cat_mode = "constant"
if "clean_impute_cat_val" not in st.session_state:
    st.session_state.clean_impute_cat_val = "No Response"

if uploaded_file is not None:
    df_loaded, err = load_dataset(uploaded_file)
    if err:
        st.error(err)
    else:
        st.session_state.df_raw = df_loaded
        st.session_state.file_name = uploaded_file.name
elif load_sample:
    sample_path = os.path.join("data", "sample_survey.csv")
    if os.path.exists(sample_path):
        st.session_state.df_raw = pd.read_csv(sample_path)
        st.session_state.file_name = "sample_survey.csv (Benchmark)"
    else:
        st.error("Benchmark dataset not found. Run python data/generate_synthetic_data.py.")

df_raw = st.session_state.df_raw

if df_raw is None:
    st.markdown("<h2 style='font-weight: 700; color: #f8fafc;'>Automated Survey Analysis Tool</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.05rem;'>Dataset-independent statistical testing, semantic schema mapping, exploratory analysis, and NLP.</p>", unsafe_allow_html=True)
    st.info("👈 Upload a survey spreadsheet (.csv, .xlsx) or click 'Load Benchmark Dataset' to begin.")
    st.stop()

# 1. Pipeline Baseline Ingestion
raw_profile = compute_dataset_profile(df_raw)
schema_df = detect_column_schema(df_raw)
raw_roles = get_role_mapping(schema_df)

cleaned_df_base, clean_summary = clean_survey_data(
    df_raw,
    schema_df,
    drop_duplicates=st.session_state.clean_drop_dups,
    impute_numeric=st.session_state.clean_impute_num,
    impute_categorical=st.session_state.clean_impute_cat_mode,
    categorical_constant=st.session_state.clean_impute_cat_val
)

# 2. Global Cohort Slicer (Sidebar Dynamic Filter)
with st.sidebar:
    st.markdown("---")
    st.markdown("**🎯 Global Cohort Slicer**")
    active_filters = {}
    if raw_roles["categorical_cols"]:
        slicer_col = st.selectbox("Cohort Variable", ["(All Cohorts)"] + raw_roles["categorical_cols"])
        if slicer_col != "(All Cohorts)":
            cohort_values = cleaned_df_base[slicer_col].dropna().unique().tolist()
            chosen_val = st.selectbox(f"Filter {slicer_col}", ["(All)"] + cohort_values)
            if chosen_val != "(All)":
                active_filters[slicer_col] = chosen_val

# Apply Cohort Filtering across cleaned data
if active_filters:
    cleaned_df = cleaned_df_base.copy()
    for col, val in active_filters.items():
        cleaned_df = cleaned_df[cleaned_df[col] == val]
    st.sidebar.info(f"Filtered Cohort: `{len(cleaned_df):,}` / `{len(cleaned_df_base):,}` records.")
else:
    cleaned_df = cleaned_df_base

cleaned_profile = compute_dataset_profile(cleaned_df)
cleaned_roles = get_role_mapping(schema_df)

# Metrics
nps_res = calculate_nps(cleaned_df[cleaned_roles["nps_col"]]) if cleaned_roles["nps_col"] else {"status": "error"}
csat_res = calculate_csat(cleaned_df[cleaned_roles["csat_col"]]) if cleaned_roles["csat_col"] else {"status": "error"}
ces_res = calculate_ces(cleaned_df[cleaned_roles["ces_col"]]) if cleaned_roles["ces_col"] else {"status": "error"}

primary_text_col = cleaned_roles["text_cols"][0] if cleaned_roles["text_cols"] else None
if primary_text_col and len(cleaned_df) > 0:
    @st.cache_data
    def run_nlp(text_series):
        sent_df = analyze_document_sentiment(text_series)
        keywords_df = extract_top_keywords(text_series, top_n=15)
        topics_lst = discover_topics(text_series, sent_df["compound_score"], n_topics=4)
        aspects_df = extract_aspect_sentiments(text_series.dropna().tolist())
        return sent_df, keywords_df, topics_lst, aspects_df

    sent_df, keywords_df, topics_lst, aspects_df = run_nlp(cleaned_df[primary_text_col])
    sentiment_counts = sent_df["sentiment"].value_counts().to_dict()
else:
    sent_df, keywords_df, aspects_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    topics_lst = []
    sentiment_counts = {}

descriptive_df = compute_descriptives(cleaned_df, cleaned_roles["numeric_cols"])
insights = generate_automated_insights(nps_res, csat_res, sentiment_counts, [], [], topics_lst)

# Executive KPI Strip
st.markdown(f"<div style='font-size: 0.9rem; font-weight: 500; color: #94a3b8; margin-bottom: 6px;'>Active Dataset: <span style='color: #38bdf8;'>{st.session_state.file_name}</span></div>", unsafe_allow_html=True)
kpi_cols = st.columns(6)

with kpi_cols[0]:
    st.markdown(f'<div class="metric-card"><h4>Active Sample</h4><h2>{cleaned_profile["total_rows"]:,}</h2><span class="badge-blue">Cohort Filtered</span></div>', unsafe_allow_html=True)
with kpi_cols[1]:
    val = f"{nps_res['nps_score']}" if nps_res.get("status") == "success" else "N/A"
    badge_cls = "badge-green" if nps_res.get("status") == "success" and nps_res['nps_score'] > 0 else "badge-red"
    st.markdown(f'<div class="metric-card"><h4>NPS Score</h4><h2>{val}</h2><span class="{badge_cls}">Scale: -100 to +100</span></div>', unsafe_allow_html=True)
with kpi_cols[2]:
    val = f"{csat_res['csat_score']}%" if csat_res.get("status") == "success" else "N/A"
    st.markdown(f'<div class="metric-card"><h4>CSAT Score</h4><h2>{val}</h2><span class="badge-blue">Top 2 Boxes</span></div>', unsafe_allow_html=True)
with kpi_cols[3]:
    val = f"{ces_res['mean_score']}" if ces_res.get("status") == "success" else "N/A"
    st.markdown(f'<div class="metric-card"><h4>Effort (CES)</h4><h2>{val}</h2><span class="badge-amber">Average (1-7)</span></div>', unsafe_allow_html=True)
with kpi_cols[4]:
    val = f"{sentiment_counts.get('Positive', 0):,}" if sentiment_counts else "N/A"
    st.markdown(f'<div class="metric-card"><h4>Positive Tone</h4><h2>{val}</h2><span class="badge-green">Favorable Comments</span></div>', unsafe_allow_html=True)
with kpi_cols[5]:
    val = f"{sentiment_counts.get('Negative', 0):,}" if sentiment_counts else "N/A"
    st.markdown(f'<div class="metric-card"><h4>Negative Tone</h4><h2>{val}</h2><span class="badge-red">Requires Attention</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tabs = st.tabs([
    "1. Profiling & Schema",
    "2. Exploratory EDA",
    "3. Cleaning Studio",
    "4. Clean Insights",
    "5. Quantitative Summary",
    "6. Hypothesis & Inference",
    "7. Drivers, Reliability & Personas",
    "8. ⚡ What-If Simulator",
    "9. Sentiment & ABSA",
    "10. Thematic Topics & Keywords",
    "11. Export Center"
])

# TAB 1: PROFILING & SCHEMA
with tabs[0]:
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.markdown('<div class="section-header">Raw Ingestion Telemetry</div>', unsafe_allow_html=True)
        st.markdown(f"""
        - **Raw Ingested Records:** `{raw_profile['total_rows']:,}` rows × `{raw_profile['total_columns']}` columns
        - **Baseline Missing Cell Rate:** `{raw_profile['overall_missing_pct']}%`
        - **Identified Duplicate Rows:** `{raw_profile['duplicate_rows']:,}`
        """)
    with col_p2:
        st.markdown('<div class="section-header">Inferred Semantic Survey Anchors</div>', unsafe_allow_html=True)
        st.json({k: v for k, v in raw_roles.items() if v is not None})
    st.markdown('<div class="section-header">Dynamic Column Schema Classifications</div>', unsafe_allow_html=True)
    st.dataframe(schema_df, use_container_width=True)

# TAB 2: EXPLORATORY DATA ANALYSIS
with tabs[1]:
    st.markdown('<div class="section-header">Exploratory Data Analysis (Unadulterated Raw Data)</div>', unsafe_allow_html=True)
    null_matrix = df_raw.head(100).isna().astype(int)
    fig_miss = px.imshow(null_matrix.T, labels=dict(x="Respondent Index (First 100)", y="Feature", color="Missing"),
                         color_continuous_scale=[[0, "#1e293b"], [1, "#f87171"]], aspect="auto")
    fig_miss.update_layout(**PRO_LAYOUT)
    fig_miss.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_miss, use_container_width=True)

    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        if raw_roles["categorical_cols"]:
            eda_cat = st.selectbox("Categorical Factor", raw_roles["categorical_cols"], key="eda_c")
            cat_counts = df_raw[eda_cat].value_counts(dropna=False).reset_index()
            cat_counts.columns = [eda_cat, "Observations"]
            cat_counts[eda_cat] = cat_counts[eda_cat].fillna("(Missing)")
            fig_cat = px.bar(cat_counts, x=eda_cat, y="Observations", color_discrete_sequence=["#38bdf8"])
            fig_cat.update_layout(**PRO_LAYOUT)
            st.plotly_chart(fig_cat, use_container_width=True)
    with c_eda2:
        if raw_roles["numeric_cols"]:
            eda_num = st.selectbox("Continuous Variable", raw_roles["numeric_cols"], key="eda_n")
            fig_dens = px.histogram(df_raw, x=eda_num, marginal="box", nbins=30, color_discrete_sequence=["#38bdf8"])
            fig_dens.update_layout(**PRO_LAYOUT)
            st.plotly_chart(fig_dens, use_container_width=True)

# TAB 3: DATA CLEANING STUDIO
with tabs[2]:
    st.markdown('<div class="section-header">Data Cleaning Studio</div>', unsafe_allow_html=True)
    c_opt1, c_opt2, c_opt3 = st.columns(3)
    with c_opt1:
        new_drop_dups = st.checkbox("Purge Exact Duplicate Rows", value=st.session_state.clean_drop_dups)
    with c_opt2:
        num_options = ["median", "mean", "mode", "interpolate", "ffill", "bfill", "drop_rows", "none"]
        curr_idx = num_options.index(st.session_state.clean_impute_num) if st.session_state.clean_impute_num in num_options else 0
        new_impute_num = st.selectbox("Numerical Missing Handler", num_options, index=curr_idx)
    with c_opt3:
        cat_options = ["constant", "mode", "ffill", "bfill", "drop_rows", "none"]
        curr_cat_idx = cat_options.index(st.session_state.clean_impute_cat_mode) if st.session_state.clean_impute_cat_mode in cat_options else 0
        new_impute_cat_mode = st.selectbox("Categorical Strategy", cat_options, index=curr_cat_idx)
        new_impute_cat_val = st.text_input("Replacement Constant", value=st.session_state.clean_impute_cat_val) if new_impute_cat_mode == "constant" else st.session_state.clean_impute_cat_val

    if (new_drop_dups != st.session_state.clean_drop_dups or
        new_impute_num != st.session_state.clean_impute_num or
        new_impute_cat_mode != st.session_state.clean_impute_cat_mode or
        new_impute_cat_val != st.session_state.clean_impute_cat_val):
        if st.button("Apply Cleaning Rules & Re-Compute Pipeline", type="primary"):
            st.session_state.clean_drop_dups = new_drop_dups
            st.session_state.clean_impute_num = new_impute_num
            st.session_state.clean_impute_cat_mode = new_impute_cat_mode
            st.session_state.clean_impute_cat_val = new_impute_cat_val
            st.rerun()

    st.markdown("---")
    col_aud1, col_aud2, col_aud3 = st.columns(3)
    with col_aud1:
        st.write(f"- **Raw Row Count:** `{raw_profile['total_rows']:,}`")
        st.write(f"- **Clean Row Count:** `{cleaned_profile['total_rows']:,}`")
    with col_aud2:
        st.write(f"- **Raw Missing Rate:** `{raw_profile['overall_missing_pct']}%`")
        st.write(f"- **Clean Missing Rate:** `{cleaned_profile['overall_missing_pct']}%`")
    with col_aud3:
        total_pii = sum(clean_summary['pii_redactions'].values())
        st.write(f"- **PII Tokens Redacted:** `{total_pii}` (`[EMAIL]`, `[PHONE]`, `[URL]`)")

# TAB 4: CLEAN INSIGHTS
with tabs[3]:
    st.markdown('<div class="section-header">Automated Insights (Cleaned Dataset)</div>', unsafe_allow_html=True)
    col_ins, col_viz = st.columns([1, 1])
    with col_ins:
        for ins in insights:
            if ins["Severity"] == "Alert":
                st.error(f"**[{ins['Category']}]** {ins['Text']}")
            elif ins["Severity"] == "Positive":
                st.success(f"**[{ins['Category']}]** {ins['Text']}")
            else:
                st.info(f"**[{ins['Category']}]** {ins['Text']}")
    with col_viz:
        if sentiment_counts:
            fig_s = px.pie(values=list(sentiment_counts.values()), names=list(sentiment_counts.keys()), hole=0.45,
                           color=list(sentiment_counts.keys()), color_discrete_map={"Positive": "#4ade80", "Neutral": "#94a3b8", "Negative": "#f87171"})
            fig_s.update_layout(**PRO_LAYOUT)
            fig_s.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_s, use_container_width=True)

# TAB 5: QUANTITATIVE SUMMARY
with tabs[4]:
    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(descriptive_df, use_container_width=True)

# TAB 6: HYPOTHESIS & INFERENCE (Auto-Recommender & Effect Sizes)
with tabs[5]:
    st.markdown('<div class="section-header">Statistical Inference & Automated Guardrails</div>', unsafe_allow_html=True)
    if cleaned_roles["categorical_cols"] and cleaned_roles["numeric_cols"]:
        c_inf1, c_inf2 = st.columns(2)
        with c_inf1:
            g_factor = st.selectbox("Grouping Factor (X)", cleaned_roles["categorical_cols"], key="inf_g")
            y_metric = st.selectbox("Continuous Metric (Y)", cleaned_roles["numeric_cols"], key="inf_y")
            
            guardrail = check_normality_and_homogeneity(cleaned_df, g_factor, y_metric)
            if guardrail["status"] == "success":
                st.markdown(f"**Automated Diagnostic:** `{guardrail['recommended_test']}`")
                st.caption(f"Normality p: `{guardrail['normality_p']}` | Levene p: `{guardrail['levene_p']}`. {guardrail['rationale']}")
            
            test_mode = st.radio("Execute Test Paradigm", ["Auto-Recommended", "Force ANOVA", "Force Kruskal-Wallis"], horizontal=True)
            if st.button("Run Group Difference Test"):
                run_mode = guardrail["recommended_test"] if test_mode == "Auto-Recommended" else test_mode
                if "ANOVA" in run_mode:
                    out = run_anova(cleaned_df, g_factor, y_metric)
                    if out["status"] == "success":
                        st.info(out["interpretation"])
                        st.metric("Effect Size: Eta-Squared (η²)", f"{out['eta_squared']} ({out['effect_size_label']})")
                        st.json(out["group_summary"])
                else:
                    out = run_kruskal_wallis(cleaned_df, g_factor, y_metric)
                    if out["status"] == "success":
                        st.info(out["interpretation"])
                        st.metric("Effect Size: Epsilon-Squared (ε²)", f"{out['epsilon_squared']} ({out['effect_size_label']})")
                        st.json(out["group_summary"])

        with c_inf2:
            st.markdown("#### Post-Hoc Pairwise & 2-Cohort Tests")
            st.caption("Pinpoint specific differing pairs or compare exactly two cohorts.")
            sub_t1, sub_t2 = st.tabs(["Tukey HSD Table", "Mann-Whitney U Test"])
            with sub_t1:
                if st.button("Compute Tukey HSD Pairwise"):
                    tukey_res = run_tukey_hsd(cleaned_df, g_factor, y_metric)
                    if not tukey_res.empty:
                        st.dataframe(tukey_res, use_container_width=True)
            with sub_t2:
                avail_cohorts = cleaned_df[g_factor].dropna().unique().tolist()
                if len(avail_cohorts) >= 2:
                    ca = st.selectbox("Cohort A", avail_cohorts, index=0, key="mw_a")
                    cb = st.selectbox("Cohort B", avail_cohorts, index=1, key="mw_b")
                    if st.button("Run Mann-Whitney U"):
                        mw_res = run_mann_whitney(cleaned_df, g_factor, y_metric, ca, cb)
                        if mw_res["status"] == "success":
                            st.info(mw_res["interpretation"])
                            st.metric("Rank-Biserial Effect (r)", mw_res["rank_biserial_r"])

    st.markdown("---")
    c_inf3, c_inf4 = st.columns(2)
    with c_inf3:
        st.markdown("#### Chi-Square Test (with Cramér's V)")
        if len(cleaned_roles["categorical_cols"]) >= 2:
            x1_c = st.selectbox("Factor A", cleaned_roles["categorical_cols"], key="chi_a")
            x2_c = st.selectbox("Factor B", cleaned_roles["categorical_cols"], index=1, key="chi_b")
            if st.button("Run Chi-Square"):
                c_out = run_chi_square(cleaned_df, x1_c, x2_c)
                if c_out["status"] == "success":
                    st.info(c_out["interpretation"])
                    st.metric("Cramér's V Association", c_out["cramers_v"])
                    st.dataframe(c_out["contingency_table"])
    with c_inf4:
        st.markdown("#### Correlation Matrix")
        if len(cleaned_roles["numeric_cols"]) >= 2:
            corr_m = st.selectbox("Correlation Metric", ["pearson", "spearman"])
            c_mat = compute_correlation_matrix(cleaned_df, cleaned_roles["numeric_cols"], method=corr_m)
            fig_c = px.imshow(c_mat, text_auto=True, color_continuous_scale="Blues", zmin=-1, zmax=1)
            fig_c.update_layout(**PRO_LAYOUT)
            fig_c.update_layout(height=340)
            st.plotly_chart(fig_c, use_container_width=True)

# TAB 7: DRIVERS, RELIABILITY & PERSONAS
with tabs[6]:
    st.markdown('<div class="section-header">Key Drivers, Scale Psychometrics & Personas</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 1. Key Drivers & Multivariate OLS")
        if len(cleaned_roles["numeric_cols"]) >= 2:
            t_col = st.selectbox("Outcome Metric (Y)", cleaned_roles["numeric_cols"], index=0, key="reg_y")
            candidates = [c for c in cleaned_roles["numeric_cols"] if c != t_col]
            p_cols = st.multiselect("Predictor Factors (X)", candidates, default=candidates[:min(3, len(candidates))])
            if st.button("Compute Drivers"):
                reg_res = run_multivariate_regression(cleaned_df, t_col, p_cols)
                if reg_res["status"] == "success":
                    st.write(f"Model **R²:** `{reg_res['r_squared']}` (Adjusted R²: `{reg_res['adj_r_squared']}`)")
                    st.dataframe(reg_res["coefficients"], use_container_width=True)
                    drivers_df = compute_key_drivers(cleaned_df, t_col, p_cols)
                    if not drivers_df.empty:
                        fig_drv = px.bar(drivers_df, x="Relative Importance %", y="Driver / Survey Item", orientation="h", color_discrete_sequence=["#38bdf8"])
                        fig_drv.update_layout(**PRO_LAYOUT)
                        fig_drv.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_drv, use_container_width=True)
    with col_d2:
        st.markdown("#### 2. Scale Reliability: Cronbach's Alpha (α)")
        if len(cleaned_roles["numeric_cols"]) >= 2:
            alpha_items = st.multiselect("Scale Items", cleaned_roles["numeric_cols"], default=cleaned_roles["numeric_cols"][:min(4, len(cleaned_roles["numeric_cols"]))], key="cron_sel")
            if st.button("Evaluate Scale Reliability (α)"):
                alpha_res = compute_cronbach_alpha(cleaned_df, alpha_items)
                if alpha_res["status"] == "success":
                    st.metric("Cronbach's Alpha (α)", alpha_res["cronbach_alpha"])
                    st.info(f"**Interpretation:** {alpha_res['interpretation']}.")

# TAB 8: WHAT-IF PRESCRIPTIVE SIMULATOR
with tabs[7]:
    st.markdown('<div class="section-header">⚡ What-If Prescriptive Simulator (Sensitivity Engine)</div>', unsafe_allow_html=True)
    st.caption("Simulate how operational interventions and detractor mitigation strategies shift high-level metrics.")
    
    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        st.markdown("#### 1. NPS Churn Mitigation Simulation")
        if nps_res.get("status") == "success":
            det_shift = st.slider("Simulate converting % of Detractors to Passives/Promoters", 0, 100, 15)
            curr_nps = nps_res["nps_score"]
            curr_det_count = nps_res["detractors"]
            shifted_det = int(curr_det_count * (det_shift / 100.0))
            new_det_count = curr_det_count - shifted_det
            new_prom_count = nps_res["promoters"] + shifted_det
            new_nps = round(((new_prom_count - new_det_count) / nps_res["total_responses"]) * 100, 1)
            
            c_m1, c_m2 = st.columns(2)
            c_m1.metric("Baseline NPS", curr_nps)
            c_m2.metric("Simulated Projected NPS", new_nps, delta=round(new_nps - curr_nps, 1))
            st.info(f"Moving **{shifted_det:,}** detractors produces a **{round(new_nps - curr_nps, 1):+} point** shift in aggregate Net Promoter Score.")
        else:
            st.info("NPS metric not detected for sensitivity simulation.")
            
    with c_sim2:
        st.markdown("#### 2. Driver Elasticity Simulation")
        if len(cleaned_roles["numeric_cols"]) >= 2:
            st.caption("Estimate CSAT / Outcome lift by modeling improvements in specific drivers.")
            sim_target = st.selectbox("Outcome Metric", cleaned_roles["numeric_cols"], key="sim_y")
            cand = [c for c in cleaned_roles["numeric_cols"] if c != sim_target]
            sim_driver = st.selectbox("Intervention Driver", cand, key="sim_d")
            driver_lift = st.slider(f"Simulate +lift in {sim_driver} (Points)", 0.1, 2.0, 0.5, step=0.1)
            
            s_clean = cleaned_df[[sim_target, sim_driver]].dropna()
            slope, intercept, r_v, p_v, _ = stats.linregress(s_clean[sim_driver], s_clean[sim_target])
            curr_mean = round(float(s_clean[sim_target].mean()), 2)
            lifted_mean = round(curr_mean + (slope * driver_lift), 2)
            
            c_el1, c_el2 = st.columns(2)
            c_el1.metric("Current Mean", curr_mean)
            c_el2.metric("Projected Mean", lifted_mean, delta=round(lifted_mean - curr_mean, 2))
            st.info(f"Each +1.0 point improvement in **{sim_driver}** yields an estimated **{slope:+.2f} point** lift in **{sim_target}** (R = {r_v:.2f}).")

# TAB 9: SENTIMENT & ABSA
with tabs[8]:
    st.markdown('<div class="section-header">Text Sentiment & Aspect Analysis</div>', unsafe_allow_html=True)
    if not sent_df.empty:
        col_ab1, col_ab2 = st.columns([3, 2])
        with col_ab1:
            st.dataframe(pd.concat([cleaned_df[[primary_text_col]].reset_index(drop=True), sent_df], axis=1).head(15), use_container_width=True)
        with col_ab2:
            st.dataframe(aspects_df, use_container_width=True)
    else:
        st.info("No text stream identified for NLP analysis.")

# TAB 10: THEMATIC TOPICS & KEYWORDS
with tabs[9]:
    st.markdown('<div class="section-header">Unsupervised Thematic Clusters (NMF) & TF-IDF Keywords</div>', unsafe_allow_html=True)
    if topics_lst:
        t_df = pd.DataFrame(topics_lst)
        st.dataframe(t_df, use_container_width=True)
        fig_top = px.bar(t_df, x="Topic Name", y="Responses", color="Mean Sentiment", color_continuous_scale="RdYlGn")
        fig_top.update_layout(**PRO_LAYOUT)
        st.plotly_chart(fig_top, use_container_width=True)
    if not keywords_df.empty:
        st.dataframe(keywords_df.head(10), use_container_width=True)

# TAB 11: EXPORT CENTER
with tabs[10]:
    st.markdown('<div class="section-header">Export Results & Executive Artifacts</div>', unsafe_allow_html=True)
    c_e1, c_e2, c_e3 = st.columns(3)
    with c_e1:
        st.markdown("#### Standalone Executive HTML")
        st.caption("Self-contained styled briefing document ready for offline executive review.")
        html_doc = generate_executive_html_report(st.session_state.file_name, cleaned_profile, nps_res, csat_res, ces_res, insights, topics_lst)
        st.download_button("📥 Download HTML Briefing", html_doc.encode("utf-8"), "Executive_Survey_Briefing.html", "text/html", use_container_width=True)
    with c_e2:
        st.markdown("#### Multi-Sheet Excel Workbook")
        st.caption("Formatted Excel summary sheets across all modules.")
        x_bytes = generate_excel_report(cleaned_profile["column_profiles"], schema_df, descriptive_df, pd.DataFrame(topics_lst), aspects_df)
        st.download_button("📥 Download Summary (.XLSX)", x_bytes, "survey_analytics.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c_e3:
        st.markdown("#### Cleaned Dataset & JSON")
        st.caption("Sanitized CSV corpus.")
        st.download_button("📥 Download Cleaned CSV", cleaned_df.to_csv(index=False).encode("utf-8"), "cleaned_survey.csv", "text/csv", use_container_width=True)
