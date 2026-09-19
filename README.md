# Automated Survey Analysis Platform 📊

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-pytest%2015%20passed-success.svg)](https://docs.pytest.org/)

An end-to-end, dataset-agnostic quantitative and qualitative survey intelligence platform. Built in Python and Streamlit, it automates the transition from raw, messy spreadsheet dumps to rigorous statistical inference, NLP aspect extraction, psychometrics, and prescriptive decision simulation.

---

## 🌟 Key Engineering Highlights

* **Dataset-Agnostic Ingestion:** Invariant role detection across arbitrary `.csv` and `.xlsx` files without hardcoded column header schemas.
* **Statistical Rigor & Automated Guardrails:** Tests distributional normality (D'Agostino-Pearson) and homoscedasticity (Levene's) to recommend and compute either Parametric (One-Way ANOVA) or Non-Parametric (Kruskal-Wallis) tests alongside explicit effect sizes (η², ε²).
* **Syntactic Aspect-Based Sentiment Analysis (ABSA):** Uses `spaCy` syntactic dependency parsing to isolate noun aspects linked to adjectival modifiers, avoiding the blind spots of document-level sentiment scoring.
* **Unsupervised Latent Topic Modeling:** Applies Non-Negative Matrix Factorization (NMF) to sparse TF-IDF matrices to surface core feedback discussion themes.
* **Prescriptive Sensitivity Engine:** An interactive What-If simulator using empirical Ordinary Least Squares (OLS) linear elasticity to model high-level metric shifts.
* **Zero-Leakage Privacy Studio:** Regex-driven redaction of Personally Identifiable Information (PII) for emails, phone numbers, and URLs.

---

## 🏗️ Architecture & Data Pipeline

Raw Spreadsheet (.csv / .xlsx)
│
├── 1. Data Ingestion & Validation (Encoding Fallbacks, Malformed Row Checks)
├── 2. Invariant Data Profiling (Missing Matrix, Duplication Ratio, Cardinality)
├── 3. Dynamic Heuristic Schema Inference (NPS [0-10], CSAT [1-5], CES [1-7], Text, Demographics)
├── 4. Raw Exploratory Data Analysis (Dropout Attrition Heatmap)
├── 5. Interactive Cleaning Studio (PII Masking, 8 Imputation Strategies, Deduplication)
├── 6. Global Cohort Slicer (Dynamic Multi-Factor Tabular Filtering)
├── 7. Quantitative Metrics Suite (NPS, CSAT Top-2-Box, CES Mean, Descriptive Moments)
├── 8. Inferential Engine with Guardrails (Normality Checks, ANOVA η², Kruskal ε², Mann-Whitney r, Tukey HSD, Chi-Square Cramér's V)
├── 9. Predictive Modeling & Psychometrics (OLS Key Drivers, VIF Multicollinearity, Cronbach's α)
├── 10. Unsupervised Persona Discovery (StandardScaler -> 2D PCA -> k-Means Clustering)
├── 11. Qualitative NLP Suite (VADER Document Polarity, spaCy Syntactic ABSA, TF-IDF n-grams, NMF Topics)
├── 12. Prescriptive Decision Simulator (NPS Churn Mitigation, Driver Elasticity Sliders)
└── 13. Multi-Format Export Center (Cleaned CSV, Results JSON, Multi-Sheet Excel, Standalone HTML Briefing)


---

## 📊 Modules & Methodologies

| Module | Core Functionality | Algorithms & Formulas |
| :--- | :--- | :--- |
| **`modules/schema_detector.py`** | Semantic Role Detection | Heuristic Range Checking ([0,10], [1,5], [1,7]), Cardinality Ratios, Regex Keyword Triggers |
| **`modules/preprocessing.py`** | Data Cleansing & PII Redaction | Regex Token Sanitization (`[EMAIL]`, `[PHONE]`, `[URL]`), Median/Mean/Mode/Linear Interpolation/FFill/BFill |
| **`modules/metrics.py`** | Core Metric Indices | NPS = %P - %D, CSAT = (N_4,5 / N_valid) * 100, CES = (1/n) * Σ x_i |
| **`modules/statistical_analysis.py`** | Inferential Guardrails | D'Agostino-Pearson Omnibus, Levene's Test, One-Way ANOVA (η²), Kruskal-Wallis (ε²), Mann-Whitney U (r), Tukey HSD, χ² (Cramér's V) |
| **`modules/statistical_analysis.py`** | Predictive & Psychometrics | OLS Multi-Regression, Variance Inflation Factor (VIF = 1 / (1 - R_j²)), Relative Driver Importance %, Cronbach's Alpha (α) |
| **`modules/statistical_analysis.py`** | Unsupervised Clustering | `StandardScaler` → Principal Component Analysis (2D Projection) → k-Means Clustering |
| **`modules/sentiment.py`** | Sentiment & Aspect ABSA | VADER Lexicon Compound Scoring, spaCy Dependency Parsing (`token.pos_ == "NOUN"` linked to `amod`/`acomp`) |
| **`modules/topic_modeling.py`** | Latent Thematic Discovery | Scikit-Learn TF-IDF Vectorizer (1–2 n-grams) → Non-Negative Matrix Factorization (V ≈ W × H) |
| **`modules/report_generator.py`** | Multi-Format Exporting | OpenPyXL Multi-tab Serialization, JSON Serializer, Zero-Dependency Offline Styled HTML Engine |

---

## 🛠️ Tech Stack

* **Core Runtime:** Python 3.11+ (Tested on Python 3.13.2)
* **Web UI / HUD:** Streamlit
* **Tabular & Array Mathematics:** Pandas, NumPy
* **Inferential Statistics & Modeling:** SciPy (`scipy.stats`), Statsmodels (`statsmodels.api`, `statsmodels.stats`)
* **Machine Learning & Dimensionality:** Scikit-Learn (`PCA`, `KMeans`, `NMF`, `TfidfVectorizer`, `StandardScaler`)
* **Natural Language Processing:** spaCy (`en_core_web_sm`), VADER (`nltk` / `vaderSentiment`)
* **Visualization Engine:** Plotly Express, Plotly Graph Objects
* **Spreadsheet Serialization:** OpenPyXL
* **Testing & Quality Assurance:** PyTest

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/Arup-415/automated-survey-analysis.git](https://github.com/Arup-415/automated-survey-analysis.git)
cd automated-survey-analysis


# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate


pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm



pytest -v


streamlit run app.py


Open http://localhost:8501 in your browser. You can click "Load Benchmark Dataset" to test with pre-built synthetic survey records.

🧪 Benchmark Datasets
A synthetic data generator is provided in data/generate_synthetic_data.py. Additionally, the interactive Survey Benchmark Data Generator Hub (survey_data_hub.html) allows you to generate and test against 5 distinct scenarios:

SaaS Product Feedback: B2B software ratings, feature usability Likert scales, and open support tickets.

HR Employee Engagement: Organizational culture ratings, departmental grouping, and retention verbatims.

E-Commerce Post-Purchase: Delivery satisfaction, product quality, return effort scores, and packaging complaints.

Healthcare Patient Experience: Clinic wait times, physician communication CSAT, and sensitive PII verbatims.

Telecom Service Experience: Network reliability, customer service NPS, plan pricing, and churn risks.

📄 AUTHOR
 ARUP BASU
