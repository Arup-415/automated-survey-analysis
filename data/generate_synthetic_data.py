"""
data/generate_synthetic_data.py
Generates a realistic synthetic enterprise survey dataset for local testing.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_survey_dataset(n_rows: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    random.seed(random_seed)

    departments = ["Engineering", "Sales", "Customer Support", "Marketing", "Product"]
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    genders = ["Female", "Male", "Non-Binary", "Prefer not to say"]
    
    # Base distributions
    dept_choices = np.random.choice(departments, size=n_rows, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    region_choices = np.random.choice(regions, size=n_rows, p=[0.40, 0.30, 0.20, 0.10])
    gender_choices = np.random.choice(genders, size=n_rows, p=[0.48, 0.46, 0.04, 0.02])
    
    # Age with slight skew
    ages = np.clip(np.random.normal(loc=36, scale=10, size=n_rows).astype(int), 18, 70)
    tenures = np.clip(np.random.exponential(scale=24, size=n_rows).astype(int) + 1, 1, 120)

    # Coordinated scores to simulate real-world statistical behavior
    # Engagement varies by department (Engineering and Product slightly higher)
    dept_engagement_bias = {
        "Engineering": 1.2,
        "Product": 0.8,
        "Marketing": 0.0,
        "Sales": -0.5,
        "Customer Support": -1.0
    }
    
    base_engagement = np.random.normal(loc=6.5, scale=1.5, size=n_rows)
    engagement_scores = np.array([
        base_engagement[i] + dept_engagement_bias[dept_choices[i]] for i in range(n_rows)
    ])
    engagement_scores = np.clip(np.round(engagement_scores, 1), 1.0, 10.0)

    # NPS (0 to 10) correlated with engagement
    nps_raw = np.round((engagement_scores / 10.0) * 10 + np.random.normal(0, 1.5, n_rows))
    nps_scores = np.clip(nps_raw, 0, 10).astype(int)

    # CSAT (1 to 5) correlated with NPS
    csat_raw = np.round((nps_scores / 10.0) * 4 + 1 + np.random.normal(0, 0.7, n_rows))
    csat_scores = np.clip(csat_raw, 1, 5).astype(int)

    # CES (1 to 7) Customer Effort Score (inversely correlated: lower score = less effort = better)
    ces_raw = np.round(8 - ((csat_scores / 5.0) * 6 + np.random.normal(0, 0.8, n_rows)))
    ces_scores = np.clip(ces_raw, 1, 7).astype(int)

    # Synthetic text feedback pools with specific aspect patterns
    positive_comments = [
        "The software platform is exceptionally intuitive and saves our team hours.",
        "Customer support was very responsive and resolved my billing issue in 10 minutes.",
        "Great UI and clean design. Very satisfied with recent feature releases.",
        "Fast performance, excellent reliability, and helpful documentation.",
        "Our account manager is fantastic. Keep up the great work!"
    ]

    neutral_comments = [
        "The application works fine, but performance slows down occasionally during peak hours.",
        "Average experience overall. Does what it promises, nothing more, nothing less.",
        "Features are acceptable, but pricing seems a bit high for small teams.",
        "Support is fine, but response times could be improved.",
        "The interface is standard. A few bugs here and there."
    ]

    negative_comments = [
        "Billing is a nightmare. I was overcharged twice and customer support was completely unresponsive.",
        "The UI update makes navigation horrible. Features crash constantly on Chrome.",
        "Terrible experience. Pricing increased without notice and the software is laggy.",
        "Customer service takes 4 days to reply. We are considering switching providers.",
        "Constant downtime and sync errors. Frustrating and unhelpful."
    ]

    pii_templates = [
        " Contact me at user{i}@example.com if you want more details.",
        " You can reach my direct desk at +1-555-01{i} to discuss.",
        " More details at https://internal-feedback-{i}.org/details.",
        ""
    ]

    text_feedback = []
    for i in range(n_rows):
        nps = nps_scores[i]
        if nps >= 9:
            base_txt = random.choice(positive_comments)
        elif nps >= 7:
            base_txt = random.choice(neutral_comments)
        else:
            base_txt = random.choice(negative_comments)

        # Inject realistic PII into ~10% of responses
        if random.random() < 0.10:
            pii_addon = random.choice(pii_templates).format(i=100 + (i % 800))
            base_txt += pii_addon

        text_feedback.append(base_txt)

    # Timestamps spread over the last 90 days
    base_date = datetime.now() - timedelta(days=90)
    timestamps = [base_date + timedelta(minutes=int(x)) for x in np.random.uniform(0, 90 * 24 * 60, n_rows)]

    # Generate IDs
    response_ids = [f"RESP-{10000 + i}" for i in range(n_rows)]

    df = pd.DataFrame({
        "response_id": response_ids,
        "submission_date": timestamps,
        "age": ages,
        "gender": gender_choices,
        "department": dept_choices,
        "region": region_choices,
        "tenure_months": tenures,
        "engagement_score": engagement_scores,
        "nps_score": nps_scores,
        "satisfaction_rating": csat_scores,
        "effort_score": ces_scores,
        "open_feedback": text_feedback
    })

    # Add a small percentage of realistic missing values
    df.loc[df.sample(frac=0.03, random_state=1).index, "age"] = np.nan
    df.loc[df.sample(frac=0.02, random_state=2).index, "open_feedback"] = np.nan
    df.loc[df.sample(frac=0.01, random_state=3).index, "satisfaction_rating"] = np.nan

    # Introduce 10 duplicate rows to test duplicate detection
    duplicates = df.iloc[:10].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "sample_survey.csv")
    print("Generating benchmark synthetic dataset...")
    df = generate_survey_dataset(n_rows=3000)
    df.to_csv(output_path, index=False)
    print(f"Success: {len(df)} rows saved to '{output_path}'.")
    