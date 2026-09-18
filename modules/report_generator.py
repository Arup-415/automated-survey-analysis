"""
modules/report_generator.py
Export facilities for JSON, Excel (xlsx), and self-contained Executive HTML reports.
"""

from typing import Dict, Any, List
import json
import io
import pandas as pd

def generate_json_report(payload: Dict[str, Any]) -> str:
    def clean_obj(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)
    return json.dumps(payload, indent=2, default=clean_obj)

def generate_excel_report(
    column_profiles: List[Dict[str, Any]],
    schema_df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    aspects_df: pd.DataFrame
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(column_profiles).to_excel(writer, sheet_name="Data Profile", index=False)
        schema_df.to_excel(writer, sheet_name="Schema & Roles", index=False)
        if not descriptive_df.empty:
            descriptive_df.to_excel(writer, sheet_name="Descriptive Stats", index=False)
        if not topic_df.empty:
            topic_df.to_excel(writer, sheet_name="Topic Modeling", index=False)
        if not aspects_df.empty:
            aspects_df.to_excel(writer, sheet_name="Aspect Sentiments", index=False)
    output.seek(0)
    return output.read()

def generate_executive_html_report(
    file_name: str,
    profile: Dict[str, Any],
    nps_res: Dict[str, Any],
    csat_res: Dict[str, Any],
    ces_res: Dict[str, Any],
    insights: List[Dict[str, Any]],
    topics: List[Dict[str, Any]]
) -> str:
    """Generates an offline, styled single-page HTML executive briefing."""
    insights_html = "".join([
        f"<li style='margin-bottom: 8px;'><strong>[{ins['Category']}]</strong> {ins['Text']}</li>"
        for ins in insights
    ])
    topics_html = "".join([
        f"<tr><td>{t.get('Topic Name','')}</td><td>{t.get('Keywords','')}</td><td>{t.get('Responses','')}</td><td>{t.get('Mean Sentiment','')}</td></tr>"
        for t in topics
    ]) if topics else "<tr><td colspan='4'>No thematic clusters extracted.</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Executive Briefing - {file_name}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; color: #1e293b; background: #f8fafc; }}
    .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }}
    h1 {{ color: #0f172a; margin-bottom: 4px; }}
    .subtitle {{ color: #64748b; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
    .card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center; }}
    .card h3 {{ margin: 0; font-size: 0.8rem; color: #475569; text-transform: uppercase; }}
    .card h2 {{ margin: 8px 0 0 0; font-size: 1.8rem; color: #0284c7; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; }}
    th {{ background: #f8fafc; color: #475569; }}
</style>
</head>
<body>
<div class="container">
    <h1>Survey Intelligence Briefing</h1>
    <div class="subtitle">Dataset: {file_name} | Generated automatically by Local Survey Analytics Engine</div>
    <div class="grid">
        <div class="card"><h3>Total Records</h3><h2>{profile.get('total_rows', 0):,}</h2></div>
        <div class="card"><h3>NPS Score</h3><h2>{nps_res.get('nps_score', 'N/A')}</h2></div>
        <div class="card"><h3>CSAT Index</h3><h2>{f"{csat_res['csat_score']}%" if csat_res.get('status') == 'success' else 'N/A'}</h2></div>
        <div class="card"><h3>CES Effort</h3><h2>{ces_res.get('mean_score', 'N/A')}</h2></div>
    </div>
    <h2>Automated Executive Insights</h2>
    <ul>{insights_html}</ul>
    <h2>Unsupervised Thematic Topics (NMF)</h2>
    <table>
        <thead><tr><th>Topic</th><th>Keywords</th><th>Volume</th><th>Mean Polarity</th></tr></thead>
        <tbody>{topics_html}</tbody>
    </table>
</div>
</body>
</html>"""
    return html
