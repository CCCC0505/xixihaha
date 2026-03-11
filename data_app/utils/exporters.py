from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd

from data_app.services.analysis import compute_basic_summary, compute_missing_summary, compute_numeric_summary


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return output.getvalue()


def _history_to_html(items: List[Dict[str, Any]], title: str) -> str:
    if not items:
        return "<p>无</p>"
    rows = []
    for item in items:
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                item.get("action", "-"),
                ", ".join(item.get("target_columns", []) or []),
                str(item.get("params", {})),
            )
        )
    return (
        "<table><thead><tr><th>{}</th><th>目标列</th><th>参数</th></tr></thead><tbody>{}</tbody></table>".format(
            title, "".join(rows)
        )
    )


def build_analysis_report_html_bytes(
    df: pd.DataFrame,
    source_name: Optional[str],
    selected_columns: Optional[List[str]],
    cleaning_history: Optional[List[Dict[str, Any]]],
    filter_conditions: Optional[List[Dict[str, Any]]],
    current_chart=None,
) -> bytes:
    display_df = df[selected_columns] if selected_columns else df
    summary = compute_basic_summary(display_df)
    missing_df = compute_missing_summary(display_df)
    numeric_df = compute_numeric_summary(display_df)

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chart_html = ""
    if current_chart is not None:
        chart_html = current_chart.to_html(full_html=False, include_plotlyjs="cdn")

    filters_html = _history_to_html(
        [
            {
                "action": item.get("operator", "-"),
                "target_columns": [item.get("column", "")],
                "params": {"value": item.get("value"), "value_to": item.get("value_to")},
            }
            for item in (filter_conditions or [])
        ],
        "筛选条件",
    )
    cleaning_html = _history_to_html(cleaning_history or [], "处理步骤")
    numeric_table = numeric_df.to_html(index=False, border=0) if not numeric_df.empty else "<p>当前没有数值列可生成描述性统计。</p>"

    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>数据分析报告</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 32px; border-radius: 16px; box-shadow: 0 10px 30px rgba(15,23,42,0.08); }}
h1, h2 {{ color: #0f172a; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 20px 0; }}
.card {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 16px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; margin-bottom: 24px; }}
th, td {{ border: 1px solid #dbeafe; padding: 8px 10px; font-size: 14px; text-align: left; }}
th {{ background: #dbeafe; }}
.meta {{ color: #475569; margin-bottom: 20px; }}
.section {{ margin-top: 28px; }}
</style>
</head>
<body>
<div class="container">
<h1>数据分析报告</h1>
<p class="meta">生成时间：{report_time}</p>
<p class="meta">数据来源：{source_name or '未命名数据文件'}</p>
<div class="grid">
  <div class="card"><strong>总行数</strong><div>{summary['row_count']}</div></div>
  <div class="card"><strong>总列数</strong><div>{summary['column_count']}</div></div>
  <div class="card"><strong>缺失值总数</strong><div>{summary['missing_count']}</div></div>
</div>
<div class="section">
  <h2>分析范围</h2>
  <p>已选分析列：{", ".join(selected_columns or list(display_df.columns))}</p>
</div>
<div class="section">
  <h2>筛选条件</h2>
  {filters_html}
</div>
<div class="section">
  <h2>处理历史</h2>
  {cleaning_html}
</div>
<div class="section">
  <h2>缺失值统计</h2>
  {missing_df.to_html(index=False, border=0)}
</div>
<div class="section">
  <h2>描述性统计</h2>
  {numeric_table}
</div>
<div class="section">
  <h2>当前图表</h2>
  {chart_html or '<p>当前未生成图表。</p>'}
</div>
</div>
</body>
</html>
"""
    return html.encode("utf-8")
