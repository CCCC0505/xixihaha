from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


NUMERIC_CHARTS = {"柱状图", "堆叠柱状图", "折线图", "面积图", "散点图", "箱线图", "小提琴图", "直方图"}


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: Optional[str],
    y_axis: Optional[str],
    color: Optional[str],
    top_n: int = 10,
    matrix_df: Optional[pd.DataFrame] = None,
    hist_bins: int = 20,
):
    if chart_type == "热力图":
        matrix = matrix_df if matrix_df is not None else df.select_dtypes(include="number").corr(numeric_only=True)
        if matrix is None or matrix.empty:
            return None
        return go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale="Blues",
            )
        )

    if chart_type == "饼图":
        if x_axis not in df.columns:
            return None
        if y_axis and y_axis in df.columns:
            chart_df = df[[x_axis, y_axis]].copy()
            chart_df[y_axis] = pd.to_numeric(chart_df[y_axis], errors="coerce")
            chart_df = chart_df.dropna(subset=[y_axis]).groupby(x_axis, as_index=False)[y_axis].sum()
            chart_df = chart_df.sort_values(y_axis, ascending=False).head(top_n)
            return px.pie(chart_df, names=x_axis, values=y_axis)
        counts = df[x_axis].astype(str).value_counts().head(top_n).reset_index()
        counts.columns = [x_axis, "count"]
        return px.pie(counts, names=x_axis, values="count")

    if chart_type == "直方图":
        if x_axis not in df.columns:
            return None
        return px.histogram(df, x=x_axis, color=color if color in df.columns else None, nbins=hist_bins)

    if x_axis not in df.columns:
        return None

    if chart_type == "柱状图":
        if y_axis and y_axis in df.columns:
            return px.bar(df, x=x_axis, y=y_axis, color=color if color in df.columns else None)
        counts = df[x_axis].astype(str).value_counts().head(top_n).reset_index()
        counts.columns = [x_axis, "count"]
        return px.bar(counts, x=x_axis, y="count")

    if chart_type == "堆叠柱状图":
        if y_axis is None or y_axis not in df.columns:
            return None
        figure = px.bar(df, x=x_axis, y=y_axis, color=color if color in df.columns else None)
        figure.update_layout(barmode="stack")
        return figure

    if chart_type == "折线图":
        if y_axis is None or y_axis not in df.columns:
            return None
        return px.line(df.sort_values(by=x_axis), x=x_axis, y=y_axis, color=color if color in df.columns else None, markers=True)

    if chart_type == "面积图":
        if y_axis is None or y_axis not in df.columns:
            return None
        return px.area(df.sort_values(by=x_axis), x=x_axis, y=y_axis, color=color if color in df.columns else None)

    if chart_type == "散点图":
        if y_axis is None or y_axis not in df.columns:
            return None
        return px.scatter(df, x=x_axis, y=y_axis, color=color if color in df.columns else None)

    if chart_type == "箱线图":
        if y_axis is None or y_axis not in df.columns:
            return None
        return px.box(df, x=x_axis, y=y_axis, color=color if color in df.columns else None)

    if chart_type == "小提琴图":
        if y_axis is None or y_axis not in df.columns:
            return None
        return px.violin(df, x=x_axis, y=y_axis, color=color if color in df.columns else None, box=True)

    return None


def chart_to_html_bytes(figure) -> bytes:
    return figure.to_html(include_plotlyjs="cdn").encode("utf-8")


def chart_to_png_bytes(figure):
    try:
        return figure.to_image(format="png")
    except Exception:
        return None
