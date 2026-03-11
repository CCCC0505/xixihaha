import re
from typing import Dict, List, Optional

import pandas as pd


AGGREGATION_FUNCTIONS = {
    "count": "count",
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
}
MONTH_PATTERN = re.compile(r"^(\d{1,2})月$")


def compute_basic_summary(df: pd.DataFrame) -> Dict[str, int]:
    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "numeric_column_count": int(len(numeric_columns)),
        "categorical_column_count": int(len(categorical_columns)),
        "missing_count": int(df.isna().sum().sum()),
    }


def compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        ratios = [0.0 for _ in df.columns]
    else:
        ratios = (df.isna().sum() / len(df)).values
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_ratio": ratios,
        }
    )
    return summary.sort_values("missing_count", ascending=False).reset_index(drop=True)


def compute_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().transpose().reset_index().rename(columns={"index": "column"})


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def _sort_dimension_values(values: List[str]) -> List[str]:
    month_hits = []
    for value in values:
        match = MONTH_PATTERN.match(str(value))
        if not match:
            return list(values)
        month_hits.append((int(match.group(1)), value))
    return [value for _, value in sorted(month_hits)]


def reshape_wide_dataframe(
    df: pd.DataFrame,
    label_column: str,
    value_columns: List[str],
    selected_labels: Optional[List[str]] = None,
    dimension_name: str = "维度",
    value_name: str = "值",
) -> pd.DataFrame:
    if label_column not in df.columns:
        return pd.DataFrame()

    valid_value_columns = [column for column in value_columns if column in df.columns and column != label_column]
    if not valid_value_columns:
        return pd.DataFrame()

    working_df = df[[label_column] + valid_value_columns].copy()
    if selected_labels:
        allowed = {str(item) for item in selected_labels}
        working_df = working_df[working_df[label_column].astype(str).isin(allowed)]
    if working_df.empty:
        return pd.DataFrame()

    melted = working_df.melt(
        id_vars=[label_column],
        value_vars=valid_value_columns,
        var_name=dimension_name,
        value_name=value_name,
    )
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    melted = melted.dropna(subset=[value_name])
    if melted.empty:
        return pd.DataFrame()

    ordered_dimensions = _sort_dimension_values(valid_value_columns)
    melted[dimension_name] = pd.Categorical(melted[dimension_name], categories=ordered_dimensions, ordered=True)
    return melted.sort_values([dimension_name, label_column]).reset_index(drop=True)


def build_wide_comparison_matrix(
    df: pd.DataFrame,
    label_column: str,
    value_columns: List[str],
    selected_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    valid_value_columns = [column for column in value_columns if column in df.columns and column != label_column]
    if label_column not in df.columns or not valid_value_columns:
        return pd.DataFrame()

    matrix_df = df[[label_column] + valid_value_columns].copy()
    if selected_labels:
        allowed = {str(item) for item in selected_labels}
        matrix_df = matrix_df[matrix_df[label_column].astype(str).isin(allowed)]
    if matrix_df.empty:
        return pd.DataFrame()

    ordered_dimensions = _sort_dimension_values(valid_value_columns)
    return matrix_df[[label_column] + ordered_dimensions].reset_index(drop=True)


def compute_grouped_summary(
    df: pd.DataFrame,
    group_columns: List[str],
    value_column: Optional[str] = None,
    agg_func: str = "count",
    top_n: Optional[int] = 10,
    sort_desc: bool = True,
) -> pd.DataFrame:
    valid_group_columns = [column for column in group_columns if column in df.columns]
    if not valid_group_columns:
        return pd.DataFrame()

    working_df = df.copy()
    if agg_func == "count" or not value_column or value_column not in working_df.columns:
        summary = working_df.groupby(valid_group_columns, dropna=False).size().reset_index(name="value")
    else:
        working_df = working_df.assign(__value=pd.to_numeric(working_df[value_column], errors="coerce"))
        summary = (
            working_df.groupby(valid_group_columns, dropna=False)["__value"]
            .agg(AGGREGATION_FUNCTIONS.get(agg_func, "sum"))
            .reset_index(name="value")
        )

    summary = summary.dropna(subset=["value"]) if "value" in summary.columns else summary
    if summary.empty:
        return pd.DataFrame()

    primary_group = valid_group_columns[0]
    if top_n and top_n > 0:
        top_keys = (
            summary.groupby(primary_group, observed=False)["value"]
            .sum()
            .sort_values(ascending=sort_desc)
            .head(top_n)
            .index
        )
        summary = summary[summary[primary_group].isin(top_keys)]
        summary[primary_group] = pd.Categorical(summary[primary_group], categories=list(top_keys), ordered=True)

    return summary.sort_values(valid_group_columns).reset_index(drop=True)


def compute_pivot_table(
    df: pd.DataFrame,
    index_column: str,
    column_column: str,
    value_column: str,
    agg_func: str = "sum",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    if index_column not in df.columns or column_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame()

    if len({index_column, column_column, value_column}) < 3:
        return pd.DataFrame()

    working_df = df.copy()
    if agg_func == "count":
        pivot = pd.pivot_table(
            working_df,
            index=index_column,
            columns=column_column,
            values=value_column,
            aggfunc="count",
            fill_value=fill_value,
        )
    else:
        working_df[value_column] = pd.to_numeric(working_df[value_column], errors="coerce")
        pivot = pd.pivot_table(
            working_df,
            index=index_column,
            columns=column_column,
            values=value_column,
            aggfunc=AGGREGATION_FUNCTIONS.get(agg_func, "sum"),
            fill_value=fill_value,
        )

    pivot.columns = [str(column) for column in pivot.columns]
    return pivot.reset_index()


def compute_time_series_summary(
    df: pd.DataFrame,
    date_column: str,
    value_column: Optional[str] = None,
    agg_func: str = "count",
    frequency: str = "M",
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    if date_column not in df.columns:
        return pd.DataFrame()

    working_df = df.copy()
    working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")
    working_df = working_df.dropna(subset=[date_column])
    if working_df.empty:
        return pd.DataFrame()

    working_df["period"] = working_df[date_column].dt.to_period(frequency).dt.to_timestamp()
    group_fields = ["period"]
    if group_column and group_column in working_df.columns:
        group_fields.append(group_column)

    if agg_func == "count" or not value_column or value_column not in working_df.columns:
        summary = working_df.groupby(group_fields, dropna=False).size().reset_index(name="value")
    else:
        working_df["__value"] = pd.to_numeric(working_df[value_column], errors="coerce")
        summary = (
            working_df.groupby(group_fields, dropna=False)["__value"]
            .agg(AGGREGATION_FUNCTIONS.get(agg_func, "sum"))
            .reset_index(name="value")
        )

    return summary.dropna(subset=["value"]).sort_values(group_fields).reset_index(drop=True)


def detect_outliers(df: pd.DataFrame, column: str, method: str = "IQR", threshold: float = 3.0) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError("列不存在：%s" % column)

    numeric_series = pd.to_numeric(df[column], errors="coerce")
    report = pd.DataFrame({"value": numeric_series})
    report["is_outlier"] = False

    if numeric_series.dropna().empty:
        return report

    if method == "Z-score":
        std = numeric_series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return report
        mean = numeric_series.mean()
        z_scores = (numeric_series - mean) / std
        report["score"] = z_scores
        report["is_outlier"] = z_scores.abs() > threshold
        return report

    q1 = numeric_series.quantile(0.25)
    q3 = numeric_series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    report["score"] = numeric_series
    report["is_outlier"] = (numeric_series < lower) | (numeric_series > upper)
    return report