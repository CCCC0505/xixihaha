from typing import List, Optional

import pandas as pd


def build_column_profile(df: pd.DataFrame, selected_columns: Optional[List[str]] = None) -> pd.DataFrame:
    selected_set = set(selected_columns or list(df.columns))
    return pd.DataFrame(
        {
            "选择": [column in selected_set for column in df.columns],
            "列名": list(df.columns),
            "数据类型": [str(dtype) for dtype in df.dtypes],
            "非空数": [int(df[column].notna().sum()) for column in df.columns],
            "缺失值": [int(df[column].isna().sum()) for column in df.columns],
            "唯一值": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )


def parse_row_indices_text(text: str, max_index: int) -> List[int]:
    if not text.strip():
        return []

    indices = set()
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                if 0 <= value <= max_index:
                    indices.add(value)
        else:
            value = int(item)
            if 0 <= value <= max_index:
                indices.add(value)
    return sorted(indices)


def select_analysis_subset(
    df: pd.DataFrame,
    selected_columns: Optional[List[str]] = None,
    row_mode: str = "全部行",
    row_indices: Optional[List[int]] = None,
    row_start: int = 0,
    row_end: Optional[int] = None,
    row_count: int = 100,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    valid_columns = [column for column in (selected_columns or list(df.columns)) if column in df.columns]
    if not valid_columns:
        return pd.DataFrame()

    subset = df.loc[:, valid_columns]
    total_rows = len(subset)

    if row_mode == "行号区间":
        end = total_rows - 1 if row_end is None else min(row_end, total_rows - 1)
        start = max(0, min(row_start, end))
        subset = subset.iloc[start : end + 1]
    elif row_mode == "前N行":
        subset = subset.head(max(1, row_count))
    elif row_mode in ("手动行号", "预览勾选"):
        valid_indices = [index for index in (row_indices or []) if 0 <= index < total_rows]
        if not valid_indices:
            return pd.DataFrame(columns=valid_columns)
        subset = subset.iloc[valid_indices]

    return subset.copy()
