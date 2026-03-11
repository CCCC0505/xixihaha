from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import pandas as pd


@dataclass
class FilterCondition:
    column: str
    operator: str
    value: Optional[Any] = None
    value_to: Optional[Any] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FilterCondition":
        return cls(
            column=payload.get("column"),
            operator=payload.get("operator"),
            value=payload.get("value"),
            value_to=payload.get("value_to"),
        )


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def apply_filters(dataframe: pd.DataFrame, conditions: Iterable[FilterCondition]) -> pd.DataFrame:
    filtered = dataframe.copy()
    for condition in conditions:
        if not condition.column or condition.column not in filtered.columns:
            continue

        series = filtered[condition.column]
        operator = condition.operator

        if operator == "等于":
            filtered = filtered[series.astype(str) == str(condition.value)]
        elif operator == "不等于":
            filtered = filtered[series.astype(str) != str(condition.value)]
        elif operator == "包含":
            filtered = filtered[series.astype(str).str.contains(str(condition.value), na=False)]
        elif operator == "数值区间":
            numeric_series = pd.to_numeric(series, errors="coerce")
            lower = float(condition.value) if condition.value not in (None, "") else None
            upper = float(condition.value_to) if condition.value_to not in (None, "") else None
            mask = pd.Series(True, index=filtered.index)
            if lower is not None:
                mask &= numeric_series >= lower
            if upper is not None:
                mask &= numeric_series <= upper
            filtered = filtered[mask]
        elif operator == "日期区间":
            datetime_series = _coerce_datetime(series)
            start = pd.to_datetime(condition.value, errors="coerce")
            end = pd.to_datetime(condition.value_to, errors="coerce")
            mask = pd.Series(True, index=filtered.index)
            if not pd.isna(start):
                mask &= datetime_series >= start
            if not pd.isna(end):
                mask &= datetime_series <= end
            filtered = filtered[mask]
        elif operator == "为空":
            filtered = filtered[series.isna() | (series.astype(str).str.strip() == "")]
        elif operator == "不为空":
            filtered = filtered[~series.isna() & (series.astype(str).str.strip() != "")]

    return filtered.reset_index(drop=True)
