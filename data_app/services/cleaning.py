from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from data_app.services.analysis import detect_outliers


@dataclass
class CleaningStep:
    action: str
    target_columns: List[str]
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_columns": self.target_columns,
            "params": self.params,
        }


@dataclass
class CleaningResult:
    dataframe: pd.DataFrame
    message: str
    outlier_report: Optional[pd.DataFrame] = None


def build_cleaning_step(action: str, target_columns: List[str], params: Dict[str, Any]) -> CleaningStep:
    return CleaningStep(action=action, target_columns=target_columns, params=params)


def _require_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    valid = [column for column in columns if column in df.columns]
    if not valid and columns:
        raise ValueError("所选列不存在。")
    return valid


def _fill_missing(df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> CleaningResult:
    strategy = params.get("strategy")
    target_df = df.copy()
    valid_columns = _require_columns(target_df, columns) if columns else list(target_df.columns)

    if strategy == "删除含缺失值的行":
        target_df = target_df.dropna(subset=valid_columns)
        return CleaningResult(target_df.reset_index(drop=True), "已删除目标列中包含缺失值的行。")

    for column in valid_columns:
        if strategy == "按均值填充":
            value = pd.to_numeric(target_df[column], errors="coerce").mean()
        elif strategy == "按中位数填充":
            value = pd.to_numeric(target_df[column], errors="coerce").median()
        elif strategy == "按众数填充":
            mode = target_df[column].mode(dropna=True)
            value = mode.iloc[0] if not mode.empty else None
        else:
            value = params.get("fill_value")
        target_df[column] = target_df[column].fillna(value)
    return CleaningResult(target_df, "缺失值处理已完成。")


def _convert_type(df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> CleaningResult:
    target_df = df.copy()
    valid_columns = _require_columns(target_df, columns)
    dtype = params.get("dtype")
    if not valid_columns:
        raise ValueError("类型转换至少需要选择一列。")

    for column in valid_columns:
        if dtype == "string":
            target_df[column] = target_df[column].astype(str)
        elif dtype == "int":
            target_df[column] = pd.to_numeric(target_df[column], errors="coerce").astype("Int64")
        elif dtype == "float":
            target_df[column] = pd.to_numeric(target_df[column], errors="coerce").astype(float)
        elif dtype == "datetime":
            target_df[column] = pd.to_datetime(target_df[column], errors="coerce")
    return CleaningResult(target_df, "列类型转换已完成。")


def _handle_outliers(df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> CleaningResult:
    target_df = df.copy()
    valid_columns = _require_columns(target_df, columns)
    if not valid_columns:
        raise ValueError("异常值处理至少需要选择一个数值列。")

    combined_report = []
    method = params.get("method", "IQR")
    threshold = float(params.get("threshold", 3.0))
    mode = params.get("mode", "标记异常值")
    rows_to_drop = set()

    for column in valid_columns:
        report = detect_outliers(target_df, column, method, threshold)
        report = report[report["is_outlier"]]
        if not report.empty:
            report = report.copy()
            report["column"] = column
            report["row_index"] = report.index
            combined_report.append(report)
            rows_to_drop.update(report.index.tolist())

    if mode == "删除异常值" and rows_to_drop:
        target_df = target_df.drop(index=list(rows_to_drop)).reset_index(drop=True)
        message = "已删除 %s 条异常值记录。" % len(rows_to_drop)
    else:
        message = "异常值检测已完成，结果已生成。"

    outlier_report = pd.concat(combined_report, axis=0).reset_index(drop=True) if combined_report else pd.DataFrame()
    return CleaningResult(target_df, message, outlier_report=outlier_report)


def _sort_dataframe(df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> CleaningResult:
    valid_columns = _require_columns(df, columns)
    if len(valid_columns) != 1:
        raise ValueError("排序时必须且只能选择一个列。")
    ascending = bool(params.get("ascending", True))
    sorted_df = df.sort_values(by=valid_columns[0], ascending=ascending, kind="mergesort").reset_index(drop=True)
    return CleaningResult(sorted_df, "排序已完成。")


def _scale_numeric_columns(df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> CleaningResult:
    valid_columns = _require_columns(df, columns)
    if not valid_columns:
        raise ValueError("数值缩放至少需要选择一个列。")

    target_df = df.copy()
    mode = params.get("mode", "Min-Max归一化")
    for column in valid_columns:
        series = pd.to_numeric(target_df[column], errors="coerce")
        if series.dropna().empty:
            continue
        if mode == "Z-score标准化":
            std = series.std(ddof=0)
            scaled = 0 if pd.isna(std) or std == 0 else (series - series.mean()) / std
        else:
            min_value = series.min()
            max_value = series.max()
            scaled = 0 if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value else (series - min_value) / (max_value - min_value)
        target_df[column] = scaled
    return CleaningResult(target_df, "数值缩放已完成。")


def apply_cleaning_step(df: pd.DataFrame, step: CleaningStep) -> CleaningResult:
    if df is None:
        raise ValueError("没有可清洗的数据。")

    action = step.action
    if action == "缺失值处理":
        return _fill_missing(df, step.target_columns, step.params)
    if action == "删除重复值":
        keep = step.params.get("keep", "first")
        subset = _require_columns(df, step.target_columns) if step.target_columns else None
        cleaned = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        return CleaningResult(cleaned, "重复值删除已完成。")
    if action == "类型转换":
        return _convert_type(df, step.target_columns, step.params)
    if action == "异常值处理":
        return _handle_outliers(df, step.target_columns, step.params)
    if action == "重命名列":
        valid_columns = _require_columns(df, step.target_columns)
        if len(valid_columns) != 1:
            raise ValueError("重命名列时必须且只能选择一个列。")
        new_name = step.params.get("new_name")
        if not new_name:
            raise ValueError("请输入新的列名。")
        renamed = df.rename(columns={valid_columns[0]: new_name})
        return CleaningResult(renamed, "列重命名已完成。")
    if action == "删除列":
        valid_columns = _require_columns(df, step.target_columns)
        if not step.params.get("confirm"):
            raise ValueError("请先确认删除所选列。")
        dropped = df.drop(columns=valid_columns)
        return CleaningResult(dropped, "所选列已删除。")
    if action == "排序数据":
        return _sort_dataframe(df, step.target_columns, step.params)
    if action == "数值缩放":
        return _scale_numeric_columns(df, step.target_columns, step.params)
    raise ValueError("暂不支持的清洗操作：%s" % action)
