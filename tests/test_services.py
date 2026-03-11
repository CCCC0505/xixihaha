from io import BytesIO

import pandas as pd

from data_app.services.analysis import (
    build_wide_comparison_matrix,
    compute_correlation,
    compute_grouped_summary,
    compute_pivot_table,
    compute_time_series_summary,
    detect_outliers,
    reshape_wide_dataframe,
)
from data_app.services.cleaning import apply_cleaning_step, build_cleaning_step
from data_app.services.filters import FilterCondition, apply_filters
from data_app.services.loader import load_dataset
from data_app.services.subset import build_column_profile, parse_row_indices_text, select_analysis_subset
from data_app.utils.exporters import (
    build_analysis_report_html_bytes,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
)
from data_app.utils.text_analysis import compute_text_frequency


class UploadedFileStub(object):
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def test_load_csv_with_utf8():
    content = "城市,销量\n上海,10\n北京,20\n".encode("utf-8")
    file_obj = UploadedFileStub("sample.csv", content)
    df = load_dataset(file_obj)
    assert list(df.columns) == ["城市", "销量"]
    assert df.iloc[0]["城市"] == "上海"


def test_load_xlsx_single_sheet():
    source = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        source.to_excel(writer, index=False, sheet_name="Sheet1")
    file_obj = UploadedFileStub("sample.xlsx", buffer.getvalue())
    df = load_dataset(file_obj, "Sheet1")
    assert df.equals(source)


def test_build_column_profile_and_row_subset_helpers():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [10, 20, 30]})
    profile = build_column_profile(df, ["A", "C"])
    assert profile["选择"].tolist() == [True, False, True]

    indices = parse_row_indices_text("0,2,1-2", 2)
    assert indices == [0, 1, 2]

    subset = select_analysis_subset(df, selected_columns=["A", "C"], row_mode="手动行号", row_indices=[0, 2])
    assert list(subset.columns) == ["A", "C"]
    assert len(subset) == 2


def test_apply_filters_for_text_and_range():
    df = pd.DataFrame({"城市": ["上海", "北京", "上海"], "销量": [10, 20, 30]})
    conditions = [
        FilterCondition(column="城市", operator="等于", value="上海"),
        FilterCondition(column="销量", operator="数值区间", value="15", value_to="35"),
    ]
    result = apply_filters(df, conditions)
    assert len(result) == 1
    assert result.iloc[0]["销量"] == 30


def test_apply_filters_for_date_range():
    df = pd.DataFrame({"日期": ["2024-01-01", "2024-02-01", "2024-03-01"], "销量": [1, 2, 3]})
    conditions = [
        FilterCondition(column="日期", operator="日期区间", value="2024-01-15", value_to="2024-02-15"),
    ]
    result = apply_filters(df, conditions)
    assert len(result) == 1
    assert result.iloc[0]["销量"] == 2


def test_cleaning_missing_fill_and_drop_duplicates():
    df = pd.DataFrame({"值": [1, None, 1], "标签": ["a", "b", "a"]})
    fill_step = build_cleaning_step("缺失值处理", ["值"], {"strategy": "按均值填充"})
    filled = apply_cleaning_step(df, fill_step).dataframe
    assert filled["值"].isna().sum() == 0

    dedupe_step = build_cleaning_step("删除重复值", ["值", "标签"], {"keep": "first"})
    deduped = apply_cleaning_step(filled, dedupe_step).dataframe
    assert len(deduped) == 2


def test_rename_drop_sort_and_scale_cleaning():
    df = pd.DataFrame({"旧列": [3, 1, 2], "保留列": [3, 4, 5]})
    rename_step = build_cleaning_step("重命名列", ["旧列"], {"new_name": "新列"})
    renamed = apply_cleaning_step(df, rename_step).dataframe
    assert "新列" in renamed.columns

    sort_step = build_cleaning_step("排序数据", ["新列"], {"ascending": True})
    sorted_df = apply_cleaning_step(renamed, sort_step).dataframe
    assert list(sorted_df["新列"]) == [1, 2, 3]

    scale_step = build_cleaning_step("数值缩放", ["新列"], {"mode": "Min-Max归一化"})
    scaled_df = apply_cleaning_step(sorted_df, scale_step).dataframe
    assert scaled_df["新列"].min() == 0
    assert scaled_df["新列"].max() == 1

    drop_step = build_cleaning_step("删除列", ["保留列"], {"confirm": True})
    dropped = apply_cleaning_step(renamed, drop_step).dataframe
    assert list(dropped.columns) == ["新列"]


def test_outlier_detection_and_cleaning():
    df = pd.DataFrame({"分数": [10, 11, 12, 300]})
    report = detect_outliers(df, "分数", "IQR")
    assert report["is_outlier"].sum() == 1

    step = build_cleaning_step("异常值处理", ["分数"], {"method": "IQR", "mode": "删除异常值"})
    cleaned = apply_cleaning_step(df, step).dataframe
    assert len(cleaned) == 3


def test_grouped_summary_and_pivot_analysis():
    df = pd.DataFrame(
        {
            "月份": ["一月", "一月", "二月", "二月"],
            "类别": ["工资", "差旅", "工资", "差旅"],
            "金额": [100, 50, 120, 60],
        }
    )
    summary = compute_grouped_summary(df, ["月份", "类别"], value_column="金额", agg_func="sum", top_n=10)
    assert summary["value"].sum() == 330

    pivot = compute_pivot_table(df, "月份", "类别", "金额", agg_func="sum", fill_value=0)
    assert list(pivot.columns) == ["月份", "工资", "差旅"]


def test_pivot_returns_empty_for_duplicate_roles():
    df = pd.DataFrame({"Id": [1, 2, 3], "值": [10, 20, 30]})
    pivot = compute_pivot_table(df, "Id", "Id", "Id", agg_func="sum", fill_value=0)
    assert pivot.empty


def test_wide_table_helpers_for_month_trend():
    df = pd.DataFrame(
        {
            "科目名称": ["工资", "差旅费"],
            "1月": [100, 20],
            "2月": [120, 30],
            "3月": [90, 25],
        }
    )
    matrix_df = build_wide_comparison_matrix(df, "科目名称", ["3月", "1月", "2月"], ["工资", "差旅费"])
    assert list(matrix_df.columns) == ["科目名称", "1月", "2月", "3月"]

    long_df = reshape_wide_dataframe(df, "科目名称", ["3月", "1月", "2月"], ["工资", "差旅费"], dimension_name="月份", value_name="金额")
    assert len(long_df) == 6
    assert list(long_df["月份"].cat.categories) == ["1月", "2月", "3月"]


def test_time_series_and_text_frequency():
    df = pd.DataFrame(
        {
            "日期": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "销量": [10, 20, 30],
            "评论": ["苹果 很 甜", "苹果 新鲜", "香蕉 也 很 甜"],
        }
    )
    series_df = compute_time_series_summary(df, "日期", "销量", agg_func="sum", frequency="M")
    assert len(series_df) == 2
    assert series_df["value"].sum() == 60

    text_df = compute_text_frequency(df["评论"], top_n=5)
    assert "苹果" in list(text_df["token"])


def test_correlation_requires_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": ["x", "y", "z"]})
    corr = compute_correlation(df)
    assert list(corr.columns) == ["a", "b"]


def test_exporters_generate_bytes_and_report():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = dataframe_to_csv_bytes(df)
    excel_bytes = dataframe_to_excel_bytes(df)
    report_bytes = build_analysis_report_html_bytes(
        df=df,
        source_name="sample.csv",
        selected_columns=["a", "b"],
        cleaning_history=[{"action": "排序数据", "target_columns": ["a"], "params": {"ascending": True}}],
        filter_conditions=[{"column": "b", "operator": "等于", "value": "x", "value_to": None}],
        current_chart=None,
    )
    assert csv_bytes.decode("utf-8-sig").startswith("a")
    assert len(excel_bytes) > 0
    assert b"<html" in report_bytes.lower()