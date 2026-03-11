import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from data_app.services.analysis import (
    compute_basic_summary,
    compute_grouped_summary,
    compute_missing_summary,
    compute_numeric_summary,
    compute_pivot_table,
    compute_time_series_summary,
    build_wide_comparison_matrix,
    detect_outliers,
    reshape_wide_dataframe,
)
from data_app.services.cleaning import apply_cleaning_step, build_cleaning_step
from data_app.services.filters import FilterCondition, apply_filters
from data_app.services.loader import load_dataset, load_sheet_names
from data_app.services.subset import build_column_profile, parse_row_indices_text, select_analysis_subset
from data_app.utils.exporters import (
    build_analysis_report_html_bytes,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
)
from data_app.utils.text_analysis import compute_text_frequency, generate_wordcloud_image, has_wordcloud_support
from data_app.utils.visualizations import build_chart, chart_to_html_bytes, chart_to_png_bytes


st.set_page_config(page_title="数据分析处理平台", layout="wide")


OPERATORS = ["等于", "不等于", "包含", "数值区间", "日期区间", "为空", "不为空"]
CLEANING_ACTIONS = ["缺失值处理", "删除重复值", "类型转换", "异常值处理", "重命名列", "删除列", "排序数据", "数值缩放"]
CHART_TYPES = ["柱状图", "堆叠柱状图", "折线图", "面积图", "散点图", "箱线图", "小提琴图", "直方图", "饼图", "热力图"]
AGGREGATION_LABELS = {
    "计数": "count",
    "求和": "sum",
    "均值": "mean",
    "中位数": "median",
    "最小值": "min",
    "最大值": "max",
}
FREQUENCY_LABELS = {"按天": "D", "按周": "W", "按月": "M"}
ROW_SELECTION_OPTIONS = ["全部行", "行号区间", "手动行号", "前N行", "预览勾选"]
MONTH_PATTERN = re.compile(r"^(\d{1,2})月$")


def init_state() -> None:
    defaults = {
        "source_name": None,
        "sheet_name": None,
        "original_df": None,
        "working_df": None,
        "selected_columns": [],
        "filter_conditions": [],
        "cleaning_history": [],
        "current_chart": None,
        "last_outlier_report": None,
        "row_selection_mode": "全部行",
        "selected_row_indices": [],
        "selected_row_text": "",
        "selected_row_start": 0,
        "selected_row_end": 99,
        "selected_row_count": 100,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_upload(uploaded_file, selected_sheet: Optional[str]) -> None:
    dataset = load_dataset(uploaded_file, selected_sheet)
    st.session_state.original_df = dataset.copy()
    st.session_state.working_df = dataset.copy()
    st.session_state.selected_columns = list(dataset.columns)
    st.session_state.filter_conditions = []
    st.session_state.cleaning_history = []
    st.session_state.current_chart = None
    st.session_state.last_outlier_report = None
    st.session_state.row_selection_mode = "全部行"
    st.session_state.selected_row_indices = []
    st.session_state.selected_row_text = ""
    st.session_state.selected_row_start = 0
    st.session_state.selected_row_end = min(len(dataset) - 1, 99)
    st.session_state.selected_row_count = min(len(dataset), 100)
    st.session_state.source_name = uploaded_file.name
    st.session_state.sheet_name = selected_sheet


def get_analysis_df() -> pd.DataFrame:
    df = st.session_state.working_df
    if df is None:
        return pd.DataFrame()

    return select_analysis_subset(
        df=df,
        selected_columns=st.session_state.selected_columns,
        row_mode=st.session_state.row_selection_mode,
        row_indices=st.session_state.selected_row_indices,
        row_start=st.session_state.selected_row_start,
        row_end=st.session_state.selected_row_end,
        row_count=st.session_state.selected_row_count,
    )


def _guess_wide_value_columns(df: pd.DataFrame, label_column: str) -> List[str]:
    ordered_columns = [column for column in df.columns if column != label_column]
    month_columns = [column for column in ordered_columns if MONTH_PATTERN.match(str(column))]
    if month_columns:
        return month_columns
    numeric_columns = [column for column in ordered_columns if pd.api.types.is_numeric_dtype(df[column])]
    return numeric_columns


def build_filter_rows(columns: List[str]) -> List[Dict[str, object]]:
    st.subheader("3. 条件筛选行")
    st.caption("支持按文本、数值、日期、空值条件筛选。筛选会作用于当前工作数据。")

    filter_count = st.number_input("筛选条件数量", min_value=1, max_value=5, value=1, step=1)
    filters = []

    for index in range(int(filter_count)):
        with st.container(border=True):
            st.markdown("条件 %s" % (index + 1))
            selected_column = st.selectbox("列名", options=columns, key="filter_column_%s" % index)
            operator = st.selectbox("操作符", options=OPERATORS, key="filter_operator_%s" % index)

            value = None
            value_to = None
            if operator in ("等于", "不等于", "包含"):
                value = st.text_input("值", key="filter_value_%s" % index)
            elif operator == "数值区间":
                col1, col2 = st.columns(2)
                value = col1.text_input("最小值", key="filter_min_%s" % index)
                value_to = col2.text_input("最大值", key="filter_max_%s" % index)
            elif operator == "日期区间":
                col1, col2 = st.columns(2)
                value = col1.date_input("开始日期", key="filter_start_%s" % index)
                value_to = col2.date_input("结束日期", key="filter_end_%s" % index)

            filters.append(
                {
                    "column": selected_column,
                    "operator": operator,
                    "value": value,
                    "value_to": value_to,
                }
            )

    return filters


def render_upload_section() -> None:
    st.header("1. 上传文件")
    uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("请先上传一个 CSV 或 XLSX 文件。")
        return

    sheet_name = None
    if uploaded_file.name.lower().endswith(".xlsx"):
        try:
            sheet_names = load_sheet_names(uploaded_file)
            sheet_name = st.selectbox("选择 Excel 工作表", options=sheet_names)
        except Exception as exc:
            st.error("读取工作表失败：%s" % exc)
            return

    if st.button("载入数据", type="primary"):
        try:
            handle_upload(uploaded_file, sheet_name)
            row_count, column_count = st.session_state.working_df.shape
            st.success("文件已载入。当前数据共有 %s 行、%s 列。" % (row_count, column_count))
        except Exception as exc:
            st.error("载入文件失败：%s" % exc)

    if st.session_state.original_df is not None:
        source_line = "当前文件：%s" % st.session_state.source_name
        if st.session_state.sheet_name:
            source_line += " | 工作表：%s" % st.session_state.sheet_name
        st.caption(source_line)


def render_subset_selector(df: pd.DataFrame) -> None:
    st.subheader("分析子集选择")
    st.caption("这里选择的是拿去分析的数据子集。你可以勾选列，再按行范围、行号或预览勾选来选取数据。")

    column_profile = build_column_profile(df, st.session_state.selected_columns)
    edited_columns = st.data_editor(
        column_profile,
        hide_index=True,
        use_container_width=True,
        disabled=["列名", "数据类型", "非空数", "缺失值", "唯一值"],
        key="column_selector_editor",
    )
    selected_columns = edited_columns.loc[edited_columns["选择"], "列名"].tolist()
    st.session_state.selected_columns = selected_columns

    row_mode = st.radio("行选择方式", ROW_SELECTION_OPTIONS, horizontal=True, key="row_selection_mode")
    total_rows = len(df)

    if row_mode == "行号区间":
        col1, col2 = st.columns(2)
        max_index = max(total_rows - 1, 0)
        st.session_state.selected_row_start = col1.number_input("起始行号", min_value=0, max_value=max_index, value=min(st.session_state.selected_row_start, max_index), step=1)
        st.session_state.selected_row_end = col2.number_input("结束行号", min_value=0, max_value=max_index, value=min(max(st.session_state.selected_row_end, st.session_state.selected_row_start), max_index), step=1)
    elif row_mode == "手动行号":
        text = st.text_input("输入行号或区间", value=st.session_state.selected_row_text, placeholder="例如：0,1,5-12,20", help="支持单个行号和区间混写，行号从 0 开始。")
        st.session_state.selected_row_text = text
        try:
            st.session_state.selected_row_indices = parse_row_indices_text(text, max(total_rows - 1, 0))
            st.caption("已识别行号：%s" % (", ".join(map(str, st.session_state.selected_row_indices[:20])) or "无"))
        except Exception as exc:
            st.error("行号解析失败：%s" % exc)
            st.session_state.selected_row_indices = []
    elif row_mode == "前N行":
        st.session_state.selected_row_count = st.number_input("选择前 N 行", min_value=1, max_value=max(total_rows, 1), value=min(max(st.session_state.selected_row_count, 1), max(total_rows, 1)), step=1)
    elif row_mode == "预览勾选":
        preview_limit = min(total_rows, 200)
        preview_columns = selected_columns[:6] if selected_columns else list(df.columns[:6])
        selector_df = df.loc[:, preview_columns].head(preview_limit).reset_index().rename(columns={"index": "原始行号"})
        selector_df.insert(0, "选择", selector_df["原始行号"].isin(st.session_state.selected_row_indices))
        edited_rows = st.data_editor(
            selector_df,
            hide_index=True,
            use_container_width=True,
            disabled=[column for column in selector_df.columns if column != "选择"],
            key="row_selector_editor",
        )
        st.session_state.selected_row_indices = edited_rows.loc[edited_rows["选择"], "原始行号"].astype(int).tolist()
        st.caption("预览勾选模式当前最多展示前 200 行。")

    analysis_df = get_analysis_df()
    st.markdown("**当前分析子集预览**")
    if analysis_df.empty:
        st.warning("当前分析子集为空。请至少选择一列，并确认行选择结果不是空集。")
        return

    metric1, metric2 = st.columns(2)
    metric1.metric("分析子集行数", len(analysis_df))
    metric2.metric("分析子集列数", len(analysis_df.columns))
    preview_df = analysis_df.reset_index().rename(columns={"index": "原始行号"})
    st.dataframe(preview_df.head(200), use_container_width=True)


def render_preview_and_selection() -> None:
    df = st.session_state.working_df
    if df is None:
        return

    st.header("2. 数据预览与选择")
    col1, col2, col3 = st.columns(3)
    col1.metric("当前行数", len(df))
    col2.metric("当前列数", len(df.columns))
    col3.metric("缺失值总数", int(df.isna().sum().sum()))

    st.dataframe(df.head(200).reset_index().rename(columns={"index": "原始行号"}), use_container_width=True)
    render_subset_selector(df)

    filters = build_filter_rows(list(df.columns))
    col_apply, col_reset = st.columns(2)
    if col_apply.button("应用筛选", use_container_width=True):
        try:
            conditions = [FilterCondition.from_dict(item) for item in filters]
            filtered_df = apply_filters(st.session_state.working_df, conditions)
            st.session_state.working_df = filtered_df
            st.session_state.filter_conditions = filters
            st.session_state.selected_row_end = min(len(filtered_df) - 1, st.session_state.selected_row_end)
            st.session_state.selected_row_count = min(max(len(filtered_df), 1), st.session_state.selected_row_count)
            st.success("筛选完成，当前剩余 %s 行。" % len(filtered_df))
        except Exception as exc:
            st.error("筛选失败：%s" % exc)
    if col_reset.button("恢复到原始数据", use_container_width=True):
        st.session_state.working_df = st.session_state.original_df.copy()
        st.session_state.selected_columns = list(st.session_state.original_df.columns)
        st.session_state.cleaning_history = []
        st.session_state.last_outlier_report = None
        st.session_state.selected_row_indices = []
        st.session_state.row_selection_mode = "全部行"
        st.success("已恢复到原始数据。")


def render_cleaning_section() -> None:
    df = st.session_state.working_df
    if df is None:
        return

    st.header("4. 数据处理与清洗")
    action = st.selectbox("选择处理操作", options=CLEANING_ACTIONS)
    selected_columns = st.multiselect("目标列", options=list(df.columns), default=st.session_state.selected_columns[:1] if st.session_state.selected_columns else [], key="cleaning_columns")

    params = {}
    if action == "缺失值处理":
        params["strategy"] = st.selectbox("处理方式", ["删除含缺失值的行", "按均值填充", "按中位数填充", "按众数填充", "自定义填充值"])
        if params["strategy"] == "自定义填充值":
            params["fill_value"] = st.text_input("填充值")
    elif action == "删除重复值":
        params["keep"] = st.selectbox("保留哪条重复记录", ["first", "last"])
    elif action == "类型转换":
        params["dtype"] = st.selectbox("目标类型", ["string", "int", "float", "datetime"])
    elif action == "异常值处理":
        params["method"] = st.selectbox("检测方式", ["IQR", "Z-score"])
        params["mode"] = st.selectbox("处理方式", ["标记异常值", "删除异常值"])
        if params["method"] == "Z-score":
            params["threshold"] = st.number_input("Z-score 阈值", min_value=1.0, value=3.0, step=0.5)
    elif action == "重命名列":
        if selected_columns:
            params["new_name"] = st.text_input("新列名", value="%s_new" % selected_columns[0])
    elif action == "删除列":
        params["confirm"] = st.checkbox("确认删除所选列")
    elif action == "排序数据":
        params["ascending"] = st.radio("排序方向", [True, False], format_func=lambda item: "升序" if item else "降序", horizontal=True)
    elif action == "数值缩放":
        params["mode"] = st.selectbox("缩放方式", ["Min-Max归一化", "Z-score标准化"])

    if st.button("执行处理", type="primary"):
        try:
            step = build_cleaning_step(action=action, target_columns=selected_columns, params=params)
            result = apply_cleaning_step(st.session_state.working_df, step)
            st.session_state.working_df = result.dataframe
            st.session_state.cleaning_history.append(step.to_dict())
            st.session_state.last_outlier_report = result.outlier_report
            st.success(result.message)
        except Exception as exc:
            st.error("处理失败：%s" % exc)

    if st.session_state.cleaning_history:
        st.caption("已执行处理步骤：%s" % len(st.session_state.cleaning_history))
        st.json(st.session_state.cleaning_history)

    if st.session_state.last_outlier_report is not None and not st.session_state.last_outlier_report.empty:
        st.subheader("最近一次异常值处理结果")
        st.dataframe(st.session_state.last_outlier_report, use_container_width=True)


def render_overview_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    basic_summary = compute_basic_summary(df)
    missing_summary = compute_missing_summary(df)
    numeric_summary = compute_numeric_summary(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("分析列数", basic_summary["column_count"])
    col2.metric("数值列数", basic_summary["numeric_column_count"])
    col3.metric("类别列数", basic_summary["categorical_column_count"])

    st.subheader("数据概览")
    st.dataframe(pd.DataFrame([basic_summary]), use_container_width=True)

    st.subheader("缺失值统计")
    st.dataframe(missing_summary, use_container_width=True)

    st.subheader("描述性统计")
    if numeric_columns and not numeric_summary.empty:
        st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.info("当前未选择数值列，无法生成描述性统计。")


def render_chart_workbench_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    st.subheader("图表工坊")
    chart_type = st.selectbox("图表类型", CHART_TYPES, key="workbench_chart_type")
    x_axis = st.selectbox("X 轴", options=list(df.columns), key="workbench_x")
    y_axis = st.selectbox("Y 轴", options=[""] + numeric_columns, key="workbench_y")
    color_column = st.selectbox("颜色或分组列（可选）", options=[""] + list(df.columns), key="workbench_color")
    top_n = st.slider("类别图最多显示前 N 个分类", min_value=5, max_value=30, value=10, key="workbench_topn")
    hist_bins = st.slider("直方图分箱数", min_value=5, max_value=60, value=20, key="workbench_bins")

    figure = build_chart(df=df, chart_type=chart_type, x_axis=x_axis, y_axis=y_axis or None, color=color_column or None, top_n=top_n, hist_bins=hist_bins)
    if figure is None:
        st.warning("当前图表参数不完整，无法生成图表。")
    else:
        st.plotly_chart(figure, use_container_width=True)
        st.session_state.current_chart = figure


def render_group_analysis_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    st.subheader("聚合分析")
    group_column = st.selectbox("主维度", options=list(df.columns), key="group_main")
    split_column = st.selectbox("拆分维度（可选）", options=[""] + list(df.columns), key="group_split")
    agg_label = st.selectbox("统计方式", options=list(AGGREGATION_LABELS.keys()), key="group_agg")
    value_options = [""] + numeric_columns if agg_label != "计数" else [""] + list(df.columns)
    value_column = st.selectbox("指标列", options=value_options, key="group_value")
    top_n = st.slider("保留前 N 个主维度", min_value=3, max_value=20, value=10, key="group_topn")

    summary = compute_grouped_summary(df, [group_column] + ([split_column] if split_column else []), value_column=value_column or None, agg_func=AGGREGATION_LABELS[agg_label], top_n=top_n)
    if summary.empty:
        st.info("当前聚合条件没有生成结果。")
        return

    st.dataframe(summary, use_container_width=True)
    chart_kind = st.selectbox("聚合结果图表", ["柱状图", "堆叠柱状图", "折线图", "面积图", "饼图"], key="group_chart")
    figure = build_chart(df=summary, chart_type=chart_kind, x_axis=group_column, y_axis="value", color=split_column or None, top_n=top_n)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
        st.session_state.current_chart = figure


def render_wide_table_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    st.subheader("宽表分析")
    st.caption("适合“科目名称在行上、月份在列上”的表。选择标签列和多个数值列后，系统会自动转成长表做多线趋势、多科目对比和热力图。")

    label_column = st.selectbox("标签列", options=list(df.columns), key="wide_label")
    default_value_columns = _guess_wide_value_columns(df, label_column)
    value_columns = st.multiselect(
        "多列值字段",
        options=[column for column in df.columns if column != label_column],
        default=default_value_columns,
        key="wide_values",
    )
    if not value_columns:
        st.info("请至少选择两个要合并分析的列，例如 1月 到 12月。")
        return

    label_options = df[label_column].dropna().astype(str).unique().tolist()
    default_labels = label_options[: min(5, len(label_options))]
    selected_labels = st.multiselect("选中要对比的行标签", options=label_options, default=default_labels, key="wide_labels")
    chart_kind = st.selectbox("宽表图表", ["多条线趋势图", "分组柱状图", "堆叠柱状图", "面积图", "对比热力图"], key="wide_chart_kind")

    matrix_df = build_wide_comparison_matrix(df, label_column, value_columns, selected_labels or None)
    if matrix_df.empty:
        st.info("当前宽表条件没有生成结果。")
        return

    st.markdown("**宽表矩阵预览**")
    st.dataframe(matrix_df, use_container_width=True)

    long_df = reshape_wide_dataframe(df, label_column, value_columns, selected_labels or None, dimension_name="分析维度", value_name="分析值")
    if long_df.empty:
        st.info("当前宽表数据无法转换为可分析结果。")
        return

    st.markdown("**转换后的长表预览**")
    st.dataframe(long_df, use_container_width=True)

    figure = None
    if chart_kind == "对比热力图":
        heatmap_df = matrix_df.set_index(label_column)
        figure = build_chart(heatmap_df, "热力图", None, None, None, matrix_df=heatmap_df)
    elif chart_kind == "多条线趋势图":
        figure = build_chart(long_df, "折线图", "分析维度", "分析值", label_column)
    elif chart_kind == "分组柱状图":
        figure = build_chart(long_df, "柱状图", "分析维度", "分析值", label_column)
    elif chart_kind == "堆叠柱状图":
        figure = build_chart(long_df, "堆叠柱状图", "分析维度", "分析值", label_column)
    elif chart_kind == "面积图":
        figure = build_chart(long_df, "面积图", "分析维度", "分析值", label_column)

    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
        st.session_state.current_chart = figure


def render_pivot_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    st.subheader("透视分析")
    if not numeric_columns:
        st.info("当前没有数值列，无法生成透视分析。")
        return

    index_column = st.selectbox("行维度", options=list(df.columns), key="pivot_index")
    column_column = st.selectbox("列维度", options=list(df.columns), key="pivot_column")
    value_column = st.selectbox("值列", options=numeric_columns, key="pivot_value")
    agg_label = st.selectbox("透视统计方式", options=list(AGGREGATION_LABELS.keys()), key="pivot_agg")
    fill_value = st.number_input("空值填充值", value=0.0, key="pivot_fill")

    if len({index_column, column_column, value_column}) < 3:
        st.warning("透视分析中，行维度、列维度和值列需要选择三个不同的列。")
        return

    pivot_df = compute_pivot_table(df, index_column=index_column, column_column=column_column, value_column=value_column, agg_func=AGGREGATION_LABELS[agg_label], fill_value=fill_value)
    if pivot_df.empty:
        st.info("当前透视条件没有生成结果。")
        return

    st.dataframe(pivot_df, use_container_width=True)
    heatmap_df = pivot_df.set_index(index_column)
    heatmap = build_chart(heatmap_df, "热力图", None, None, None, matrix_df=heatmap_df)
    if heatmap is not None:
        st.plotly_chart(heatmap, use_container_width=True)
        st.session_state.current_chart = heatmap


def render_time_series_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    st.subheader("时间趋势")
    date_column = st.selectbox("日期列", options=list(df.columns), key="time_date")
    agg_label = st.selectbox("时间聚合方式", options=list(AGGREGATION_LABELS.keys()), key="time_agg")
    value_options = [""] + numeric_columns if agg_label != "计数" else [""] + list(df.columns)
    value_column = st.selectbox("指标列", options=value_options, key="time_value")
    frequency_label = st.selectbox("时间粒度", options=list(FREQUENCY_LABELS.keys()), key="time_freq")
    split_column = st.selectbox("拆分维度（可选）", options=[""] + list(df.columns), key="time_split")

    summary = compute_time_series_summary(df, date_column=date_column, value_column=value_column or None, agg_func=AGGREGATION_LABELS[agg_label], frequency=FREQUENCY_LABELS[frequency_label], group_column=split_column or None)
    if summary.empty:
        st.info("当前时间趋势条件没有生成结果，请确认日期列格式可识别。")
        return

    st.dataframe(summary, use_container_width=True)
    figure = build_chart(df=summary, chart_type="折线图", x_axis="period", y_axis="value", color=split_column or None)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
        st.session_state.current_chart = figure


def render_text_analysis_tab(df: pd.DataFrame) -> None:
    st.subheader("文本分析与词云")
    text_columns = list(df.select_dtypes(exclude="number").columns)
    if not text_columns:
        st.info("当前没有可用于文本分析的非数值列。")
        return

    text_column = st.selectbox("文本列", options=text_columns, key="text_column")
    top_n = st.slider("高频词数量", min_value=10, max_value=50, value=20, key="text_topn")
    bg_color = st.selectbox("词云背景色", options=["white", "black"], key="wordcloud_bg")

    frequency_df = compute_text_frequency(df[text_column], top_n=top_n)
    if frequency_df.empty:
        st.info("当前文本列没有可提取的高频词。")
        return

    col_table, col_chart = st.columns(2)
    col_table.dataframe(frequency_df, use_container_width=True)
    bar_figure = build_chart(frequency_df, "柱状图", "token", "count", None, top_n=top_n)
    if bar_figure is not None:
        col_chart.plotly_chart(bar_figure, use_container_width=True)
        st.session_state.current_chart = bar_figure

    if has_wordcloud_support():
        image = generate_wordcloud_image(df[text_column], max_words=top_n * 2, background_color=bg_color)
        if image is not None:
            st.image(image, caption="文本词云")
        else:
            st.info("词云暂时没有可生成的数据。")
    else:
        st.warning("当前环境未安装 wordcloud，已展示高频词统计。如需词云，请安装 wordcloud 包。")


def render_outlier_tab(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    st.subheader("异常值检测")
    if not numeric_columns:
        st.info("当前没有数值列，无法进行异常值检测。")
        return

    outlier_column = st.selectbox("选择异常值检测列", options=numeric_columns, key="outlier_column")
    outlier_method = st.selectbox("异常值算法", options=["IQR", "Z-score"], key="outlier_method")
    z_threshold = 3.0
    if outlier_method == "Z-score":
        z_threshold = st.number_input("检测阈值", min_value=1.0, value=3.0, step=0.5, key="analysis_z")
    outlier_report = detect_outliers(df, outlier_column, outlier_method, z_threshold)
    st.dataframe(outlier_report, use_container_width=True)


def render_analysis_section() -> None:
    df = get_analysis_df()
    if df.empty:
        st.header("5. 可视化与分析")
        st.warning("当前没有可分析的数据子集，请先在上方完成列选择和行选择。")
        return

    numeric_columns = list(df.select_dtypes(include="number").columns)

    st.header("5. 可视化与分析")
    st.caption("以下分析全部基于你在“数据预览与选择”里选定的当前数据子集。宽表分析专门处理“月份在列上、科目在行上”的表。")
    tabs = st.tabs(["数据概览", "图表工坊", "聚合分析", "宽表分析", "透视分析", "时间趋势", "文本分析", "异常值"])
    with tabs[0]:
        render_overview_tab(df, numeric_columns)
    with tabs[1]:
        render_chart_workbench_tab(df, numeric_columns)
    with tabs[2]:
        render_group_analysis_tab(df, numeric_columns)
    with tabs[3]:
        render_wide_table_tab(df, numeric_columns)
    with tabs[4]:
        render_pivot_tab(df, numeric_columns)
    with tabs[5]:
        render_time_series_tab(df, numeric_columns)
    with tabs[6]:
        render_text_analysis_tab(df)
    with tabs[7]:
        render_outlier_tab(df, numeric_columns)


def render_export_section() -> None:
    df = st.session_state.working_df
    analysis_df = get_analysis_df()
    if df is None:
        return

    st.header("6. 导出结果")
    csv_bytes = dataframe_to_csv_bytes(df)
    xlsx_bytes = dataframe_to_excel_bytes(df)

    col1, col2 = st.columns(2)
    col1.download_button("下载当前处理后数据 CSV", data=csv_bytes, file_name="cleaned_data.csv", mime="text/csv", use_container_width=True)
    col2.download_button("下载当前处理后数据 Excel", data=xlsx_bytes, file_name="cleaned_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    if not analysis_df.empty:
        subset_csv_bytes = dataframe_to_csv_bytes(analysis_df)
        subset_xlsx_bytes = dataframe_to_excel_bytes(analysis_df)
        report_bytes = build_analysis_report_html_bytes(df=analysis_df, source_name=st.session_state.source_name, selected_columns=list(analysis_df.columns), cleaning_history=st.session_state.cleaning_history, filter_conditions=st.session_state.filter_conditions, current_chart=st.session_state.current_chart)

        col3, col4, col5 = st.columns(3)
        col3.download_button("下载当前分析子集 CSV", data=subset_csv_bytes, file_name="analysis_subset.csv", mime="text/csv", use_container_width=True)
        col4.download_button("下载当前分析子集 Excel", data=subset_xlsx_bytes, file_name="analysis_subset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        col5.download_button("下载分析报告 HTML", data=report_bytes, file_name="analysis_report.html", mime="text/html", use_container_width=True)

    if st.session_state.current_chart is not None:
        chart_html = chart_to_html_bytes(st.session_state.current_chart)
        png_bytes = chart_to_png_bytes(st.session_state.current_chart)
        col_html, col_png = st.columns(2)
        col_html.download_button("下载当前图表 HTML", data=chart_html, file_name="chart.html", mime="text/html", use_container_width=True)
        if png_bytes is not None:
            col_png.download_button("下载当前图表 PNG", data=png_bytes, file_name="chart.png", mime="image/png", use_container_width=True)
        else:
            col_png.info("安装 kaleido 后可导出 PNG 图表。")
    else:
        st.info("生成图表后即可导出当前图表。")


def main() -> None:
    init_state()
    st.title("本地数据分析与清洗平台")
    st.write("上传 CSV 或 Excel 文件，选择列与数据子集，并完成清洗、宽表趋势分析、聚合分析、时间趋势分析、词云分析和导出。")

    render_upload_section()
    render_preview_and_selection()
    render_cleaning_section()
    render_analysis_section()
    render_export_section()


if __name__ == "__main__":
    main()