# 本地数据分析与清洗平台

一个基于 Python 和 Streamlit 的本地单机 Web 应用，用于上传 CSV/Excel 文件，进行列选择、条件筛选、数据处理、可视化分析和结果导出。

## 功能

- 上传 `CSV` 与 `XLSX` 文件，Excel 支持选择单个工作表
- 选择指定列参与后续分析
- 按文本、数值区间、日期区间、空值条件筛选指定行
- 执行缺失值处理、删除重复值、类型转换、异常值处理、重命名列、删除列、排序、数值缩放
- 查看数据概览、缺失统计、描述性统计、更多图表类型
- 执行聚合分析、透视分析、时间趋势分析、文本高频词和词云分析
- 下载清洗后的 CSV/Excel 文件，以及当前图表的 HTML 或 PNG 文件

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Windows 也可以直接运行：

```bash
start_app.bat
```

## 正确启动方式

这是一个 Streamlit 应用，不能使用 `python app.py` 直接启动。

错误示例：

```bash
python app.py
```

正确示例：

```bash
streamlit run app.py
```

如果你在虚拟环境中，推荐使用：

```bash
.\.venv\Scripts\streamlit run app.py
```

启动成功后，在浏览器打开：

```text
http://localhost:8501
```

## 新增分析能力

- 图表工坊：柱状图、堆叠柱状图、折线图、面积图、散点图、箱线图、小提琴图、直方图、饼图、热力图
- 聚合分析：按维度统计计数、求和、均值、中位数、最小值、最大值
- 透视分析：按行维度和列维度生成透视表并展示热力图
- 时间趋势：按天、周、月聚合数值变化趋势
- 文本分析：高频词统计和词云图

## 目录结构

```text
app.py
data_app/
  services/
  utils/
tests/
requirements.txt
start_app.bat
```

## 测试

```bash
pytest
```
