from io import BytesIO
from typing import BinaryIO, List, Optional

import pandas as pd


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "gbk"):
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError("CSV 文件编码无法识别，请确认文件为 UTF-8 或 GBK。") from last_error


def load_sheet_names(uploaded_file: BinaryIO) -> List[str]:
    file_bytes = uploaded_file.getvalue()
    excel_file = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    return list(excel_file.sheet_names)


def load_dataset(uploaded_file: BinaryIO, sheet_name: Optional[str] = None) -> pd.DataFrame:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return _read_csv(file_bytes)
    if file_name.endswith(".xlsx"):
        target_sheet = sheet_name or 0
        return pd.read_excel(BytesIO(file_bytes), sheet_name=target_sheet, engine="openpyxl")
    raise ValueError("仅支持 CSV 和 XLSX 文件。")
