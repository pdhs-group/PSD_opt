# -*- coding: utf-8 -*-
"""
功能1：批量读取一个或多个结果文件夹中的 pure_CB_result_*.npz
- 按迭代数排序
- 打印 opt_score + opt_params(剔除 __exp_paths)
- 导出每个文件夹一个美观格式的 Excel

依赖：
  pip install openpyxl numpy
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class ResultRecord:
    iteration: int
    opt_score: Optional[float]
    opt_params: Dict[str, Any]
    npz_path: str


# -----------------------------
# 文件扫描与排序
# -----------------------------
def parse_iteration_from_name(filename: str, prefix: str) -> Optional[int]:
    """
    从文件名中提取迭代数：
      e.g. prefix='pure_CB_result_' -> pure_CB_result_400.npz => 400
    """
    # 允许前缀相同，后面是数字，最后 .npz
    pattern = rf"^{re.escape(prefix)}(\d+)\.npz$"
    m = re.match(pattern, filename)
    if not m:
        return None
    return int(m.group(1))


def find_result_files(folder: Path, prefix: str = "pure_CB_result_") -> List[Tuple[int, Path]]:
    """
    返回 (iteration, file_path) 列表，并按 iteration 升序排序
    """
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    pairs: List[Tuple[int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        it = parse_iteration_from_name(p.name, prefix=prefix)
        if it is None:
            continue
        pairs.append((it, p))

    pairs.sort(key=lambda x: x[0])
    return pairs


# -----------------------------
# 读取 npz
# -----------------------------
def load_single_result(npz_path: Path, 
                       drop_param_keys: Tuple[str, ...] = ("__exp_paths", "actor_wait", "max_reuse", "wait_time", "__known_params", )
                       ) -> ResultRecord:
    """
    读取单个 npz，返回 ResultRecord
    约定：npz 内 key 'results' 是 dict (pickle)，其中包含 opt_score/opt_params/file_path 等
    """
    with np.load(npz_path, allow_pickle=True) as data:
        if "results" not in data:
            raise KeyError(f"'results' not found in {npz_path}")
        results = data["results"].item()

    opt_score = results.get("opt_score", None)
    opt_params = results.get("opt_params", {}) or {}

    # 过滤掉不想展示的字段
    opt_params_clean = {k: v for k, v in opt_params.items() if k not in drop_param_keys}

    # 迭代数从文件名拿（更稳）
    iteration = parse_iteration_from_name(npz_path.name, prefix="pure_CB_result_")
    if iteration is None:
        # 兜底：从 results['file_path'] 或者给 -1
        iteration = -1

    return ResultRecord(
        iteration=int(iteration),
        opt_score=float(opt_score) if opt_score is not None else None,
        opt_params=opt_params_clean,
        npz_path=str(npz_path),
    )


def load_folder_results(folder: Path, prefix: str = "pure_CB_result_") -> List[ResultRecord]:
    """
    扫描并读取一个文件夹内所有结果文件
    """
    files = find_result_files(folder, prefix=prefix)
    records: List[ResultRecord] = []
    for it, fp in files:
        rec = load_single_result(fp)
        # 用扫描得到的 it 覆盖更可靠
        rec.iteration = it
        records.append(rec)
    return records


# -----------------------------
# 终端美观打印
# -----------------------------
def _stringify_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        # 参数 float 用紧凑格式
        return f"{v:.6g}"
    return str(v)


def print_records(records: List[ResultRecord], title: str = "") -> None:
    """
    控制台表格打印：iteration | opt_score | opt_params...
    为了美观：用动态列宽 + 分隔线
    """
    if title:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))

    if not records:
        print("(no result files found)")
        return

    # 收集所有参数键（不同迭代可能不同）
    all_param_keys = sorted({k for r in records for k in r.opt_params.keys()})

    headers = ["iter", "opt_score"] + all_param_keys
    rows: List[List[str]] = []

    for r in records:
        row = [str(r.iteration), "" if r.opt_score is None else f"{r.opt_score:.6g}"]
        for k in all_param_keys:
            row.append(_stringify_value(r.opt_params.get(k, "")))
        rows.append(row)

    # 计算列宽
    col_widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    def fmt_row(row_: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row_))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


# -----------------------------
# Excel 导出（美观格式）
# -----------------------------
def export_records_to_excel(
    records: List[ResultRecord],
    xlsx_path: Path,
    sheet_name: str = "Results",
) -> None:
    """
    每个文件夹导出一个 xlsx，字段扁平化：
      iter | opt_score | <all opt_params keys...> | npz_path
    """
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not records:
        ws["A1"] = "No results"
        wb.save(xlsx_path)
        return

    all_param_keys = sorted({k for r in records for k in r.opt_params.keys()})
    headers = ["iter", "opt_score"] + all_param_keys + ["npz_path"]

    # 写表头
    ws.append(headers)

    # 表头样式
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4F81BD")
    for col_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # 写数据行
    for r in records:
        row = [r.iteration, r.opt_score]
        for k in all_param_keys:
            v = r.opt_params.get(k, None)
            row.append(v)
        row.append(r.npz_path)
        ws.append(row)

    # 冻结首行
    ws.freeze_panes = "A2"

    # 自动筛选
    ws.auto_filter.ref = ws.dimensions

    # 数字格式：opt_score 科学计数
    opt_score_col = 2
    for row_idx in range(2, ws.max_row + 1):
        c = ws.cell(row=row_idx, column=opt_score_col)
        if isinstance(c.value, (int, float)) and c.value is not None:
            c.number_format = "0.000000E+00"

    # 对齐与自动换行（路径列左对齐）
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            if cell.column == ws.max_column:  # npz_path
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # 列宽自适应（有上限，避免超宽）
    max_width_cap = 42
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for row_idx in range(1, ws.max_row + 1):
            v = ws.cell(row=row_idx, column=col_idx).value
            s = "" if v is None else str(v)
            max_len = max(max_len, len(s))
        # 给一点 padding
        width = min(max_width_cap, max(10, max_len + 2))
        # 路径列更宽一点
        if col_idx == ws.max_column:
            width = min(80, max(30, max_len + 2))
        ws.column_dimensions[col_letter].width = width

    # 条件格式：opt_score 用色阶（可选，美观）
    # 仅当有至少2行数据才加
    if ws.max_row >= 3:
        score_range = f"{get_column_letter(opt_score_col)}2:{get_column_letter(opt_score_col)}{ws.max_row}"
        rule = ColorScaleRule(
            start_type="min", start_color="63BE7B",   # 绿
            mid_type="percentile", mid_value=50, mid_color="FFEB84",  # 黄
            end_type="max", end_color="F8696B",       # 红
        )
        ws.conditional_formatting.add(score_range, rule)

    wb.save(xlsx_path)


# -----------------------------
# 批处理入口：一个或多个文件夹
# -----------------------------
def process_folders(
    folder_list: List[str],
    prefix: str = "pure_CB_result_",
    output_dir: Optional[str] = None,
) -> None:
    """
    对 folder_list 中每个文件夹：
      - 读取结果
      - 打印
      - 导出 Excel
    """
    out_base = Path(output_dir).resolve() if output_dir else None

    for folder in folder_list:
        base_dir = Path(__file__).resolve().parent / data_group
        folder_path = (base_dir / folder).resolve()
        records = load_folder_results(folder_path, prefix=prefix)

        # 终端打印
        print_records(records, title=f"[Folder] {folder_path}")

        # 输出 excel 路径
        if out_base is None:
            # 默认放在该文件夹下
            xlsx_path = folder_path / f"{folder_path.name}_summary.xlsx"
        else:
            out_base.mkdir(parents=True, exist_ok=True)
            xlsx_path = out_base / f"{folder_path.name}_summary.xlsx"

        export_records_to_excel(records, xlsx_path=xlsx_path)
        print(f"\nExcel saved: {xlsx_path}\n")


# -----------------------------
# main：手动填写文件夹（支持多个）
# -----------------------------
if __name__ == "__main__":
    data_group = "data_mcpbe_CB_S1"
    # ✅ 在这里手动输入一个或多个结果文件夹
    RESULT_FOLDERS = [
        # "opt_results_N2000_Q0",
        "opt_results_N2000_Q0_N",
        # "opt_results_N2000_Q0_S1",
        # "opt_results_N2000_Q0_cut",
        # "opt_results_N2000_Q3",
        # "opt_results_N5000_Q0",
        # "opt_results_N5000_Q3",
        "opt_results_N10000_Q0",
        # "opt_results_N10000_Q0_S1",
        "opt_results_N10000_Q3",
        # "opt_results_N10000_Q3_S1",
        "opt_results_N10000_Q3_cut",
        # "opt_results_N10000_Q3_cut_S1",
        # "opt_results_N10000_Q3_func",
        "opt_results_N10000_Q3_min_max",
        # "opt_results_N10000_Q3_min_max_S1",
        # "opt_results_N10000_Q0_mix",
        # "opt_results_N10000_Q3_x50",
        # "opt_results_N10000_Q3",
        # "opt_results_N20000_Q0",
        # "opt_results_N20000_Q3",
        "opt_results_N20000_Q3_x50",
        # "opt_results_N20000_Q3_x50_S1",
        # "opt_results_N50000_Q0",
        # "opt_results_N50000_Q3",
        "opt_results_N50000_Q3_cut",
        # "opt_results_N50000_Q3_cut_S1",
        # "opt_results_N100000_Q0",
        # "opt_results_N100000_Q3",
        # "opt_results_N200000_Q0",
        # "opt_results_N200000_Q3",
        # "opt_results",
    ]

    # 可选：把所有 excel 统一输出到某个目录；不填则每个文件夹各自生成
    OUTPUT_DIR = None  # e.g. r"./postprocess_excels"

    if not RESULT_FOLDERS:
        raise SystemExit("Please set RESULT_FOLDERS in the script before running.")

    process_folders(RESULT_FOLDERS, prefix="pure_CB_result_", output_dir=OUTPUT_DIR)
