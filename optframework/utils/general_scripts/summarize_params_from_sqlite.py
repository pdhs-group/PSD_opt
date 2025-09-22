# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:48:23 2025

@author: Haoran Ji (px2030@kit.edu)

离线读取 Ray Tune 过程中写入的 warm_params SQLite 数据库，
按每个 data_name（即你的文件名）在“前 N 步”的范围内统计最佳结果，
并输出与原代码 result_dict 结构一致的字典列表到 N.npz。

两种用法：
1) 命令行：
    python summarize_warm_params.py \
        --db /path/to/1600.sqlite \
        --steps 50 100 200 400 800 1600 \
        --outdir ./summaries \
        --filter nameA nameB

2) Debug 交互模式（无命令行参数时自动触发，适合 Spyder/Jupyter）：
    运行脚本后按提示输入参数。
"""

import argparse
import json
import os
import sqlite3
import sys
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# 核心功能
# -----------------------------
def load_all_records(db_path: str) -> Dict[str, List[Tuple[dict, float]]]:
    """从 SQLite 读取所有行，按 data_name 分组，并按插入顺序排序。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite 文件不存在：{db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT data_name, param_json, score, id
        FROM warm_params
        ORDER BY data_name ASC, id ASC
    """)
    rows = cursor.fetchall()
    conn.close()

    grouped: Dict[str, List[Tuple[dict, float]]] = defaultdict(list)
    for data_name, param_json, score, _id in rows:
        try:
            params = json.loads(param_json) if param_json is not None else {}
        except json.JSONDecodeError:
            params = {}
        grouped[data_name].append((params, float(score) if score is not None else float("inf")))
    return dict(grouped)


def best_so_far_prefix(records: List[Tuple[dict, float]], n: int) -> Tuple[dict, float]:
    """在前 n 条记录中找到 score 最小的 (params, score)。"""
    if not records:
        return {}, float("inf")

    end = min(n, len(records))
    best_idx = 0
    best_score = records[0][1]
    for i in range(1, end):
        sc = records[i][1]
        if sc < best_score:
            best_idx, best_score = i, sc
    return records[best_idx][0], best_score


def build_result_dict(opt_params: dict, opt_score: float, data_name: str) -> dict:
    """构造与原程序一致的 result_dict，file_path 使用 data_name。"""
    return {
        "opt_score": opt_score,
        "opt_params": opt_params,
        "file_path": data_name,
    }


def run_summarize(db_path: str,
                  steps: List[int],
                  outdir: str,
                  filters: Optional[List[str]],
                  prefix: str = "") -> None:
    """
    执行统计并保存 N.npz。
    若提供 prefix，则输出文件名为 {prefix}_{N}.npz，否则为 {N}.npz。
    """
    steps = sorted(set(int(s) for s in steps if int(s) > 0))
    os.makedirs(outdir, exist_ok=True)

    grouped = load_all_records(db_path)
    if not grouped:
        print(f"[WARN] 数据库中没有任何记录：{db_path}")
        return

    data_names = sorted(grouped.keys())
    if filters:
        filt = set(filters)
        missing = [name for name in filt if name not in grouped]
        if missing:
            print(f"[WARN] 下列 data_name 在数据库中不存在，将被忽略：{missing}")
        data_names = [name for name in data_names if name in filt]

    if not data_names:
        print("[WARN] 没有匹配到任何 data_name。")
        return

    print(f"[INFO] 从 {db_path} 读取到 {len(data_names)} 个 data_name。")
    print(f"[INFO] 将统计步数：{steps}")
    if prefix:
        print(f"[INFO] 输出文件前缀：{prefix}_")

    for n in steps:
        results_for_n: List[dict] = []

        for name in data_names:
            recs = grouped[name]
            if not recs:
                rd = build_result_dict({}, float("inf"), name)
            else:
                params, score = best_so_far_prefix(recs, n)
                rd = build_result_dict(params, score, name)

            results_for_n.append(rd)

        fname = f"{prefix}_{n}.npz" if prefix else f"{n}.npz"
        save_path = os.path.join(outdir, fname)
        np.savez_compressed(save_path, results=np.array(results_for_n, dtype=object))
        print(f"[OK] 已保存：{save_path}（包含 {len(results_for_n)} 个 data_name 的前 {n} 步最佳结果）")

    print("[DONE] 全部步数统计完成。")
    print("读取示例：")
    print("  import numpy as np")
    print("  npz = np.load('N.npz', allow_pickle=True'); results = npz['results'].tolist()")
    print("  # results 是 list[dict]，每个 dict 的结构与原始 result_dict 一致。")


def run_summarize_for_root(results_root: str,
                           steps: List[int],
                           outdir: str,
                           db_name: str = "1600.sqlite",
                           filters: Optional[List[str]] = None) -> None:
    """
    遍历 results_root 下的**一级子目录**，查找形如 {subdir}/{db_name} 的 sqlite，
    对每个子目录执行汇总，并将子目录名作为输出文件前缀：
        {subdir}_{N}.npz
    """
    if not os.path.isdir(results_root):
        raise NotADirectoryError(f"不是有效目录：{results_root}")

    os.makedirs(outdir, exist_ok=True)

    subdirs = [d for d in os.listdir(results_root)
               if os.path.isdir(os.path.join(results_root, d))]
    # subdirs = [os.path.join(results_root, "KL")]

    if not subdirs:
        print(f"[WARN] 根目录下没有子目录：{results_root}")
        return

    found_any = False
    for sub in sorted(subdirs):
        db_path = os.path.join(results_root, sub, db_name)
        if not os.path.exists(db_path):
            print(f"[SKIP] 子目录 {sub} 未发现 {db_name}")
            continue

        found_any = True
        print(f"\n====== 处理子目录：{sub} | DB: {db_path} ======")
        # 输出到共享 outdir，文件名加前缀
        run_summarize(db_path=db_path,
                      steps=steps,
                      outdir=outdir,
                      filters=filters,
                      prefix=sub)

    if not found_any:
        print(f"[WARN] 未在任何子目录中找到 {db_name}")


# -----------------------------
# 命令行主函数（支持单库或遍历根目录）
# -----------------------------
def cli_main():
    parser = argparse.ArgumentParser(
        description="从 warm_params sqlite 统计前 N 步最佳结果，支持单库或遍历根目录（输出 {prefix}_{N}.npz）。"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--db", help="单个 SQLite 文件路径，例如 /path/to/1600.sqlite")
    mode.add_argument("--root", help="根目录，遍历其一级子目录，查找 {sub}/{db_name}")

    parser.add_argument("--db-name", default="1600.sqlite",
                        help="在 --root 模式下要查找的 sqlite 文件名（默认：1600.sqlite）")
    parser.add_argument("--steps", nargs="+", type=int, required=True,
                        help="需要统计的步数列表，例如：50 100 200 400 800 1600")
    parser.add_argument("--outdir", default=".",
                        help="输出目录（默认当前目录）")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="只统计指定的 data_name，留空则统计全部")

    args = parser.parse_args()

    if args.root:
        run_summarize_for_root(results_root=args.root,
                               steps=args.steps,
                               outdir=args.outdir,
                               db_name=args.db_name,
                               filters=args.filter)
    else:
        # 单库模式不带前缀
        run_summarize(db_path=args.db,
                      steps=args.steps,
                      outdir=args.outdir,
                      filters=args.filter,
                      prefix="")


# -----------------------------
# Debug 交互主函数（适合 Spyder/Jupyter）
# -----------------------------
def debug_main():
    print("=== Debug 交互模式 ===")
    print("将遍历指定根目录的一级子目录，查找 1600.sqlite 并输出 {子目录名}_{N}.npz")

    # 你自己的默认路径
    results_root = results_path  # 根目录，包含 MSE、MAE 等子目录
    db_name = "3200.sqlite"
    steps_str = "50,100,200,400,800,1600,2400,3200"
    # steps_str = "5,10,15,20,25,30,35,40,45,50,\
    #             55,60,65,70,75,80,85,90,95,100,\
    #             110,120,130,140,150,160,170,180,190,200,\
    #             220,240,260,280,300,320,340,360,380,400,\
    #             440,480,520,560,600,640,680,720,760,800,\
    #             880,960,1040,1120,1200,1280,1360,1440,1520,1600,\
    #             1680,1760,1840,1920,2000,2080,2160,2240,2320,2400,\
    #             2480,2560,2640,2720,2800,2880,2960,3040,3120,3200,\
    #             3360,3520,3680,3840,4000,4160,4320,4480,4640,4800,\
    #             4960,5120,5280,5440,5600,5760,5920,6080,6240,6400"
    outdir = os.path.join(results_root, "summaries_array")
    filters_str = ""  # 逗号分隔多个 data_name；留空表示全部

    # 解析步数
    steps = []
    for tok in steps_str.split(","):
        tok = tok.strip()
        if tok:
            try:
                v = int(tok)
                if v > 0:
                    steps.append(v)
            except ValueError:
                print(f"[WARN] 无法解析步数：{tok}，已忽略。")

    # 解析过滤 data_name
    filters = [s.strip() for s in filters_str.split(",") if s.strip()] if filters_str else None

    print("\n[DEBUG] 参数确认：")
    print(f"  root    : {results_root}")
    print(f"  db_name : {db_name}")
    print(f"  steps   : {steps}")
    print(f"  outdir  : {outdir}")
    print(f"  filters : {filters}")
    print()

    run_summarize_for_root(results_root=results_root,
                           steps=steps,
                           outdir=outdir,
                           db_name=db_name,
                           filters=filters)

def compare_npz(script_npz_path: str, framework_npz_path: str):
    """
    比较脚本汇总的 npz 与框架直接输出的 npz。
    
    参数:
        script_npz_path : str  脚本汇总结果 npz 文件路径 (如 "1600.npz")
        framework_npz_path : str  框架直接输出结果 npz 文件路径 (如 "1600_opt.npz")
    
    返回:
        pandas.DataFrame : 包含 data_name、score_script、score_framework、equal 四列
    """
    # 读取 npz
    script_npz_path = os.path.join(results_path, script_npz_path)
    framework_npz_path = os.path.join(results_path, framework_npz_path)
    script_npz = np.load(script_npz_path, allow_pickle=True)["results"].tolist()
    framework_npz = np.load(framework_npz_path, allow_pickle=True)["results"].tolist()

    # 建立脚本数据的字典 {data_name -> opt_score}
    script_dict = {
        item["file_path"]: item["opt_score"] for item in script_npz
    }
    # script_dict = {}
    # for idx, item in enumerate(script_npz):
    #     data_name = item["file_path"]  # 脚本里的 file_path 就是 data_name
    #     if data_name in script_dict:
    #         print(f"[WARN] Script npz 重复: {data_name} "
    #               f"(旧score={script_dict[data_name]}, 新score={item['opt_score']}, index={idx})")
    #     script_dict[data_name] = item["opt_score"]

    # 框架数据 -> 提取 data_name (从 file_path[0] 提取 Sim_...xlsx 中间的部分)
    framework_dict = {}
    i = 0
    for idx, item in enumerate(framework_npz):
        file_paths = item["file_path"]
        if not isinstance(file_paths, list):
            continue
        first_path = os.path.basename(file_paths[0])  # 取文件名
        if first_path.startswith("Sim_") and first_path.endswith(".xlsx"):
            data_name = first_path[len("Sim_"):-len(".xlsx")]
        else:
            data_name = first_path  # fallback
        if data_name in framework_dict:
            i += 1
            print(f"[WARN] Framework npz 重复: {data_name} "
                  f"(旧score={framework_dict[data_name]}, 新score={item['opt_score']}, index={idx})")
        print(f"一共有 {i} 个不一样的")
        framework_dict[data_name] = item["opt_score"]

    # 对齐并合并
    rows = []
    for data_name, score_s in script_dict.items():
        score_f = framework_dict.get(data_name, None)
        equal = (score_s == score_f) if score_f is not None else False
        rows.append({
            "data_name": data_name,
            "score_script": score_s,
            "score_framework": score_f,
            "equal": equal
        })

    df = pd.DataFrame(rows)
    return df

# -----------------------------
# 入口：根据是否有命令行参数选择模式
# -----------------------------
if __name__ == "__main__":
    # 设为 True 可强制进入 debug 交互模式（即使有命令行参数）
    FORCE_DEBUG = True
    results_path = r"C:\Users\px2030\Code\Ergebnisse\opt_para_study\study_results\New_CAMES_results\results210925"
    
    if FORCE_DEBUG or len(sys.argv) == 1:
        # 无命令行参数 -> 进入交互模式（Spyder/Jupyter 友好）
        debug_main()
    else:
        cli_main()
        
    
    # df = compare_npz("MSE_3200.npz", "MSE_3200_opt.npz")
    # print(df.head())
    # csv_save_path = os.path.join(results_path, "compare_3200.csv")
    # df.to_csv(csv_save_path, index=False)

