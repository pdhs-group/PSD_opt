# compare_lmc_mcpbe_methods.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# 适配器
from optframework.mcpbe.lmc_adapter import (
    LMCTableAdapter,
    LMCRankAdapter,
    LMCLiveAdapter,
    LMCLiveFallback,
    LMCLiveDisable,
    LMCCopulaAdapter,
    LMCFlowAdapter,    # ← 新增：flow 适配器
)

# ---------------------------
# 可配置区域
# ---------------------------
A_parent = 500.0
X1_parent = 0.5

N_OUTER = 200
N_INNER = 200

USE_TABLES = True
USE_RANK   = True
USE_LIVE   = True
USE_COPULA = True
USE_FLOW   = True   # ← 新增

TABLES_NPZ_PATH = "lmc_tables_grid.npz"
RANK_NPZ_PATH   = "lmc_rank_tables_grid.npz"
COPULA_NPZ_PATH = "lmc_copula_grid.npz"
FLOW_MIX_PT_PATH    = "lmc_cond_flow_mix.pt"   # ← 你刚训练出来的 .pt
FLOW_PURE_PT_PATH   = "lmc_cond_flow_pure.pt"

A0_run = 1.0

RANK_K_USE: Optional[int] = None
TAIL_STRATEGY = "equal"

LIVE_NO_FRAG = 4
LIVE_SMALL_PARTICLE_POLICY = "fallback"
LIVE_DELTA_CELLS = 0.1

BASE_SEED = 42
# ---------------------------


def _spawn_rngs(base_seed: int, count: int) -> List[np.random.Generator]:
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(count)
    return [np.random.default_rng(s) for s in children]


def _from_rA_rB_to_fragments(rA: List[float], rB: List[float],
                             A: float, X1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X3 = 1.0 - X1
    VA = np.asarray(rA, dtype=float) * (A * X1)
    VB = np.asarray(rB, dtype=float) * (A * X3)
    VT = VA + VB
    return VA, VB, VT


def _collect_stats(VA_all: List[float], VB_all: List[float], VT_all: List[float],
                   nfrags_per_event: List[int],
                   A: float, X1: float) -> Dict[str, float]:
    sVA = np.sum(VA_all)
    sVB = np.sum(VB_all)
    sVT = np.sum(VT_all)
    n_events = len(nfrags_per_event)
    X3 = 1.0 - X1

    targetA = n_events * A * X1
    targetB = n_events * A * X3
    targetT = n_events * A

    errA = abs(sVA - targetA) / max(targetA, 1e-16)
    errB = abs(sVB - targetB) / max(targetB, 1e-16)
    errT = abs(sVT - targetT) / max(targetT, 1e-16)

    return {
        "events": float(n_events),
        "sum_VA": float(sVA), "sum_VB": float(sVB), "sum_VT": float(sVT),
        "rel_err_A": float(errA),
        "rel_err_B": float(errB),
        "rel_err_T": float(errT),
        "avg_N": float(np.mean(nfrags_per_event)) if n_events > 0 else 0.0,
        "std_N": float(np.std(nfrags_per_event)) if n_events > 0 else 0.0,
    }


def _plot_results(fig_title: str,
                  methods: List[str],
                  VT_rel_by_method: Dict[str, np.ndarray],
                  ZA_by_method: Dict[str, np.ndarray],
                  ZB_by_method: Dict[str, np.ndarray],
                  bins_1d: int = 60,
                  bins_2d: int = 60,
                  vmax = None):
    nrow = 2
    ncol = 1 + len(methods)

    fig = plt.figure(figsize=(5 * ncol, 4 * nrow))
    fig.suptitle(fig_title)

    # 1D
    ax = plt.subplot(nrow, ncol, 1)
    for name in methods:
        vt_rel = VT_rel_by_method[name]
        ax.hist(vt_rel, bins=bins_1d, histtype='step', density=True, label=name)
    ax.set_xlabel("Fragment total fraction (VT/A)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D
    idx = 2
    for name in methods:
        ax2 = plt.subplot(nrow, ncol, idx)
        ZA = ZA_by_method[name]
        ZB = ZB_by_method[name]
        hb = ax2.hexbin(ZA, ZB, gridsize=bins_2d, mincnt=1, vmin=0, vmax=vmax)
        ax2.set_xlabel("ZA = VA/A")
        ax2.set_ylabel("ZB = VB/A")
        ax2.set_title(f"{name}: 2D distribution")
        cb = fig.colorbar(hb, ax=ax2)
        cb.set_label("Counts")
        ax2.grid(True, alpha=0.3)
        idx += 1

    plt.tight_layout()
    plt.show()


# ---------------------------
# 各方法的事件采样
# ---------------------------
def run_with_tables(adapter: LMCTableAdapter,
                    A: float, X1: float,
                    rngs: List[np.random.Generator]):
    VA_all: List[float] = []
    VB_all: List[float] = []
    VT_all: List[float] = []
    N_per_event: List[int] = []

    for rng in rngs:
        rel1, rel3, rowsum_cdf, row_cdf, zmin1, zmin3, pe = adapter.get_2d(A, X1)
        fl = int(math.floor(pe))
        ce = int(math.ceil(pe))
        N = ce if (rng.random() < (pe - fl)) else fl
        N = max(N, 2)

        rA_list: List[float] = []
        rB_list: List[float] = []
        for _ in range(N - 1):
            u1 = float(rng.random())
            i = int(np.searchsorted(rowsum_cdf, u1, side="right"))
            i = min(i, rel1.size - 1)
            row = row_cdf[i]
            u2 = float(rng.random())
            j = int(np.searchsorted(row, u2, side="right"))
            j = min(j, rel3.size - 1)
            rAi = float(rel1[i]); rBj = float(rel3[j])
            rA_list.append(rAi); rB_list.append(rBj)

        sA = float(np.sum(rA_list)); sB = float(np.sum(rB_list))
        rA_list.append(max(0.0, 1.0 - sA))
        rB_list.append(max(0.0, 1.0 - sB))

        VA, VB, VT = _from_rA_rB_to_fragments(rA_list, rB_list, A, X1)
        VA_all.extend(VA.tolist())
        VB_all.extend(VB.tolist())
        VT_all.extend(VT.tolist())
        N_per_event.append(N)

    return VA_all, VB_all, VT_all, N_per_event


def run_with_rank(adapter: LMCRankAdapter,
                  A: float, X1: float,
                  rngs: List[np.random.Generator]):
    VA_all: List[float] = []
    VB_all: List[float] = []
    VT_all: List[float] = []
    N_per_event: List[int] = []

    for rng in rngs:
        rA_list, rB_list = adapter.sample_one_shot(
            A=A, X1=X1, rng=rng, N=None, K_use=RANK_K_USE, tail_strategy=TAIL_STRATEGY
        )
        N = len(rA_list)
        VA, VB, VT = _from_rA_rB_to_fragments(rA_list, rB_list, A, X1)
        VA_all.extend(VA.tolist())
        VB_all.extend(VB.tolist())
        VT_all.extend(VT.tolist())
        N_per_event.append(N)

    return VA_all, VB_all, VT_all, N_per_event


def run_with_copula(adapter: LMCCopulaAdapter,
                    A: float, X1: float,
                    rngs: List[np.random.Generator]):
    VA_all: List[float] = []
    VB_all: List[float] = []
    VT_all: List[float] = []
    N_per_event: List[int] = []

    for rng in rngs:
        rA_list, rB_list = adapter.sample_one_shot(
            A=A, X1=X1, rng=rng, N=None, K_use=None, tail_strategy="equal"
        )
        N = len(rA_list)
        VA, VB, VT = _from_rA_rB_to_fragments(rA_list, rB_list, A, X1)
        VA_all.extend(VA.tolist())
        VB_all.extend(VB.tolist())
        VT_all.extend(VT.tolist())
        N_per_event.append(N)

    return VA_all, VB_all, VT_all, N_per_event


def run_with_flow(adapter: LMCFlowAdapter,
                  A: float, X1: float,
                  rngs: List[np.random.Generator]):
    """
    新增：使用条件 flow 一次性生成整次破碎事件
    """
    VA_all: List[float] = []
    VB_all: List[float] = []
    VT_all: List[float] = []
    N_per_event: List[int] = []

    for rng in rngs:
        rA_list, rB_list = adapter.sample_one_shot(A=A, X1=X1, rng=rng, N=None)
        N = len(rA_list)
        VA, VB, VT = _from_rA_rB_to_fragments(rA_list, rB_list, A, X1)
        VA_all.extend(VA.tolist())
        VB_all.extend(VB.tolist())
        VT_all.extend(VT.tolist())
        N_per_event.append(N)

    return VA_all, VB_all, VT_all, N_per_event


def run_with_live(adapter: LMCLiveAdapter,
                  A: float, X1: float,
                  rngs: List[np.random.Generator]):
    VA_all: List[float] = []
    VB_all: List[float] = []
    VT_all: List[float] = []
    N_per_event: List[int] = []

    V_parent = np.array([A * X1, A * (1.0 - X1)], dtype=float)

    for rng in rngs:
        try:
            frags, _E = adapter.sample_one_shot(V_parent, rng)
        except (LMCLiveFallback, LMCLiveDisable):
            continue

        if len(frags) == 0:
            continue
        N = len(frags)
        N_per_event.append(N)
        for f in frags:
            if f.size == 1:
                VA_all.append(float(f[0])); VB_all.append(0.0); VT_all.append(float(f[0]))
            else:
                a = float(f[0]); b = float(f[1])
                VA_all.append(a); VB_all.append(b); VT_all.append(a + b)

    return VA_all, VB_all, VT_all, N_per_event


# ---------------------------
# 主流程
# ---------------------------
def main():
    total_events = N_OUTER * N_INNER
    rngs = _spawn_rngs(BASE_SEED, total_events)

    methods: List[str] = []
    VT_rel_by_method: Dict[str, np.ndarray] = {}
    ZA_by_method: Dict[str, np.ndarray] = {}
    ZB_by_method: Dict[str, np.ndarray] = {}
    stats_by_method: Dict[str, Dict[str, float]] = {}

    # --- TABLE ---
    if USE_TABLES:
        if not os.path.exists(TABLES_NPZ_PATH):
            print(f"[WARN] tables npz not found: {TABLES_NPZ_PATH} -> skip.")
        else:
            tab = LMCTableAdapter(TABLES_NPZ_PATH, interp="bilinear", A0_run=A0_run, cache_enabled=True)
            VA_all, VB_all, VT_all, N_event = run_with_tables(tab, A_parent, X1_parent, rngs)
            methods.append("tables")
            VT_rel_by_method["tables"] = np.asarray(VT_all) / max(A_parent, 1e-16)
            ZA_by_method["tables"] = np.asarray(VA_all) / max(A_parent, 1e-16)
            ZB_by_method["tables"] = np.asarray(VB_all) / max(A_parent, 1e-16)
            stats_by_method["tables"] = _collect_stats(VA_all, VB_all, VT_all, N_event, A_parent, X1_parent)

    # --- RANK ---
    if USE_RANK:
        if not os.path.exists(RANK_NPZ_PATH):
            print(f"[WARN] rank npz not found: {RANK_NPZ_PATH} -> skip.")
        else:
            rk = LMCRankAdapter(RANK_NPZ_PATH, interp="bilinear", A0_run=A0_run, cache_enabled=True)
            VA_all, VB_all, VT_all, N_event = run_with_rank(rk, A_parent, X1_parent, rngs)
            methods.append("rank")
            VT_rel_by_method["rank"] = np.asarray(VT_all) / max(A_parent, 1e-16)
            ZA_by_method["rank"] = np.asarray(VA_all) / max(A_parent, 1e-16)
            ZB_by_method["rank"] = np.asarray(VB_all) / max(A_parent, 1e-16)
            stats_by_method["rank"] = _collect_stats(VA_all, VB_all, VT_all, N_event, A_parent, X1_parent)

    # --- COPULA ---
    if USE_COPULA:
        if not os.path.exists(COPULA_NPZ_PATH):
            print(f"[WARN] copula npz not found: {COPULA_NPZ_PATH} -> skip.")
        else:
            cp = LMCCopulaAdapter(COPULA_NPZ_PATH, interp="bilinear", A0_run=A0_run, cache_enabled=True)
            VA_all, VB_all, VT_all, N_event = run_with_copula(cp, A_parent, X1_parent, rngs)
            methods.append("copula")
            VT_rel_by_method["copula"] = np.asarray(VT_all) / max(A_parent, 1e-16)
            ZA_by_method["copula"] = np.asarray(VA_all) / max(A_parent, 1e-16)
            ZB_by_method["copula"] = np.asarray(VB_all) / max(A_parent, 1e-16)
            stats_by_method["copula"] = _collect_stats(VA_all, VB_all, VT_all, N_event, A_parent, X1_parent)

    # --- FLOW ---
    if USE_FLOW:
        if not os.path.exists(FLOW_MIX_PT_PATH) or not os.path.exists(FLOW_PURE_PT_PATH):
            print("[WARN] flow model not found -> skip.")
        else:
            flw = LMCFlowAdapter(pure_model_path=FLOW_PURE_PT_PATH, mix_model_path=FLOW_MIX_PT_PATH, 
                                 A0_run=A0_run, cache_enabled=True)
            VA_all, VB_all, VT_all, N_event = run_with_flow(flw, A_parent, X1_parent, rngs)
            methods.append("flow")
            VT_rel_by_method["flow"] = np.asarray(VT_all) / max(A_parent, 1e-16)
            ZA_by_method["flow"] = np.asarray(VA_all) / max(A_parent, 1e-16)
            ZB_by_method["flow"] = np.asarray(VB_all) / max(A_parent, 1e-16)
            stats_by_method["flow"] = _collect_stats(VA_all, VB_all, VT_all, N_event, A_parent, X1_parent)

    # --- LIVE ---
    if USE_LIVE:
        live = LMCLiveAdapter()
        live.configure_simulator(
            NO_FRAG=LIVE_NO_FRAG,
            A0_run=A0_run,
            small_particle_policy=LIVE_SMALL_PARTICLE_POLICY,
            delta_cells=LIVE_DELTA_CELLS,
            rebuild=True,
        )
        VA_all, VB_all, VT_all, N_event = run_with_live(live, A_parent, X1_parent, rngs)
        if len(N_event) == 0:
            print("[WARN] live sampling produced no events.")
        else:
            methods.append("live")
            VT_rel_by_method["live"] = np.asarray(VT_all) / max(A_parent, 1e-16)
            ZA_by_method["live"] = np.asarray(VA_all) / max(A_parent, 1e-16)
            ZB_by_method["live"] = np.asarray(VB_all) / max(A_parent, 1e-16)
            stats_by_method["live"] = _collect_stats(VA_all, VB_all, VT_all, N_event, A_parent, X1_parent)

    # --- 输出统计 ---
    print("\n=== Summary ===")
    for name in methods:
        s = stats_by_method.get(name, {})
        print(f"[{name}] events={s.get('events',0):.0f}  "
              f"avg_N={s.get('avg_N',0):.3f}±{s.get('std_N',0):.3f}  "
              f"rel_err_A={s.get('rel_err_A',0):.3e}  "
              f"rel_err_B={s.get('rel_err_B',0):.3e}  "
              f"rel_err_T={s.get('rel_err_T',0):.3e}")

    # --- 绘图 ---
    live_H, _, _ = np.histogram2d(ZA_by_method["live"], ZB_by_method["live"], bins=50, range=[[0,1],[0,0.3]])
    live_max = live_H.max()
    if methods:
        _plot_results(
            fig_title=f"Compare methods @ A={A_parent}, X1={X1_parent}",
            methods=methods,
            VT_rel_by_method=VT_rel_by_method,
            ZA_by_method=ZA_by_method,
            ZB_by_method=ZB_by_method,
            bins_1d=80,
            bins_2d=50,
            vmax=live_max,
        )


if __name__ == "__main__":
    main()
