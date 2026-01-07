# -*- coding: utf-8 -*-
"""
Energy–volume scan using LMC + aggregate pool (1D case, X1 fixed).

- 并行版本 run_energy_scan_from_pool
- 扫描 NO_FRAG / int_bre / gamma 的 run_full_parameter_scan
- 结果保存到 HDF5:
    /runs/<param_key>
        attrs: NO_FRAG, int_bre, gamma, Df, MAS, ...
        datasets:
            Np        [n_Np]
            V         [n_Np]
            E_mean    [n_Np]
            E_std     [n_Np]
            E_samples [n_Np, n_runs_per_np]
"""

import os
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py

from lmcann.core import LMCSimulator
from aggregates_sampler import NP_LIST


# =============================
# worker：每个任务对应“一个 grid + N_FRACS 次断裂”
# =============================
def _energy_scan_worker_one_grid(args: Tuple[Any, ...]) -> Tuple[int, int, np.ndarray]:
    """
    单个 worker 处理一个 grid：
    - 创建自己的 LMCSimulator
    - 调用 mc_breakage_from_pool(N_GRIDS=1, N_FRACS=N_FRACS)
    - 返回该 grid 上 N_FRACS 次断裂的能量数组

    返回值:
        (idx_np, Np, energies)
    """
    (
        idx_np,
        Np,
        A,
        seed_grid,
        pool_dir,
        Df,
        MAS,
        X1,
        A0,
        int_bre,
        STR,
        NO_FRAG,
        gamma,
        N_FRACS,
    ) = args

    # 每个 worker 自己构建一个模拟器
    sim = LMCSimulator(
        STR=STR,
        NO_FRAG=NO_FRAG,
        gamma=gamma,
        allow_loops=False,
        accept_all_cracks=False,
        use_weighted_start=True,
        plotter=None,
        pool_dir=pool_dir,
    )

    # 只做 1 个 grid，但有 N_FRACS 次重复
    F = sim.mc_breakage_from_pool(
        pool_dir=pool_dir,
        Df=Df,
        MAS=MAS,
        A=A,
        X1=X1,
        N_GRIDS=1,
        N_FRACS=N_FRACS,
        A0=A0,
        int_bre=int_bre,
        seed=seed_grid,
        plot_each=False,
        interp="knn",
        KNN=1,
        sigma=0.35,
    )

    # F 形状：(1 * N_FRACS * NO_FRAG, 4)
    n_runs_local = 1 * N_FRACS
    try:
        F_run = F.reshape(n_runs_local, NO_FRAG, 4)
    except ValueError:
        raise RuntimeError(
            f"[worker] Unexpected F shape for Np={Np}: "
            f"F.shape={F.shape}, expected {n_runs_local * NO_FRAG} rows"
        )

    energies = F_run[:, 0, 3].copy()

    # 清理 pool cache（可选）
    if sim.agg_pool is not None:
        sim.agg_pool.close_pool_cache()

    return idx_np, Np, energies


@dataclass
class EnergyScanResult:
    V: np.ndarray
    Np: np.ndarray
    E_mean: np.ndarray
    E_std: np.ndarray
    E_all: Dict[int, np.ndarray]  # key=Np, value=所有样本
    params: Dict[str, Any]


def build_simulator_for_pool(
    STR: np.ndarray,
    NO_FRAG: int,
    gamma: float = 1.0,
    allow_loops: bool = False,
    accept_all_cracks: bool = False,
    use_weighted_start: bool = True,
    pool_dir: str | None = None,
) -> LMCSimulator:
    sim = LMCSimulator(
        STR=STR,
        NO_FRAG=NO_FRAG,
        gamma=gamma,
        allow_loops=allow_loops,
        accept_all_cracks=accept_all_cracks,
        use_weighted_start=use_weighted_start,
        plotter=None,
        pool_dir=pool_dir,
    )
    return sim


def run_energy_scan_from_pool(
    pool_dir: str,
    Df: float,
    MAS: float,
    *,
    np_list: Sequence[int] | None = None,
    X1: float = 1.0,
    A0: float = 1.0,
    int_bre: float = 0.0,
    STR: np.ndarray | None = None,
    NO_FRAG: int = 4,
    gamma: float = 1.0,
    N_GRIDS: int = 10,
    N_FRACS: int = 5,
    base_seed: int = 42,
    workers: int = 1,
) -> EnergyScanResult:
    """
    在若干 Np (来自 aggregate pool 的 NP_LIST) 上，使用 mc_breakage_from_pool
    多次重复断裂模拟，收集每次完整破碎的能量 E_need，并给出统计量。

    workers:
        =1  -> 串行（与原实现等价）
        >1  -> 并行：把 N_GRIDS 个 grid 任务平均分配到各个进程，
               每个任务负责 N_FRACS 次断裂。
    """
    if STR is None:
        STR = np.array([1.0, 0.1, 1.0], dtype=float)

    if np_list is None:
        np_list = NP_LIST
    np_list = list(np_list)

    rng_master = np.random.default_rng(base_seed)

    E_all: Dict[int, np.ndarray] = {}
    V_list: List[float] = []
    Np_list_used: List[int] = []

    n_runs_per_np = N_GRIDS * N_FRACS

    # ========== 串行路径 ==========
    if workers is None or workers <= 1:
        sim = build_simulator_for_pool(
            STR=STR,
            NO_FRAG=NO_FRAG,
            gamma=gamma,
            allow_loops=False,
            accept_all_cracks=False,
            use_weighted_start=True,
            pool_dir=pool_dir,
        )

        for Np in np_list:
            A = float(Np) * float(A0)
            seed_np = int(rng_master.integers(0, 2**31 - 1))

            F = sim.mc_breakage_from_pool(
                pool_dir=pool_dir,
                Df=Df,
                MAS=MAS,
                A=A,
                X1=X1,
                N_GRIDS=N_GRIDS,
                N_FRACS=N_FRACS,
                A0=A0,
                int_bre=int_bre,
                seed=seed_np,
                plot_each=False,
                interp="knn",
                KNN=1,
                sigma=0.35,
            )

            try:
                F_run = F.reshape(n_runs_per_np, sim.NO_FRAG, 4)
            except ValueError:
                raise RuntimeError(
                    f"Unexpected F shape for Np={Np}: F.shape={F.shape}, "
                    f"expected {n_runs_per_np * sim.NO_FRAG} rows"
                )

            energies = F_run[:, 0, 3].copy()
            E_all[Np] = energies
            V_list.append(A)
            Np_list_used.append(Np)

            print(
                f"[SCAN-SEQ] Np={Np:6d}, V={A:8.1f}, "
                f"E_mean={energies.mean():.4f}, E_std={energies.std(ddof=1):.4f}"
            )

        if sim.agg_pool is not None:
            sim.agg_pool.close_pool_cache()

    # ========== 并行路径 ==========
    else:
        # 为每个 (Np, grid_index) 准备一个任务
        jobs: list[tuple] = []
        for idx_np, Np in enumerate(np_list):
            A = float(Np) * float(A0)
            # 每个 Np 用一个独立的 RNG 序列，保证不同 Np 的 seed 不同
            rng_np = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

            for _g in range(N_GRIDS):
                seed_grid = int(rng_np.integers(0, 2**31 - 1))
                jobs.append(
                    (
                        idx_np,
                        Np,
                        A,
                        seed_grid,
                        pool_dir,
                        Df,
                        MAS,
                        X1,
                        A0,
                        int_bre,
                        STR,
                        NO_FRAG,
                        gamma,
                        N_FRACS,
                    )
                )

        # 用 idx_np 做索引，把不同 grid 的结果收集到一起
        tmp_collect: Dict[int, List[np.ndarray]] = {}

        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_job = {
                ex.submit(_energy_scan_worker_one_grid, job): job for job in jobs
            }
            for fut in as_completed(future_to_job):
                idx_np, Np, energies = fut.result()
                tmp_collect.setdefault(idx_np, []).append(energies)

        # 汇总每个 Np 的能量
        for idx_np, Np in enumerate(np_list):
            if idx_np not in tmp_collect:
                raise RuntimeError(f"No energies collected for idx_np={idx_np}, Np={Np}")

            # (num_grids_for_this_Np, N_FRACS) -> (N_GRIDS * N_FRACS,)
            arr_list = tmp_collect[idx_np]
            energies = np.concatenate(arr_list, axis=0)

            if energies.size != n_runs_per_np:
                print(
                    f"[WARN] Np={Np}: collected {energies.size} energies, "
                    f"expected {n_runs_per_np}"
                )

            E_all[Np] = energies
            A = float(Np) * float(A0)
            V_list.append(A)
            Np_list_used.append(Np)

            print(
                f"[SCAN-PAR] Np={Np:6d}, V={A:8.1f}, "
                f"E_mean={energies.mean():.4f}, E_std={energies.std(ddof=1):.4f}"
            )

    # ===== 统一收尾 =====
    V_arr = np.array(V_list, dtype=float)
    Np_arr = np.array(Np_list_used, dtype=int)
    E_mean = np.array([E_all[int(Np)].mean() for Np in Np_arr], dtype=float)
    E_std = np.array([E_all[int(Np)].std(ddof=1) for Np in Np_arr], dtype=float)

    params = dict(
        pool_dir=os.path.abspath(pool_dir),
        Df=Df,
        MAS=MAS,
        X1=X1,
        A0=A0,
        int_bre=float(int_bre),
        STR=np.array(STR, copy=True),
        NO_FRAG=int(NO_FRAG),
        gamma=float(gamma),
        N_GRIDS=int(N_GRIDS),
        N_FRACS=int(N_FRACS),
        base_seed=int(base_seed),
        workers=int(workers),
    )

    return EnergyScanResult(
        V=V_arr,
        Np=Np_arr,
        E_mean=E_mean,
        E_std=E_std,
        E_all=E_all,
        params=params,
    )


def plot_loglog_and_fit_sigma(result: EnergyScanResult, show: bool = True):
    V = result.V
    E = result.E_mean

    mask = (V > 0.0) & (E > 0.0)
    V = V[mask]
    E = E[mask]

    logV = np.log(V)
    logE = np.log(E)

    # Pearson correlation
    r = np.corrcoef(logV, logE)[0, 1]

    # linear regression
    A_mat = np.vstack([np.ones_like(logV), logV]).T
    coef, *_ = np.linalg.lstsq(A_mat, logE, rcond=None)
    a, sigma = coef

    if show:
        plt.figure()
        plt.scatter(logV, logE, label=f"data (r={r:.3f})")
        plt.plot(logV, a + sigma * logV, label=f"fit: sigma={sigma:.3f}")
        plt.xlabel("log(V)")
        plt.ylabel("log(E_mean)")
        plt.legend()
        plt.grid(True)
        plt.title("E_need vs V (log–log)")
        plt.show()

    print(f"[FIT] log(E) ≈ {a:.3f} + {sigma:.3f} * log(V),  Pearson r={r:.4f}")
    return float(sigma), float(r)


# =============================
# HDF5 保存工具
# =============================
def _make_param_key(
    NO_FRAG: int,
    int_bre: float,
    gamma: float,
    Df: float,
    MAS: float,
) -> str:
    """根据参数生成一个 group key，用于 HDF5 内去重/索引."""
    ib = round(float(int_bre), 6)
    g = round(float(gamma), 6)
    return f"NOF_{NO_FRAG}_GB_{g}_BRE_{ib}_Df_{Df}_MAS_{MAS}"


def save_result_to_h5(
    h5_path: str,
    result: EnergyScanResult,
    NO_FRAG: int,
    int_bre: float,
    gamma: float,
):
    """将一次参数组合的扫描结果保存到 HDF5 文件中。"""
    params = result.params
    Df = params["Df"]
    MAS = params["MAS"]

    key = _make_param_key(NO_FRAG, int_bre, gamma, Df, MAS)
    grp_path = f"/runs/{key}"

    # 预先构造 E_samples 2D 数组: shape (n_Np, n_runs_per_np)
    Np_arr = result.Np.astype(int)
    E_samples = np.stack([result.E_all[int(Np)] for Np in Np_arr], axis=0)

    with h5py.File(h5_path, "a") as f:
        if grp_path in f:
            print(f"[H5] Group {grp_path} already exists, skip saving.")
            return

        grp = f.create_group(grp_path)

        # 写 attributes：参数信息
        grp.attrs["NO_FRAG"] = int(NO_FRAG)
        grp.attrs["int_bre"] = float(int_bre)
        grp.attrs["gamma"] = float(gamma)
        grp.attrs["Df"] = float(Df)
        grp.attrs["MAS"] = float(MAS)
        grp.attrs["X1"] = float(params["X1"])
        grp.attrs["A0"] = float(params["A0"])
        grp.attrs["N_GRIDS"] = int(params["N_GRIDS"])
        grp.attrs["N_FRACS"] = int(params["N_FRACS"])
        grp.attrs["base_seed"] = int(params["base_seed"])
        grp.attrs["workers"] = int(params["workers"])
        grp.attrs["STR"] = np.array(params["STR"], dtype=float)

        # 写 datasets：Np, V, E_mean, E_std, E_samples
        grp.create_dataset("Np", data=Np_arr, compression="gzip")
        grp.create_dataset("V", data=result.V, compression="gzip")
        grp.create_dataset("E_mean", data=result.E_mean, compression="gzip")
        grp.create_dataset("E_std", data=result.E_std, compression="gzip")
        grp.create_dataset("E_samples", data=E_samples, compression="gzip")

        print(f"[H5] Saved result to {grp_path}")


# =============================
# 参数扫描示例：写入 HDF5
# =============================
def run_full_parameter_scan(
    h5_path: str,
    pool_dir: str,
    Df: float,
    MAS: float,
    STR: np.ndarray,
    A0: float,
    X1: float,
    np_list: Sequence[int],
    no_frag_list: Sequence[int],
    int_bre_list: np.ndarray,
    gamma_list: np.ndarray,
    N_GRIDS: int,
    N_FRACS: int,
    base_seed: int,
    workers: int,
):
    """对参数组合进行扫描，优先检查 H5 中是否已有数据，若有则跳过。"""

    for NO_FRAG in no_frag_list:
        for int_bre in int_bre_list:
            for gamma in gamma_list:

                # ==== 构造 group key ====
                key = _make_param_key(NO_FRAG, float(int_bre), float(gamma), Df, MAS)
                grp_path = f"/runs/{key}"

                # ==== 检查是否已经存在 ====
                with h5py.File(h5_path, "a") as f:
                    if grp_path in f:
                        print(f"[SKIP] {grp_path} already exists, skip running LMC.")
                        continue

                # ==== 需要运行 LMC ====
                print("\n=====================================================")
                print(f"SCAN: NO_FRAG={NO_FRAG}, int_bre={int_bre:.3f}, gamma={gamma:.3f}")
                print("=====================================================\n")

                result = run_energy_scan_from_pool(
                    pool_dir=pool_dir,
                    Df=Df,
                    MAS=MAS,
                    np_list=np_list,
                    X1=X1,
                    A0=A0,
                    int_bre=float(int_bre),
                    STR=STR,
                    NO_FRAG=int(NO_FRAG),
                    gamma=float(gamma),
                    N_GRIDS=N_GRIDS,
                    N_FRACS=N_FRACS,
                    base_seed=base_seed,
                    workers=workers,
                )

                # ==== 拟合 sigma, r ====
                sigma, r = plot_loglog_and_fit_sigma(result, show=False)
                print(f"[SCAN] sigma={sigma:.4f}, r={r:.4f}")

                result.params["sigma"] = float(sigma)
                result.params["pearson_r"] = float(r)

                # ==== 保存到 H5 ====
                save_result_to_h5(
                    h5_path=h5_path,
                    result=result,
                    NO_FRAG=int(NO_FRAG),
                    int_bre=float(int_bre),
                    gamma=float(gamma),
                )

    print("\nAll scans finished!")
    print(f"Results saved to {h5_path}")

def print_h5_structure(h5_path):
    """
    递归列出 HDF5 文件的所有 group / dataset / attributes。
    """

    def _print(name, obj):
        indent = "  " * (name.count("/") - 1)

        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group ] {name}")
            # 打印 group attrs
            for k, v in obj.attrs.items():
                print(f"{indent}    (attr) {k}: {v}")

        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")
            # 打印 dataset attrs（一般少见，但可能存在）
            for k, v in obj.attrs.items():
                print(f"{indent}    (attr) {k}: {v}")

    with h5py.File(h5_path, "r") as f:
        print(f"--- HDF5 structure of {h5_path} ---")
        f.visititems(_print)

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    pool_dir = r""  # 你的 aggregate pool 目录
    Df = 1.8
    MAS = 0.40

    A0 = 1.0
    X1 = 1.0
    STR = np.array([1.0, 0.1, 1.0], dtype=float)

    N_GRIDS = 100
    N_FRACS = 200
    base_seed = 42
    workers = 12

    np_list = NP_LIST  # 或 np_list = NP_LIST[:6]
    no_frag_list = [2,3,4,5,6,7,8]
    int_bre_list = np.linspace(0.0, 1.0, 6)
    gamma_list = np.logspace(-1, 1, 6)

    output_h5 = "psd_data.h5"

    # run_full_parameter_scan(
    #     h5_path=output_h5,
    #     pool_dir=pool_dir,
    #     Df=Df,
    #     MAS=MAS,
    #     STR=STR,
    #     A0=A0,
    #     X1=X1,
    #     np_list=np_list,
    #     no_frag_list=no_frag_list,
    #     int_bre_list=int_bre_list,
    #     gamma_list=gamma_list,
    #     N_GRIDS=N_GRIDS,
    #     N_FRACS=N_FRACS,
    #     base_seed=base_seed,
    #     workers=workers,
    # )
    print_h5_structure(output_h5)
