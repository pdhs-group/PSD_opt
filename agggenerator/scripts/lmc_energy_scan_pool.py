# -*- coding: utf-8 -*-
"""
Energy-volume scan using LMC + aggregate pool.

This version supports parameter-grid scans over:
- NO_FRAG
- int_bre
- gamma
- MAS
- X1
- STR

Each run is stored in HDF5 with a unique key and full parameter metadata,
so the saved data can be used directly as training data later.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from lmc import LMCSimulator
from aggregates_sampler import NP_LIST


def _ensure_sequence(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_str_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"STR must be a 1D array-like, got shape={arr.shape}")
    return arr.copy()


def _format_float_for_key(value: float) -> str:
    return np.format_float_positional(
        float(value), precision=6, unique=False, fractional=False, trim="-"
    ).replace(".", "p").replace("-", "m")


def _format_str_for_key(str_values: np.ndarray) -> str:
    return "x".join(_format_float_for_key(v) for v in _coerce_str_array(str_values))


# =============================
# Worker: one grid + N_FRACS repeats
# =============================
def _energy_scan_worker_one_grid(args: Tuple[Any, ...]) -> Tuple[int, int, np.ndarray]:
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

    n_runs_local = N_FRACS
    try:
        F_run = F.reshape(n_runs_local, NO_FRAG, 4)
    except ValueError as exc:
        raise RuntimeError(
            f"[worker] Unexpected F shape for Np={Np}: "
            f"F.shape={F.shape}, expected {n_runs_local * NO_FRAG} rows"
        ) from exc

    energies = F_run[:, 0, 3].copy()

    if sim.agg_pool is not None:
        sim.agg_pool.close_pool_cache()

    return idx_np, Np, energies


@dataclass
class EnergyScanResult:
    V: np.ndarray
    Np: np.ndarray
    E_mean: np.ndarray
    E_std: np.ndarray
    E_all: Dict[int, np.ndarray]
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
    return LMCSimulator(
        STR=STR,
        NO_FRAG=NO_FRAG,
        gamma=gamma,
        allow_loops=allow_loops,
        accept_all_cracks=accept_all_cracks,
        use_weighted_start=use_weighted_start,
        plotter=None,
        pool_dir=pool_dir,
    )


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
    if STR is None:
        STR = np.array([1.0, 0.1, 1.0], dtype=float)
    else:
        STR = _coerce_str_array(STR)

    if np_list is None:
        np_list = NP_LIST
    np_list = list(np_list)

    rng_master = np.random.default_rng(base_seed)

    E_all: Dict[int, np.ndarray] = {}
    V_list: List[float] = []
    Np_list_used: List[int] = []

    n_runs_per_np = N_GRIDS * N_FRACS

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
            except ValueError as exc:
                raise RuntimeError(
                    f"Unexpected F shape for Np={Np}: F.shape={F.shape}, "
                    f"expected {n_runs_per_np * sim.NO_FRAG} rows"
                ) from exc

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
    else:
        jobs: List[Tuple[Any, ...]] = []
        for idx_np, Np in enumerate(np_list):
            A = float(Np) * float(A0)
            rng_np = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

            for _ in range(N_GRIDS):
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

        tmp_collect: Dict[int, List[np.ndarray]] = {}

        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_job = {
                ex.submit(_energy_scan_worker_one_grid, job): job for job in jobs
            }
            for fut in as_completed(future_to_job):
                idx_np, Np, energies = fut.result()
                tmp_collect.setdefault(idx_np, []).append(energies)

        for idx_np, Np in enumerate(np_list):
            if idx_np not in tmp_collect:
                raise RuntimeError(f"No energies collected for idx_np={idx_np}, Np={Np}")

            energies = np.concatenate(tmp_collect[idx_np], axis=0)
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

    V_arr = np.array(V_list, dtype=float)
    Np_arr = np.array(Np_list_used, dtype=int)
    E_mean = np.array([E_all[int(Np)].mean() for Np in Np_arr], dtype=float)
    E_std = np.array([E_all[int(Np)].std(ddof=1) for Np in Np_arr], dtype=float)

    params = dict(
        pool_dir=os.path.abspath(pool_dir),
        Df=float(Df),
        MAS=float(MAS),
        X1=float(X1),
        A0=float(A0),
        int_bre=float(int_bre),
        STR=_coerce_str_array(STR),
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

    r = np.corrcoef(logV, logE)[0, 1]

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
        plt.title("E_need vs V (log-log)")
        plt.show()

    print(f"[FIT] log(E) ~= {a:.3f} + {sigma:.3f} * log(V), Pearson r={r:.4f}")
    return float(sigma), float(r)


# =============================
# HDF5 save helpers
# =============================
def _make_param_key(
    NO_FRAG: int,
    int_bre: float,
    gamma: float,
    Df: float,
    MAS: float,
    X1: float,
    STR: np.ndarray,
) -> str:
    return (
        f"NOF_{int(NO_FRAG)}"
        f"_GB_{_format_float_for_key(gamma)}"
        f"_BRE_{_format_float_for_key(int_bre)}"
        f"_Df_{_format_float_for_key(Df)}"
        f"_MAS_{_format_float_for_key(MAS)}"
        f"_X1_{_format_float_for_key(X1)}"
        f"_STR_{_format_str_for_key(STR)}"
    )


def save_result_to_h5(
    h5_path: str,
    result: EnergyScanResult,
    NO_FRAG: int,
    int_bre: float,
    gamma: float,
):
    params = result.params
    Df = float(params["Df"])
    MAS = float(params["MAS"])
    X1 = float(params["X1"])
    STR = _coerce_str_array(params["STR"])

    key = _make_param_key(NO_FRAG, int_bre, gamma, Df, MAS, X1, STR)
    grp_path = f"/runs/{key}"

    Np_arr = result.Np.astype(int)
    E_samples = np.stack([result.E_all[int(Np)] for Np in Np_arr], axis=0)

    with h5py.File(h5_path, "a") as f:
        if grp_path in f:
            print(f"[H5] Group {grp_path} already exists, skip saving.")
            return

        grp = f.create_group(grp_path)
        grp.attrs["NO_FRAG"] = int(NO_FRAG)
        grp.attrs["int_bre"] = float(int_bre)
        grp.attrs["gamma"] = float(gamma)
        grp.attrs["Df"] = Df
        grp.attrs["MAS"] = MAS
        grp.attrs["X1"] = X1
        grp.attrs["A0"] = float(params["A0"])
        grp.attrs["N_GRIDS"] = int(params["N_GRIDS"])
        grp.attrs["N_FRACS"] = int(params["N_FRACS"])
        grp.attrs["base_seed"] = int(params["base_seed"])
        grp.attrs["workers"] = int(params["workers"])
        grp.attrs["STR"] = STR
        if "sigma" in params:
            grp.attrs["sigma"] = float(params["sigma"])
        if "pearson_r" in params:
            grp.attrs["pearson_r"] = float(params["pearson_r"])

        grp.create_dataset("Np", data=Np_arr, compression="gzip")
        grp.create_dataset("V", data=result.V, compression="gzip")
        grp.create_dataset("E_mean", data=result.E_mean, compression="gzip")
        grp.create_dataset("E_std", data=result.E_std, compression="gzip")
        grp.create_dataset("E_samples", data=E_samples, compression="gzip")

        print(f"[H5] Saved result to {grp_path}")


# =============================
# Parameter scan
# =============================
def run_full_parameter_scan(
    h5_path: str,
    pool_dir: str,
    Df: float,
    MAS: float | Sequence[float],
    STR: np.ndarray | Sequence[np.ndarray | Sequence[float]],
    A0: float,
    X1: float | Sequence[float],
    np_list: Sequence[int],
    no_frag_list: Sequence[int],
    int_bre_list: np.ndarray,
    gamma_list: np.ndarray,
    N_GRIDS: int,
    N_FRACS: int,
    base_seed: int,
    workers: int,
):
    mas_list = [float(v) for v in _ensure_sequence(MAS)]
    x1_list = [float(v) for v in _ensure_sequence(X1)]
    str_list = [_coerce_str_array(v) for v in _ensure_sequence(STR)]

    for NO_FRAG in no_frag_list:
        for int_bre in int_bre_list:
            for gamma in gamma_list:
                for MAS_value in mas_list:
                    for X1_value in x1_list:
                        for STR_value in str_list:
                            key = _make_param_key(
                                NO_FRAG,
                                float(int_bre),
                                float(gamma),
                                Df,
                                MAS_value,
                                X1_value,
                                STR_value,
                            )
                            grp_path = f"/runs/{key}"

                            with h5py.File(h5_path, "a") as f:
                                if grp_path in f:
                                    print(f"[SKIP] {grp_path} already exists, skip running LMC.")
                                    continue

                            print("\n=====================================================")
                            print(
                                "SCAN: "
                                f"NO_FRAG={NO_FRAG}, int_bre={float(int_bre):.3f}, "
                                f"gamma={float(gamma):.3f}, MAS={MAS_value:.6g}, "
                                f"X1={X1_value:.6g}, STR={STR_value.tolist()}"
                            )
                            print("=====================================================\n")

                            result = run_energy_scan_from_pool(
                                pool_dir=pool_dir,
                                Df=Df,
                                MAS=MAS_value,
                                np_list=np_list,
                                X1=X1_value,
                                A0=A0,
                                int_bre=float(int_bre),
                                STR=STR_value,
                                NO_FRAG=int(NO_FRAG),
                                gamma=float(gamma),
                                N_GRIDS=N_GRIDS,
                                N_FRACS=N_FRACS,
                                base_seed=base_seed,
                                workers=workers,
                            )

                            sigma, r = plot_loglog_and_fit_sigma(result, show=False)
                            print(f"[SCAN] sigma={sigma:.4f}, r={r:.4f}")

                            result.params["sigma"] = float(sigma)
                            result.params["pearson_r"] = float(r)

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
    def _print(name, obj):
        indent = "  " * (name.count("/") - 1)

        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group ] {name}")
            for k, v in obj.attrs.items():
                print(f"{indent}    (attr) {k}: {v}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")
            for k, v in obj.attrs.items():
                print(f"{indent}    (attr) {k}: {v}")

    with h5py.File(h5_path, "r") as f:
        print(f"--- HDF5 structure of {h5_path} ---")
        f.visititems(_print)


if __name__ == "__main__":
    pool_dir = r""
    # pool_dir = os.environ.get("STORAGE_PATH")
    Df = 1.8
    MAS_list = [0.40]

    A0 = 1.0
    X1_list = [1.0]
    STR_list = [np.array([1.0, 0.1, 1.0], dtype=float)]

    N_GRIDS = 100
    N_FRACS = 200
    base_seed = 42
    workers = 12

    np_list = NP_LIST
    no_frag_list = [2, 3, 4, 5, 6, 7, 8]
    int_bre_list = np.linspace(0.0, 1.0, 6)
    gamma_list = np.logspace(-3, 3, 6)

    output_h5 = "psd_data.h5"

    run_full_parameter_scan(
        h5_path=output_h5,
        pool_dir=pool_dir,
        Df=Df,
        MAS=MAS_list,
        STR=STR_list,
        A0=A0,
        X1=X1_list,
        np_list=np_list,
        no_frag_list=no_frag_list,
        int_bre_list=int_bre_list,
        gamma_list=gamma_list,
        N_GRIDS=N_GRIDS,
        N_FRACS=N_FRACS,
        base_seed=base_seed,
        workers=workers,
    )
    # print_h5_structure(output_h5)
