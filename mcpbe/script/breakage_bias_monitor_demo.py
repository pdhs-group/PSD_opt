from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "mcpbe" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wmcpbe_backup import MCPBESolver  # noqa: E402


def exact_moments_1d_uniform_breakage(
    t_vec: np.ndarray,
    V0: np.ndarray,
    W0: np.ndarray,
    lam: float,
) -> np.ndarray:
    mu = np.zeros((3, 1, t_vec.size), dtype=float)
    m0_0 = float(np.sum(W0))
    m1_0 = float(np.sum(W0 * V0))
    m2_0 = float(np.sum(W0 * V0 * V0))

    mu[0, 0, :] = m0_0 * np.exp(lam * t_vec)
    mu[1, 0, :] = m1_0
    mu[2, 0, :] = m2_0 * np.exp(-(lam / 3.0) * t_vec)
    return mu


def build_solver(
    V_init: np.ndarray,
    W_init: np.ndarray,
    t_vec: np.ndarray,
    lam: float,
    delta_w: float,
    seed: int,
) -> MCPBESolver:
    solver = MCPBESolver(dim=1, t_vec=t_vec, verbose=False, load_attr=False, init=False, seed=seed)
    solver.process_type = "breakage"
    solver.BREAKRVAL = 1
    solver.BREAKFVAL = 2
    solver.pl_P1 = float(lam)
    solver.pl_P2 = 1
    solver.pl_v = 1.0
    solver.pl_q = 1.0
    solver.break_dW_min = float(delta_w)
    solver.break_dW_max = float(delta_w)
    solver.bias_enable = True
    solver.Vc = 1.0
    solver.a0 = int(V_init.shape[1])
    solver.recon_enable = True
    solver.recon_N_max = 2000
    solver.recon_method = "2PM"
    solver.recon_bins = 100
    solver.recon_RS_target = 1000

    solver._initialize_particles(init_Vc=False, V_flat=V_init)
    a = int(solver.a_tot)
    solver.W[:a] = np.asarray(W_init, dtype=float)
    solver.W0 = solver.W[:a].copy()
    solver.W0_save = [solver.W0.copy()]
    solver.W_save = [solver.W[:a].copy()]
    solver.W_save_left = [solver.W[:a].copy()]

    solver._init_lmc()
    solver._initialize_samplers()
    return solver


def run_repeats(
    n_runs: int,
    V_init: np.ndarray,
    W_init: np.ndarray,
    t_vec: np.ndarray,
    lam: float,
    delta_w: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_all = []
    bias_cum_m2_all = []
    bias_err_pred_m2_all = []
    bias_ratio_m2_all = []

    for k in range(n_runs):
        solver = build_solver(V_init, W_init, t_vec, lam, delta_w, seed=42 + k)
        solver.solve(maxiter=int(2e6))
        mu, _ = solver.calc_moments_over_time(max_i=2, max_j=0, normalize=True)
        mu_all.append(mu)
        bias_cum_m2_all.append(np.asarray(solver.bias_cum_M2[: mu.shape[2]], dtype=float))
        bias_err_pred_m2_all.append(np.asarray(solver.bias_err_pred_M2[: mu.shape[2]], dtype=float))
        bias_ratio_m2_all.append(np.asarray(solver.bias_ratio_M2[: mu.shape[2]], dtype=float))

    return (
        np.asarray(mu_all, dtype=float),
        np.asarray(bias_cum_m2_all, dtype=float),
        np.asarray(bias_err_pred_m2_all, dtype=float),
        np.asarray(bias_ratio_m2_all, dtype=float),
    )


def main() -> None:
    n_init = 1000
    t_vec = np.linspace(0.0, 41.0, 21)
    lam = 0.1
    delta_w = 100.0
    n_runs = 1000

    V = np.geomspace(1.0, 40.0, n_init)
    W = np.linspace(100.0, 200.0, n_init)

    V_init = np.zeros((2, n_init), dtype=float)
    V_init[0, :] = V
    V_init[1, :] = V

    mu_exact = exact_moments_1d_uniform_breakage(t_vec, V, W, lam)
    mu_all, bias_cum_m2_all, bias_err_pred_m2_all, bias_ratio_m2_all = run_repeats(
        n_runs=n_runs,
        V_init=V_init,
        W_init=W,
        t_vec=t_vec,
        lam=lam,
        delta_w=delta_w,
    )

    mu_mean = np.mean(mu_all, axis=0)
    err_m0 = mu_mean[0, 0, :] - mu_exact[0, 0, :]
    err_m2 = mu_mean[2, 0, :] - mu_exact[2, 0, :]
    bias_cum_m2_mean = np.mean(bias_cum_m2_all, axis=0)
    bias_err_pred_m2_mean = np.mean(bias_err_pred_m2_all, axis=0)
    bias_ratio_m2_mean = np.nanmean(bias_ratio_m2_all, axis=0)

    print("Breakage bias monitor demo (backup solver)")
    print(f"n_runs      = {n_runs}")
    print(f"lambda      = {lam}")
    print(f"delta_w     = {delta_w}")
    print(f"initial M0  = {mu_exact[0,0,0]:.6e}")
    print(f"initial M1  = {mu_exact[1,0,0]:.6e}")
    print(f"initial M2  = {mu_exact[2,0,0]:.6e}")
    print(f"max |E_M0|  = {np.max(np.abs(err_m0)):.6e}")
    print(f"max |E_M2|  = {np.max(np.abs(err_m2)):.6e}")
    print(f"max |B_M2|  = {np.max(np.abs(bias_cum_m2_mean)):.6e}")
    print(f"max |Ehat_M2| = {np.max(np.abs(bias_err_pred_m2_mean)):.6e}")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # axes[0].plot(t_vec, mu_exact[0, 0, :], "k-", label="Exact M0")
    # axes[0].plot(t_vec, mu_mean[0, 0, :], "r--", label="MC mean M0")
    # axes[0].set_ylabel("M0")
    # axes[0].grid(True, alpha=0.3)
    # axes[0].legend()
    
    axes[0].plot(t_vec, mu_exact[2, 0, :], "k-", label="Exact M2")
    axes[0].plot(t_vec, mu_mean[2, 0, :], "r--", label="MC mean M2")
    axes[0].set_ylabel("M2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_vec, err_m2, "r-", label="Actual error in M2")
    axes[1].plot(t_vec, bias_err_pred_m2_mean, "b--", label="Propagated bias-based prediction")
    axes[1].plot(t_vec, bias_cum_m2_mean, color="0.5", linestyle=":", label="Raw cumulative bias")
    axes[1].set_ylabel("M2 Error / Prediction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_vec, bias_ratio_m2_mean, "m-", label="Relative bias ratio r2")
    axes[2].axhline(0.0, color="k", linewidth=0.8)
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("r2")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
