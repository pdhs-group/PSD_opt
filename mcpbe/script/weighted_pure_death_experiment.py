from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DeathRunResult:
    t_vec: np.ndarray
    weights_over_time: np.ndarray
    exact_over_time: np.ndarray


class WeightedPureDeathMC:
    """
    Minimal weighted Monte Carlo experiment for a pure death process.

    Each compute particle i carries weight W_i representing the number of
    real particles in that class. A packet event consumes delta_W of class i:

        W_i <- W_i - min(delta_W, W_i)

    Event propensities follow the same weighted-MC logic as WMCPBE breakage:

        a_i = W_i * S_i

    where S_i is the single-particle decay rate of class i.

    Optionally, a delta-corrected packet formulation can be enabled:

        delta_i = min(delta_W, W_i)
        lambda_i = W_i * S_i / delta_i

    Then one event of class i consumes exactly delta_i real particles and the
    class-selection distribution is corrected for the varying packet size.
    """

    def __init__(
        self,
        W0: np.ndarray,
        S: np.ndarray,
        t_vec: np.ndarray,
        delta_w: float,
        use_delta_packet_correction: bool = False,
        seed: int | None = None,
    ) -> None:
        self.W0 = np.asarray(W0, dtype=float).copy()
        self.S = np.asarray(S, dtype=float).copy()
        self.t_vec = np.asarray(t_vec, dtype=float).copy()
        self.delta_w = float(delta_w)
        self.use_delta_packet_correction = bool(use_delta_packet_correction)
        self.rng = np.random.default_rng(seed)

        if self.W0.ndim != 1 or self.S.ndim != 1 or self.W0.size != self.S.size:
            raise ValueError("W0 and S must be 1D arrays with the same length.")
        if np.any(self.W0 < 0.0) or np.any(~np.isfinite(self.W0)):
            raise ValueError("W0 must be finite and nonnegative.")
        if np.any(self.S < 0.0) or np.any(~np.isfinite(self.S)):
            raise ValueError("S must be finite and nonnegative.")
        if self.delta_w <= 0.0 or not np.isfinite(self.delta_w):
            raise ValueError("delta_w must be positive and finite.")
        if self.t_vec.ndim != 1 or self.t_vec.size == 0 or not np.all(np.diff(self.t_vec) > 0.0):
            raise ValueError("t_vec must be a strictly increasing 1D array.")

        self.n_classes = int(self.W0.size)

    @staticmethod
    def _log_mean_positive(x: float, y: float) -> float:
        if (not np.isfinite(x)) or (not np.isfinite(y)) or x <= 0.0 or y <= 0.0:
            return float("nan")
        if np.isclose(x, y, rtol=1e-12, atol=0.0):
            return 0.5 * (x + y)
        return (y - x) / math.log(y / x)

    def _propensities(self, W: np.ndarray, delta: np.ndarray | None = None) -> np.ndarray:
        if self.use_delta_packet_correction:
            if delta is None:
                delta = self._event_deltas(W)
            prop = np.divide(
                W * self.S,
                delta,
                out=np.zeros_like(W, dtype=float),
                where=delta > 0.0,
            )
        else:
            prop = W * self.S
        prop[prop < 0.0] = 0.0
        return prop

    def _event_deltas(self, W: np.ndarray) -> np.ndarray:
        delta = np.minimum(self.delta_w, W)
        delta = np.where(np.isfinite(delta) & (delta > 0.0), delta, 0.0)
        return delta

    def _sample_class(self, prop: np.ndarray) -> int:
        total = float(np.sum(prop))
        if total <= 0.0:
            return -1
        u = float(self.rng.random()) * total
        cdf = np.cumsum(prop)
        return int(np.searchsorted(cdf, u, side="right"))

    def _do_one_event(self, W: np.ndarray, delta: np.ndarray | None = None) -> tuple[bool, float]:
        if self.use_delta_packet_correction:
            if delta is None:
                raise ValueError("delta array must be provided when delta packet correction is enabled.")
            delta_before = delta
        else:
            delta_before = self._event_deltas(W)

        prop_before = self._propensities(W, delta_before)
        sum_before = float(np.sum(prop_before))
        if sum_before <= 0.0:
            return False, float("inf")

        i = self._sample_class(prop_before)
        if i < 0:
            return False, float("inf")

        dW = float(delta_before[i])
        if dW <= 0.0:
            return False, float("inf")

        W[i] -= dW
        if W[i] < 0.0:
            W[i] = 0.0

        if self.use_delta_packet_correction:
            delta[i] = min(self.delta_w, float(W[i])) if W[i] > 0.0 else 0.0

        prop_after = self._propensities(W, delta if self.use_delta_packet_correction else None)
        sum_after = float(np.sum(prop_after))
        prop_eff = self._log_mean_positive(sum_before, sum_after)

        if self.use_delta_packet_correction:
            if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
                dt = 1.0 / sum_before
            else:
                dt = 1.0 / prop_eff
        else:
            if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
                dt = dW / sum_before
            else:
                dt = dW / prop_eff
        return True, float(dt)

    def exact_solution(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.W0[:, None] * np.exp(-self.S[:, None] * t[None, :])

    def solve(self, max_events: int = int(1e7)) -> DeathRunResult:
        W = self.W0.copy()
        delta = self._event_deltas(W) if self.use_delta_packet_correction else None
        saved = np.zeros((self.n_classes, self.t_vec.size), dtype=float)
        saved[:, 0] = W

        current_time = 0.0
        next_save_idx = 1
        event_count = 0

        while current_time < float(self.t_vec[-1]) and event_count < max_events:
            W_prev = W.copy()
            t_prev = current_time

            ok, dt = self._do_one_event(W, delta=delta)
            if not ok or not np.isfinite(dt):
                break

            current_time += dt
            while next_save_idx < self.t_vec.size and current_time >= self.t_vec[next_save_idx]:
                saved[:, next_save_idx] = W_prev
                next_save_idx += 1

            event_count += 1
            if float(np.sum(W)) <= 0.0:
                break

        while next_save_idx < self.t_vec.size:
            saved[:, next_save_idx] = W
            next_save_idx += 1

        exact = self.exact_solution(self.t_vec)
        return DeathRunResult(
            t_vec=self.t_vec.copy(),
            weights_over_time=saved,
            exact_over_time=exact,
        )

    def solve_repeats(self, n_runs: int, base_seed: int = 42, max_events: int = int(1e7)) -> list[DeathRunResult]:
        results: list[DeathRunResult] = []
        for k in range(n_runs):
            child = WeightedPureDeathMC(
                W0=self.W0,
                S=self.S,
                t_vec=self.t_vec,
                delta_w=self.delta_w,
                use_delta_packet_correction=self.use_delta_packet_correction,
                seed=base_seed + k,
            )
            results.append(child.solve(max_events=max_events))
        return results


def plot_results(results: list[DeathRunResult], class_indices: list[int]) -> None:
    if len(results) == 0:
        return

    t = results[0].t_vec
    mc_stack = np.stack([r.weights_over_time for r in results], axis=0)
    mc_mean = np.mean(mc_stack, axis=0)
    exact = results[0].exact_over_time

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for idx in class_indices:
        axes[0].plot(t, exact[idx], "-", label=f"Exact class {idx}")
        axes[0].plot(t, mc_mean[idx], "--", label=f"MC mean class {idx}")
    axes[0].set_ylabel("Weight")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    total_exact = np.sum(exact, axis=0)
    total_mc = np.sum(mc_mean, axis=0)
    axes[1].plot(t, total_exact, "-", label="Exact total weight")
    axes[1].plot(t, total_mc, "--", label="MC mean total weight")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("Total Weight")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    # -------------------------
    # Manual experiment inputs
    # -------------------------
    n_classes = 12
    t_vec = np.linspace(0.0, 24.0, 21)
    delta_w = 10.0
    use_delta_packet_correction = True
    n_runs = 1000
    max_events = int(2e6)

    class_id = np.arange(n_classes, dtype=float)

    # Example initial weights and class-dependent decay rates.
    W0 = 5.0 + 10.0 * class_id
    S = 0.08 + 0.015 * class_id

    solver = WeightedPureDeathMC(
        W0=W0,
        S=S,
        t_vec=t_vec,
        delta_w=delta_w,
        use_delta_packet_correction=use_delta_packet_correction,
        seed=42,
    )
    results = solver.solve_repeats(n_runs=n_runs, base_seed=42, max_events=max_events)

    mc_stack = np.stack([r.weights_over_time for r in results], axis=0)
    mc_mean = np.mean(mc_stack, axis=0)
    exact = results[0].exact_over_time

    max_abs_err = float(np.max(np.abs(mc_mean - exact)))
    max_rel_err = float(np.max(np.abs(mc_mean - exact) / np.maximum(exact, 1e-12)))

    print("Weighted pure death experiment")
    print(f"n_classes   = {n_classes}")
    print(f"delta_w     = {delta_w}")
    print(f"delta_fix   = {use_delta_packet_correction}")
    print(f"n_runs      = {n_runs}")
    print(f"max_abs_err = {max_abs_err:.6e}")
    print(f"max_rel_err = {max_rel_err:.6e}")

    # plot_results(results, class_indices=[0, n_classes // 2, n_classes - 1])
    plot_results(results, class_indices=[0, 1, 2])

if __name__ == "__main__":
    main()
