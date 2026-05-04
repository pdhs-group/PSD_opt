from __future__ import annotations

import math

import numpy as np


def ensure_delta_array(solver, attr_name: str) -> np.ndarray:
    cap = int(solver._cap)
    arr = np.zeros(cap, dtype=float)
    if attr_name == "_delta_agg":
        solver._delta_agg = arr
    elif attr_name == "_delta_break":
        solver._delta_break = arr
    else:
        raise ValueError(f"Unsupported delta array name: {attr_name!r}")
    return arr


def prepare_process_delta_config(
    solver,
    *,
    process_name: str,
    dW_attr_name: str,
    cache_attr_name: str,
) -> float:
    if dW_attr_name == "agg_dW_max":
        dW_const = float(solver.agg_dW_max)
    elif dW_attr_name == "break_dW_max":
        dW_const = float(solver.break_dW_max)
    else:
        raise ValueError(f"Unsupported delta config attribute: {dW_attr_name!r}")

    if (not np.isfinite(dW_const)) or dW_const <= 0.0:
        raise ValueError(f"`{dW_attr_name}` must be a positive finite value.")
    setattr(solver, cache_attr_name, dW_const)
    return dW_const


def delta_from_weights(W: np.ndarray, dW_const: float) -> np.ndarray:
    delta = np.minimum(np.asarray(W, dtype=float), float(dW_const))
    delta = np.where(np.isfinite(delta) & (delta > 0.0), delta, 0.0)
    return delta


def update_delta_single(
    solver,
    i: int,
    *,
    attr_name: str,
    dW_const: float,
) -> float:
    if attr_name == "_delta_agg":
        arr = solver._delta_agg
    elif attr_name == "_delta_break":
        arr = solver._delta_break
    else:
        raise ValueError(f"Unsupported delta array name: {attr_name!r}")

    a_tot = int(solver.a_tot)
    if i < 0 or i >= a_tot:
        raise IndexError("delta update index out of active particle range.")
    if arr.shape[0] < int(solver._cap):
        raise ValueError(f"`{attr_name}` capacity is smaller than solver._cap.")
    Wi = float(solver.W[i])
    delta = min(float(dW_const), Wi) if (np.isfinite(Wi) and Wi > 0.0) else 0.0
    arr[i] = delta
    return float(delta)


def log_mean_positive(x: float, y: float) -> float:
    if (not np.isfinite(x)) or (not np.isfinite(y)) or x <= 0.0 or y <= 0.0:
        return float("nan")
    if np.isclose(x, y, rtol=1e-12, atol=0.0):
        return 0.5 * (x + y)
    return (y - x) / math.log(y / x)


def dt_agg_from_sum_prop(a_tot: int, Vc: float, sum_prop: float) -> float:
    if a_tot < 2 or sum_prop <= 0.0:
        return float("inf")
    return 2.0 * float(Vc) * (a_tot - 1) / (a_tot * sum_prop)


def dt_agg_from_sum_prop_pair(a_tot: int, Vc: float, sum_prop_before: float, sum_prop_after: float) -> float:
    if a_tot < 2:
        return float("inf")
    prop_eff = log_mean_positive(float(sum_prop_before), float(sum_prop_after))
    if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
        return float("inf")
    return 2.0 * float(Vc) * (a_tot - 1) / (a_tot * prop_eff)


def dt_break_from_sum_prop(sum_prop: float) -> float:
    if sum_prop <= 0.0:
        return float("inf")
    return 1.0 / sum_prop

def dt_break_from_sum_prop_pair(sum_prop_before: float, sum_prop_after: float) -> float:
    prop_eff = log_mean_positive(float(sum_prop_before), float(sum_prop_after))
    if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
        return float("inf")
    return 1.0 / prop_eff


def agg_rate_from_sum_prop(a_tot: int, Vc: float, sum_prop: float) -> float:
    if a_tot < 2 or sum_prop <= 0.0:
        return 0.0
    return float(a_tot) * float(sum_prop) / (2.0 * float(Vc) * (float(a_tot) - 1.0))


def mix_total_rate_from_sum_prop(a_tot: int, Vc: float, sum_prop_agg: float, sum_prop_break: float) -> float:
    return agg_rate_from_sum_prop(a_tot, Vc, sum_prop_agg) + max(float(sum_prop_break), 0.0)

class MCPBETimeHelper:
    def _draw_time_multiplier(self) -> float:
        if not bool(self.exp_time_step):
            return 1.0
        u = max(float(self._rng.random()), 1e-300)
        return -math.log(u)

    def _build_agg_dt_strategy(self):
        use_pair = bool(self.sum_prop_pair)

        def initial_dt(sum_prop: float) -> float:
            return self._dt_agg_from_sum_prop(float(sum_prop)) * self._draw_time_multiplier()

        if use_pair:
            def event_dt(sum_prop_before: float, sum_prop_after: float) -> float:
                return self._dt_agg_from_sum_prop_pair(
                    float(sum_prop_before), float(sum_prop_after)
                ) * self._draw_time_multiplier()
        else:
            def event_dt(sum_prop_before: float, _sum_prop_after: float) -> float:
                return self._dt_agg_from_sum_prop(float(sum_prop_before)) * self._draw_time_multiplier()

        return initial_dt, event_dt

    def _build_break_dt_strategy(self):
        use_pair = bool(self.sum_prop_pair)

        def initial_dt(sum_prop: float) -> float:
            return self._dt_break_from_sum_prop(float(sum_prop)) * self._draw_time_multiplier()

        if use_pair:
            def event_dt(sum_prop_before: float, sum_prop_after: float) -> float:
                return self._dt_break_from_sum_prop_pair(
                    float(sum_prop_before), float(sum_prop_after)
                ) * self._draw_time_multiplier()
        else:
            def event_dt(sum_prop_before: float, _sum_prop_after: float) -> float:
                return self._dt_break_from_sum_prop(float(sum_prop_before)) * self._draw_time_multiplier()

        return initial_dt, event_dt

    def _build_mix_dt_strategy(self):
        use_pair = bool(self.sum_prop_pair)

        def initial_dt(total_rate: float) -> float:
            if total_rate <= 0.0:
                return float("inf")
            return (1.0 / float(total_rate)) * self._draw_time_multiplier()

        if use_pair:
            def event_dt(total_rate_before: float, total_rate_after: float) -> float:
                prop_eff = self._log_mean_positive(float(total_rate_before), float(total_rate_after))
                if (not np.isfinite(prop_eff)) or prop_eff <= 0.0:
                    return float("inf")
                return (1.0 / prop_eff) * self._draw_time_multiplier()
        else:
            def event_dt(total_rate_before: float, _total_rate_after: float) -> float:
                if total_rate_before <= 0.0:
                    return float("inf")
                return (1.0 / float(total_rate_before)) * self._draw_time_multiplier()

        return initial_dt, event_dt

    def _prepare_agg_delta_config(self) -> float:
        ensure_delta_array(self, "_delta_agg")
        return prepare_process_delta_config(
            self,
            process_name="agglomeration",
            dW_attr_name="agg_dW_max",
            cache_attr_name="_agg_dW_const",
        )

    def _prepare_break_delta_config(self) -> float:
        ensure_delta_array(self, "_delta_break")
        return prepare_process_delta_config(
            self,
            process_name="breakage",
            dW_attr_name="break_dW_max",
            cache_attr_name="_break_dW_const",
        )

    def _delta_from_weights(self, W: np.ndarray, *, dW_const: float) -> np.ndarray:
        return delta_from_weights(W, dW_const)

    def _update_delta_single(self, i: int, *, attr_name: str, dW_const: float) -> float:
        return update_delta_single(self, i, attr_name=attr_name, dW_const=dW_const)

    @staticmethod
    def _log_mean_positive(x: float, y: float) -> float:
        return log_mean_positive(x, y)

    def _dt_agg(self) -> float:
        a = self.a_tot
        if a < 2:
            return float("inf")
        return self._dt_agg_from_sum_prop(float(np.sum(self._r_agg[:a])))

    def _dt_agg_from_sum_prop(self, sum_prop: float) -> float:
        return dt_agg_from_sum_prop(int(self.a_tot), float(self.Vc), float(sum_prop))

    def _dt_agg_from_sum_prop_pair(self, sum_prop_before: float, sum_prop_after: float) -> float:
        return dt_agg_from_sum_prop_pair(
            int(self.a_tot), float(self.Vc), float(sum_prop_before), float(sum_prop_after)
        )

    def _dt_break(self) -> float:
        if self.a_tot <= 0:
            return float("inf")
        return self._dt_break_from_sum_prop(float(np.sum(self._break_rate[:self.a_tot])))

    def _dt_break_from_sum_prop(self, sum_prop: float) -> float:
        return dt_break_from_sum_prop(float(sum_prop))

    def _dt_break_from_sum_prop_pair(self, sum_prop_before: float, sum_prop_after: float) -> float:
        return dt_break_from_sum_prop_pair(float(sum_prop_before), float(sum_prop_after))

    def _agg_rate_from_sum_prop(self, sum_prop: float) -> float:
        return agg_rate_from_sum_prop(int(self.a_tot), float(self.Vc), float(sum_prop))

    def _mix_total_rate_from_sum_prop(self, sum_prop_agg: float, sum_prop_break: float) -> float:
        return mix_total_rate_from_sum_prop(
            int(self.a_tot), float(self.Vc), float(sum_prop_agg), float(sum_prop_break)
        )
