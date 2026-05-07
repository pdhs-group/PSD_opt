from __future__ import annotations

import math

import numpy as np


class MCPBETimeHelper:
    def _draw_time_multiplier(self) -> float:
        if not bool(self.exp_time_step):
            return 1.0
        u = max(float(self._rng.random()), 1e-300)
        return -math.log(u)

    @staticmethod
    def _log_mean_positive(x: float, y: float) -> float:
        if (not np.isfinite(x)) or (not np.isfinite(y)) or x <= 0.0 or y <= 0.0:
            return float("nan")
        if np.isclose(x, y, rtol=1e-12, atol=0.0):
            return 0.5 * (x + y)
        return (y - x) / math.log(y / x)

    def _dt_from_rate(self, rate: float) -> float:
        if (not np.isfinite(rate)) or rate <= 0.0:
            return float("inf")
        return self._draw_time_multiplier() / float(rate)

    def _build_rate_dt_strategy(self, rate_from_propensity):
        use_pair = bool(self.sum_prop_pair)

        def initial_dt(propensity: float) -> float:
            return self._dt_from_rate(rate_from_propensity(float(propensity)))

        if use_pair:
            def event_dt(propensity_before: float, propensity_after: float) -> float:
                rate_before = rate_from_propensity(float(propensity_before))
                rate_after = rate_from_propensity(float(propensity_after))
                return self._dt_from_rate(self._log_mean_positive(rate_before, rate_after))
        else:
            def event_dt(propensity_before: float, _propensity_after: float) -> float:
                return self._dt_from_rate(rate_from_propensity(float(propensity_before)))

        return initial_dt, event_dt

    def _build_agg_dt_strategy(self):
        return self._build_rate_dt_strategy(self._agg_event_rate_from_sum_prop)

    def _build_break_dt_strategy(self):
        return self._build_rate_dt_strategy(self._break_event_rate_from_sum_prop)

    def _build_mix_dt_strategy(self):
        return self._build_rate_dt_strategy(lambda total_rate: max(float(total_rate), 0.0))

    def _agg_event_rate_from_sum_prop(self, sum_prop: float) -> float:
        a_tot = int(self.a_tot)
        if a_tot < 2 or sum_prop <= 0.0:
            return 0.0
        return (
            float(a_tot)
            * float(sum_prop)
            / (2.0 * float(self.Vc) * (float(a_tot) - 1.0))
        )

    @staticmethod
    def _break_event_rate_from_sum_prop(sum_prop: float) -> float:
        return max(float(sum_prop), 0.0)

    def _mix_event_rate_from_sum_prop(self, sum_prop_agg: float, sum_prop_break: float) -> float:
        return (
            self._agg_event_rate_from_sum_prop(float(sum_prop_agg))
            + self._break_event_rate_from_sum_prop(float(sum_prop_break))
        )

    def _prepare_agg_delta_config(self) -> float:
        dW_const = float(self.agg_dW_max)
        if (not np.isfinite(dW_const)) or dW_const <= 0.0:
            raise ValueError("`agg_dW_max` must be a positive finite value.")

        self._delta_agg = np.zeros(int(self._cap), dtype=float)
        self._agg_dW_const = dW_const
        return dW_const

    @staticmethod
    def _delta_from_weights(W: np.ndarray, *, dW_const: float) -> np.ndarray:
        delta = np.minimum(np.asarray(W, dtype=float), float(dW_const))
        return np.where(np.isfinite(delta) & (delta > 0.0), delta, 0.0)

    def _update_delta_single(self, i: int, *, attr_name: str, dW_const: float) -> float:
        i = int(i)
        if i < 0 or i >= int(self.a_tot):
            raise IndexError("delta update index out of active particle range.")

        Wi = float(self.W[i])
        delta = min(float(dW_const), Wi) if (np.isfinite(Wi) and Wi > 0.0) else 0.0
        getattr(self, attr_name)[i] = delta
        return float(delta)
