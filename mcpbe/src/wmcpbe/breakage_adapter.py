# -*- coding: utf-8 -*-
"""Energy-surrogate bridge for the wmcpbe breakage-rate interface.

All supported energy surrogates receive the canonical full feature vector::

    [logV, log_gamma, log_NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]

Each surrogate selects its persisted active features internally.  This keeps
the PBE/LMC interface independent of the model family.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

from breakage_rate_model.ann_model import ANNEnergyModel
from breakage_rate_model.base import BaseEnergyModel
from breakage_rate_model.mlp_model import MLPEnergyModel
from breakage_rate_model.parametric_model import ParametricEnergyModel
from breakage_rate_model.powerlaw_separable import PowerLawSeparableModel


_MODEL_TYPES = {
    "mlp": MLPEnergyModel,
    "ann": ANNEnergyModel,
    "powerlaw": PowerLawSeparableModel,
    "parametric": ParametricEnergyModel,
}
_NEURAL_MODEL_TYPES = (MLPEnergyModel, ANNEnergyModel)


class BreakageRateAdapter:
    """Convert a trained energy surrogate into PBE single-particle rates.

    ``model_kind`` is deliberately explicit so a mismatched pickle fails
    during solver setup rather than being used as an unintended model family.
    ``model_logV_bounds`` refers to the normalized ``logV`` feature supplied
    to the surrogate, namely ``log(V / A0_run)``.  Current legacy pickles do
    not persist this range, so it must be configured here when extrapolation
    warnings are required for such a model.
    """

    def __init__(
        self,
        *,
        model_kind: str = "mlp",
        model: Optional[BaseEnergyModel] = None,
        model_path: Optional[str] = None,
        lambda_E: float = 1.0,
        energy_exp: float = 1.0,
        energy_in_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        gamma: float = 1.0,
        NO_FRAG: float = 4.0,
        int_bre: float = 0.0,
        Df: float = 2.5,
        MAS: float = 0.0,
        STR: Sequence[float] = (1.0, 1.0, 1.0),
        rate_min: float = 0.0,
        rate_max: Optional[float] = None,
        eps_E: float = 1e-30,
        A0_run: float = 1.0,
        model_logV_bounds: Optional[Sequence[float]] = None,
        warn_model_extrapolation: bool = True,
    ):
        normalized_kind = str(model_kind).lower().strip()
        if normalized_kind not in _MODEL_TYPES:
            raise ValueError(
                f"model_kind must be one of {tuple(_MODEL_TYPES)}, got {model_kind!r}."
            )
        self.model_kind = normalized_kind
        expected_type = _MODEL_TYPES[self.model_kind]

        if model is None:
            if model_path is None:
                raise ValueError("Either 'model' or 'model_path' must be provided.")
            model = BaseEnergyModel.load(model_path, device="cpu")
        if not isinstance(model, expected_type):
            raise TypeError(
                f"model_kind={self.model_kind!r} requires {expected_type.__name__}, "
                f"got {type(model).__name__}."
            )
        if not model.is_fitted:
            raise RuntimeError(f"{type(model).__name__} must be fitted before use in wmcpbe.")
        self.model: BaseEnergyModel = model
        self._move_neural_model_to_cpu()

        self.lambda_E = float(lambda_E)
        self.energy_exp = float(energy_exp)
        self.energy_in_fn = energy_in_fn

        self.gamma_default = float(gamma)
        self.NO_FRAG_default = float(NO_FRAG)
        self.int_bre_default = float(int_bre)
        self.Df_default = float(Df)
        self.MAS_default = float(MAS)
        self.STR_default = self._coerce_str(STR)

        self.rate_min = float(rate_min)
        self.rate_max = None if rate_max is None else float(rate_max)
        self.eps_E = float(eps_E)
        self.A0_run = float(A0_run)

        self.model_logV_bounds = self._coerce_logV_bounds(model_logV_bounds)
        self.warn_model_extrapolation = bool(warn_model_extrapolation)
        self._warned_below_model_range = False
        self._warned_above_model_range = False

    def _move_neural_model_to_cpu(self) -> None:
        """Keep only neural model inference on CPU inside the PBE process."""
        if isinstance(self.model, _NEURAL_MODEL_TYPES):
            self.model.device = torch.device("cpu")
            if self.model._net is not None:
                self.model._net.to("cpu")

    @staticmethod
    def _coerce_str(value: Sequence[float] | np.ndarray) -> np.ndarray:
        str_values = np.asarray(value, dtype=float).reshape(-1)
        if str_values.size != 3:
            raise ValueError(f"STR must have length 3, got shape={np.asarray(value).shape}")
        return str_values.copy()

    @staticmethod
    def _coerce_logV_bounds(
        value: Optional[Sequence[float]],
    ) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        bounds = np.asarray(value, dtype=float).reshape(-1)
        if bounds.size != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(
                "model_logV_bounds must be two finite increasing values "
                "for log(V / A0_run)."
            )
        return float(bounds[0]), float(bounds[1])

    def _get_lmc_params_from_pbe(
        self, pbe
    ) -> tuple[float, float, float, float, float, np.ndarray]:
        """Read runtime LMC parameters, retaining the adapter construction defaults."""
        gamma = float(getattr(pbe, "lmc_gamma", self.gamma_default))
        NO_FRAG = float(getattr(pbe, "lmc_NO_FRAG", self.NO_FRAG_default))
        int_bre = float(getattr(pbe, "lmc_int_bre", self.int_bre_default))
        Df = float(getattr(pbe, "lmc_Df", self.Df_default))
        MAS = float(getattr(pbe, "lmc_MAS", self.MAS_default))
        STR = self._coerce_str(getattr(pbe, "lmc_STR", self.STR_default))
        return gamma, NO_FRAG, int_bre, Df, MAS, STR

    def _energy_in(self, V: np.ndarray) -> np.ndarray:
        """Compute the incident energy for a batch of normalized volumes."""
        V = np.asarray(V, dtype=float)
        if self.energy_in_fn is not None:
            E_in = self.energy_in_fn(V)
            return np.asarray(E_in, dtype=float)
        return self.lambda_E * (V**self.energy_exp)

    def _warn_if_model_extrapolating(self, logV: np.ndarray) -> None:
        """Emit one warning per extrapolation direction without changing rates."""
        if not self.warn_model_extrapolation or self.model_logV_bounds is None:
            return
        lower, upper = self.model_logV_bounds
        logV = np.asarray(logV, dtype=float)
        if np.any(logV < lower) and not self._warned_below_model_range:
            print(
                "[BreakageRateAdapter][WARNING] Energy-model volume extrapolation "
                f"below its configured training range: requested log(V/A0_run) down to "
                f"{float(np.min(logV)):.6g}, training range [{lower:.6g}, {upper:.6g}]."
            )
            self._warned_below_model_range = True
        if np.any(logV > upper) and not self._warned_above_model_range:
            print(
                "[BreakageRateAdapter][WARNING] Energy-model volume extrapolation "
                f"above its configured training range: requested log(V/A0_run) up to "
                f"{float(np.max(logV)):.6g}, training range [{lower:.6g}, {upper:.6g}]."
            )
            self._warned_above_model_range = True

    def _build_features_batch(
        self,
        pbe,
        indices: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the canonical full feature matrix for active PBE particles."""
        dim = int(getattr(pbe, "dim", 1))
        a = int(getattr(pbe, "a_tot", 0))
        if a <= 0:
            return np.zeros((0, 10), dtype=float), np.zeros((0,), dtype=float)

        V_flat = np.asarray(pbe.V_flat, dtype=float)
        if indices is None:
            idx = np.arange(a, dtype=int)
        else:
            idx = np.asarray(indices, dtype=int)
            idx = idx[(idx >= 0) & (idx < a)]
        if idx.size == 0:
            return np.zeros((0, 10), dtype=float), np.zeros((0,), dtype=float)

        if dim == 1:
            V = V_flat[-1, idx]
            X1 = np.ones_like(V)
        elif dim == 2:
            v1 = V_flat[0, idx]
            v3 = V_flat[1, idx]
            V = v1 + v3
            X1 = np.where(V > 0.0, v1 / V, 0.5)
        else:
            raise NotImplementedError(
                f"BreakageRateAdapter only supports dim=1 or 2 (got dim={dim})"
            )

        V = V / self.A0_run
        gamma, NO_FRAG, int_bre, Df, MAS, STR = self._get_lmc_params_from_pbe(pbe)
        logV = np.log(np.maximum(V, 1e-30))
        self._warn_if_model_extrapolating(logV)

        X = np.stack(
            [
                logV,
                np.full_like(logV, np.log(gamma), dtype=float),
                np.full_like(logV, np.log(NO_FRAG), dtype=float),
                np.full_like(logV, int_bre, dtype=float),
                np.full_like(logV, Df, dtype=float),
                np.full_like(logV, MAS, dtype=float),
                X1,
                np.full_like(logV, STR[0], dtype=float),
                np.full_like(logV, STR[1], dtype=float),
                np.full_like(logV, STR[2], dtype=float),
            ],
            axis=1,
        )
        return X, V

    def compute_rates_full(self, pbe) -> np.ndarray:
        """Compute breakage rates for all active particles in the solver."""
        X, V = self._build_features_batch(pbe, indices=None)
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=float)

        E_need = np.exp(np.asarray(self.model.predict(X), dtype=float))
        E_need = np.maximum(E_need, self.eps_E)
        rates = self._energy_in(V) / E_need
        rates = np.maximum(rates, self.rate_min)
        if self.rate_max is not None:
            rates = np.minimum(rates, self.rate_max)
        return rates

    def compute_rate_single(self, pbe, i: int) -> float:
        """Compute the breakage rate for one active particle index."""
        a = int(getattr(pbe, "a_tot", 0))
        if i < 0 or i >= a:
            return 0.0

        X, V = self._build_features_batch(pbe, indices=[i])
        if X.shape[0] == 0:
            return 0.0

        E_need = max(float(np.exp(self.model.predict(X)[0])), self.eps_E)
        rate = float(self._energy_in(np.array([V[0]], dtype=float))[0]) / E_need
        rate = max(rate, self.rate_min)
        if self.rate_max is not None:
            rate = min(rate, self.rate_max)
        return float(rate)
