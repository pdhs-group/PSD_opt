# -*- coding: utf-8 -*-
"""Mixed/pure energy-surrogate bridge for wmcpbe breakage rates."""

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
_PURE_ACTIVE_FEATURE_NAMES = ("logV", "log_gamma")


class BreakageRateAdapter:
    """Convert mixed/pure energy models into PBE single-particle rates.

    All callers supply the canonical ten-feature vector.  Mixed particles use
    the mixed model output ``E/S0``.  Pure phase-1/phase-2 particles use the
    shared pure reference model scaled by ``STR0/S0`` or ``STR2/S0``.
    """

    def __init__(
        self,
        *,
        mixed_model_kind: str,
        pure_model_kind: str,
        mixed_model: Optional[BaseEnergyModel] = None,
        mixed_model_path: Optional[str] = None,
        pure_model: Optional[BaseEnergyModel] = None,
        pure_model_path: Optional[str] = None,
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
        mixed_model_logV_bounds: Optional[Sequence[float]] = None,
        pure_model_logV_bounds: Optional[Sequence[float]] = None,
        warn_model_extrapolation: bool = True,
    ):
        self.mixed_model_kind, self.mixed_model = self._load_model(
            "mixed", mixed_model_kind, mixed_model, mixed_model_path
        )
        self.pure_model_kind, self.pure_model = self._load_model(
            "pure", pure_model_kind, pure_model, pure_model_path
        )
        self._validate_model_contracts()
        self._move_neural_model_to_cpu(self.mixed_model)
        self._move_neural_model_to_cpu(self.pure_model)

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
        if not np.isfinite(self.A0_run) or self.A0_run <= 0.0:
            raise ValueError("A0_run must be finite and positive.")
        if not np.isfinite(self.eps_E) or self.eps_E <= 0.0:
            raise ValueError("eps_E must be finite and positive.")

        self.mixed_model_logV_bounds = self._coerce_logV_bounds(mixed_model_logV_bounds)
        self.pure_model_logV_bounds = self._coerce_logV_bounds(pure_model_logV_bounds)
        self.warn_model_extrapolation = bool(warn_model_extrapolation)
        self._warned_ranges: set[tuple[str, str]] = set()

    @staticmethod
    def _load_model(
        label: str,
        model_kind: str,
        model: Optional[BaseEnergyModel],
        model_path: Optional[str],
    ) -> tuple[str, BaseEnergyModel]:
        kind = str(model_kind).lower().strip()
        if kind not in _MODEL_TYPES:
            raise ValueError(f"{label}_model_kind must be one of {tuple(_MODEL_TYPES)}, got {model_kind!r}.")
        if (model is None) == (model_path is None):
            raise ValueError(f"Provide exactly one of {label}_model or {label}_model_path.")
        loaded = model if model is not None else BaseEnergyModel.load(model_path, device="cpu")
        expected_type = _MODEL_TYPES[kind]
        if not isinstance(loaded, expected_type):
            raise TypeError(f"{label}_model_kind={kind!r} requires {expected_type.__name__}, got {type(loaded).__name__}.")
        if not loaded.is_fitted:
            raise RuntimeError(f"{label} energy model must be fitted before use in wmcpbe.")
        return kind, loaded

    def _validate_model_contracts(self) -> None:
        if self.mixed_model.strength_normalization != "geometric_mean_relative":
            raise ValueError("mixed energy model must use strength_normalization='geometric_mean_relative'.")
        if self.pure_model.strength_normalization != "none":
            raise ValueError("pure energy model must use strength_normalization='none'.")
        if tuple(self.pure_model.active_feature_names) != _PURE_ACTIVE_FEATURE_NAMES:
            raise ValueError("pure energy model must activate exactly ('logV', 'log_gamma').")

    @staticmethod
    def _move_neural_model_to_cpu(model: BaseEnergyModel) -> None:
        if isinstance(model, _NEURAL_MODEL_TYPES):
            model.device = torch.device("cpu")
            if model._net is not None:
                model._net.to("cpu")

    @staticmethod
    def _coerce_str(value: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.size != 3 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("STR must contain three finite positive values.")
        return values.copy()

    @staticmethod
    def _coerce_logV_bounds(value: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        bounds = np.asarray(value, dtype=float).reshape(-1)
        if bounds.size != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError("model logV bounds must be two finite increasing log(V / A0_run) values.")
        return float(bounds[0]), float(bounds[1])

    def _get_lmc_params_from_pbe(self, pbe) -> tuple[float, float, float, float, float, np.ndarray]:
        gamma = float(pbe.lmc_gamma)
        NO_FRAG = float(pbe.lmc_NO_FRAG)
        int_bre = float(pbe.lmc_int_bre)
        Df = float(pbe.lmc_Df)
        MAS = float(pbe.lmc_mixed_MAS)
        STR = self._coerce_str(pbe.lmc_STR)
        return gamma, NO_FRAG, int_bre, Df, MAS, STR

    def _energy_in(self, V: np.ndarray) -> np.ndarray:
        values = np.asarray(V, dtype=float)
        if self.energy_in_fn is not None:
            return np.asarray(self.energy_in_fn(values), dtype=float)
        return self.lambda_E * values**self.energy_exp

    def _warn_if_model_extrapolating(self, label: str, logV: np.ndarray) -> None:
        if not self.warn_model_extrapolation:
            return
        bounds = self.mixed_model_logV_bounds if label == "mixed" else self.pure_model_logV_bounds
        if bounds is None:
            return
        lower, upper = bounds
        if np.any(logV < lower) and (label, "below") not in self._warned_ranges:
            print(f"[BreakageRateAdapter][WARNING] {label} energy-model volume extrapolation below its configured training range: requested log(V/A0_run) down to {float(np.min(logV)):.6g}, training range [{lower:.6g}, {upper:.6g}].")
            self._warned_ranges.add((label, "below"))
        if np.any(logV > upper) and (label, "above") not in self._warned_ranges:
            print(f"[BreakageRateAdapter][WARNING] {label} energy-model volume extrapolation above its configured training range: requested log(V/A0_run) up to {float(np.max(logV)):.6g}, training range [{lower:.6g}, {upper:.6g}].")
            self._warned_ranges.add((label, "above"))

    def _build_features_batch(self, pbe, indices: Optional[Sequence[int]] = None) -> tuple[np.ndarray, np.ndarray]:
        dim = int(pbe.dim)
        a = int(pbe.a_tot)
        if a <= 0:
            return np.empty((0, 10), dtype=float), np.empty(0, dtype=float)
        idx = np.arange(a, dtype=int) if indices is None else np.asarray(indices, dtype=int)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= a):
            raise IndexError("Requested breakage-rate index is outside the active particle range.")
        V_flat = np.asarray(pbe.V_flat, dtype=float)
        if dim == 1:
            V = V_flat[-1, idx]
            X1 = np.ones_like(V)
        elif dim == 2:
            phase_1, phase_2 = V_flat[0, idx], V_flat[1, idx]
            V = phase_1 + phase_2
            if np.any(V <= 0.0):
                raise ValueError("Active 2D particles must have strictly positive total volume.")
            X1 = phase_1 / V
        else:
            raise NotImplementedError(f"BreakageRateAdapter only supports dim=1 or 2 (got dim={dim}).")
        if not np.all(np.isfinite(V)) or np.any(V <= 0.0) or not np.all(np.isfinite(X1)) or np.any((X1 < 0.0) | (X1 > 1.0)):
            raise ValueError("Active particle volumes and phase fractions must be finite and physical.")
        V = V / self.A0_run
        gamma, NO_FRAG, int_bre, Df, MAS, STR = self._get_lmc_params_from_pbe(pbe)
        if not np.isfinite(gamma) or gamma <= 0.0 or not np.isfinite(NO_FRAG) or NO_FRAG <= 0.0:
            raise ValueError("gamma and NO_FRAG must be finite and positive.")
        logV = np.log(V)
        X = np.stack((logV, np.full_like(logV, np.log(gamma)), np.full_like(logV, np.log(NO_FRAG)), np.full_like(logV, int_bre), np.full_like(logV, Df), np.full_like(logV, MAS), X1, np.full_like(logV, STR[0]), np.full_like(logV, STR[1]), np.full_like(logV, STR[2])), axis=1)
        return X, V

    def _predict_log_energy(self, X: np.ndarray) -> np.ndarray:
        X1 = X[:, 6]
        mixed_mask = (X1 > 0.0) & (X1 < 1.0)
        pure_1_mask = X1 == 1.0
        pure_2_mask = X1 == 0.0
        if not np.all(mixed_mask | pure_1_mask | pure_2_mask):
            raise ValueError("X1 must be exactly 0 or 1 for pure particles, or strictly between them for mixed particles.")
        log_energy = np.empty(X.shape[0], dtype=float)
        if np.any(mixed_mask):
            self._warn_if_model_extrapolating("mixed", X[mixed_mask, 0])
            log_energy[mixed_mask] = np.asarray(self.mixed_model.predict(X[mixed_mask]), dtype=float)
        if np.any(pure_1_mask | pure_2_mask):
            pure_mask = pure_1_mask | pure_2_mask
            self._warn_if_model_extrapolating("pure", X[pure_mask, 0])
            log_reference = np.asarray(self.pure_model.predict(X[pure_mask]), dtype=float)
            log_s0 = np.mean(np.log(X[pure_mask, 7:10]), axis=1)
            log_scale = np.where(pure_1_mask[pure_mask], np.log(X[pure_mask, 7]) - log_s0, np.log(X[pure_mask, 9]) - log_s0)
            log_energy[pure_mask] = log_reference + log_scale
        if not np.all(np.isfinite(log_energy)):
            raise FloatingPointError("Energy model produced non-finite log-energy predictions.")
        return log_energy

    def compute_rates_full(self, pbe) -> np.ndarray:
        X, V = self._build_features_batch(pbe)
        if X.shape[0] == 0:
            return np.empty(0, dtype=float)
        E_need = np.maximum(np.exp(self._predict_log_energy(X)), self.eps_E)
        rates = self._energy_in(V) / E_need
        if not np.all(np.isfinite(rates)):
            raise FloatingPointError("Computed non-finite energy-model breakage rates.")
        rates = np.maximum(rates, self.rate_min)
        return np.minimum(rates, self.rate_max) if self.rate_max is not None else rates

    def compute_rate_single(self, pbe, i: int) -> float:
        X, V = self._build_features_batch(pbe, indices=[i])
        E_need = max(float(np.exp(self._predict_log_energy(X)[0])), self.eps_E)
        rate = max(float(self._energy_in(V)[0]) / E_need, self.rate_min)
        return min(rate, self.rate_max) if self.rate_max is not None else rate
