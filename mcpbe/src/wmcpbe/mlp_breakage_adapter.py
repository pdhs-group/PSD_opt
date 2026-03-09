# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

from breakage_rate_model.base import BaseEnergyModel
from breakage_rate_model.mlp_model import MLPEnergyModel


class MLPBreakageRateAdapter:
    """Bridge a trained MLP energy model into the PBE breakage-rate interface.

    The trained model expects the same 10D feature vector used during training:
        [logV, log_gamma, log_NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
    """

    def __init__(
        self,
        *,
        model: Optional[MLPEnergyModel] = None,
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
    ):
        if model is None:
            if model_path is None:
                raise ValueError("Either 'model' or 'model_path' must be provided.")
            loaded = BaseEnergyModel.load(model_path, device="cpu")
            if not isinstance(loaded, MLPEnergyModel):
                raise TypeError(
                    f"Loaded model from {model_path} is not MLPEnergyModel (got {type(loaded)})"
                )
            loaded.device = torch.device("cpu")
            if getattr(loaded, "_net", None) is not None:
                loaded._net.to("cpu")
            self.model: MLPEnergyModel = loaded
        else:
            model.device = torch.device("cpu")
            if getattr(model, "_net", None) is not None:
                model._net.to("cpu")
            self.model = model

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

    @staticmethod
    def _coerce_str(value: Sequence[float] | np.ndarray) -> np.ndarray:
        str_values = np.asarray(value, dtype=float).reshape(-1)
        if str_values.size != 3:
            raise ValueError(f"STR must have length 3, got shape={np.asarray(value).shape}")
        return str_values.copy()

    def _get_lmc_params_from_pbe(
        self, pbe
    ) -> tuple[float, float, float, float, float, np.ndarray]:
        """Read LMC parameters from the solver, falling back to adapter defaults."""
        gamma = float(getattr(pbe, "lmc_gamma", self.gamma_default))
        NO_FRAG = float(getattr(pbe, "lmc_NO_FRAG", self.NO_FRAG_default))
        int_bre = float(getattr(pbe, "lmc_int_bre", self.int_bre_default))
        Df = float(getattr(pbe, "lmc_Df", self.Df_default))
        MAS = float(getattr(pbe, "lmc_MAS", self.MAS_default))
        STR = self._coerce_str(getattr(pbe, "lmc_STR", self.STR_default))
        return gamma, NO_FRAG, int_bre, Df, MAS, STR

    def _energy_in(self, V: np.ndarray) -> np.ndarray:
        """Compute the incident energy for a batch of particle volumes."""
        V = np.asarray(V, dtype=float)
        if self.energy_in_fn is not None:
            E_in = self.energy_in_fn(V)
            return np.asarray(E_in, dtype=float)
        return self.lambda_E * (V**self.energy_exp)

    def _build_features_batch(
        self,
        pbe,
        indices: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the MLP feature matrix for a set of active particles.

        Returns
        -------
        X : ndarray, shape (n, 10)
            Model features in the exact training order.
        V : ndarray, shape (n,)
            Normalized particle volume used later to compute incident energy.
        """
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
            A = v1 + v3
            V = A
            X1 = np.where(A > 0.0, v1 / A, 0.5)
        else:
            raise NotImplementedError(
                f"MLPBreakageRateAdapter currently only supports dim=1 or 2 (got dim={dim})"
            )

        V = V / self.A0_run
        gamma, NO_FRAG, int_bre, Df, MAS, STR = self._get_lmc_params_from_pbe(pbe)

        logV = np.log(np.maximum(V, 1e-30))
        log_gamma = np.log(gamma)
        log_NOFRAG = np.log(NO_FRAG)

        log_gamma_v = np.full_like(logV, log_gamma, dtype=float)
        log_nf_v = np.full_like(logV, log_NOFRAG, dtype=float)
        int_bre_v = np.full_like(logV, int_bre, dtype=float)
        Df_v = np.full_like(logV, Df, dtype=float)
        MAS_v = np.full_like(logV, MAS, dtype=float)
        STR0_v = np.full_like(logV, STR[0], dtype=float)
        STR1_v = np.full_like(logV, STR[1], dtype=float)
        STR2_v = np.full_like(logV, STR[2], dtype=float)

        X = np.stack(
            [
                logV,
                log_gamma_v,
                log_nf_v,
                int_bre_v,
                Df_v,
                MAS_v,
                X1,
                STR0_v,
                STR1_v,
                STR2_v,
            ],
            axis=1,
        )

        return X, V

    def compute_rates_full(self, pbe) -> np.ndarray:
        """Compute breakage rates for all active particles in the solver."""
        X, V = self._build_features_batch(pbe, indices=None)
        n = X.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=float)

        logE_need = self.model.predict(X)
        E_need = np.exp(np.asarray(logE_need, dtype=float))
        E_need = np.maximum(E_need, self.eps_E)

        E_in = self._energy_in(V)

        rates = E_in / E_need
        rates = np.maximum(rates, self.rate_min)
        if self.rate_max is not None:
            rates = np.minimum(rates, self.rate_max)

        return rates

    def compute_rate_single(self, pbe, i: int) -> float:
        """Compute the breakage rate for a single particle index."""
        a = int(getattr(pbe, "a_tot", 0))
        if i < 0 or i >= a:
            return 0.0

        X, V = self._build_features_batch(pbe, indices=[i])
        if X.shape[0] == 0:
            return 0.0

        logE_need = self.model.predict(X[0:1, :])
        E_need = float(np.exp(logE_need[0]))
        if E_need < self.eps_E:
            E_need = self.eps_E

        E_in = float(self._energy_in(np.array([V[0]], dtype=float))[0])
        rate = E_in / E_need
        if rate < self.rate_min:
            rate = self.rate_min
        if self.rate_max is not None and rate > self.rate_max:
            rate = self.rate_max

        return float(rate)
