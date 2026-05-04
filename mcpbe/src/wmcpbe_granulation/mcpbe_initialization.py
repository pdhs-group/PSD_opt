from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass
class _InitialParticlePayload:
    V: np.ndarray
    W: Optional[np.ndarray] = None
    source: str = "model"
    weights_explicit: bool = False


class InitialParticleMixin:
    def _prepare_initial_counts(self, init_Vc: bool, require_model_init: bool) -> Optional[int]:
        """Prepare derived counts used by model-based initialization."""
        if not init_Vc and not require_model_init:
            return None

        self._validate_input_arrays()
        self.c = np.asarray(self.c, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.PGV = np.asarray(self.PGV)
        self.SIG = np.asarray(self.SIG, dtype=float)
        self.v = (self.x ** 3) * math.pi / 6.0
        self.n = np.round(self.c / self.v)
        self.n0 = float(np.sum(self.n))
        if self.n0 <= 0:
            raise ValueError("Total primary particle count `n0` must be > 0 (check c and x).")

        if init_Vc:
            self.Vc = self.a0 / self.n0
        else:
            self.Vc = float(self.Vc)

        self.a = np.round(self.n * self.Vc).astype(int)
        total_cols = int(np.sum(self.a))
        if require_model_init and total_cols <= 0:
            raise ValueError("No particles to initialize (sum(a) == 0). Check c/x/PGV/SIG.")
        return total_cols

    def _build_initial_particle_payload(
        self,
        total_cols: Optional[int],
        V_flat: Optional[np.ndarray],
        W_init: Optional[np.ndarray],
    ) -> _InitialParticlePayload:
        if V_flat is None:
            if total_cols is None:
                raise ValueError("Model-based initialization requires derived particle counts.")
            V = self._build_initial_particles_from_model(total_cols)
            return _InitialParticlePayload(V=V, source="model", weights_explicit=False)

        V = self._build_initial_particles_from_arrays(V_flat)
        return _InitialParticlePayload(
            V=V,
            W=W_init,
            source="array",
            weights_explicit=(W_init is not None),
        )

    def _build_initial_particles_from_model(self, total_cols: int) -> np.ndarray:
        dim = int(self.dim)
        V_init = np.zeros((dim + 1, int(total_cols)), dtype=float)
        cnt = 0
        for i in range(dim):
            ai = int(self.a[i])
            if ai <= 0:
                continue
            V_init[i, cnt : cnt + ai] = self._sample_initial_component_volumes(i, ai)
            cnt += ai

        V_init[-1, :] = np.sum(V_init[:dim, :], axis=0)
        return V_init

    def _sample_initial_component_volumes(self, i: int, size: int) -> np.ndarray:
        pgv = str(self.PGV[i]).strip().lower()
        if pgv == "mono":
            return np.full(size, float(self.v[i]), dtype=float)
        if pgv == "norm":
            mu = float(self.v[i])
            sig = float(self.SIG[i]) * mu
            return self._rng.normal(mu, sig, size)
        if pgv == "weibull":
            return self._rng.weibull(2.0, size) * (float(self.SIG[i]) * float(self.v[i]))
        raise ValueError(f"Unsupported PGV[{i}]='{pgv}'. Use 'mono' | 'norm' | 'weibull'.")

    def _build_initial_particles_from_arrays(self, V_flat: np.ndarray) -> np.ndarray:
        V = np.asarray(V_flat, dtype=float)
        if V.ndim != 2:
            raise ValueError("V_flat must be a 2D array with shape (dim+1, N).")
        expected_rows = int(self.dim) + 1
        if V.shape[0] != expected_rows:
            raise ValueError(f"V_flat must have shape ({expected_rows}, N), got {V.shape}.")
        return V

    def _normalize_initial_particle_payload(
        self,
        payload: _InitialParticlePayload,
    ) -> _InitialParticlePayload:
        V = np.asarray(payload.V, dtype=float)
        if V.ndim != 2:
            raise ValueError("Initial particle volumes must be a 2D array.")
        if V.shape[0] != int(self.dim) + 1:
            raise ValueError(
                f"Initial particle volumes must have {int(self.dim) + 1} rows, got {V.shape[0]}."
            )

        keep = np.all(np.isfinite(V), axis=0) & (V[-1, :] > 0.0)
        W = None
        if payload.W is not None:
            W = np.asarray(payload.W, dtype=float).ravel()
            if W.size != V.shape[1]:
                raise ValueError("W_init must have the same number of entries as initial particles.")
            keep = keep & np.isfinite(W) & (W > 0.0)

        V = V[:, keep].copy()
        if W is not None:
            W = W[keep].copy()
        if V.shape[1] <= 0:
            raise ValueError("No particles initialized after filtering non-positive volumes.")

        return _InitialParticlePayload(
            V=V,
            W=W,
            source=payload.source,
            weights_explicit=payload.weights_explicit,
        )

    def _maybe_compress_initial_particle_payload(
        self,
        payload: _InitialParticlePayload,
    ) -> _InitialParticlePayload:
        if payload.weights_explicit:
            return payload

        V_eff_init = int(self.V_eff_init)
        V_eff_mod = str(self.V_eff_mod)
        n_particles = int(payload.V.shape[1])
        if V_eff_init > 0 and V_eff_init < n_particles:
            V_new, W_new = self._compress_init_by_quantile(payload.V, V_eff_init, V_eff_mod)
            return _InitialParticlePayload(
                V=V_new,
                W=W_new,
                source=payload.source,
                weights_explicit=True,
            )

        W_new = np.ones(n_particles, dtype=float)
        return _InitialParticlePayload(
            V=payload.V,
            W=W_new,
            source=payload.source,
            weights_explicit=True,
        )

    def _commit_initial_particle_payload(self, payload: _InitialParticlePayload):
        V_init = np.asarray(payload.V, dtype=float)
        W_new = np.asarray(payload.W, dtype=float).ravel()
        a0_eff = int(V_init.shape[1])
        if W_new.size != a0_eff:
            raise ValueError("Initial particle weights must match the number of particles.")

        cap = max(a0_eff + max(8, a0_eff // 10), 16)
        self._cap = int(cap)

        dim = int(self.dim)
        self.V_flat = np.zeros((dim + 1, self._cap), dtype=float)
        self.V_flat[:, :a0_eff] = V_init
        self.a_tot = a0_eff

        self.X = np.zeros(self._cap, dtype=float)
        self.X[:a0_eff] = self._vol2diam(self.V_flat[-1, :a0_eff])

        self.W = np.zeros(self._cap, dtype=float)
        self.W[:a0_eff] = W_new

        if self.t_vec is None:
            steps = max(1, int(self.t_total // max(1, self.t_write)))
            self.t_vec = np.linspace(0.0, float(self.t_total), steps + 1)

        self._compute_frag_num()

        self.V0 = self.V_flat[:, :self.a_tot].copy()
        self.X0 = self.X[:self.a_tot].copy()
        self.W0 = self.W[:self.a_tot].copy()

        self.V0_save = [self.V0.copy()]
        self.W0_save = [self.W0.copy()]

        self.V_save = [self.V_flat[:, :self.a_tot].copy()]
        self.W_save = [self.W[:self.a_tot].copy()]

        self.Vc_save = [float(self.Vc)]
        self.step = 1

        self.V_save_left = [self.V_flat[:, :self.a_tot].copy()]
        self.W_save_left = [self.W[:self.a_tot].copy()]
        self.t_left = [0.0]
        self.t_right = [0.0]

        self._cv_a_ref = int(self.a_tot)

    def _compress_init_by_quantile(
        self,
        V_init: np.ndarray,
        V_eff_init: int,
        V_eff_mod: str = "Q0",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compress initial particles to representatives by uniform quantiles.
        Q0 uses number CDF; Q3 uses volume CDF.
        """
        if V_eff_init <= 0:
            raise ValueError("V_eff_init must be > 0 for compression.")

        V_eff_mod = str(V_eff_mod).strip().upper()
        V = np.asarray(V_init, dtype=float)
        if V.ndim != 2:
            raise ValueError("V_init must be a 2D array (dim+1, N).")

        N = int(V.shape[1])
        if V_eff_init >= N:
            return V.copy(), np.ones(N, dtype=float)

        Vtot = np.asarray(V[-1, :], dtype=float)
        if np.any(~np.isfinite(Vtot)) or np.any(Vtot <= 0.0):
            raise ValueError("Compression requires finite, positive Vtot in V_init[-1,:].")

        order = np.argsort(Vtot)
        V_sorted = V[:, order]
        Vtot_sorted = Vtot[order]
        q = (np.arange(V_eff_init, dtype=float) + 0.5) / float(V_eff_init)

        def pick_indices_from_cdf(cdf: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(cdf, qgrid, side="left")
            return np.clip(idx, 0, cdf.size - 1).astype(int)

        if V_eff_mod == "Q0":
            cdf = (np.arange(N, dtype=float) + 1.0) / float(N)
            pick = pick_indices_from_cdf(cdf, q)
            V_new = V_sorted[:, pick].copy()
            w_each = float(N) / float(V_eff_init)
            W_new = np.full(V_eff_init, w_each, dtype=float)
            return V_new, W_new

        if V_eff_mod == "Q3":
            tot_vol = float(np.sum(Vtot_sorted))
            if not np.isfinite(tot_vol) or tot_vol <= 0.0:
                raise ValueError("Invalid total volume for Q3 compression.")

            cdf = np.cumsum(Vtot_sorted) / tot_vol
            pick = pick_indices_from_cdf(cdf, q)
            V_new = V_sorted[:, pick].copy()
            rep_vol_each = tot_vol / float(V_eff_init)
            Vp = np.maximum(V_new[-1, :].astype(float), 1e-300)
            W_new = rep_vol_each / Vp
            return V_new, W_new

        raise ValueError(f"Unknown V_eff_mod='{V_eff_mod}'. Use 'Q0' or 'Q3'.")
