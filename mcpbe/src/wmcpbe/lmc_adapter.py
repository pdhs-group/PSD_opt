"""Live mixed/pure aggregate-pool fragment generation for active wmcpbe."""

from __future__ import annotations

from typing import Optional

import numpy as np

from lmc import LMCSimulator


class LMCLiveUnbreakable(RuntimeError):
    """A live-LMC event cannot yield valid fragments for this parent."""


class LMCLiveAdapter:
    """Route mixed and pure parents to independent aggregate-pool simulators."""

    def __init__(self) -> None:
        self.STR = np.ones(3, dtype=float)
        self.NO_FRAG = 4
        self.gamma = 1.0
        self.allow_loops = False
        self.accept_all_cracks = False
        self.use_weighted_start = True
        self.aspect_ratio = 1.0
        self.int_bre = 0
        self.A0_run = 1.0
        self.delta_cells = 0.1
        self.warn_pool_out_of_bounds = True
        self.mixed_pool_dir: Optional[str] = None
        self.pure_pool_dir: Optional[str] = None
        self.Df: Optional[float] = None
        self.mixed_MAS: Optional[float] = None
        self.pure_MAS: Optional[float] = None
        self._mixed_sim: Optional[LMCSimulator] = None
        self._pure_sim: Optional[LMCSimulator] = None

    def close(self) -> None:
        for simulator in (self._mixed_sim, self._pure_sim):
            if simulator is not None:
                simulator.close()

    def configure_simulator(
        self,
        *,
        STR: np.ndarray,
        NO_FRAG: int,
        gamma: float,
        allow_loops: bool,
        accept_all_cracks: bool,
        use_weighted_start: bool,
        aspect_ratio: float,
        int_bre: int,
        A0_run: float,
        delta_cells: float,
        mixed_pool_dir: str,
        pure_pool_dir: str,
        Df: float,
        mixed_MAS: float,
        pure_MAS: float,
        warn_pool_out_of_bounds: bool,
        rebuild: bool = True,
    ) -> None:
        values = np.asarray(STR, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("STR must contain three finite positive values.")
        if not isinstance(NO_FRAG, (int, np.integer)) or NO_FRAG < 2:
            raise ValueError("NO_FRAG must be an integer greater than or equal to 2.")
        if not np.isfinite(gamma) or gamma <= 0.0 or not np.isfinite(A0_run) or A0_run <= 0.0:
            raise ValueError("gamma and A0_run must be finite and positive.")
        if not np.isfinite(delta_cells) or delta_cells < 0.0:
            raise ValueError("delta_cells must be finite and non-negative.")
        if not isinstance(mixed_pool_dir, str) or not mixed_pool_dir:
            raise ValueError("mixed_pool_dir must be a non-empty string.")
        if not isinstance(pure_pool_dir, str) or not pure_pool_dir:
            raise ValueError("pure_pool_dir must be a non-empty string.")
        if not np.all(np.isfinite((Df, mixed_MAS, pure_MAS))):
            raise ValueError("Df, mixed_MAS and pure_MAS must be finite.")

        self.STR = values.copy()
        self.NO_FRAG = int(NO_FRAG)
        self.gamma = float(gamma)
        self.allow_loops = bool(allow_loops)
        self.accept_all_cracks = bool(accept_all_cracks)
        self.use_weighted_start = bool(use_weighted_start)
        self.aspect_ratio = float(aspect_ratio)
        self.int_bre = int(int_bre)
        self.A0_run = float(A0_run)
        self.delta_cells = float(delta_cells)
        self.mixed_pool_dir = mixed_pool_dir
        self.pure_pool_dir = pure_pool_dir
        self.Df = float(Df)
        self.mixed_MAS = float(mixed_MAS)
        self.pure_MAS = float(pure_MAS)
        self.warn_pool_out_of_bounds = bool(warn_pool_out_of_bounds)
        if rebuild:
            self.close()
            common = dict(
                STR=self.STR,
                NO_FRAG=self.NO_FRAG,
                gamma=self.gamma,
                allow_loops=self.allow_loops,
                accept_all_cracks=self.accept_all_cracks,
                use_weighted_start=self.use_weighted_start,
                plotter=None,
                warn_pool_out_of_bounds=self.warn_pool_out_of_bounds,
            )
            self._mixed_sim = LMCSimulator(pool_dir=self.mixed_pool_dir, **common)
            self._pure_sim = LMCSimulator(pool_dir=self.pure_pool_dir, **common)

    @staticmethod
    def _AX1_from_Vparent(V_parent: np.ndarray) -> tuple[float, float]:
        V = np.asarray(V_parent, dtype=float)
        if V.ndim != 1 or V.size not in (1, 2):
            raise ValueError("V_parent must be a one- or two-component vector.")
        if not np.all(np.isfinite(V)) or np.any(V < 0.0):
            raise ValueError("V_parent must contain finite non-negative volumes.")
        A = float(np.sum(V))
        if A <= 0.0:
            raise ValueError("V_parent must have a strictly positive total volume.")
        return A, 1.0 if V.size == 1 else float(V[0] / A)

    def sample_one_shot(
        self, V_parent: np.ndarray, rng: np.random.Generator, *, seed: Optional[int] = None
    ) -> tuple[list[np.ndarray], float]:
        if self._mixed_sim is None or self._pure_sim is None:
            raise RuntimeError("LMCLiveAdapter has not been configured.")
        if self.mixed_pool_dir is None or self.pure_pool_dir is None or self.Df is None or self.mixed_MAS is None or self.pure_MAS is None:
            raise RuntimeError("LMCLiveAdapter pool parameters have not been configured.")
        V = np.asarray(V_parent, dtype=float)
        A, X1 = self._AX1_from_Vparent(V)
        available_cells = int(np.floor(max(A / self.A0_run - self.delta_cells, 0.0)))
        if available_cells < self.NO_FRAG:
            raise LMCLiveUnbreakable(f"Live LMC parent is too small for the configured fragment count: available_cells={available_cells}, requested_NO_FRAG={self.NO_FRAG}.")

        pure_phase: Optional[int]
        if 0.0 < X1 < 1.0:
            simulator, pool_dir, MAS, pool_X1, pure_phase = self._mixed_sim, self.mixed_pool_dir, self.mixed_MAS, X1, None
        elif X1 == 1.0:
            simulator, pool_dir, MAS, pool_X1, pure_phase = self._pure_sim, self.pure_pool_dir, self.pure_MAS, 0.0, 1
        elif X1 == 0.0:
            simulator, pool_dir, MAS, pool_X1, pure_phase = self._pure_sim, self.pure_pool_dir, self.pure_MAS, 0.0, 2
        else:
            raise ValueError("X1 must be exactly 0 or 1 for pure particles, or strictly between them for mixed particles.")
        seed_use = int(seed) if seed is not None else int(rng.integers(0, 2**31 - 1))
        fragments = np.asarray(simulator.mc_breakage_from_pool(pool_dir=pool_dir, Df=self.Df, MAS=MAS, A=A, X1=pool_X1, N_GRIDS=1, N_FRACS=1, A0=self.A0_run, int_bre=self.int_bre, interp="bilinear", seed=seed_use, plot_each=False), dtype=float)
        if fragments.ndim != 2 or fragments.shape[1] < 4 or fragments.size == 0:
            raise LMCLiveUnbreakable("Live LMC returned no fragment data.")
        total = fragments[:, 0]
        valid = np.isfinite(total) & (total > 0.0)
        if not np.any(valid):
            raise LMCLiveUnbreakable("Live LMC returned no positive-volume fragments.")
        energy = float(fragments[0, 3])
        if not np.isfinite(energy) or energy <= 0.0:
            raise LMCLiveUnbreakable("Live LMC returned an invalid breakage energy.")
        if V.size == 1:
            sizes = total[valid] * (V[0] / float(np.sum(total[valid])))
            return [np.array([size], dtype=float) for size in sizes], energy
        if pure_phase == 1:
            return [np.array([size, 0.0], dtype=float) for size in total[valid]], energy
        if pure_phase == 2:
            return [np.array([0.0, size], dtype=float) for size in total[valid]], energy
        phase_1, phase_2 = fragments[valid, 1], fragments[valid, 2]
        if not np.all(np.isfinite(phase_1)) or not np.all(np.isfinite(phase_2)) or np.any(phase_1 < 0.0) or np.any(phase_2 < 0.0):
            raise LMCLiveUnbreakable("Live LMC returned invalid two-phase fragment volumes.")
        return [np.array([a, b], dtype=float) for a, b in zip(phase_1, phase_2)], energy
