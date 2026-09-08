"""Live LMC fragment-generation adapter used by the active wmcpbe solver.

The active solver deliberately supports only the aggregate-pool-backed LMC
path.  The analytical PBE fragment distribution is implemented separately in
``mcpbe_break.py`` and does not use this module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lmc import LMCSimulator


class LMCLiveUnbreakable(RuntimeError):
    """The live LMC calculation cannot produce a valid breakage event.

    This is a normal event-level outcome for an undersized aggregate or for a
    sampled aggregate without a usable crack.  ``MCPBEBreak`` catches only
    this signal and sets that particle's breakage propensity to zero.
    """


class LMCLiveAdapter:
    """Generate one fragment set directly from the aggregate pool."""

    def __init__(self) -> None:
        self.STR = np.array([1.0, 1.0, 1.0], dtype=float)
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

        self.pool_dir: Optional[str] = None
        self.Df: Optional[float] = None
        self.MAS: Optional[float] = None
        self._sim: Optional[LMCSimulator] = None

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
        pool_dir: str,
        Df: float,
        MAS: float,
        warn_pool_out_of_bounds: bool,
        rebuild: bool = True,
    ) -> None:
        """Store the active LMC configuration and construct its simulator."""
        STR_arr = np.asarray(STR, dtype=float)
        if STR_arr.shape != (3,) or not np.all(np.isfinite(STR_arr)) or np.any(STR_arr <= 0.0):
            raise ValueError("STR must contain three finite positive values.")
        if not isinstance(NO_FRAG, (int, np.integer)) or NO_FRAG < 2:
            raise ValueError("NO_FRAG must be an integer greater than or equal to 2.")
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be finite and positive.")
        if not np.isfinite(A0_run) or A0_run <= 0.0:
            raise ValueError("A0_run must be finite and positive.")
        if not np.isfinite(delta_cells) or delta_cells < 0.0:
            raise ValueError("delta_cells must be finite and non-negative.")
        if not isinstance(pool_dir, str) or not pool_dir:
            raise ValueError("pool_dir must be a non-empty string.")
        if not np.isfinite(Df) or not np.isfinite(MAS):
            raise ValueError("Df and MAS must be finite.")

        self.STR = STR_arr.copy()
        self.NO_FRAG = int(NO_FRAG)
        self.gamma = float(gamma)
        self.allow_loops = bool(allow_loops)
        self.accept_all_cracks = bool(accept_all_cracks)
        self.use_weighted_start = bool(use_weighted_start)
        self.aspect_ratio = float(aspect_ratio)
        self.int_bre = int(int_bre)
        self.A0_run = float(A0_run)
        self.delta_cells = float(delta_cells)
        self.pool_dir = pool_dir
        self.Df = float(Df)
        self.MAS = float(MAS)
        self.warn_pool_out_of_bounds = bool(warn_pool_out_of_bounds)

        if rebuild or self._sim is None:
            self._sim = LMCSimulator(
                STR=self.STR,
                NO_FRAG=self.NO_FRAG,
                gamma=self.gamma,
                allow_loops=self.allow_loops,
                accept_all_cracks=self.accept_all_cracks,
                use_weighted_start=self.use_weighted_start,
                plotter=None,
                pool_dir=self.pool_dir,
                warn_pool_out_of_bounds=self.warn_pool_out_of_bounds,
            )

    @staticmethod
    def _AX1_from_Vparent(V_parent: np.ndarray) -> tuple[float, float]:
        """Return aggregate area and phase-1 fraction from a PBE state vector."""
        V = np.asarray(V_parent, dtype=float)
        if V.ndim != 1 or V.size not in (1, 2):
            raise ValueError("V_parent must be a one- or two-component vector.")
        if not np.all(np.isfinite(V)) or np.any(V < 0.0):
            raise ValueError("V_parent must contain finite non-negative volumes.")

        if V.size == 1:
            A = float(V[0])
            X1 = 1.0
        else:
            A = float(V[0] + V[1])
            X1 = float(V[0] / A) if A > 0.0 else 0.5

        if A <= 0.0:
            raise ValueError("V_parent must have a strictly positive total volume.")
        return A, X1

    def sample_one_shot(
        self,
        V_parent: np.ndarray,
        rng: np.random.Generator,
        *,
        seed: Optional[int] = None,
    ) -> tuple[list[np.ndarray], float]:
        """Return live-LMC fragments, or raise :class:`LMCLiveUnbreakable`."""
        if self._sim is None:
            raise RuntimeError("LMCLiveAdapter has not been configured.")
        if self.pool_dir is None or self.Df is None or self.MAS is None:
            raise RuntimeError("LMCLiveAdapter pool parameters have not been configured.")

        V = np.asarray(V_parent, dtype=float)
        A, X1 = self._AX1_from_Vparent(V)
        n_cells = A / self.A0_run
        NO_FRAG_raw = int(np.floor(max(n_cells - self.delta_cells, 0.0)))
        if NO_FRAG_raw < self.NO_FRAG:
            raise LMCLiveUnbreakable(
                "Live LMC parent is too small for the configured fragment count: "
                f"available_cells={NO_FRAG_raw}, requested_NO_FRAG={self.NO_FRAG}."
            )

        seed_use = int(seed) if seed is not None else int(rng.integers(0, 2**31 - 1))
        result = self._sim.mc_breakage_from_pool(
            pool_dir=self.pool_dir,
            Df=self.Df,
            MAS=self.MAS,
            A=A,
            X1=X1,
            N_GRIDS=1,
            N_FRACS=1,
            A0=self.A0_run,
            int_bre=self.int_bre,
            interp="bilinear",
            seed=seed_use,
            plot_each=False,
        )

        fragments = np.asarray(result, dtype=float)
        if fragments.ndim != 2 or fragments.shape[1] < 4 or fragments.size == 0:
            raise LMCLiveUnbreakable("Live LMC returned no fragment data.")

        total_volume = fragments[:, 0]
        valid = np.isfinite(total_volume) & (total_volume > 0.0)
        if not np.any(valid):
            raise LMCLiveUnbreakable("Live LMC returned no positive-volume fragments.")

        energy = float(fragments[0, 3])
        if not np.isfinite(energy) or energy <= 0.0:
            raise LMCLiveUnbreakable("Live LMC returned an invalid breakage energy.")

        if V.size == 1:
            sizes = total_volume[valid]
            sizes *= V[0] / float(np.sum(sizes))
            return [np.array([size], dtype=float) for size in sizes], energy

        if fragments.shape[1] < 3:
            raise LMCLiveUnbreakable("Live LMC returned no two-phase fragment volumes.")
        phase_1 = fragments[valid, 1]
        phase_2 = fragments[valid, 2]
        if (
            not np.all(np.isfinite(phase_1))
            or not np.all(np.isfinite(phase_2))
            or np.any(phase_1 < 0.0)
            or np.any(phase_2 < 0.0)
        ):
            raise LMCLiveUnbreakable("Live LMC returned invalid two-phase fragment volumes.")
        return [np.array([a, b], dtype=float) for a, b in zip(phase_1, phase_2)], energy
