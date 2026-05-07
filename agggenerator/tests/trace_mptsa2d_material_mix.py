# -*- coding: utf-8 -*-
"""
Trace-style visualizer for the 2D MPTSA and material-mix workflows.

This script intentionally reimplements the two high-level driver functions
while reusing the existing dataclasses and low-level helper functions from
agggenerator.mptsa2d and agggenerator.material_mix.

It saves PNG snapshots instead of writing aggregate data files. For MPTSA, it
caches the aggregate after every ``particle_plot`` placed particles, then
renders all cached states on the final bounding grid so the background does
not jump between frames. For material mixing, it saves the material-label image
every time MAS is evaluated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


AGGGENERATOR_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGGGENERATOR_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agggenerator.mptsa2d import (  # noqa: E402
    MPTSALatticeParams2D,
    _choose_base_index_lattice,
    _empty_orth_neighbors,
    _to_min_bounding_grid,
    estimate_fractal_dimension_2d,
    fill_small_holes_and_compensate,
    radius_of_gyration_2d,
)
from agggenerator.material_mix import (  # noqa: E402
    MASPhysicalParams,
    MaterialMixParams,
    _build_neighbors_4,
    _evaluate_mas_on_labels,
    _make_low_mas_label_candidates,
    _make_random_exact_labels,
    _mcmc_exchange,
)


def _fmt_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _labels_to_image(
    shape: Tuple[int, int],
    occ_y: np.ndarray,
    occ_x: np.ndarray,
    lbl_flat: np.ndarray,
) -> np.ndarray:
    labels = np.full(shape, fill_value=-1, dtype=np.int8)
    labels[occ_y, occ_x] = lbl_flat.astype(np.int8, copy=False)
    return labels


class SnapshotWriter:
    """Save trace frames using the same plotting style as the original helpers."""

    def __init__(
        self,
        *,
        snapshot_dir: str | Path = "snapshot",
        dpi: int = 120,
        gridline_limit: int = 140,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.dpi = int(dpi)
        self.gridline_limit = int(gridline_limit)
        self.mptsa_frame = 0
        self.mix_frame = 0

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    def _style_grid_axis(self, ax: plt.Axes, grid: np.ndarray) -> None:
        h, w = grid.shape
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if max(h, w) <= self.gridline_limit:
            ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
            ax.grid(which="minor", linewidth=0.3)
        else:
            ax.grid(False)

    def _save_and_close(self, fig: plt.Figure, prefix: str, frame: int) -> Path:
        fig.tight_layout()
        path = self.snapshot_dir / f"{prefix}_{frame}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_mptsa(
        self,
        grid: np.ndarray,
        *,
        title: str,
        origin: Tuple[int, int],
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(grid, origin="lower", interpolation="nearest", alpha=0.35)
        ax.set_title(title, fontsize=20, pad=8)
        self._style_grid_axis(ax, grid)
        path = self._save_and_close(fig, "mptsa2d", self.mptsa_frame)
        self.mptsa_frame += 1
        return path

    def save_materials(
        self,
        grid: np.ndarray,
        labels: np.ndarray,
        *,
        title: str,
        origin: Tuple[int, int],
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(grid, origin="lower", interpolation="nearest", alpha=0.35)
        show = np.ma.masked_where(labels < 0, labels)
        ax.imshow(show, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=20, pad=8)
        self._style_grid_axis(ax, grid)
        path = self._save_and_close(fig, "material_mix", self.mix_frame)
        self.mix_frame += 1
        return path


def _record_mptsa_state(
    *,
    note: str,
    occ: Set[Tuple[int, int]],
    positions: np.ndarray,
    Ns: List[float],
    Rgs: List[float],
    target_df: float,
    snapshots: List[Dict[str, Any]],
) -> None:
    grid, origin = _to_min_bounding_grid(occ)
    df_est, slope = estimate_fractal_dimension_2d(
        np.asarray(Ns, dtype=float),
        np.asarray(Rgs, dtype=float),
    )
    current_rg = float(Rgs[-1]) if Rgs else float("nan")
    print(
        "\n"
        f"[MPTSA] {note} | "
        f"N={positions.shape[0]} | "
        f"target_Df={target_df:.4f} | "
        f"Df_est={_fmt_float(df_est)} | "
        f"slope={_fmt_float(slope)} | "
        f"Rg={_fmt_float(current_rg)} | "
        f"grid_shape={grid.shape} | origin={origin}"
    )
    snapshots.append(
        dict(
            note=str(note),
            positions=positions.copy(),
            Ns=np.asarray(Ns, dtype=float).copy(),
            Rgs=np.asarray(Rgs, dtype=float).copy(),
            N=int(positions.shape[0]),
            Rg=float(current_rg),
            Df_est=float(df_est),
            target_Df=float(target_df),
            slope=float(slope),
        )
    )


def _positions_to_final_canvas_grid(
    positions: np.ndarray,
    final_shape: Tuple[int, int],
    final_origin: Tuple[int, int],
) -> np.ndarray:
    frame = np.zeros(final_shape, dtype=np.uint8)
    x0, y0 = int(final_origin[0]), int(final_origin[1])
    xs = np.rint(positions[:, 0]).astype(int) - x0
    ys = np.rint(positions[:, 1]).astype(int) - y0
    ok = (0 <= ys) & (ys < final_shape[0]) & (0 <= xs) & (xs < final_shape[1])
    frame[ys[ok], xs[ok]] = 1
    return frame


def _save_mptsa_snapshots(
    snapshots: List[Dict[str, Any]],
    *,
    final_grid: np.ndarray,
    final_origin: Tuple[int, int],
    writer: SnapshotWriter,
) -> None:
    print(f"\n[MPTSA-SNAPSHOT] saving {len(snapshots)} frames to {writer.snapshot_dir}")
    for snap in snapshots:
        frame_grid = _positions_to_final_canvas_grid(
            np.asarray(snap["positions"], dtype=float),
            final_grid.shape,
            final_origin,
        )
        path = writer.save_mptsa(
            frame_grid,
            title=(
                f"N={snap['N']} | "
                f"Df={_fmt_float(float(snap['Df_est']))} | "
                f"target Df={_fmt_float(float(snap['target_Df']))}"
            ),
            origin=final_origin,
        )
        print(f"[MPTSA-SNAPSHOT] {path}")


def generate_mptsa_lattice_2d_trace(
    params: MPTSALatticeParams2D,
    *,
    particle_plot: int = 10,
    writer: SnapshotWriter,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    rng = np.random.default_rng(params.seed)
    Df = float(params.Df)
    k = float(params.k)
    particle_plot = max(1, int(particle_plot))

    occ: Set[Tuple[int, int]] = {(0, 0), (1, 0)}
    positions = np.array([(0, 0), (1, 0)], dtype=float)
    Ns: List[float] = [2.0]
    Rgs: List[float] = [float(radius_of_gyration_2d(positions))]
    snapshots: List[Dict[str, Any]] = []

    _record_mptsa_state(
        note="initial seed",
        occ=occ,
        positions=positions,
        Ns=Ns,
        Rgs=Rgs,
        target_df=Df,
        snapshots=snapshots,
    )

    while positions.shape[0] < int(params.Np):
        placed = False

        for _ in range(int(params.max_attempts)):
            base_idx = _choose_base_index_lattice(positions, rng, Df=Df, k=k)
            bx, by = map(int, positions[base_idx])

            empties = _empty_orth_neighbors((bx, by), occ)
            if not empties:
                continue

            cx, cy = empties[int(rng.integers(0, len(empties)))]
            occ.add((cx, cy))
            positions = np.vstack([positions, [cx, cy]])
            placed = True
            break

        if not placed:
            raise RuntimeError(
                "Placement failed after max_attempts. Increase max_attempts "
                "or adjust the MPTSA parameters."
            )

        Ns.append(float(positions.shape[0]))
        Rgs.append(float(radius_of_gyration_2d(positions)))

        n_now = int(positions.shape[0])
        if n_now % particle_plot == 0 or n_now == int(params.Np):
            _record_mptsa_state(
                note=f"after placing {n_now} particles",
                occ=occ,
                positions=positions,
                Ns=Ns,
                Rgs=Rgs,
                target_df=Df,
                snapshots=snapshots,
            )

    grid, origin = _to_min_bounding_grid(occ)
    final_grid = grid.copy()
    final_origin = origin

    if params.fill_hole:
        grid, n_fill, n_del = fill_small_holes_and_compensate(
            grid,
            area_max=int(params.hole_area_max),
            alpha=float(params.compensate_alpha),
            beta=float(params.compensate_beta),
            verbose=bool(params.verbose),
        )
        ys, xs = np.nonzero(grid)
        positions = np.column_stack([xs + origin[0], ys + origin[1]]).astype(float)
        Ns = list(np.arange(2, positions.shape[0] + 1, dtype=float))
        Rgs = [float(radius_of_gyration_2d(positions[: int(n)])) for n in Ns]
        print(
            "\n"
            f"[MPTSA-POST] fill_hole=True | filled={n_fill} | "
            f"deleted={n_del} | final_N={positions.shape[0]}"
        )
        final_grid = grid.copy()
        final_origin = origin
        snapshots.append(
            dict(
                note=f"post-process filled={n_fill}, deleted={n_del}",
                positions=positions.copy(),
                Ns=np.asarray(Ns, dtype=float).copy(),
                Rgs=np.asarray(Rgs, dtype=float).copy(),
                N=int(positions.shape[0]),
                Rg=float(Rgs[-1]) if Rgs else float("nan"),
                Df_est=float(estimate_fractal_dimension_2d(
                    np.asarray(Ns, dtype=float),
                    np.asarray(Rgs, dtype=float),
                )[0]),
                target_Df=float(Df),
                slope=float(estimate_fractal_dimension_2d(
                    np.asarray(Ns, dtype=float),
                    np.asarray(Rgs, dtype=float),
                )[1]),
            )
        )

    _save_mptsa_snapshots(
        snapshots,
        final_grid=final_grid,
        final_origin=final_origin,
        writer=writer,
    )

    return (
        positions,
        np.asarray(Ns, dtype=float),
        np.asarray(Rgs, dtype=float),
        grid,
        origin,
    )


def assign_materials_with_target_mas_trace(
    grid: np.ndarray,
    params: MaterialMixParams,
    phys: MASPhysicalParams = MASPhysicalParams(),
    *,
    origin: Tuple[int, int] = (0, 0),
    writer: SnapshotWriter,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(params.seed)
    occ_y, occ_x = np.nonzero(grid)
    M = len(occ_y)
    if M == 0:
        raise ValueError("Empty aggregate grid.")

    nA = int(round(float(params.frac_A) * M))
    nA = max(0, min(M, nA))
    nB = M - nA
    H, W = grid.shape
    to_id = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(occ_y, occ_x))}
    neighbors = _build_neighbors_4(occ_y, occ_x, to_id, H, W)

    target = float(np.clip(params.target_MAS, 0.0, 1.0))
    eval_count = 0

    def evaluate_and_print(
        stage: str,
        lbl_flat: np.ndarray,
        lam: float | None,
        bisection: int = 0,
    ) -> Tuple[float, float, float, float]:
        nonlocal eval_count
        eval_count += 1
        MAS, sigma2, sig0, sigz = _evaluate_mas_on_labels(
            grid, occ_y, occ_x, lbl_flat, params, phys
        )
        label_img = _labels_to_image(grid.shape, occ_y, occ_x, lbl_flat)
        lam_text = "none" if lam is None else _fmt_float(float(lam), digits=6)
        print(
            "\n"
            f"[MIX-EVAL {eval_count:03d}] {stage} | "
            f"lambda={lam_text} | "
            f"MAS={_fmt_float(MAS, digits=6)} | "
            f"target={target:.6f} | "
            f"err={abs(float(MAS) - target):.6f} | "
            f"sigma2={_fmt_float(sigma2, digits=6)} | "
            f"sigma0={_fmt_float(sig0, digits=6)} | "
            f"sigmaz={_fmt_float(sigz, digits=6)}"
        )
        path = writer.save_materials(
            grid,
            label_img,
            title=(
                f"Bisection {int(bisection)} | "
                f"MAS={_fmt_float(MAS, digits=6)} | "
                f"target MAS={target:.6f}"
            ),
            origin=origin,
        )
        print(f"[MIX-SNAPSHOT] {path}")
        return float(MAS), float(sigma2), float(sig0), float(sigz)

    print(
        "\n"
        f"[MIX] occupied={M} | nA={nA} | nB={nB} | "
        f"frac_A_actual={nA / M:.6f} | target_MAS={target:.6f} | "
        f"tol={float(params.tol_MAS):.6f} | window={params.window} | stride={params.stride}"
    )

    candidates: List[Tuple[str, np.ndarray]] = [
        ("random", _make_random_exact_labels(M, nA, rng))
    ]
    if 0 < nA < M and target < float(params.low_mas_init_threshold):
        candidates.extend(
            _make_low_mas_label_candidates(
                occ_y=occ_y,
                occ_x=occ_x,
                nA=nA,
                neighbors=neighbors,
                rng=rng,
                n_bfs_candidates=int(max(0, params.low_mas_init_candidates)),
            )
        )

    best_lbl = candidates[0][1].copy()
    best_mas = float("nan")
    best_lam = 0.0
    best_err = float("inf")
    best_init_strategy = candidates[0][0]

    for idx, (name, labels) in enumerate(candidates):
        mas, _, _, _ = evaluate_and_print(
            stage=f"initial candidate {idx + 1}/{len(candidates)} ({name})",
            lbl_flat=labels,
            lam=None,
            bisection=0,
        )
        err = abs(mas - target) if np.isfinite(mas) else float("inf")
        if err < best_err:
            best_lbl = labels.copy()
            best_mas = mas
            best_lam = 0.0
            best_err = err
            best_init_strategy = str(name)

    lbl_flat = best_lbl.copy()
    mas_initial = best_mas

    def record_best(candidate_lbl: np.ndarray, candidate_mas: float, candidate_lam: float) -> None:
        nonlocal best_lbl, best_mas, best_lam, best_err
        if not np.isfinite(candidate_mas):
            return
        err = abs(float(candidate_mas) - target)
        if err < best_err:
            best_lbl = candidate_lbl.copy()
            best_mas = float(candidate_mas)
            best_lam = float(candidate_lam)
            best_err = float(err)

    lam_lo = float(params.lambda_min)
    lam_hi = float(params.lambda_max)

    lbl_work = lbl_flat.copy()
    _mcmc_exchange(
        lbl_work,
        neighbors,
        lam_lo,
        sweeps=int(params.sweeps_per_eval),
        T=float(params.temperature),
        rng=rng,
    )
    mas_lo, _, _, _ = evaluate_and_print(
        "lambda lower bound after MCMC",
        lbl_work,
        lam_lo,
        bisection=0,
    )
    record_best(lbl_work, mas_lo, lam_lo)

    lbl_work2 = lbl_flat.copy()
    _mcmc_exchange(
        lbl_work2,
        neighbors,
        lam_hi,
        sweeps=int(params.sweeps_per_eval),
        T=float(params.temperature),
        rng=rng,
    )
    mas_hi, _, _, _ = evaluate_and_print(
        "lambda upper bound after MCMC",
        lbl_work2,
        lam_hi,
        bisection=0,
    )
    record_best(lbl_work2, mas_hi, lam_hi)

    if mas_lo > mas_hi:
        lam_lo, lam_hi = lam_hi, lam_lo
        lbl_work, lbl_work2 = lbl_work2, lbl_work
        mas_lo, mas_hi = mas_hi, mas_lo

    converged = False
    for i_bisect in range(int(params.max_bisect)):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        lbl_mid = lbl_work.copy() if target <= mas_hi else lbl_work2.copy()
        _mcmc_exchange(
            lbl_mid,
            neighbors,
            lam_mid,
            sweeps=max(1, int(params.sweeps_per_eval)),
            T=float(params.temperature),
            rng=rng,
        )
        mas_mid, _, _, _ = evaluate_and_print(
            stage=f"bisection {i_bisect + 1}/{int(params.max_bisect)}",
            lbl_flat=lbl_mid,
            lam=lam_mid,
            bisection=i_bisect + 1,
        )
        record_best(lbl_mid, mas_mid, lam_mid)

        if abs(mas_mid - target) <= float(params.tol_MAS):
            converged = True
            break

        if mas_mid < target:
            lam_lo, lbl_work = lam_mid, lbl_mid
            mas_lo = mas_mid
        else:
            lam_hi, lbl_work2 = lam_mid, lbl_mid
            mas_hi = mas_mid

    final_lbl = best_lbl.copy()
    final_mas, sigma2, sigma0, sigmaz = evaluate_and_print(
        stage=f"final selected (converged={converged})",
        lbl_flat=final_lbl,
        lam=best_lam,
        bisection=0,
    )

    labels = _labels_to_image(grid.shape, occ_y, occ_x, final_lbl)
    stats = dict(
        N_total=int(M),
        nA=int((labels == 0).sum()),
        nB=int((labels == 1).sum()),
        frac_A=float((labels == 0).sum() / M),
        lambda_used=float(best_lam),
        MAS=float(final_mas),
        sigma2=float(sigma2),
        sigma0_sq=float(sigma0),
        sigmaz_sq=float(sigmaz),
        window=int(params.window),
        stride=int(params.stride),
        initial_MAS=float(mas_initial),
        initial_strategy=str(best_init_strategy),
        initial_candidates=int(len(candidates)),
        evaluations=int(eval_count),
        converged=bool(converged),
    )
    return labels, stats


def _auto_window_stride(grid: np.ndarray) -> Tuple[int, int]:
    shorter = int(min(grid.shape[0], grid.shape[1]))
    window = max(2, min(12, shorter // 5))
    stride = max(1, min(3, window // 4))
    return int(window), int(stride)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save trace PNG snapshots for MPTSA growth and MAS material mixing."
    )
    parser.add_argument("--Np", type=int, default=2000)
    parser.add_argument("--Df", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--particle-plot", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=50000)
    parser.add_argument("--fill-hole", action="store_true")

    parser.add_argument("--frac-A", type=float, default=0.5)
    parser.add_argument("--target-MAS", type=float, default=0.35)
    parser.add_argument("--tol-MAS", type=float, default=0.005)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min-occupancy-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-min", type=float, default=-6.0)
    parser.add_argument("--lambda-max", type=float, default=6.0)
    parser.add_argument("--sweeps-per-eval", type=int, default=8)
    parser.add_argument("--max-bisect", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--low-mas-init-threshold", type=float, default=0.3)
    parser.add_argument("--low-mas-init-candidates", type=int, default=8)

    parser.add_argument("--snapshot-dir", type=str, default="snapshot")
    parser.add_argument("--frame-dpi", type=int, default=120)
    parser.add_argument("--gridline-limit", type=int, default=140)
    parser.add_argument("--skip-material-mix", action="store_true")
    return parser.parse_args()


def main() -> Tuple[np.ndarray, np.ndarray | None, Dict[str, Any] | None]:
    args = _parse_args()
    mptsa_params = MPTSALatticeParams2D(
        Np=5000,
        Df=2.0,
        k=2.0,
        max_attempts=50000,
        seed=9,
        fill_hole=True,
        hole_area_max=4,
        compensate_alpha=1.0,
        compensate_beta=0.25,
        verbose=True, 
    )

    print(
        f"[CONFIG] Np={mptsa_params.Np} | Df={mptsa_params.Df} | "
        f"seed={mptsa_params.seed} | particle_plot={args.particle_plot}"
    )
    writer = SnapshotWriter(
        snapshot_dir=Path(args.snapshot_dir),
        dpi=int(args.frame_dpi),
        gridline_limit=int(args.gridline_limit),
    )
    _positions, Ns, Rgs, grid, origin = generate_mptsa_lattice_2d_trace(
        mptsa_params,
        particle_plot=int(args.particle_plot),
        writer=writer,
    )
    df_est, slope = estimate_fractal_dimension_2d(Ns, Rgs)
    print(
        "\n"
        f"[MPTSA-FINAL] grid_shape={grid.shape} | origin={origin} | "
        f"Df_est={_fmt_float(df_est)} | slope={_fmt_float(slope)}"
    )

    if args.skip_material_mix:
        return grid, None, None

    window, stride = _auto_window_stride(grid)
    if int(args.window) > 0:
        window = int(args.window)
    if int(args.stride) > 0:
        stride = int(args.stride)

    mix_params = MaterialMixParams(
        frac_A=0.5,
        target_MAS=0.35,
        tol_MAS=0.005,
        window=12,
        stride=3,
        sweeps_per_eval=12,
        max_bisect=20,
        seed=42,
        min_occupancy_ratio=0.5,
        # lower/upper bounds for lambda
        lambda_min = -6.0,
        lambda_max = 6.0,
        temperature = 1.0,
        low_mas_init_threshold = 0.3,
        low_mas_init_candidates = 8,
    )
    phys_params = MASPhysicalParams()

    labels, stats = assign_materials_with_target_mas_trace(
        grid,
        mix_params,
        phys_params,
        origin=origin,
        writer=writer,
    )
    print("\n[MIX-FINAL] stats:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")

    return grid, labels, stats


if __name__ == "__main__":
    main()
