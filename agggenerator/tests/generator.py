# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:49:14 2025

@author: px2030
"""
# import numpy as np
from agggenerator.mptsa2d import MPTSALatticeParams2D, generate_mptsa_lattice_2d, estimate_fractal_dimension_2d, plot_aggregate_grid
from agggenerator.material_mix import MaterialMixParams, MASPhysicalParams, assign_materials_with_target_mas, plot_materials_grid
import cProfile, pstats

def generate_mptsa():
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
    
    print(f"[MPTSA-2D-Lattice] Np={mptsa_params.Np}, Df={mptsa_params.Df}, k={mptsa_params.k}, seed={mptsa_params.seed}, fill_hole={mptsa_params.fill_hole}")
    positions, Ns, Rgs, grid, origin = generate_mptsa_lattice_2d(mptsa_params)
    Df_est, slope = estimate_fractal_dimension_2d(Ns, Rgs)
    print(f"[MPTSA-2D-Lattice] Estimated Df ≈ {Df_est:.4f} (slope={slope:.4f})")
    plot_aggregate_grid(grid, origin)
    return grid, origin
    
def assign_materials(grid, origin):
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
    phys_params = MASPhysicalParams(
        # keep defaults unless you need the transmission upper bound or particle-size effects
        # transmission_weights=(0.0, 0.0, 1.0)  # example: 100% [1 1] case
    )

    labels, stats = assign_materials_with_target_mas(grid, mix_params, phys_params)
    print("[Materials] stats:", stats)
    plot_materials_grid(grid, labels, origin)
    return labels
    
def main():
    grid, origin = generate_mptsa()
    labels = assign_materials(grid, origin)
    return labels
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    labels = main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    # stats.print_stats(20)