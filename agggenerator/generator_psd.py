# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 12:49:14 2025

@author: px2030
"""
import psd_growth_mvp as mvp
import cProfile, pstats

def main():
    # --- Example pipeline using .npz with cumulative Q0 ---
    npz_path = "psd_example.npz"

    # Save synthetic example cumulative PSDs to .npz
    mvp.save_example_psd_npz(npz_path)

    # Load PSDs from .npz as *cumulative* Q0
    psdA = mvp.load_psd_from_npz(npz_path, diam_key="diam_A", psd_key="Q0_A")
    psdB = mvp.load_psd_from_npz(npz_path, diam_key="diam_B", psd_key="Q0_B")

    Np = 10000
    Df = 1.6
    cell_um = 1.0
    seed = None
    frac_A = 0.6
    s_min = 1
    s_max = 20
    
    params = mvp.GrowthParams(
        Np=Np, Df=Df, k=1.0, cell_um=cell_um,
        w_rg=1.0, w_slope=0.25, w_quota=0.1,
        num_trials=24, seed=seed, pad_margin=16,
        # When set to True, both “overfitting” and “underfitting” are penalized
        # |placed+1 - target| / max(1, target)
        bidirectional_quota_penalty=True,
        # When set to False, the gap is always filled with s=1; 
        # when True, the final stage prioritizes filling gaps according to the PSD.
        endfill_by_psd_deficit=True,
    )
    scheduler = mvp.OnlinePSDScheduler(Np=Np, frac_A=frac_A, psd_A=psdA, psd_B=psdB, 
                                   cell_um=cell_um, s_min=s_min, s_max=s_max,
                                   # Specifies whether to enable the Hamilton largest remainder method.
                                   integer_quota=True)

    result = mvp.growth_mvp(params, scheduler)

    print(f"[Result] N={int(result.grid.sum())}, Df_est≈{result.Df_est:.3f}, slope≈{result.slope:.3f}, comp={scheduler.composition_area()}")
    cmp = mvp.compare_psd_input_vs_achieved(result.scheduler)
    for m in ["A", "B"]:
        print(f"Material {m}: TV distance (number-PSD) = {cmp[m]['total_variation']:.3f}")
        # print table head
        head = sorted(cmp[m]['table'], key=lambda t: t[0])[:5]
        print(" ", head, "...")
    mvp.plot_labels_grid(result.grid, result.labels, result.origin)
    return result.labels
    
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    labels = main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    # stats.print_stats(20)