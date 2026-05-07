# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:56:09 2025

@author: px2030
"""
import numpy as np
import matplotlib.pyplot as plt
import importlib
from pathlib import Path
from lmc import LMCSimulator
from lmc import Plotter
from lmc import CrackStepSnapshotWriter
import time


GENERATOR_MODULE = "generator"
# GENERATOR_MODULE = "generator_psd"
LABELS_NPZ_PATH = Path(__file__).with_name("udp_labels_cache.npz")
REGENERATE_LABELS = False


def load_or_create_labels(cache_path=LABELS_NPZ_PATH, *, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            labels = data["labels"]
        print(f"[labels] loaded from {cache_path} shape={labels.shape} dtype={labels.dtype}")
        return labels

    generator = importlib.import_module(GENERATOR_MODULE)
    labels = generator.main()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        labels=labels,
        generator_module=np.asarray(GENERATOR_MODULE),
    )
    print(f"[labels] generated and saved to {cache_path} shape={labels.shape} dtype={labels.dtype}")
    return labels


def live_generator():
    labels = load_or_create_labels(force=REGENERATE_LABELS)
    
    writer = CrackStepSnapshotWriter(
        output_dir="crack_snapshots",
        prefix="udp_crack",
        material_style="mask",      # "phases" / "mask" / "none"
        show_intact_bonds=False,
        show_axes=False,
        padding=3.0,
        dpi=140,
    )
    writer = None
    sim = LMCSimulator(STR=STR, NO_FRAG=NO_FRAG, gamma=gamma,
                       allow_loops=True, accept_all_cracks=False,
                       use_weighted_start=False, plotter=Plotter())
    start_time = time.time()
    F = sim.mc_breakage_udp(mat=labels, N_FRACS=N_FRACS,
                            a_code=0, b_code=1, empty_code=-1,
                            A0=A0,int_bre=int_bre,seed=seed,
                            plot_intermediate=False,plot_each=False, 
                            crack_step_callback=writer, trace_stride=1,)
    end_time = time.time()
    sim_time = end_time - start_time
    print(f"simulation time = {sim_time:6f}", )
    sim.plotter.plot_F(F)
    plt.show()
    
def from_pool():
    sim = LMCSimulator(STR=STR, NO_FRAG=NO_FRAG, gamma=gamma,
                       allow_loops=False, accept_all_cracks=False,
                       use_weighted_start=True, plotter=Plotter())
    F = sim.mc_breakage_from_pool(
        pool_dir=r"C:\Users\px2030\Code\LMC_ANN\agggenerator",
        Df=1.8,
        MAS=0.40,
        A=A,
        X1=X1,
        N_GRIDS=N_GRIDS,
        N_FRACS=N_FRACS,
        A0=A0,
        int_bre=int_bre,
        seed=seed,
        plot_each=False,
    )
    sim.agg_pool.close_pool_cache()
    sim.plotter.plot_F(F)
    plt.show()
if __name__ == "__main__":
    A = 10000.0
    A0 = 1.0
    X1 = 0.6
    X2 = 1 - X1
    STR = np.array([1.0, 0.1, 1.0], dtype=float)
    NO_FRAG = 4
    aspect_ratio = 2.0
    int_bre = 0
    gamma = 10.0
    
    N_GRIDS = 1
    N_FRACS = 2000000
    seed = 33
    # for i in range(10,50):
    #     seed=i
    live_generator()
    # from_pool()
