"""Basic example for the new validation workflow."""

from __future__ import annotations

import cProfile, pstats

import numpy as np

from validation import (
    CaseConfig,
    DPBEVariantConfig,
    QMOMVariantConfig,
    ValidationConfig,
    ValidationPlotter,
    ValidationRunner,
    WMCPBEVariantConfig,
)


if __name__ == "__main__":
    case = CaseConfig(
        dim=2,
        kernel="const",
        process="mix",
        t_vec=np.arange(0.0, 10.0, 2.0, dtype=float),
        c=1.0,
        x=2e-1,
        beta0=9e-4,
        p1=1e-1,
        p2=1.0,
        use_psd=False,
    )

    dpbe_variants = [
        DPBEVariantConfig(name="dPBE (NS=15)", grid="geo", ns=15, s=2,enabled=False),
    ]

    wmcpbe_variants = [
        # WMCPBEVariantConfig(
        #     name="WMCPBE (coarse)",
        #     repeats=10,
        #     attrs={
        #         "a0": 50000,
        #         "V_eff_init": 800,
        #         "recon_N_max": 2500,
        #         "recon_bins": 20,
        #         "recon_method": "4PMC",
        #     },
        # ),
        WMCPBEVariantConfig(
            name="WMCPBE (fine)",
            repeats=10,
            attrs={
                "a0": 1e5,
                "V_eff_init": 1000,
                "recon_enable": True,
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_method": "4PMC",
                "break_dW_max": 10.0,
                "agg_dW_max": 10.0
            },
        ),
    ]

    qmom_variants = [
    ]

    config = ValidationConfig(
        case=case,
        dpbe_variants=dpbe_variants,
        wmcpbe_variants=wmcpbe_variants,
        qmom_variants=qmom_variants,
        reference_dpbe_name="dPBE (NS=15)",
    )

    profiler = cProfile.Profile()
    profiler.enable()
    result = ValidationRunner(config).run()
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    stats.print_stats(20)

    plotter = ValidationPlotter(result)
    plotter.plot_all_moments(relative=True, include_total_volume=False)
    plotter.show()
