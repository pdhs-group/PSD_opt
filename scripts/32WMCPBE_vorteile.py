"""Basic entry script for the advanced 2D validation workflow."""

from __future__ import annotations

# import cProfile, pstats

import numpy as np

from pbe_validation_advance import (
    DirichletInitialCondition,
    PBEValidationAdvanced,
)
from validation import (
    CaseConfig,
    DPBEVariantConfig,
    ValidationConfig,
    WMCPBEVariantConfig,
)

if __name__ == "__main__":
    case = CaseConfig(
        dim=2,
        kernel="const",
        process="mix",
        t_vec=np.arange(0.0, 30.0 + 1e-12, 2.0),
        x=2e-3,
        beta0=1e-6,
        p1=1e-1,
        p2=1.0,
        use_psd=False,
    )

    config = ValidationConfig(
        case=case,
        dpbe_variants=[
            DPBEVariantConfig(name="dPBE", grid="geo", ns=20, s=2),
        ],
        wmcpbe_variants=[
            WMCPBEVariantConfig(
                name="MCPBE",
                repeats=40,
                attrs={
                    "a0": 1000,
                    "V_eff_init": 0,
                    "recon_enable": False,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 1,
                    "agg_dW_max": 1,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE",
                repeats=40,
                attrs={
                    "a0": 10000,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 20,
                    "agg_dW_max": 2,
                },
            ),
        ],
        qmom_variants=[],
        reference_dpbe_name="dPBE",
    )

    init_dist = DirichletInitialCondition(
        alpha_x=1.5,
        alpha_y=3.0,
        alpha_rest=3.0,
        x_min_scale=2.0,
        x_max_scale=0.5,
        y_min_scale=2.0,
        y_max_scale=0.5,
        total_number=1e4,
        volume_concentration=None,
        inverse=True,
    )

    advanced = PBEValidationAdvanced(config=config, init_dist=init_dist, export_dir="exports_WMCPBE_vorteile")
    result = advanced.run()

    advanced.print_moment_error_summary(result)
    advanced.plot_selected_moments(result, relative=True)
    advanced.plot_psd_snapshot(result, t_index=-1, two_d=True, marginal=True, total=True, q0=True, q3=True)
    advanced.plot_error_time_pareto(result)
    advanced.plot_moment_variances(result)
    advanced.show()
