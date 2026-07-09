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
        process="breakage",
        t_vec=np.arange(0.0, 10.0 + 1e-12, 1.0),
        x=2e-3,
        beta0=1.05e-6,
        p1=4.1e-2,
        p2=1.0,
        use_psd=False,
    )

    config = ValidationConfig(
        case=case,
        dpbe_variants=[
            DPBEVariantConfig(name="dPBE", grid="geo", ns=50, s=1.5, enabled=False),
        ],
        wmcpbe_variants=[
            WMCPBEVariantConfig(
                name="WMCPBE dW=2",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 2, 
                    "agg_dW_max": 2,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=4",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 4,
                    "agg_dW_max": 4,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=8",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 8,
                    "agg_dW_max": 8,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=12",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 12,
                    "agg_dW_max": 12,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=16",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 16,
                    "agg_dW_max": 16,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=24",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 24,
                    "agg_dW_max": 24,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=32",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 32,
                    "agg_dW_max": 32,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=40",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 40,
                    "agg_dW_max": 40,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE dW=50",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 50,
                    "agg_dW_max": 50,
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
        x_max_scale=1e-2,
        y_min_scale=2.0,
        y_max_scale=1e-2,
        total_number=2e5,
        volume_concentration=None,
    )

    advanced = PBEValidationAdvanced(config=config, init_dist=init_dist, export_dir="exports_pure_break")
    result = advanced.run()

    advanced.print_moment_error_summary(result)
    advanced.plot_selected_moments(result, relative=True)
    advanced.plot_psd_snapshot(result, t_index=-1, two_d=True, marginal=True, total=True, q0=True, q3=True)
    advanced.plot_error_time_pareto(result)
    advanced.plot_moment_variances(result)
    advanced.show()
