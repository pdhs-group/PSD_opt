"""Basic entry script for the advanced 2D validation workflow."""

from __future__ import annotations

import sys
from pathlib import Path
# import cProfile, pstats

import numpy as np

def _bootstrap_project_paths() -> None:
    root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        root,
        root / "scripts" / "pbe_validation" / "new",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_project_paths()

from pbe_validation_advance import (  # noqa: E402
    DirichletInitialCondition,
    PBEValidationAdvanced,
)
from validation import (  # noqa: E402
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
        t_vec=np.arange(0.0, 10.0 + 1e-12, 1.0),
        x=2e-3,
        beta0=5.2e-7,
        p1=5.1e-2,
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
                name="WMCPBE 10-5",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 5,
                    "agg_dW_max": 10,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 10-8",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 8,
                    "agg_dW_max": 10,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 10-10",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 10,
                    "agg_dW_max": 10,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 10-15",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 15,
                    "agg_dW_max": 10,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 10-20",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 20,
                    "agg_dW_max": 10,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 20-10",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 10,
                    "agg_dW_max": 20,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 20-15",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 15,
                    "agg_dW_max": 20,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 20-20",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 20,
                    "agg_dW_max": 20,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 20-30",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 30,
                    "agg_dW_max": 20,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 20-40",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 40,
                    "agg_dW_max": 20,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 50-25",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 25,
                    "agg_dW_max": 50,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 50-40",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 40,
                    "agg_dW_max": 50,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 50-50",
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
            WMCPBEVariantConfig(
                name="WMCPBE 50-75",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 75,
                    "agg_dW_max": 50,
                },
            ),
            WMCPBEVariantConfig(
                name="WMCPBE 50-100",
                repeats=100,
                attrs={
                    "a0": 2e5,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_N_max": 4000,
                    "recon_bins": 30,
                    "recon_method": "4PMC",
                    "break_dW_max": 100,
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

    advanced = PBEValidationAdvanced(config=config, init_dist=init_dist, export_dir="exports_dWavsdWb")
    result = advanced.run()

    advanced.print_moment_error_summary(result)
    advanced.plot_selected_moments(result, relative=True)
    advanced.plot_psd_snapshot(result, t_index=-1, two_d=True, marginal=True, total=True, q0=True, q3=True)
    advanced.plot_error_time_pareto(result)
    advanced.plot_moment_variances(result)
    advanced.show()