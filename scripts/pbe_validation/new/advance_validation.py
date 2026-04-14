"""Basic entry script for the advanced 2D validation workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pbe_core.plotter.plotter_new import PaperPlotter


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
                repeats=20,
                attrs={
                    "a0": 10000,
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
                repeats=20,
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
            WMCPBEVariantConfig(
                name="WMCPBE2",
                repeats=100,
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
        x_max_scale=1e-2,
        y_min_scale=2.0,
        y_max_scale=1e-2,
        total_number=1e4,
        volume_concentration=None,
    )

    advanced = PBEValidationAdvanced(config=config, init_dist=init_dist)
    result = advanced.run()

    advanced.print_moment_error_summary(result)
    advanced.plot_selected_moments(result, relative=True)
    advanced.plot_psd_snapshot(result, t_index=-1, two_d=True, marginal=True, total=True, q0=True, q3=True)
    advanced.plot_error_time_pareto(result)
    advanced.plot_moment_variances(result)
    advanced.show()