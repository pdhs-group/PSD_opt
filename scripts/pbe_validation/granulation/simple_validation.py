"""Small WMCPBE granulation validation example.

Edit CaseConfig and WMCPBEVariantConfig entries below to compare different
WMCPBE parameter choices from the same explicit weighted initial particles.
"""

from __future__ import annotations

import numpy as np

from validation import (
    CaseConfig,
    GranulationValidationConfig,
    GranulationValidationRunner,
    ValidationPlotter,
    WMCPBEVariantConfig,
)


def main() -> None:
    case = CaseConfig(
        dim=2,
        kernel="const",
        process="breakage",
        t_vec=np.arange(0.0, 10.0, 2.0, dtype=float),
        x=2e-1,
        beta0=1e-3,
        p1=3e-2,
        p2=1.0,
        initial_number_density=1.0,
        initial_total_weight=100000.0,
    )

    wmcpbe_variants = [
        WMCPBEVariantConfig(
            name="WMCPBE baseline",
            repeats=1,
            base_seed=42,
            attrs={
                "recon_enable": False,
                "break_dW_max": 1.0,
                "agg_dW_max": 1.0,
            },
        ),
        WMCPBEVariantConfig(
            name="WMCPBE reconstructed",
            repeats=1,
            base_seed=43,
            attrs={
                "recon_enable": True,
                "recon_method": "4PMC",
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_RS_target": 1000,
                "break_dW_max": 1.0,
                "agg_dW_max": 1.0,
            },
        ),
    ]

    config = GranulationValidationConfig(
        case=case,
        wmcpbe_variants=wmcpbe_variants,
        verbose=False,
    )

    result = GranulationValidationRunner(config).run()
    plotter = ValidationPlotter(result)
    plotter.plot_all_moments(relative=True, include_total_volume=False)
    plotter.show()


if __name__ == "__main__":
    main()
