"""Basic example for the new validation workflow."""

from __future__ import annotations

import sys
from pathlib import Path

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

from validation import (  # noqa: E402
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
        process="agglomeration",
        t_vec=np.arange(0.0, 25.0, 2.0, dtype=float),
        c=1.0,
        x=2e-1,
        beta0=1e-3,
        p1=3e-2,
        p2=1.0,
        use_psd=False,
    )

    dpbe_variants = [
        DPBEVariantConfig(name="dPBE (NS=15)", grid="geo", ns=15, s=2),
        # DPBEVariantConfig(name="dPBE (NS=25)", grid="geo", ns=25, s=2),
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
                "a0": 100000,
                "V_eff_init": 1000,
                "recon_enable": False,
                "recon_N_max": 4000,
                "recon_bins": 30,
                "recon_method": "4PMC",
                "break_dW_max": 50.0,
                "agg_dW_max": 10.0
            },
        ),
    ]

    qmom_variants = [
        # QMOMVariantConfig(name="QMOM", n_order=2, n_add=10),
    ]

    config = ValidationConfig(
        case=case,
        dpbe_variants=dpbe_variants,
        wmcpbe_variants=wmcpbe_variants,
        qmom_variants=qmom_variants,
        reference_dpbe_name="dPBE (NS=15)",
    )

    result = ValidationRunner(config).run()

    plotter = ValidationPlotter(result)
    plotter.plot_all_moments(relative=True, include_total_volume=False)
    plotter.show()
