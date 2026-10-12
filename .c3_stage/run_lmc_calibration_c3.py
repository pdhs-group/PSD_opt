"""Spyder entry: reuse C1/C2 states for the C3 damage and morphology comparison."""
from datetime import datetime
from pathlib import Path
from lmc_mod.c3_analysis import run, self_check

ROOT = Path(__file__).resolve().parents[1]
STRICT_CHECKS = False
ACTION = "analyze"  # analyze | smoke; neither runs DEM or LMC fracture simulations
CONFIG = {
    "output": str(ROOT / "runs" / ("lmc_calibration_C3_" + datetime.now().strftime("%d%m%y")) / datetime.now().strftime("%H%M%S_%f")),
    "c2_root": str(ROOT / "runs/lmc_calibration_C2_121026/133644_264497"),
    "dem_case": str(ROOT / "runs/lmc_parameter_calibration_111026/194626_656598/C1/reference_dt"),
    "dem_history": str(ROOT / "runs/reference_dt_extension_121026/115542_571249/analysis/reference_dt"),
    "strict_checks": STRICT_CHECKS,
    "primary_threshold": 0.05,
    "reference_dt_s": 25e-9,
    "reference_speed_m_s": 0.005,
    # Descriptive resolution windows, not physical acceptance tolerances.
    "resolution_windows": {"f2_initial": 1 / 400, "alpha": 1 / 400, "eta_cut": 1 / 20},
    "comparison_protocol": {
        "candidates": "Pareto frontier of absolute coarse-group mean deviations in f2_initial, alpha, eta_cut; no weighted score",
        "damage": "report separately, including mother-induced-edge denominator and child-interior partition; no state-equivalence claim from size alone",
        "near_case": "joint resolution-window count divided by ALL realizations; nonreach counts as not near; descriptive only",
        "confirmation": "report 500-new-seed groups separately; do not pool with coarse groups or budget repeats",
        "orientation": "report loading alignment; exclude from shape-only ranking because current LMC has no loading input",
        "dem_scope": "single fixed 25 ns / 0.005 m/s realization; no inference of physical population variability or dt-as-random-seed",
    },
}

if __name__ == "__main__":
    if ACTION == "smoke":
        print(self_check())
    elif ACTION == "analyze":
        OUTPUT = run(CONFIG)
        print(OUTPUT)
    else:
        raise ValueError("ACTION must be analyze or smoke")
