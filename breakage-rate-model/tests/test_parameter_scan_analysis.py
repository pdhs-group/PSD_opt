# -*- coding: utf-8 -*-
"""Small contract checks for the standalone phase-1 analysis script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "LMC_MCPBE" / "parameter_scan_analysis.py"
SPEC = importlib.util.spec_from_file_location("parameter_scan_analysis", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def _write_minimal_results(path: Path) -> None:
    t_vec = np.array([0.0, 1.0, 2.0])
    x_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cdf = np.tile(np.array([0.0, 0.1, 0.5, 0.9, 1.0]), (t_vec.size, 1))
    requested = set()
    for MAS in ANALYSIS.MAS_VALUES:
        requested.add((MAS, ANALYSIS.BASE_X1, *ANALYSIS.BASE_STR, ANALYSIS.BASE_GAMMA))
    for X1 in ANALYSIS.X1_VALUES:
        requested.add((ANALYSIS.BASE_MAS, X1, *ANALYSIS.BASE_STR, ANALYSIS.BASE_GAMMA))
    for gamma in ANALYSIS.GAMMA_VALUES:
        requested.add((ANALYSIS.BASE_MAS, ANALYSIS.BASE_X1, *ANALYSIS.BASE_STR, gamma))
    for STR in ANALYSIS.STR_COMPARISONS:
        stored_STR, _ = ANALYSIS._canonical_str_request(STR, ANALYSIS.BASE_X1)
        requested.add((ANALYSIS.BASE_MAS, ANALYSIS.BASE_X1, *stored_STR, ANALYSIS.BASE_GAMMA))

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("t_vec", data=t_vec)
        h5_file.create_dataset("psd_x_grid", data=x_grid)
        conditions = h5_file.create_group("conditions")
        for index, (MAS, X1, STR0, STR1, STR2, gamma) in enumerate(sorted(requested)):
            group = conditions.create_group(f"case_{index:04d}")
            group.attrs["complete"] = True
            group.attrs["MAS"] = MAS
            group.attrs["X1"] = X1
            group.attrs["STR"] = (STR0, STR1, STR2)
            group.attrs["gamma"] = gamma
            group.create_dataset("psd_Q_mean", data=cdf)
            group.create_dataset("x50_mean", data=np.full(t_vec.size, 3.0))
            moments = np.empty((3, 3, t_vec.size))
            for i in range(3):
                for j in range(3):
                    moments[i, j] = (i + 1) * 10 + j + t_vec
            group.create_dataset("moments_mean", data=moments)


class TestParameterScanAnalysis(unittest.TestCase):
    def test_relative_changes_keep_sign_and_use_symlog_axes(self) -> None:
        np.testing.assert_allclose(ANALYSIS._relative_change_percent(np.array([10.0, 12.0, 8.0]), 10.0), [0.0, 20.0, -20.0])
        with self.assertRaises(ValueError):
            ANALYSIS._relative_change_percent(np.array([1.0]), 0.0)

        cdf = np.array([[0.0, 0.1, 0.5, 0.9, 1.0], [0.0, 0.05, 0.4, 0.9, 1.0]])
        data = {"psd_Q": cdf, "moments": np.full((3, 3, 2), [10.0, 12.0])}
        grid = np.arange(1.0, 6.0)
        with patch.object(ANALYSIS, "_save_figure", side_effect=lambda fig, path: path):
            ANALYSIS._plot_final_diameters("MAS", (0.5,), [data], grid, Path("plots"))
            figure = plt.gcf()
            left, right = figure.axes
            self.assertEqual((left.get_yscale(), right.get_yscale()), ("symlog", "symlog"))
            self.assertAlmostEqual(left.lines[0].get_ydata()[0], -6.66666666666667)
            self.assertAlmostEqual(right.lines[0].get_ydata()[0], 6.66666666666667)
            plt.close(figure)

            ANALYSIS._plot_time_series("MAS", (0.5,), [data], np.array([0.0, 1.0]), Path("plots"), grid, (2, 0))
            figure = plt.gcf()
            self.assertEqual(figure.axes[0].get_yscale(), "symlog")
            np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [0.0, 20.0])
            plt.close(figure)

    def test_reversed_str_swaps_phase_moments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "phase1_results.h5"
            _write_minimal_results(result_path)
            with h5py.File(result_path, "r") as h5_file:
                index = ANALYSIS._condition_index(h5_file)
                mirrored = ANALYSIS._read_requested_str_condition(
                    index, (1000.0, 1000.0, 1.0), 3, 5
                )["moments"]
                stored = ANALYSIS._read_condition(
                    index, 0.5, 0.5, (1.0, 1000.0, 1000.0), 1.0, 3, 5
                )["moments"]
            np.testing.assert_array_equal(mirrored[2, 0], stored[0, 2])
            np.testing.assert_array_equal(mirrored[0, 2], stored[2, 0])
            np.testing.assert_array_equal(mirrored[1, 1], stored[1, 1])
            with self.assertRaises(ValueError):
                ANALYSIS._canonical_str_request((1000.0, 1000.0, 1.0), 0.1)

    def test_writes_all_requested_figures_and_inverts_cdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "phase1_results.h5"
            output_directory = Path(directory) / "plots"
            _write_minimal_results(result_path)
            outputs = ANALYSIS.run_analysis(result_path, output_directory)
            self.assertEqual(len(outputs), 18)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertEqual(
                len(ANALYSIS.run_analysis(result_path, output_directory, run_str_moment_comparisons=False)), 15
            )
            self.assertEqual(
                len(ANALYSIS.run_analysis(result_path, output_directory, run_primary_comparisons=False)), 3
            )
            self.assertEqual(ANALYSIS._invert_cdf(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.5, 1.0]), 0.5), 2.0)
            figure, axis = plt.subplots()
            ANALYSIS._set_scan_x_axis(axis, "gamma")
            self.assertEqual(axis.get_xscale(), "log")
            plt.close(figure)

    def test_invalid_cdf_and_missing_dataset_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "phase1_results.h5"
            _write_minimal_results(result_path)
            with h5py.File(result_path, "r+") as h5_file:
                group = next(iter(h5_file["conditions"].values()))
                group["psd_Q_mean"][0, 2] = 0.05
            with self.assertRaises(ValueError):
                ANALYSIS.run_analysis(result_path, Path(directory) / "plots")

            _write_minimal_results(result_path)
            with h5py.File(result_path, "r+") as h5_file:
                group = next(iter(h5_file["conditions"].values()))
                del group["x50_mean"]
            with self.assertRaises(KeyError):
                ANALYSIS.run_analysis(result_path, Path(directory) / "plots")


if __name__ == "__main__":
    unittest.main()
