# -*- coding: utf-8 -*-
"""Unit tests for the phase-1 LMC--MCPBE scan orchestration helpers."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCAN_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "LMC_MCPBE" / "parameter_scan.py"
SCAN_SPEC = importlib.util.spec_from_file_location("phase1_parameter_scan", SCAN_SCRIPT_PATH)
assert SCAN_SPEC is not None and SCAN_SPEC.loader is not None
SCAN_MODULE = importlib.util.module_from_spec(SCAN_SPEC)
sys.modules[SCAN_SPEC.name] = SCAN_MODULE
SCAN_SPEC.loader.exec_module(SCAN_MODULE)


def _fake_result(config, case):
    repeats = config.n_repeats
    time_count = config.n_time_points
    grid_count = len(config.psd_x_grid)
    order_count = config.moment_max_order + 1
    return {
        "case": case.as_dict(),
        "seed": np.arange(10, 10 + repeats, dtype=np.uint64),
        "moments": np.full((repeats, order_count, order_count, time_count), 2.0),
        "psd_Q": np.full((repeats, time_count, grid_count), 0.5),
        "x50": np.full((repeats, time_count), 4.0),
        "psd_support": np.tile(np.array([1.0, 10.0]), (repeats, time_count, 1)),
        "events": np.full((repeats, time_count, 4), 3.0),
        "initial_phase_volume": np.full((repeats, 2), 500.0),
        "final_phase_volume": np.full((repeats, 2), 500.0),
        "phase_volume_relative_error": np.zeros(repeats),
        "reconstruction_count": np.arange(repeats, dtype=int),
        "machine_seconds": np.arange(1, repeats + 1, dtype=float),
    }


def _full_scan_config():
    """Make tests independent from Spyder's editable single-test switch."""
    return replace(
        SCAN_MODULE.DEFAULT_CONFIG,
        single_test_case=False,
        result_filename=SCAN_MODULE.RESULT_FILENAME,
        mas_values=(0.1, 0.5, 0.9),
        x1_values=(0.1, 0.5, 0.9),
        str_values=(1.0, 10.0, 100.0, 1000.0),
        gamma_values=(1.0e-3, 1.0, 1.0e3),
        case_indices=None,
    )


class TestPhase1ParameterScan(unittest.TestCase):
    def test_case_enumeration_and_deterministic_repeat_seeds(self) -> None:
        config = _full_scan_config()
        cases = SCAN_MODULE.build_scan_cases(config)
        self.assertEqual(len(cases), 1080)
        self.assertEqual(len({case.case_id for case in cases}), 1080)
        str_combinations = {case.STR for case in cases}
        self.assertEqual(len(str_combinations), 40)
        self.assertTrue(all(STR0 <= STR2 for STR0, _, STR2 in str_combinations))

        first = SCAN_MODULE.derive_repeat_seeds(config, cases[17].index)
        second = SCAN_MODULE.derive_repeat_seeds(config, cases[17].index)
        self.assertEqual(first, second)
        self.assertEqual(len(first), config.n_repeats)
        self.assertEqual(len(set(first)), config.n_repeats)
        self.assertNotEqual(first, SCAN_MODULE.derive_repeat_seeds(config, cases[18].index))

    def test_single_test_branch_overrides_scan_space_and_case_selection(self) -> None:
        original_flag = SCAN_MODULE.RUN_SINGLE_TEST_CASE
        try:
            SCAN_MODULE.RUN_SINGLE_TEST_CASE = True
            config = SCAN_MODULE.build_default_config()
        finally:
            SCAN_MODULE.RUN_SINGLE_TEST_CASE = original_flag

        self.assertTrue(config.single_test_case)
        self.assertEqual(config.result_filename, SCAN_MODULE.SINGLE_TEST_RESULT_FILENAME)
        self.assertEqual(config.mas_values, (0.5,))
        self.assertEqual(config.x1_values, (0.5,))
        self.assertEqual(config.str_values, (1.0,))
        self.assertEqual(config.gamma_values, (1.0,))
        self.assertIsNone(config.case_indices)

        cases = SCAN_MODULE.build_scan_cases(config)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].MAS, 0.5)
        self.assertEqual(cases[0].X1, 0.5)
        self.assertEqual(cases[0].STR, (1.0, 1.0, 1.0))
        self.assertEqual(cases[0].gamma, 1.0)
        SCAN_MODULE._validate_config_values(config, cases)

    def test_validation_rejects_invalid_values_before_asset_access(self) -> None:
        config = _full_scan_config()
        cases = SCAN_MODULE.build_scan_cases(config)
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(replace(config, model_kind="unknown"), cases)
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, model_logV_bounds=(2.0, 2.0)), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, psd_x_grid=(1.0, 0.5)), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, break_dW_const=0.0), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, agg_dW_const=float("inf")), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, initial_compute_particles=1.5), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, initial_weight_per_compute_particle=0.0), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, recon_method="unknown"), cases
            )
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(
                replace(config, recon_QMX_q_small=0.9, recon_QMX_q_tail=0.2), cases
            )
        bad_gamma = replace(config, gamma_values=(1.0e-3, 0.0, 1.0e3))
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(bad_gamma, SCAN_MODULE.build_scan_cases(bad_gamma))
        bad_str = replace(config, str_values=(0.0, 1.0e1, 1.0e2, 1.0e3))
        with self.assertRaises(ValueError):
            SCAN_MODULE._validate_config_values(bad_str, SCAN_MODULE.build_scan_cases(bad_str))

        with tempfile.TemporaryDirectory() as directory:
            missing_assets = replace(
                config,
                model_path=Path(directory) / "missing.pkl",
                aggregate_pool_root=Path(directory) / "missing_pool",
            )
            with self.assertRaises(FileNotFoundError):
                SCAN_MODULE._validate_assets(missing_assets)
            existing_model = Path(directory) / "model.pkl"
            existing_model.touch()
            missing_pool = replace(
                config,
                model_path=existing_model,
                aggregate_pool_root=Path(directory) / "missing_pool",
            )
            with self.assertRaises(FileNotFoundError):
                SCAN_MODULE._validate_assets(missing_pool)

    def test_refresh_classification(self) -> None:
        lmc_refresh = SCAN_MODULE.classify_parameter_changes(("MAS", "gamma"))
        self.assertFalse(lmc_refresh.rebuild_particles)
        self.assertTrue(lmc_refresh.rebuild_lmc)
        self.assertTrue(lmc_refresh.rebuild_samplers)

        particle_refresh = SCAN_MODULE.classify_parameter_changes(("X1",))
        self.assertTrue(particle_refresh.rebuild_particles)
        self.assertFalse(particle_refresh.rebuild_lmc)
        self.assertTrue(particle_refresh.rebuild_samplers)

        time_only = SCAN_MODULE.classify_parameter_changes(("end_time",))
        self.assertFalse(time_only.rebuild_particles)
        self.assertFalse(time_only.rebuild_lmc)
        self.assertFalse(time_only.rebuild_samplers)

        packet_refresh = SCAN_MODULE.classify_parameter_changes(
            ("break_dW_const", "agg_dW_const")
        )
        self.assertFalse(packet_refresh.rebuild_particles)
        self.assertFalse(packet_refresh.rebuild_lmc)
        self.assertTrue(packet_refresh.rebuild_samplers)

        self.assertFalse(hasattr(SCAN_MODULE.DEFAULT_CONFIG, "break_dW_max"))
        self.assertFalse(hasattr(SCAN_MODULE.DEFAULT_CONFIG, "agg_dW_max"))
        self.assertFalse(hasattr(SCAN_MODULE.DEFAULT_CONFIG, "agg_dW_min"))

        reconstruction_only = SCAN_MODULE.classify_parameter_changes(
            ("recon_enable", "recon_N_max", "recon_RS_target")
        )
        self.assertFalse(reconstruction_only.rebuild_particles)
        self.assertFalse(reconstruction_only.rebuild_lmc)
        self.assertFalse(reconstruction_only.rebuild_samplers)

    def test_explicit_monodisperse_initial_state_and_reconstruction_transfer(self) -> None:
        config = SCAN_MODULE.DEFAULT_CONFIG
        volumes, weights = SCAN_MODULE.build_initial_state(config, X1=0.25)
        self.assertEqual(volumes.shape, (3, config.initial_compute_particles))
        np.testing.assert_allclose(volumes[0], 0.25 * config.initial_particle_volume)
        np.testing.assert_allclose(volumes[1], 0.75 * config.initial_particle_volume)
        np.testing.assert_allclose(volumes[2], config.initial_particle_volume)
        np.testing.assert_allclose(weights, config.initial_weight_per_compute_particle)
        self.assertEqual(
            config.initial_represented_particle_count,
            config.initial_compute_particles * config.initial_weight_per_compute_particle,
        )
        self.assertEqual(
            config.initial_represented_total_volume,
            config.initial_compute_particles
            * config.initial_weight_per_compute_particle
            * config.initial_particle_volume,
        )

        modified = replace(
            config,
            initial_compute_particles=3,
            initial_weight_per_compute_particle=2.5,
            initial_particle_volume=80.0,
            recon_enable=True,
            recon_method="4PM",
            recon_N_max=90,
            recon_every_events=7,
            recon_tail_protect=4,
        )
        volumes, weights = SCAN_MODULE.build_initial_state(modified, X1=0.4)
        self.assertEqual(volumes.shape, (3, 3))
        np.testing.assert_allclose(volumes[:, 0], (32.0, 48.0, 80.0))
        np.testing.assert_allclose(weights, 2.5)
        self.assertEqual(modified.initial_represented_particle_count, 7.5)
        self.assertEqual(modified.initial_represented_total_volume, 600.0)

        solver = SCAN_MODULE.MCPBESolver(
            dim=2,
            t_vec=np.array([0.0, 1.0]),
            load_attr=False,
            init=False,
            seed=42,
        )
        case = SCAN_MODULE.build_scan_cases(modified)[0]
        SCAN_MODULE.apply_case_to_solver(solver, modified, case)
        self.assertEqual(solver.a0, 7.5)
        self.assertTrue(solver.recon_enable)
        self.assertEqual(solver.recon_method, "4PM")
        self.assertEqual(solver.recon_N_max, 90)
        self.assertEqual(solver.recon_every_events, 7)
        self.assertEqual(solver.recon_tail_protect, 4)

    def test_h5_write_csv_and_strict_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                _full_scan_config(),
                output_directory=Path(directory),
                result_filename="phase1_test.h5",
                case_indices=(0,),
                print_every_condition=False,
                plot_case_ids=(),
            )
            cases = SCAN_MODULE.build_scan_cases(config)
            pending = SCAN_MODULE.prepare_result_file(config, cases)
            self.assertEqual([case.index for case in pending], [0])

            result = _fake_result(config, cases[0])
            with h5py.File(config.result_path, "a") as h5_file:
                SCAN_MODULE.write_condition_result(h5_file, result, config)

            self.assertEqual(SCAN_MODULE.prepare_result_file(config, cases), [])
            condition_csv, time_csv = SCAN_MODULE.export_csv_summaries(config.result_path)
            self.assertTrue(condition_csv.is_file())
            self.assertTrue(time_csv.is_file())

            with h5py.File(config.result_path, "r") as h5_file:
                self.assertEqual(h5_file.attrs["format_version"], 2)
                self.assertEqual(h5_file.attrs["initial_state_kind"], "monodisperse_2d_explicit")
                self.assertEqual(
                    h5_file.attrs["initial_compute_particles"],
                    config.initial_compute_particles,
                )
                self.assertEqual(
                    h5_file.attrs["initial_represented_particle_count"],
                    config.initial_represented_particle_count,
                )
                group = h5_file["conditions"][cases[0].case_id]
                self.assertTrue(bool(group.attrs["complete"]))
                self.assertEqual(
                    group["moments"].shape,
                    (
                        config.n_repeats,
                        config.moment_max_order + 1,
                        config.moment_max_order + 1,
                        config.n_time_points,
                    ),
                )
                self.assertEqual(
                    group["psd_Q"].shape,
                    (config.n_repeats, config.n_time_points, len(config.psd_x_grid)),
                )
                expected_reconstruction_counts = np.arange(config.n_repeats, dtype=int)
                np.testing.assert_array_equal(
                    group["reconstruction_count"], expected_reconstruction_counts
                )
                self.assertEqual(
                    group["reconstruction_count_mean"][()],
                    float(np.mean(expected_reconstruction_counts)),
                )
                np.testing.assert_allclose(group["x50_mean"], 4.0)

            full_selection = replace(config, case_indices=None)
            full_pending = SCAN_MODULE.prepare_result_file(
                full_selection, SCAN_MODULE.build_scan_cases(full_selection)
            )
            self.assertEqual(len(full_pending), 1079)
            self.assertNotIn(cases[0].case_id, {case.case_id for case in full_pending})

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                changed = replace(config, recon_N_max=config.recon_N_max + 1)
                SCAN_MODULE.prepare_result_file(changed, SCAN_MODULE.build_scan_cases(changed))


if __name__ == "__main__":
    unittest.main()
