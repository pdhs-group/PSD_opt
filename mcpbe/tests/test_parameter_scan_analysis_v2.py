"""No PBE solve: exact finite-grid examples and small saved-trajectory fixtures."""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[2] / "scripts" / "LMC_MCPBE"
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import parameter_scan_analysis_v2 as analysis


def write_fixture(path):
    """48 complete conditions, three independent repeat trajectories."""
    settings = dict(mas_values=(.1, .9), x1_values=(.25, .75), str_values=(1., 2.),
                    gamma_values=(.1, 10.), n_repeats=3, mass_conservation_rtol=1e-12)
    strengths = tuple(s for s in product(settings["str_values"], repeat=3) if s[0] <= s[2])
    levels = (settings["mas_values"], settings["x1_values"], strengths, settings["gamma_values"])
    t = np.array([0., 1., 2.])
    volume = 50000.
    diameter = (6 * volume / np.pi) ** (1 / 3)
    grid = np.geomspace(.1, 100., 64)
    initial_cdf = (grid >= diameter).astype(float)
    fine_cdf = (grid >= .75 * diameter).astype(float)
    with h5py.File(path, "w") as f:
        f.attrs.update(initial_state_kind="monodisperse_2d_explicit",
                       initial_particle_volume=volume, config_fingerprint="synthetic-full-grid")
        f.create_dataset("config_json", data=json.dumps(settings).encode())
        f.create_dataset("t_vec", data=t)
        f.create_dataset("psd_x_grid", data=grid)
        groups = f.create_group("conditions")
        for ci, (mas, x1, strength, gamma) in enumerate(product(*levels)):
            g = groups.create_group(f"case_{ci:04d}")
            g.attrs.update(complete=True, MAS=mas, X1=x1, STR=strength, gamma=gamma)
            progress = .005 * (ci + 1) * np.array([.9, 1., 1.1])[:, None] * t
            m = np.empty((3, 3, 3, len(t)))
            for i, j in product(range(3), repeat=2):
                initial = 100 * (volume * x1) ** i * (volume * (1-x1)) ** j
                m[:, i, j] = initial
                if i + j >= 2:
                    m[:, i, j] *= 1 - progress / 10
            m[:, 0, 0] *= 1 + progress
            x = diameter * (1 - progress / 10)
            x[:, 0] = np.nan  # The recorded solver does not define its mono x50 at t=0.
            q = (1-progress[:, :, None]) * initial_cdf + progress[:, :, None] * fine_cdf
            events = np.zeros((3, len(t), 4))
            events[:, :, 1] = events[:, :, 3] = 100 * progress
            phase_volume = np.tile([100 * volume * x1, 100 * volume * (1-x1)], (3, 1))
            support = np.tile([.75*diameter, diameter], (3, len(t), 1))
            support[:, 0] = diameter
            values = dict(moments=m, moments_mean=m.mean(axis=0), x50=x, psd_Q=q,
                          psd_support=support, events=events, events_mean=events.mean(axis=0),
                          seed=np.arange(ci*3, ci*3+3), initial_phase_volume=phase_volume,
                          final_phase_volume=phase_volume, phase_volume_relative_error=np.zeros(3),
                          reconstruction_count=np.zeros(3))
            for name, array in values.items():
                g.create_dataset(name, data=array)
    return levels


def tiny_config(folder, **overrides):
    config = analysis.AnalysisConfig(result_h5=folder / "results.h5", output_directory=folder / "analysis",
        strict_checks=True, bootstrap_n=20, interaction_times=(1., 2.),
        plot_conditional_responses=False, plot_marginal_responses=False,
        plot_sensitivity=False, plot_interactions=False, n_plot_workers=1, figure_dpi=50)
    return replace(config, **overrides)


class FactorDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.signs = [np.array([-1., 1.]).reshape(tuple(2 if i == j else 1 for i in range(4)))
                      for j in range(4)]

    def test_each_pure_order_and_total_effect(self):
        for subset in analysis.SUBSETS:
            with self.subTest(subset=subset):
                values = np.ones((2, 2, 2, 2))
                for axis in subset:
                    values *= self.signs[axis]
                result = analysis.functional_anova(values[..., None, None], True)
                expected = np.zeros((15, 1, 1))
                expected[analysis.SUBSETS.index(subset)] = 1
                np.testing.assert_array_equal(result["components"], expected)
                np.testing.assert_array_equal(result["total"][:, 0, 0], [int(i in subset) for i in range(4)])

    def test_additive_and_interaction_closure(self):
        values = np.broadcast_to(2*self.signs[0] + 3*self.signs[1] + self.signs[2]*self.signs[3], (2, 2, 2, 2))
        result = analysis.functional_anova(values[..., None, None], True)
        self.assertEqual(result["variance"].item(), 14)
        self.assertEqual(result["components"][0].item(), 4)
        self.assertEqual(result["components"][1].item(), 9)
        self.assertEqual(result["components"][analysis.SUBSETS.index((2, 3))].item(), 1)

    def test_zero_variance_stays_undefined(self):
        result = analysis.functional_anova(np.full((2, 3, 2, 2, 3, 1), .1), True)
        np.testing.assert_array_equal(result["variance"], 0)
        self.assertTrue(np.isnan(analysis.defined_ratio(result["components"], result["variance"])).all())
        intervals, valid = analysis.percentile_interval(np.full((10, 3, 1), np.nan))
        self.assertTrue(np.isnan(intervals).all())
        np.testing.assert_array_equal(valid, 0)
        self.assertEqual(analysis._json_value({"undefined": np.nan}), {"undefined": None})

    def test_bootstrap_seed_and_whole_trajectory_association(self):
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(2, 2, 2, 2, 3, 1, 1)) * np.array([0., 1., 2.])[None, None, None, None, None, :, None]
        scan = dict(responses=raw, t=np.array([0., 1., 2.]))
        config = analysis.AnalysisConfig(bootstrap_n=20, strict_checks=True)
        first = analysis.bootstrap_analysis(scan, config)
        second = analysis.bootstrap_analysis(scan, config)
        for key in ("means", "curve_ci", "component_ci", "total_ci", "integrated_total_ci"):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first["curve_ci"][..., 2, :, :], 2 * first["curve_ci"][..., 1, :, :], rtol=0, atol=0)
        np.testing.assert_allclose(first["component_ci"][:, 1], first["component_ci"][:, 2], rtol=0, atol=0)

    def test_time_integration_uses_variances_not_average_indices(self):
        a = np.broadcast_to(self.signs[0], (2, 2, 2, 2))[..., None, None]
        b = np.broadcast_to(self.signs[1], (2, 2, 2, 2))[..., None, None]
        means = a * np.array([0., 1., 1.])[:, None] + b * np.array([0., 0., 2.])[:, None]
        scan = dict(responses=np.repeat(means[..., None, :, :], 3, axis=4), t=np.array([0., 1., 3.]))
        result = analysis.bootstrap_analysis(scan, analysis.AnalysisConfig(bootstrap_n=3, strict_checks=True))
        self.assertAlmostEqual(result["integrated_components"][0, 0], 2.5 / 6.5)
        self.assertNotAlmostEqual(result["integrated_components"][0, 0], (.2 + 1) / 2)


class SavedResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPT_DIRECTORY)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.config = tiny_config(self.folder)
        self.levels = write_fixture(self.config.result_h5)

    def test_strict_and_fast_have_identical_numeric_data(self):
        strict = analysis.read_results(self.config)
        with patch.object(analysis, "_sha256", side_effect=AssertionError("Fast mode must not hash")):
            fast = analysis.read_results(replace(self.config, strict_checks=False))
        self.assertIsNotNone(strict["input_sha256"])
        self.assertIsNone(fast["input_sha256"])
        np.testing.assert_array_equal(strict["responses"], fast["responses"])
        self.assertEqual(strict["responses"].shape, (2, 2, 6, 2, 3, 3, 8))
        one = analysis.bootstrap_analysis(strict, self.config)
        two = analysis.bootstrap_analysis(fast, replace(self.config, strict_checks=False))
        for key in ("means", "sd", "curve_ci", "component_index", "component_ci", "integrated_total_ci"):
            np.testing.assert_array_equal(one[key], two[key])

    def test_eight_response_formulas_and_initial_diameter(self):
        with h5py.File(self.config.result_h5, "r") as f:
            g = f["conditions/case_0000"]
            m, x, q, grid = g["moments"][()], g["x50"][()], g["psd_Q"][()], f["psd_x_grid"][()]
            diameter = (6*f.attrs["initial_particle_volume"] / np.pi) ** (1/3)
        y = analysis.compute_responses(m, x, q, grid, diameter, self.config)
        for column, (i, j) in enumerate(((0, 0), (2, 0), (0, 2), (1, 1))):
            np.testing.assert_array_equal(y[:, :, column], 100*(m[:, i, j] / m[:, i, j, :1] - 1))
        second = m[:, 2, 0] + 2*m[:, 1, 1] + m[:, 0, 2]
        np.testing.assert_array_equal(y[:, :, 4], 100*(second / second[:, :1] - 1))
        np.testing.assert_array_equal(y[:, 0], 0)
        np.testing.assert_array_equal(y[:, 1:, 5], 100*(x[:, 1:] / diameter - 1))
        ratio = np.array([[analysis._invert_cdf(grid, curve, .9) / analysis._invert_cdf(grid, curve, .1)
                           for curve in repeat] for repeat in q])
        np.testing.assert_array_equal(y[:, :, 6], 100*(ratio / ratio[:, :1] - 1))
        fine = np.array([[np.interp(.9*diameter, grid, curve) for curve in repeat] for repeat in q])
        np.testing.assert_array_equal(y[:, :, 7], 100*(fine - fine[:, :1]))

    def test_missing_and_unfinished_conditions_fail_even_in_fast_mode(self):
        with h5py.File(self.config.result_h5, "r+") as f:
            f["conditions/case_0000"].attrs["complete"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete condition"):
            analysis.read_results(replace(self.config, strict_checks=False))
        with h5py.File(self.config.result_h5, "r+") as f:
            del f["conditions/case_0000"]
        with self.assertRaisesRegex(ValueError, "Incomplete full factorial"):
            analysis.read_results(self.config)

    def test_shape_errors_fail(self):
        with h5py.File(self.config.result_h5, "r+") as f:
            g = f["conditions/case_0000"]
            del g["x50"]
            g.create_dataset("x50", data=np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            analysis.read_results(replace(self.config, strict_checks=False))

    def test_strict_nonfinite_and_nonmonotone_inputs_fail(self):
        with h5py.File(self.config.result_h5, "r+") as f:
            f["conditions/case_0000/moments"][0, 2, 0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            analysis.read_results(self.config)
        write_fixture(self.config.result_h5)
        with h5py.File(self.config.result_h5, "r+") as f:
            f["conditions/case_0000/psd_Q"][0, 1, :2] = [1, 0]
        with self.assertRaisesRegex(ValueError, "Invalid PSD"):
            analysis.read_results(self.config)

    def test_initial_value_and_quantile_domain_fail(self):
        with h5py.File(self.config.result_h5, "r+") as f:
            f["conditions/case_0000/moments"][0, 0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "positive initial"):
            analysis.read_results(replace(self.config, strict_checks=False))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            analysis._invert_cdf(np.array([1, 2]), np.array([.2, 1]), .1)

    def test_exports_undefined_values_and_output_guards(self):
        output = analysis.run_analysis(self.config)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["png_count"], 0)
        self.assertEqual(len(manifest["csv_rows"]), 8)
        self.assertEqual(manifest["csv_rows"]["condition_response_time.csv"], 48*3)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["n_conditions"], 48)
        self.assertIn("integrated_total", summary["responses"]["mu00"])
        self.assertIn("zero_variance", (output / "sensitivity_time.csv").read_text())
        with self.assertRaises(FileExistsError):
            analysis.run_analysis(self.config)
        scan = analysis.read_results(self.config)
        with self.assertRaisesRegex(ValueError, "different input/configuration"):
            analysis.prepare_output(scan, replace(self.config, overwrite_outputs=True, analysis_seed=43))
        with self.assertRaisesRegex(ValueError, "contents changed"):
            analysis.prepare_output(dict(scan, input_sha256="different"), replace(self.config, overwrite_outputs=True))
        analysis.prepare_output(scan, replace(self.config, overwrite_outputs=True, strict_checks=False))

    def test_all_constant_response_exports_null_sensitivities(self):
        with h5py.File(self.config.result_h5, "r+") as f:
            for g in f["conditions"].values():
                m = g["moments"][()]
                m[:, 0, 0] = 100
                g["moments"][:] = m
                g["moments_mean"][:] = m.mean(axis=0)
        config = replace(self.config, response_names=("mu00",), bootstrap_n=3)
        output = analysis.run_analysis(config)
        summary = json.loads((output / "summary.json").read_text())
        for scope in ("final", "integrated"):
            self.assertEqual(summary["responses"]["mu00"][f"{scope}_first"], [None]*4)
            self.assertEqual(summary["responses"]["mu00"][f"{scope}_total"], [None]*4)
            self.assertEqual(summary["responses"]["mu00"]["rankings"][f"{scope}_total"], [])

    def test_conditional_counts_and_png_layouts(self):
        default_levels = ((.1, .5, .9), (.1, .5, .9), tuple(range(40)), (.001, 1., 1000.))
        jobs = list(analysis.plot_jobs(dict(levels=default_levels), analysis.AnalysisConfig()))
        self.assertEqual(len(jobs), 1247)
        for factor, expected in zip(range(4), (360, 360, 108, 360)):
            self.assertEqual(sum(j[0] == "conditional" and j[1] == factor for j in jobs), expected)
        self.assertEqual(sum(j[0] == "marginal" for j in jobs), 7)
        self.assertEqual(sum(j[0] == "interaction" for j in jobs), 48)
        scan = analysis.read_results(self.config)
        result = analysis.bootstrap_analysis(scan, self.config)
        analysis._initialize_plot_worker(dict(scan=scan, result=result, config=self.config))
        representative_jobs = (("conditional", 2, (0, 0, 0), 0, tuple(range(6))),
            ("marginal", 0, (), 0, (0, 1)), ("first",), ("ranking",),
            ("interaction", (0, 2), 0), ("interaction", (2, 3), 0), ("interaction", (0, 1), 0))
        for job in representative_jobs:
            relative = analysis._render_job(job)
            with Image.open(self.config.output_directory / relative) as image:
                self.assertGreater(min(image.size), 200)
                image.verify()

    def test_invalid_configuration(self):
        for change in (dict(bootstrap_n=1), dict(response_names=("unknown",)),
                       dict(response_names=("mu00", "mu00")), dict(fine_diameter_fraction=0),
                       dict(n_plot_workers=0), dict(symlog_linthresh=np.nan)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis._validate_config(replace(self.config, **change))


if __name__ == "__main__":
    unittest.main()
