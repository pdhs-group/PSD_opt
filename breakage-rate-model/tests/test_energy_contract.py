# -*- coding: utf-8 -*-
"""Regression tests for the v2 energy-pool and full-feature surrogate contract."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "breakage-rate-model" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "mcpbe" / "src"))

from breakage_rate_model.data_io import load_energy_groups_from_h5
from breakage_rate_model.data_io import EnergyGroupRecord
from breakage_rate_model.ann_model import ANNEnergyModel
from breakage_rate_model.datasets import build_energy_dataset
from breakage_rate_model.mlp_model import MLPEnergyModel
from breakage_rate_model.parametric_model import ParametricEnergyModel
from breakage_rate_model.powerlaw_separable import PowerLawSeparableModel
from wmcpbe.breakage_adapter import BreakageRateAdapter


MERGE_SCRIPT_PATH = PROJECT_ROOT / "agggenerator" / "scripts" / "merge_energy_pool_h5.py"
os.environ.setdefault("STORAGE_PATH", str(PROJECT_ROOT))
MERGE_SPEC = importlib.util.spec_from_file_location("merge_energy_pool_h5", MERGE_SCRIPT_PATH)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
MERGE_MODULE = importlib.util.module_from_spec(MERGE_SPEC)
MERGE_SPEC.loader.exec_module(MERGE_MODULE)

TRAINING_SCRIPT_PATH = PROJECT_ROOT / "breakage-rate-model" / "tests" / "train_4_models.py"
TRAINING_SPEC = importlib.util.spec_from_file_location("energy_training_entry", TRAINING_SCRIPT_PATH)
assert TRAINING_SPEC is not None and TRAINING_SPEC.loader is not None
TRAINING_MODULE = importlib.util.module_from_spec(TRAINING_SPEC)
sys.modules[TRAINING_SPEC.name] = TRAINING_MODULE
TRAINING_SPEC.loader.exec_module(TRAINING_MODULE)


def create_completed_run(h5_file: h5py.File, key: str, *, complete: bool = True) -> h5py.Group:
    """Create a small, strict-version-2 run suitable for contract tests."""
    group = h5_file.require_group("runs").create_group(key)
    n_np, n_grids, n_fracs = 3, 2, 3
    np_values = np.array([1, 2, 3], dtype=np.int64)
    a0 = 2.0
    samples = np.arange(1, n_np * n_grids * n_fracs + 1, dtype=float).reshape(
        n_np, n_grids, n_fracs
    )
    group.attrs.update(
        {
            "NO_FRAG": 2,
            "int_bre": 0.0,
            "gamma": 6.0,
            "Df": 1.8,
            "MAS": 0.1,
            "X1": 0.2,
            "A0": a0,
            "N_GRIDS": n_grids,
            "N_FRACS": n_fracs,
            "base_seed": 123,
            "workers": 1,
            "STR": np.array([1.0, 1.1, 1.2]),
            "checkpoint_version": 2,
            "expected_tasks": n_np * n_grids,
            "completed_tasks": n_np * n_grids if complete else n_np * n_grids - 1,
        }
    )
    group.create_dataset("Np", data=np_values, compression="gzip")
    group.create_dataset("V", data=np_values.astype(float) * a0, compression="gzip")
    group.create_dataset("E_samples", data=samples, compression="gzip")
    group.create_dataset("completed_mask", data=np.full((n_np, n_grids), complete), compression="gzip")
    group.create_dataset("E_mean", data=np.mean(samples, axis=(1, 2)), compression="gzip")
    group.create_dataset("E_std", data=np.std(samples, axis=(1, 2)), compression="gzip")
    return group


class TestEnergyGroupReader(unittest.TestCase):
    def test_default_reader_omits_samples_and_per_sample_is_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.h5"
            with h5py.File(path, "w") as h5_file:
                create_completed_run(h5_file, "curve")

            groups = load_energy_groups_from_h5(str(path), load_samples=False)
            self.assertIsNone(groups[0].E_samples)
            mean_dataset = build_energy_dataset(groups, target="log_mean")
            self.assertEqual(mean_dataset.X.shape, (3, 10))

            sampled_groups = load_energy_groups_from_h5(str(path), load_samples=True)
            quantile_dataset = build_energy_dataset(
                sampled_groups, target="quantile", quantile=0.5
            )
            sample_dataset = build_energy_dataset(sampled_groups, per_sample=True, target="log_mean")
            self.assertEqual(quantile_dataset.y.shape, (3,))
            self.assertEqual(sample_dataset.y.ndim, 1)
            self.assertEqual(sample_dataset.y.size, 3 * 2 * 3)
            np.testing.assert_array_equal(sample_dataset.meta_idx[:6, 2], np.arange(6))

            with self.assertRaisesRegex(ValueError, "load_samples=True"):
                build_energy_dataset(groups, target="quantile", quantile=0.5)
            with self.assertRaisesRegex(ValueError, "load_samples=True"):
                build_energy_dataset(groups, per_sample=True)

    def test_invalid_v2_inputs_fail_strictly(self) -> None:
        cases = ("missing_attr", "old_samples", "unfinished", "bad_mean", "wrong_count")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{case}.h5"
                with h5py.File(path, "w") as h5_file:
                    group = create_completed_run(h5_file, "curve")
                    if case == "missing_attr":
                        del group.attrs["STR"]
                    elif case == "old_samples":
                        del group["E_samples"]
                        group.create_dataset("E_samples", data=np.ones((3, 2)))
                    elif case == "unfinished":
                        group["completed_mask"][0, 0] = False
                    elif case == "bad_mean":
                        group["E_mean"][0] = np.nan
                    elif case == "wrong_count":
                        group.attrs["completed_tasks"] = 1

                with self.assertRaises(ValueError):
                    load_energy_groups_from_h5(str(path), load_samples=False)

    def test_nonfinite_samples_fail_when_samples_are_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_samples.h5"
            with h5py.File(path, "w") as h5_file:
                group = create_completed_run(h5_file, "curve")
                group["E_samples"][0, 0, 0] = np.nan
            with self.assertRaises(ValueError):
                load_energy_groups_from_h5(str(path), load_samples=True)

    def test_group_split_selects_rows_by_meta_index_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.h5"
            with h5py.File(path, "w") as h5_file:
                for index in range(4):
                    create_completed_run(h5_file, f"curve-{index}")
            groups = load_energy_groups_from_h5(str(path), load_samples=False)
            dataset = build_energy_dataset(groups, target="log_mean")
            split = TRAINING_MODULE.split_train_val_by_group(
                groups, dataset, val_ratio=0.5, seed=3
            )
            self.assertFalse(
                np.intersect1d(split.train_group_indices, split.val_group_indices).size
            )
            self.assertTrue(np.isin(split.meta_train[:, 0], split.train_group_indices).all())
            self.assertTrue(np.isin(split.meta_val[:, 0], split.val_group_indices).all())


class TestFeatureSelectionAndAdapter(unittest.TestCase):
    def test_powerlaw_ignores_excluded_features_and_uses_str(self) -> None:
        model = PowerLawSeparableModel(active_feature_names=("logV", "STR0"))
        model._theta_dim = 1
        model._coef_erosion = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        model._theta_scaler_erosion = (np.zeros(1), np.ones(1))
        model._is_fitted = True

        first = np.array([[np.log(2.0), np.log(6.0), np.log(2.0), 0.0, 1.8, 0.1, 0.2, 1.0, 1.1, 1.2]])
        changed_excluded = first.copy()
        changed_excluded[0, 2] = np.log(7.0)
        changed_excluded[0, 4] = 2.6
        changed_str = first.copy()
        changed_str[0, 7] = 2.0

        first_prediction = model.predict(first)
        np.testing.assert_allclose(model.predict(changed_excluded), first_prediction)
        self.assertNotAlmostEqual(float(model.predict(changed_str)[0]), float(first_prediction[0]))

        model._coef_erosion = None
        with self.assertRaisesRegex(RuntimeError, "not trained"):
            model.predict(first)

    def test_adapter_passes_full_features_to_default_active_mlp(self) -> None:
        X = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.8, 0.0, 0.1, 1.0, 1.1, 1.2],
                [0.2, 1.1, 0.0, 0.0, 1.8, 0.1, 0.2, 1.1, 1.2, 1.3],
                [0.4, 1.2, 0.0, 0.0, 1.8, 0.2, 0.3, 1.2, 1.3, 1.4],
                [0.6, 1.3, 0.0, 0.0, 1.8, 0.3, 0.4, 1.3, 1.4, 1.5],
            ],
            dtype=float,
        )
        y = np.array([0.0, 0.1, 0.2, 0.3], dtype=float)
        model = MLPEnergyModel(
            hidden_sizes=(4,), max_epochs=1, batch_size=2, patience=None, seed=1
        ).fit(X, y)

        class PBE:
            dim = 1
            a_tot = 2
            V_flat = np.array([[1.0, 2.0]])

        adapter = BreakageRateAdapter(
            model_kind="mlp", model=model, gamma=6.0, NO_FRAG=2.0, Df=1.8
        )
        features, _ = adapter._build_features_batch(PBE())
        self.assertEqual(features.shape, (2, 10))
        rates = adapter.compute_rates_full(PBE())
        self.assertEqual(rates.shape, (2,))
        self.assertTrue(np.all(np.isfinite(rates)))

        with self.assertRaisesRegex(TypeError, "model_kind='ann'"):
            BreakageRateAdapter(model_kind="ann", model=model)

    def test_adapter_model_volume_warning_is_once_and_can_be_disabled(self) -> None:
        X = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.8, 0.0, 0.1, 1.0, 1.1, 1.2],
                [0.2, 1.1, 0.0, 0.0, 1.8, 0.1, 0.2, 1.1, 1.2, 1.3],
                [0.4, 1.2, 0.0, 0.0, 1.8, 0.2, 0.3, 1.2, 1.3, 1.4],
                [0.6, 1.3, 0.0, 0.0, 1.8, 0.3, 0.4, 1.3, 1.4, 1.5],
            ],
            dtype=float,
        )
        model = MLPEnergyModel(
            hidden_sizes=(4,), max_epochs=1, batch_size=2, patience=None, seed=1
        ).fit(X, np.array([0.0, 0.1, 0.2, 0.3], dtype=float))

        class PBE:
            dim = 1
            a_tot = 2
            V_flat = np.array([[1.0, 2.0]])

        adapter = BreakageRateAdapter(
            model_kind="mlp", model=model, model_logV_bounds=(0.1, 0.5)
        )
        from contextlib import redirect_stdout
        from io import StringIO

        with StringIO() as stream, redirect_stdout(stream):
            adapter.compute_rates_full(PBE())
            first_output = stream.getvalue()
            stream.seek(0)
            stream.truncate(0)
            adapter.compute_rates_full(PBE())
            second_output = stream.getvalue()
        self.assertIn("below", first_output)
        self.assertIn("above", first_output)
        self.assertEqual(second_output, "")

        silent_adapter = BreakageRateAdapter(
            model_kind="mlp",
            model=model,
            model_logV_bounds=(0.1, 0.5),
            warn_model_extrapolation=False,
        )
        with StringIO() as stream, redirect_stdout(stream):
            silent_adapter.compute_rates_full(PBE())
            self.assertEqual(stream.getvalue(), "")


class TestFourModelGroupComparison(unittest.TestCase):
    @staticmethod
    def _record(index: int) -> EnergyGroupRecord:
        np_values = np.arange(1, 7, dtype=np.int64)
        scale = 1.0 + 0.05 * index
        energy = scale * np_values.astype(float) ** 0.8
        return EnergyGroupRecord(
            key=f"training-record-{index}",
            NO_FRAG=2,
            int_bre=0.0,
            gamma=2.0 + index,
            Df=1.8,
            MAS=0.1 * index,
            X1=0.2 + 0.01 * index,
            A0=1.0,
            N_GRIDS=1,
            N_FRACS=1,
            base_seed=index,
            workers=1,
            STR=np.array([1.0 + index, 2.0 + index, 3.0 + index]),
            sigma_attr=None,
            pearson_r_attr=None,
            Np=np_values,
            V=np_values.astype(float),
            E_mean=energy,
            E_std=np.zeros_like(energy),
            E_samples=None,
            sigma_fit=0.8,
            b_fit=float(np.log(scale)),
            pearson_r_fit=1.0,
        )

    @classmethod
    def _save_all_models(cls, directory: Path) -> dict[str, object]:
        groups = [cls._record(index) for index in range(12)]
        dataset = build_energy_dataset(groups, target="log_mean")
        models = {
            "powerlaw": PowerLawSeparableModel(
                enable_tail=True,
                regress_type="ridge",
                ridge_lambda=1.0,
            ).fit(None, None, groups=groups),
            "parametric": ParametricEnergyModel(
                enable_tail=True,
                residual_type="ridge",
                residual_lambda=1.0,
            ).fit(None, None, groups=groups),
            "mlp": MLPEnergyModel(
                hidden_sizes=(4,),
                max_epochs=1,
                batch_size=128,
                patience=None,
                seed=1,
            ).fit(dataset.X, dataset.y),
            "ann": ANNEnergyModel(
                hidden_sizes=(4, 4, 2),
                dropout=0.0,
                max_epochs=1,
                batch_size=128,
                patience=None,
                seed=1,
            ).fit(dataset.X, dataset.y),
        }
        for kind, model in models.items():
            model.save(str(directory / f"{kind}_model.pkl"))
        return models

    def test_compares_all_saved_models_with_mean_curve_and_errorbars(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            h5_path = directory / "energy_pool.h5"
            with h5py.File(h5_path, "w") as h5_file:
                create_completed_run(h5_file, "curve")
            self._save_all_models(directory)

            with mock.patch.object(
                TRAINING_MODULE,
                "load_energy_groups_from_h5",
                wraps=load_energy_groups_from_h5,
            ) as reader:
                figure = TRAINING_MODULE.compare_models_on_group(str(h5_path), 0, show=False)
            reader.assert_called_once_with(str(h5_path), load_samples=False)

            axis = figure.axes[0]
            self.assertEqual(axis.get_xscale(), "log")
            self.assertEqual(axis.get_yscale(), "log")
            self.assertEqual(len(axis.containers), 0)
            model_lines = [
                line for line in axis.lines
                if line.get_label() in {"powerlaw", "parametric", "mlp", "ann"}
            ]
            self.assertEqual(len(model_lines), 4)
            raw_lines = [line for line in axis.lines if line.get_label() == "E_mean"]
            self.assertEqual(len(raw_lines), 1)
            for line in model_lines:
                self.assertEqual(line.get_xdata().shape, (3,))
                self.assertTrue(np.all(np.isfinite(line.get_ydata())))
                self.assertTrue(np.all(line.get_ydata() > 0.0))

            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_comparison_rejects_invalid_group_and_saved_model_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            h5_path = directory / "energy_pool.h5"
            with h5py.File(h5_path, "w") as h5_file:
                create_completed_run(h5_file, "curve")

            with self.assertRaises(IndexError):
                TRAINING_MODULE.compare_models_on_group(str(h5_path), -1, show=False)
            with self.assertRaises(IndexError):
                TRAINING_MODULE.compare_models_on_group(str(h5_path), 1, show=False)
            with self.assertRaises(FileNotFoundError):
                TRAINING_MODULE.compare_models_on_group(str(h5_path), 0, show=False)

            models = self._save_all_models(directory)
            models["powerlaw"].save(str(directory / "parametric_model.pkl"))
            with self.assertRaisesRegex(TypeError, "requested as 'parametric'"):
                TRAINING_MODULE.compare_models_on_group(str(h5_path), 0, show=False)


class TestStructuralTraining(unittest.TestCase):
    @staticmethod
    def _record(index: int) -> EnergyGroupRecord:
        np_values = np.arange(1, 7, dtype=np.int64)
        scale = 1.0 + 0.05 * index
        energy = scale * np_values.astype(float) ** 0.8
        return EnergyGroupRecord(
            key=f"record-{index}", NO_FRAG=2, int_bre=0.0, gamma=2.0 + index,
            Df=1.8, MAS=0.1 * index, X1=0.2 + 0.01 * index, A0=1.0,
            N_GRIDS=1, N_FRACS=1, base_seed=index, workers=1,
            STR=np.array([1.0 + index, 2.0 + index, 3.0 + index]),
            sigma_attr=None, pearson_r_attr=None, Np=np_values, V=np_values.astype(float),
            E_mean=energy, E_std=np.zeros_like(energy), E_samples=None,
            sigma_fit=0.8, b_fit=float(np.log(scale)), pearson_r_fit=1.0,
        )

    def test_structural_models_fit_only_training_groups(self) -> None:
        groups = [self._record(index) for index in range(12)]
        train_groups = groups[:10]
        validation = build_energy_dataset(groups[10:], target="log_mean")

        powerlaw = PowerLawSeparableModel(enable_tail=True, regress_type="ridge", ridge_lambda=1.0)
        powerlaw.fit(None, None, groups=train_groups)
        self.assertTrue(np.all(np.isfinite(powerlaw.predict(validation.X))))

        parametric = ParametricEnergyModel(residual_type="ridge", residual_lambda=1.0)
        parametric.fit(None, None, groups=train_groups)
        self.assertTrue(np.all(np.isfinite(parametric.predict(validation.X))))


class TestMergeEnergyPool(unittest.TestCase):
    def _create_source(self, path: Path, key: str, *, complete: bool = True) -> None:
        with h5py.File(path, "w") as h5_file:
            create_completed_run(h5_file, key, complete=complete)

    def test_merge_and_reject_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_source(root / "one.h5", "one")
            self._create_source(root / "two.h5", "two")
            output = MERGE_MODULE.merge_energy_pool(root, ("one.h5", "two.h5"), "merged.h5")
            with h5py.File(output, "r") as merged:
                self.assertEqual(set(merged["runs"].keys()), {"one", "two"})
                self.assertEqual(merged["runs"]["one"]["E_samples"].shape, (3, 2, 3))
                self.assertEqual(merged["runs"]["one"]["E_samples"].compression, "gzip")

            with self.assertRaises(FileExistsError):
                MERGE_MODULE.merge_energy_pool(root, ("one.h5", "two.h5"), "merged.h5")

    def test_merge_rejects_duplicate_and_unfinished_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_source(root / "one.h5", "same")
            self._create_source(root / "two.h5", "same")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                MERGE_MODULE.merge_energy_pool(root, ("one.h5", "two.h5"), "duplicate.h5")

            self._create_source(root / "unfinished.h5", "unfinished", complete=False)
            with self.assertRaisesRegex(ValueError, "incomplete|unfinished"):
                MERGE_MODULE.merge_energy_pool(root, ("one.h5", "unfinished.h5"), "bad.h5")


if __name__ == "__main__":
    unittest.main()
