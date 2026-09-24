# -*- coding: utf-8 -*-
"""Small regression tests for the Optuna CMA-ES search entry point."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "breakage-rate-model" / "src"))

from breakage_rate_model.data_io import EnergyGroupRecord
from breakage_rate_model.datasets import build_energy_dataset
from breakage_rate_model.base import BaseEnergyModel


SEARCH_SCRIPT_PATH = Path(__file__).with_name("hyperparameter_search.py")
SEARCH_SPEC = importlib.util.spec_from_file_location("hyperparameter_search_entry", SEARCH_SCRIPT_PATH)
assert SEARCH_SPEC is not None and SEARCH_SPEC.loader is not None
SEARCH_MODULE = importlib.util.module_from_spec(SEARCH_SPEC)
sys.modules[SEARCH_SPEC.name] = SEARCH_MODULE
SEARCH_SPEC.loader.exec_module(SEARCH_MODULE)


class TestHyperparameterSearch(unittest.TestCase):
    @staticmethod
    def _record(index: int, *, pure: bool = False) -> EnergyGroupRecord:
        np_values = np.arange(1, 9, dtype=np.int64)
        scale = 1.0 + 0.1 * index
        energy = scale * np_values.astype(float) ** 0.8
        return EnergyGroupRecord(
            key=f"record-{index}",
            NO_FRAG=2,
            int_bre=0.0,
            gamma=2.0 + index,
            Df=1.8,
            MAS=0.5 if pure else 0.1 * index,
            X1=0.0 if pure else 0.2 + 0.02 * index,
            A0=1.0,
            N_GRIDS=1,
            N_FRACS=1,
            base_seed=index,
            workers=1,
            STR=np.ones(3) if pure else np.array([1.0 + index, 2.0 + index, 3.0 + index]),
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

    def _context(self, directory: Path, *, dataset_kind: str = "mixed"):
        source = directory / "synthetic_energy_pool.h5"
        source.touch()
        groups = [self._record(index, pure=dataset_kind == "pure") for index in range(8)]
        dataset = build_energy_dataset(groups, target="log_mean")
        training_module = SEARCH_MODULE._load_training_module()
        split = training_module.split_train_val_by_group(
            groups, dataset, val_ratio=0.25, seed=3
        )
        return SEARCH_MODULE.SearchContext(
            h5_file=source.resolve(),
            output_directory=(directory / "results" / dataset_kind).resolve(),
            split=split,
            training_module=training_module,
            n_trials=1,
            sampler_seed=1,
            objective_metric="group_mae_log",
            show_training_output=False,
            dataset_kind=dataset_kind,
            strength_normalization=(
                "none" if dataset_kind == "pure" else "geometric_mean_relative"
            ),
        )

    def _assert_saved_result(self, result, *, strength_normalization: str) -> None:
        self.assertTrue(result.database_path.is_file())
        self.assertTrue(result.trials_csv_path.is_file())
        self.assertTrue(result.best_result_path.is_file())
        self.assertTrue(result.best_model_path.is_file())
        self.assertIn("group_mae_log", result.validation_metrics)
        with result.best_result_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertIn("best_trial_validation_metrics", payload)
        self.assertIn("group_mae_log", payload["best_trial_validation_metrics"])
        self.assertEqual(payload["configuration"]["strength_normalization"], strength_normalization)
        self.assertEqual(
            BaseEnergyModel.load(str(result.best_model_path)).strength_normalization,
            strength_normalization,
        )

    def _run_all_model_searches(self, context, active_feature_names):
        powerlaw = SEARCH_MODULE.search_powerlaw(
            context,
            ridge_lambda_range=(1e-2, 1e-1),
            plateau_weight_range=(1.0, 2.0),
            frac_threshold_range=(0.1, 0.2),
            abs_threshold_range=(0.05, 0.1),
            pearson_min=None,
            fit_mode="logE",
            pure_powerlaw_if_no_plateau=True,
            enable_tail=True,
            sigma_bounds=(0.0, 10.0),
            log_vc_margin=2.0,
            log_emax_margin=2.0,
            alpha_bounds=(0.0, 0.5),
            max_v=None,
            regress_type="ridge",
            active_feature_names=active_feature_names,
        )
        parametric = SEARCH_MODULE.search_parametric(
            context,
            residual_lambda_range=(1e-2, 1e-1),
            plateau_weight_range=(1.0, 2.0),
            frac_threshold_range=(0.1, 0.2),
            abs_threshold_range=(0.05, 0.1),
            pearson_min=None,
            fit_mode="logE",
            pure_powerlaw_if_no_plateau=True,
            residual_type="ridge",
            enable_tail=True,
            tol_int_bre=1e-12,
            active_feature_names=active_feature_names,
        )
        mlp = SEARCH_MODULE.search_mlp(
            context,
            hidden_1_range=(2, 4),
            hidden_2_range=(2, 4),
            lr_range=(1e-3, 1e-2),
            weight_decay_range=(1e-6, 1e-5),
            patience_range=(1, 2),
            activation="relu",
            max_epochs=1,
            batch_size=2,
            device="cpu",
            seed=1,
            active_feature_names=active_feature_names,
        )
        ann = SEARCH_MODULE.search_ann(
            context,
            hidden_1_range=(2, 4),
            hidden_2_range=(2, 4),
            hidden_3_range=(2, 4),
            dropout_range=(0.0, 0.2),
            lr_range=(1e-3, 1e-2),
            weight_decay_range=(1e-6, 1e-5),
            patience_range=(1, 2),
            activation="silu",
            max_epochs=1,
            batch_size=2,
            device="cpu",
            seed=1,
            active_feature_names=active_feature_names,
        )
        return powerlaw, parametric, mlp, ann

    def test_mixed_and_pure_model_searches_write_separate_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            mixed_results = self._run_all_model_searches(
                self._context(directory, dataset_kind="mixed"),
                SEARCH_MODULE.MIXED_ACTIVE_FEATURE_NAMES,
            )
            pure_results = self._run_all_model_searches(
                self._context(directory, dataset_kind="pure"),
                SEARCH_MODULE.PURE_ACTIVE_FEATURE_NAMES,
            )
            for result in mixed_results:
                self._assert_saved_result(result, strength_normalization="geometric_mean_relative")
            for result in pure_results:
                self._assert_saved_result(result, strength_normalization="none")
            self.assertNotEqual(mixed_results[0].database_path.parent, pure_results[0].database_path.parent)

    def test_changed_configuration_cannot_resume_existing_study(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            context = self._context(Path(directory_text))
            kwargs = {
                "ridge_lambda_range": (1e-2, 1e-1),
                "plateau_weight_range": (1.0, 2.0),
                "frac_threshold_range": (0.1, 0.2),
                "abs_threshold_range": (0.05, 0.1),
                "pearson_min": None,
                "fit_mode": "logE",
                "pure_powerlaw_if_no_plateau": True,
                "enable_tail": True,
                "sigma_bounds": (0.0, 10.0),
                "log_vc_margin": 2.0,
                "log_emax_margin": 2.0,
                "alpha_bounds": (0.0, 0.5),
                "max_v": None,
                "regress_type": "ridge",
                "active_feature_names": ("logV", "log_gamma", "MAS", "X1", "STR0", "STR1", "STR2"),
            }
            SEARCH_MODULE.search_powerlaw(context, **kwargs)
            pure_context = replace(context, dataset_kind="pure", strength_normalization="none")
            pure_kwargs = {**kwargs, "active_feature_names": SEARCH_MODULE.PURE_ACTIVE_FEATURE_NAMES}
            with self.assertRaisesRegex(ValueError, "different data, split, search-space"):
                SEARCH_MODULE.search_powerlaw(pure_context, **pure_kwargs)
            kwargs["ridge_lambda_range"] = (1e-3, 1e-1)
            with self.assertRaisesRegex(ValueError, "different data, split, search-space"):
                SEARCH_MODULE.search_powerlaw(context, **kwargs)

    def test_build_pure_context_validates_pure_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            source = directory / "pure.h5"
            source.touch()
            groups = [self._record(index, pure=True) for index in range(4)]
            dataset = build_energy_dataset(groups, target="log_mean")
            training_module = SEARCH_MODULE._load_training_module()
            calls = []

            def validate(received):
                calls.append(received)
                training_module._validate_pure_groups(received)

            proxy = mock.Mock(
                load_data=lambda _: (groups, dataset),
                split_train_val_by_group=training_module.split_train_val_by_group,
                _validate_pure_groups=validate,
            )
            with mock.patch.object(SEARCH_MODULE, "_load_training_module", return_value=proxy):
                context = SEARCH_MODULE.build_search_context(
                    source,
                    output_directory=directory / "results" / "pure",
                    n_trials=1,
                    sampler_seed=1,
                    validation_ratio=0.25,
                    split_seed=3,
                    objective_metric="group_mae_log",
                    show_training_output=False,
                    dataset_kind="pure",
                    strength_normalization="none",
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(context.dataset_kind, "pure")
            self.assertEqual(context.strength_normalization, "none")

    def test_mixed_objective_is_invariant_to_global_strength_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            groups = [self._record(index) for index in range(4)]
            scaled_groups = [
                replace(
                    group,
                    STR=group.STR * 10.0,
                    E_mean=group.E_mean * 10.0,
                    E_std=group.E_std * 10.0,
                    b_fit=group.b_fit + np.log(10.0),
                )
                for group in groups
            ]
            training_module = SEARCH_MODULE._load_training_module()

            def context_for(records, name):
                dataset = build_energy_dataset(records, target="log_mean")
                return SEARCH_MODULE.SearchContext(
                    h5_file=(directory / name).resolve(),
                    output_directory=(directory / name).resolve(),
                    split=training_module.split_train_val_by_group(
                        records, dataset, val_ratio=0.25, seed=3
                    ),
                    training_module=training_module,
                    n_trials=1,
                    sampler_seed=1,
                    objective_metric="group_mae_log",
                    show_training_output=False,
                    dataset_kind="mixed",
                    strength_normalization="geometric_mean_relative",
                )

            class ZeroModel:
                def predict(self, X):
                    return np.zeros(X.shape[0], dtype=float)

            metrics, _, _ = SEARCH_MODULE._fit_and_evaluate(
                ZeroModel(), context_for(groups, "base.h5"), lambda _: None
            )
            scaled_metrics, _, _ = SEARCH_MODULE._fit_and_evaluate(
                ZeroModel(), context_for(scaled_groups, "scaled.h5"), lambda _: None
            )
            self.assertAlmostEqual(metrics["group_mae_log"], scaled_metrics["group_mae_log"])


if __name__ == "__main__":
    unittest.main()
