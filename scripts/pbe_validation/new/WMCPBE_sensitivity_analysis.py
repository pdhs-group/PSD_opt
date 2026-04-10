"""Variance-based sensitivity analysis for key WMCPBE parameters."""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample


def _bootstrap_project_paths() -> None:
    root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        root,
        root / "scripts" / "pbe_validation" / "new",
        root / "dpbe" / "src",
        root / "mcpbe" / "src",
        root / "qmom" / "src",
        root / "pbe-core" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_project_paths()

from validation import (  # noqa: E402
    MIN,
    CaseConfig,
    DPBEVariantConfig,
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
    WMCPBEVariantConfig,
)
from pbe_validation_advance import Dirichlet2DValidationRunner, DirichletInitialCondition  # noqa: E402


@dataclass
class SensitivityParameter:
    name: str
    bounds: Tuple[float, float]
    kind: str = "float"  # "float" or "int"
    targets: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.targets) == 0:
            self.targets = (self.name,)
        if self.kind not in ("float", "int"):
            raise ValueError(f"Unsupported parameter kind: {self.kind}")


@dataclass
class SensitivityAnalysisResult:
    metric_name: str
    sample_records: pd.DataFrame
    first_order: pd.DataFrame
    total_order: pd.DataFrame
    second_order: Optional[pd.DataFrame]
    metadata: Dict[str, object]


def _build_validation_runner_for_sensitivity(
    config: ValidationConfig,
    init_dist: Optional[DirichletInitialCondition],
) -> ValidationRunner:
    if config.case.dim == 2 and init_dist is not None:
        return Dirichlet2DValidationRunner(config, copy.deepcopy(init_dist))
    return ValidationRunner(config)


def _metric_aggregated_moment_error(result: ValidationResult, wm_name: str) -> float:
    reference = result.methods["Analytical Solution"].moments
    target = result.methods[wm_name].moments
    if result.dim == 1:
        indices = [(0, 0), (1, 0), (2, 0)]
    else:
        indices = [(0, 0), (0, 1), (1, 1), (0, 2)]
    terms = []
    for i, j in indices:
        rel = (target[i, j, :] - reference[i, j, :]) / (reference[i, j, :] + MIN)
        terms.append(float(np.max(rel ** 2)))
    return float(np.sqrt(np.sum(terms)))


def _evaluate_sensitivity_task(task: Dict[str, object]) -> Dict[str, object]:
    _bootstrap_project_paths()
    cfg = copy.deepcopy(task["config"])
    init_dist = copy.deepcopy(task.get("init_dist"))
    metric_name = str(task["metric_name"])
    sample_index = int(task["sample_index"])
    cast_sample = copy.deepcopy(task["cast_sample"])

    runner = _build_validation_runner_for_sensitivity(cfg, init_dist)
    time_start = time.time()
    result = runner.run()
    elapsed = time.time() - time_start

    wm_name = cfg.wmcpbe_variants[0].name
    if metric_name == "aggregated_moment_error":
        metric_value = _metric_aggregated_moment_error(result, wm_name)
    else:
        raise ValueError(f"Unknown metric '{metric_name}'.")

    row: Dict[str, object] = {
        "sample_index": sample_index,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "cpu_time_s": elapsed,
    }
    row.update(cast_sample)
    return row


class WMCPBESensitivityAnalyzer:
    """Sobol/Saltelli sensitivity analysis on one WMCPBE template configuration."""

    def __init__(
        self,
        config: ValidationConfig,
        parameters: Sequence[SensitivityParameter],
        init_dist: Optional[DirichletInitialCondition] = None,
        metric_name: str = "aggregated_moment_error",
        sample_size: int = 32,
        calc_second_order: bool = False,
        workers: Optional[int] = None,
        export_dir: Optional[Path] = None,
    ) -> None:
        enabled_wm = [variant for variant in config.wmcpbe_variants if variant.enabled]
        if len(enabled_wm) != 1:
            raise ValueError("Sensitivity analysis expects exactly one enabled WMCPBE variant as the template.")
        if len(parameters) == 0:
            raise ValueError("At least one sensitivity parameter must be provided.")
        self.base_config = copy.deepcopy(config)
        self.template_variant = copy.deepcopy(enabled_wm[0])
        self.parameters = list(parameters)
        self.init_dist = copy.deepcopy(init_dist)
        self.metric_name = metric_name
        self.sample_size = int(sample_size)
        self.calc_second_order = bool(calc_second_order)
        self.workers = max(1, int(workers)) if workers is not None else max(1, os.cpu_count() or 1)
        self.export_dir = Path(export_dir) if export_dir is not None else Path(__file__).resolve().parent / "exports_wmcpbe_sensitivity"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.metric_registry: Dict[str, Callable[[ValidationResult, str], float]] = {
            "aggregated_moment_error": self._metric_aggregated_moment_error,
        }
        if metric_name not in self.metric_registry:
            raise ValueError(f"Unknown metric '{metric_name}'.")
        if self.base_config.case.dim == 2 and self.init_dist is None:
            raise ValueError("For dim=2 sensitivity analysis, init_dist must be provided as DirichletInitialCondition.")
        self._validate_parameterization()

    def run(self) -> SensitivityAnalysisResult:
        problem = self._build_problem()
        samples = sobol_sample.sample(problem, self.sample_size, calc_second_order=self.calc_second_order)

        tasks: List[Dict[str, object]] = []
        for idx, sample in enumerate(samples):
            attrs, cast_sample = self._attrs_from_sample(sample)
            cfg = self._build_config_for_attrs(attrs, sample_index=idx)
            tasks.append(
                {
                    "sample_index": idx,
                    "metric_name": self.metric_name,
                    "config": cfg,
                    "init_dist": copy.deepcopy(self.init_dist),
                    "cast_sample": cast_sample,
                }
            )

        records = self._evaluate_tasks(tasks)
        responses = np.asarray([float(row["metric_value"]) for row in records], dtype=float)

        sobol_result = sobol.analyze(problem, responses, calc_second_order=self.calc_second_order, print_to_console=False)
        first_df = pd.DataFrame(
            {
                "parameter": problem["names"],
                "S1": sobol_result["S1"],
                "S1_conf": sobol_result["S1_conf"],
            }
        )
        total_df = pd.DataFrame(
            {
                "parameter": problem["names"],
                "ST": sobol_result["ST"],
                "ST_conf": sobol_result["ST_conf"],
            }
        )
        second_df = None
        if self.calc_second_order:
            second_rows: List[Dict[str, object]] = []
            names = problem["names"]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    second_rows.append(
                        {
                            "parameter_i": names[i],
                            "parameter_j": names[j],
                            "S2": sobol_result["S2"][i, j],
                            "S2_conf": sobol_result["S2_conf"][i, j],
                        }
                    )
            second_df = pd.DataFrame(second_rows)

        sample_df = pd.DataFrame(records)
        result_obj = SensitivityAnalysisResult(
            metric_name=self.metric_name,
            sample_records=sample_df,
            first_order=first_df,
            total_order=total_df,
            second_order=second_df,
            metadata={
                "metric_name": self.metric_name,
                "sample_size": self.sample_size,
                "calc_second_order": self.calc_second_order,
                "workers": self.workers,
                "num_parameters": len(self.parameters),
                "num_model_evaluations": int(samples.shape[0]),
                "template_variant": self.template_variant.name,
                "process": self.base_config.case.process,
                "kernel": self.base_config.case.kernel,
            },
        )
        self._write_excel(result_obj)
        self._print_summary(result_obj)
        return result_obj

    def _build_problem(self) -> Dict[str, object]:
        return {
            "num_vars": len(self.parameters),
            "names": [item.name for item in self.parameters],
            "bounds": [list(item.bounds) for item in self.parameters],
        }

    def _validate_parameterization(self) -> None:
        names = {item.name for item in self.parameters}
        if "recon_capacity_factor" in names:
            if "recon_bins" not in names:
                raise ValueError(
                    "Using recon_capacity_factor requires recon_bins to also be sampled."
                )
            if "recon_N_max" in names:
                raise ValueError(
                    "Do not sample recon_N_max directly when using recon_capacity_factor. "
                    "Use recon_capacity_factor to derive recon_N_max from recon_bins."
                )

    def _apply_derived_attrs(
        self,
        attrs: Dict[str, object],
        cast_values: Dict[str, object],
    ) -> tuple[Dict[str, object], Dict[str, object]]:
        attrs_out = copy.deepcopy(attrs)
        values_out = copy.deepcopy(cast_values)

        if "recon_bins" in values_out and "recon_capacity_factor" in values_out:
            recon_bins = int(values_out["recon_bins"])
            capacity_factor = float(values_out["recon_capacity_factor"])
            derived_recon_n_max = int(math.ceil(2.0 * recon_bins ** 2 * capacity_factor))
            attrs_out["recon_N_max"] = derived_recon_n_max
            values_out["recon_N_max_derived"] = derived_recon_n_max

        return attrs_out, values_out

    def _attrs_from_sample(self, sample: np.ndarray) -> tuple[Dict[str, object], Dict[str, object]]:
        attrs = copy.deepcopy(self.template_variant.attrs)
        cast_values: Dict[str, object] = {}
        for param, value in zip(self.parameters, sample):
            if param.kind == "int":
                cast_value: object = int(round(float(value)))
            else:
                cast_value = float(value)
            for target in param.targets:
                attrs[target] = cast_value
            cast_values[param.name] = cast_value
        return self._apply_derived_attrs(attrs, cast_values)

    def _build_config_for_attrs(self, attrs: Dict[str, object], sample_index: int) -> ValidationConfig:
        cfg = copy.deepcopy(self.base_config)
        cfg.qmom_variants = []
        cfg.wmcpbe_variants = [
            WMCPBEVariantConfig(
                name=f"{self.template_variant.name} sample {sample_index:04d}",
                repeats=self.template_variant.repeats,
                base_seed=self.template_variant.base_seed,
                maxiter=self.template_variant.maxiter,
                enabled=True,
                attrs=attrs,
            )
        ]
        if cfg.reference_dpbe_name is None:
            cfg.reference_dpbe_name = cfg.dpbe_variants[0].name
        return cfg

    def _build_runner(self, config: ValidationConfig) -> ValidationRunner:
        return _build_validation_runner_for_sensitivity(config, self.init_dist)

    def _metric_aggregated_moment_error(self, result: ValidationResult, wm_name: str) -> float:
        return _metric_aggregated_moment_error(result, wm_name)

    def _evaluate_tasks(self, tasks: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        if self.workers <= 1:
            records = [_evaluate_sensitivity_task(task) for task in tasks]
        else:
            records = [None] * len(tasks)
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                future_to_index = {
                    executor.submit(_evaluate_sensitivity_task, task): int(task["sample_index"])
                    for task in tasks
                }
                for future in as_completed(future_to_index):
                    sample_index = future_to_index[future]
                    records[sample_index] = future.result()
        return [record for record in records if record is not None]

    def _write_excel(self, result: SensitivityAnalysisResult) -> Path:
        path = self.export_dir / f"wmcpbe_sensitivity_{self.metric_name}.xlsx"
        with pd.ExcelWriter(path) as writer:
            meta_df = pd.DataFrame(
                [{"key": key, "value": value if np.isscalar(value) or value is None else str(value)} for key, value in result.metadata.items()]
            )
            meta_df.to_excel(writer, sheet_name="metadata", index=False)
            result.sample_records.to_excel(writer, sheet_name="sample_records", index=False)
            result.first_order.to_excel(writer, sheet_name="first_order", index=False)
            result.total_order.to_excel(writer, sheet_name="total_order", index=False)
            if result.second_order is not None:
                result.second_order.to_excel(writer, sheet_name="second_order", index=False)
        print(f"Saved Excel export: {path}")
        return path

    def _print_summary(self, result: SensitivityAnalysisResult) -> None:
        print("\nWMCPBE variance-based sensitivity analysis")
        print("-" * 88)
        print(f"Metric              : {result.metric_name}")
        print(f"Template WMCPBE     : {result.metadata['template_variant']}")
        print(f"Model evaluations   : {result.metadata['num_model_evaluations']}")
        print("")
        print("First-order Sobol indices")
        for _, row in result.first_order.sort_values("S1", ascending=False).iterrows():
            print(f"  {row['parameter']:<20} S1={row['S1']:.6e}  (+/- {row['S1_conf']:.6e})")
        print("")
        print("Total-order Sobol indices")
        for _, row in result.total_order.sort_values("ST", ascending=False).iterrows():
            print(f"  {row['parameter']:<20} ST={row['ST']:.6e}  (+/- {row['ST_conf']:.6e})")


if __name__ == "__main__":
    case = CaseConfig(
        dim=2,
        kernel="const",
        process="mix",
        t_vec=np.arange(0.0, 30.0 + 1e-12, 1.0),
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
                name="WMCPBE template",
                repeats=10,
                attrs={
                    "a0": 100000,
                    "V_eff_init": 1000,
                    "recon_enable": True,
                    "recon_method": "4PMC",
                    "break_dW_min": 10.0,
                    "break_dW_max": 10.0,
                    "agg_dW_min": 10.0,
                    "agg_dW_max": 10.0,
                    "recon_bins": 24,
                    "recon_N_max": 3500,
                    "recon_RS_target": 1200,
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

    parameters = [
        SensitivityParameter(
            name="break_dW_max",
            bounds=(5.0, 500.0),
            kind="float",
            targets=("break_dW_max",),
        ),
        SensitivityParameter(
            name="agg_dW_max",
            bounds=(2.0, 100.0),
            kind="float",
            targets=("agg_dW_max",),
        ),
        SensitivityParameter(
            name="recon_bins",
            bounds=(20.0, 50.0),
            kind="int",
            targets=("recon_bins",),
        ),
        SensitivityParameter(
            name="recon_capacity_factor",
            bounds=(1.1, 4.0),
            kind="float",
        ),
    ]

    analyzer = WMCPBESensitivityAnalyzer(
        config=config,
        parameters=parameters,
        init_dist=init_dist,
        metric_name="aggregated_moment_error",
        sample_size=64,
        # N_sample - N(2D+2) or N(D+2)
        calc_second_order=True,
        workers=20,
    )
    analyzer.run()
