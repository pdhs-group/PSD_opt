"""Variance-based sensitivity analysis for key WMCPBE parameters."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
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
    task_id = str(task["task_id"])
    sample_index = int(task["sample_index"])
    cast_sample = copy.deepcopy(task["cast_sample"])
    npz_path = Path(task["npz_path"])

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
        "task_id": task_id,
        "sample_index": sample_index,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "cpu_time_s": elapsed,
        "npz_relpath": str(npz_path.name),
    }
    row.update(cast_sample)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npz_path.with_suffix(".tmp.npz")
    npz_payload: Dict[str, object] = {
        "task_id": np.asarray(task_id),
        "sample_index": np.asarray(sample_index, dtype=np.int64),
        "metric_name": np.asarray(metric_name),
        "metric_value": np.asarray(metric_value, dtype=np.float64),
        "cpu_time_s": np.asarray(elapsed, dtype=np.float64),
    }
    for key, value in cast_sample.items():
        npz_payload[f"param_{key}"] = np.asarray(value)
    np.savez_compressed(tmp_path, **npz_payload)
    tmp_path.replace(npz_path)
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
        self.run_dir = self.export_dir / f"SA_{self.sample_size}"
        self.npz_dir = self.run_dir / "samples_npz"
        self.sqlite_path = self.run_dir / "results_index.sqlite"
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
        self._prepare_run_directory(problem, samples)

        tasks: List[Dict[str, object]] = []
        for idx, sample in enumerate(samples):
            attrs, cast_sample = self._attrs_from_sample(sample)
            cfg = self._build_config_for_attrs(attrs, sample_index=idx)
            task_id = self._task_id_for_sample(idx, cast_sample)
            tasks.append(
                {
                    "task_id": task_id,
                    "sample_index": idx,
                    "metric_name": self.metric_name,
                    "config": cfg,
                    "init_dist": copy.deepcopy(self.init_dist),
                    "cast_sample": cast_sample,
                    "npz_path": str(self._task_npz_path(task_id)),
                }
            )

        self._write_samples_manifest(tasks)
        completed_map = self._load_completed_task_results()
        recovered_count = self._recover_completed_from_npz(tasks, completed_map)
        if recovered_count > 0:
            print(f"Recovered {recovered_count} completed samples from NPZ files.", flush=True)
        pending_tasks = [task for task in tasks if str(task["task_id"]) not in completed_map]
        if pending_tasks:
            print(
                f"Resumable SA: {len(completed_map)}/{len(tasks)} samples already finished, "
                f"running remaining {len(pending_tasks)}.",
                flush=True,
            )
            self._evaluate_tasks(
                pending_tasks,
                on_result=lambda row: self._handle_completed_sample(row, completed_map),
            )
        else:
            print(
                f"Resumable SA: all {len(tasks)} samples already finished, reusing saved results.",
                flush=True,
            )

        records = self._records_from_completed_map(completed_map)
        if len(records) != int(samples.shape[0]):
            raise RuntimeError(
                f"Sensitivity analysis results are incomplete: expected {int(samples.shape[0])} "
                f"samples but found {len(records)} completed results in {self.run_dir}."
            )
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
                "run_directory": str(self.run_dir),
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

    def _handle_completed_sample(
        self,
        row: Dict[str, object],
        completed_map: Dict[str, Dict[str, object]],
    ) -> None:
        self._upsert_result_sqlite(row)
        completed_map[str(row["task_id"])] = row
        self._write_results_snapshot(completed_map)
        print("Completed one sample.", flush=True)

    def _evaluate_tasks(
        self,
        tasks: Sequence[Dict[str, object]],
        on_result: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> List[Dict[str, object]]:
        if len(tasks) == 0:
            return []
        if self.workers <= 1:
            records = []
            for task in tasks:
                row = _evaluate_sensitivity_task(task)
                records.append(row)
                if on_result is not None:
                    on_result(row)
        else:
            record_map: Dict[int, Dict[str, object]] = {}
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                future_to_index = {
                    executor.submit(_evaluate_sensitivity_task, task): int(task["sample_index"])
                    for task in tasks
                }
                for future in as_completed(future_to_index):
                    sample_index = future_to_index[future]
                    row = future.result()
                    record_map[sample_index] = row
                    if on_result is not None:
                        on_result(row)
            records = [record_map[idx] for idx in sorted(record_map.keys())]
        return [record for record in records if record is not None]

    def _normalize_for_json(self, obj: object) -> object:
        if is_dataclass(obj):
            return self._normalize_for_json(asdict(obj))
        if isinstance(obj, dict):
            return {str(key): self._normalize_for_json(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._normalize_for_json(value) for value in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return [self._normalize_for_json(value) for value in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def _run_metadata_payload(self, problem: Dict[str, object], samples: np.ndarray) -> Dict[str, object]:
        payload = {
            "metric_name": self.metric_name,
            "sample_size": self.sample_size,
            "calc_second_order": self.calc_second_order,
            "workers": self.workers,
            "problem": problem,
            "num_model_evaluations": int(samples.shape[0]),
            "parameters": [self._normalize_for_json(item) for item in self.parameters],
            "base_config": self._normalize_for_json(self.base_config),
            "template_variant": self._normalize_for_json(self.template_variant),
            "init_dist": self._normalize_for_json(self.init_dist),
        }
        sig_src = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        payload["run_signature"] = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()
        return payload

    def _prepare_run_directory(self, problem: Dict[str, object], samples: np.ndarray) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.npz_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.run_dir / "metadata.json"
        payload = self._run_metadata_payload(problem, samples)
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("run_signature") != payload["run_signature"]:
                raise ValueError(
                    f"Existing resume directory has different configuration: {self.run_dir}. "
                    "Please clear it or use a different export_dir/sample_size."
                )
        else:
            metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        conn = self._open_db()
        try:
            self._init_db(conn)
            self._set_db_meta(conn, "run_signature", payload["run_signature"])
            self._set_db_meta(conn, "metric_name", payload["metric_name"])
            self._set_db_meta(conn, "sample_size", payload["sample_size"])
            self._set_db_meta(conn, "calc_second_order", payload["calc_second_order"])
            self._set_db_meta(conn, "num_model_evaluations", payload["num_model_evaluations"])
        finally:
            conn.close()

    def _task_id_for_sample(self, sample_index: int, cast_sample: Dict[str, object]) -> str:
        src = json.dumps(self._normalize_for_json(cast_sample), sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
        return f"sample_{sample_index:06d}_{digest}"

    def _write_samples_manifest(self, tasks: Sequence[Dict[str, object]]) -> None:
        rows: List[Dict[str, object]] = []
        for task in tasks:
            row = {
                "task_id": str(task["task_id"]),
                "sample_index": int(task["sample_index"]),
                "npz_relpath": str(Path(str(task["npz_path"])).name),
            }
            row.update(copy.deepcopy(task["cast_sample"]))
            rows.append(row)
        pd.DataFrame(rows).sort_values("sample_index").to_csv(
            self.run_dir / "samples.csv",
            index=False,
        )

    def _task_npz_path(self, task_id: str) -> Path:
        return self.npz_dir / f"{task_id}.npz"

    def _recover_completed_from_npz(
        self,
        tasks: Sequence[Dict[str, object]],
        completed_map: Dict[str, Dict[str, object]],
    ) -> int:
        recovered = 0
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in completed_map:
                continue
            npz_path = Path(str(task["npz_path"]))
            if not npz_path.exists():
                continue
            row = self._load_row_from_npz(npz_path)
            if row is None:
                continue
            if str(row.get("task_id", "")) != task_id:
                continue
            self._upsert_result_sqlite(row)
            completed_map[task_id] = row
            recovered += 1
        if recovered > 0:
            self._write_results_snapshot(completed_map)
        return recovered

    def _load_row_from_npz(self, npz_path: Path) -> Optional[Dict[str, object]]:
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                required = {"task_id", "sample_index", "metric_name", "metric_value", "cpu_time_s"}
                if not required.issubset(set(data.files)):
                    return None
                row: Dict[str, object] = {
                    "task_id": str(np.asarray(data["task_id"]).item()),
                    "sample_index": int(np.asarray(data["sample_index"]).item()),
                    "metric_name": str(np.asarray(data["metric_name"]).item()),
                    "metric_value": float(np.asarray(data["metric_value"]).item()),
                    "cpu_time_s": float(np.asarray(data["cpu_time_s"]).item()),
                    "npz_relpath": str(npz_path.name),
                }
                for key in data.files:
                    if not key.startswith("param_"):
                        continue
                    value = np.asarray(data[key])
                    if value.ndim == 0:
                        row[key[6:]] = value.item()
                    else:
                        row[key[6:]] = value.tolist()
                return row
        except Exception:
            return None

    def _load_completed_task_results(self) -> Dict[str, Dict[str, object]]:
        completed_map: Dict[str, Dict[str, object]] = {}
        if not self.sqlite_path.exists():
            return completed_map
        conn = self._open_db()
        try:
            self._init_db(conn)
            query = """
                SELECT task_id, sample_index, metric_name, metric_value, cpu_time_s, npz_relpath, attrs_json
                FROM sample_results
                WHERE status = 'done'
                ORDER BY sample_index
            """
            for task_id, sample_index, metric_name, metric_value, cpu_time_s, npz_relpath, attrs_json in conn.execute(query):
                row = {
                    "task_id": str(task_id),
                    "sample_index": int(sample_index),
                    "metric_name": str(metric_name),
                    "metric_value": float(metric_value),
                    "cpu_time_s": float(cpu_time_s),
                    "npz_relpath": str(npz_relpath),
                }
                row.update(json.loads(str(attrs_json)))
                completed_map[str(task_id)] = row
        finally:
            conn.close()
        return completed_map

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sample_results (
                task_id TEXT PRIMARY KEY,
                sample_index INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                cpu_time_s REAL NOT NULL,
                npz_relpath TEXT NOT NULL,
                attrs_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sample_results_index ON sample_results(sample_index);
            CREATE INDEX IF NOT EXISTS idx_sample_results_status ON sample_results(status);
            """
        )
        conn.commit()

    def _set_db_meta(self, conn: sqlite3.Connection, key: str, value: object) -> None:
        conn.execute(
            """
            INSERT INTO meta(key, value_json) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (str(key), json.dumps(self._normalize_for_json(value), ensure_ascii=False, sort_keys=True)),
        )
        conn.commit()

    def _upsert_result_sqlite(self, row: Dict[str, object]) -> None:
        conn = self._open_db()
        try:
            self._init_db(conn)
            attrs = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "task_id",
                    "sample_index",
                    "metric_name",
                    "metric_value",
                    "cpu_time_s",
                    "npz_relpath",
                }
            }
            conn.execute(
                """
                INSERT INTO sample_results(
                    task_id, sample_index, status, metric_name, metric_value, cpu_time_s, npz_relpath, attrs_json
                )
                VALUES (?, ?, 'done', ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    sample_index = excluded.sample_index,
                    status = excluded.status,
                    metric_name = excluded.metric_name,
                    metric_value = excluded.metric_value,
                    cpu_time_s = excluded.cpu_time_s,
                    npz_relpath = excluded.npz_relpath,
                    attrs_json = excluded.attrs_json
                """,
                (
                    str(row["task_id"]),
                    int(row["sample_index"]),
                    str(row["metric_name"]),
                    float(row["metric_value"]),
                    float(row["cpu_time_s"]),
                    str(row["npz_relpath"]),
                    json.dumps(self._normalize_for_json(attrs), ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _records_from_completed_map(self, completed_map: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
        records = list(completed_map.values())
        records.sort(key=lambda row: int(row["sample_index"]))
        return records

    def _write_results_snapshot(self, completed_map: Dict[str, Dict[str, object]]) -> None:
        records = self._records_from_completed_map(completed_map)
        if not records:
            return
        pd.DataFrame(records).to_csv(self.run_dir / "results_snapshot.csv", index=False)

    def _write_excel(self, result: SensitivityAnalysisResult) -> Path:
        path = self.run_dir / f"wmcpbe_sensitivity_{self.metric_name}.xlsx"
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
        t_vec=np.arange(0.0, 30.0 + 1e-12, 3.0),
        x=2e-3,
        beta0=1e-6,
        p1=1e-1,
        p2=1.0,
        use_psd=False,
    )

    config = ValidationConfig(
        case=case,
        dpbe_variants=[
            DPBEVariantConfig(name="dPBE", grid="geo", ns=50, s=1.5, enabled=False),
        ],
        wmcpbe_variants=[
            WMCPBEVariantConfig(
                name="WMCPBE template",
                repeats=1,
                attrs={
                    "a0": 1e4,
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
            bounds=(1.0, 100.0),
            kind="float",
            targets=("break_dW_max",),
        ),
        SensitivityParameter(
            name="agg_dW_max",
            bounds=(1.0, 20.0),
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
            bounds=(1.1, 5.0),
            kind="float",
        ),
    ]

    analyzer = WMCPBESensitivityAnalyzer(
        config=config,
        parameters=parameters,
        init_dist=init_dist,
        metric_name="aggregated_moment_error",
        sample_size=32,
        # N_sample - N(2D+2) or N(D+2)
        calc_second_order=True,
        workers=10,
        # export_dir=os.environ.get('STORAGE_PATH'),
    )
    analyzer.run()
