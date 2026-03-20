from __future__ import annotations

import csv
import gc
import os
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


class PoolMemoryLogger:
    def __init__(self) -> None:
        self._call_counter = 0
        self._csv_path: Path | None = None
        self._fieldnames: List[str] | None = None
        self._announced = False

    def maybe_log(self, sim, tag: str) -> None:
        if not _env_flag("LMC_POOL_DEBUG_MEMORY", False):
            return

        self._call_counter += 1
        every = int(os.environ.get("LMC_POOL_DEBUG_EVERY", "100") or "100")
        every = max(1, every)
        if (self._call_counter % every) != 0:
            return

        if not tracemalloc.is_tracing():
            tracemalloc.start(10)

        stats = self._runtime_memory_stats(
            sim,
            force_gc=_env_flag("LMC_POOL_DEBUG_GC", False),
        )
        row: Dict[str, float | int | str] = {"tag": tag, "call": int(self._call_counter)}
        row.update(stats)
        csv_path = self._append_row(row)
        if not self._announced:
            print(f"[LMC memory] Debug CSV: {csv_path}")
            self._announced = True

    def _runtime_memory_stats(self, sim, *, force_gc: bool = False) -> Dict[str, float | int]:
        if force_gc:
            gc.collect()

        stats: Dict[str, float | int] = {}
        if getattr(sim, "M", None) is not None:
            stats["M_bytes"] = int(sim.M.nbytes)
        if getattr(sim, "Hbond", None) is not None:
            stats["Hbond_bytes"] = int(sim.Hbond.nbytes)
        if getattr(sim, "Vbond", None) is not None:
            stats["Vbond_bytes"] = int(sim.Vbond.nbytes)

        agg_pool = getattr(sim, "agg_pool", None)
        if agg_pool is not None and hasattr(agg_pool, "debug_stats"):
            for key, value in agg_pool.debug_stats().items():
                stats[f"aggpool_{key}"] = int(value)

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            stats["py_current_bytes"] = int(current)
            stats["py_peak_bytes"] = int(peak)

        try:
            import psutil  # type: ignore
            stats["rss_bytes"] = int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass

        return stats

    def _log_dir(self) -> Path:
        raw = os.environ.get("LMC_POOL_DEBUG_DIR", "").strip()
        if raw:
            return Path(raw)
        return Path(r"C:\Users\px2030\Code\PSD_opt\lmc\tests")

    def _ensure_csv(self, fieldnames: List[str]) -> Path:
        if self._csv_path is not None:
            return self._csv_path

        log_dir = self._log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"lmc_pool_memory_{timestamp}"
        csv_path = log_dir / f"{base_name}.csv"
        suffix = 1
        while csv_path.exists():
            csv_path = log_dir / f"{base_name}_{suffix:02d}.csv"
            suffix += 1

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

        self._csv_path = csv_path
        self._fieldnames = list(fieldnames)
        return csv_path

    def _append_row(self, row: Dict[str, float | int | str]) -> Path:
        fieldnames = list(row.keys())
        csv_path = self._ensure_csv(fieldnames)
        expected = self._fieldnames or fieldnames
        if fieldnames != expected:
            raise RuntimeError(f"[LMC memory] CSV field mismatch. expected={expected}, got={fieldnames}")

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=expected)
            writer.writerow(row)
        return csv_path
