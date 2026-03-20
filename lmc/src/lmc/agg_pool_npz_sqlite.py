# -*- coding: utf-8 -*-
import math
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np


SQLITE_NAME = "pool_index.sqlite"
POOL_SUFFIX = "_npz_single"


def _format_pool_dirname(Df: float, MAS: float, suffix: str = POOL_SUFFIX) -> str:
    df_str = str(Df).replace(".", "p")
    mas_str = f"{MAS:.2f}".replace(".", "p")
    return f"aggregate_pool_Df{df_str}_MAS{mas_str}{suffix}"


class AggPool:
    def __init__(
        self,
        pool_dir: str,
        *,
        max_open_pools: int = 2,
        sample_cache_size: int = 0,
        pool_suffix: str = POOL_SUFFIX,
        sqlite_name: str = SQLITE_NAME,
    ):
        self.pool_dir = pool_dir
        self.max_open_pools = max(1, int(max_open_pools))
        self.sample_cache_size = max(0, int(sample_cache_size))
        self.pool_suffix = str(pool_suffix)
        self.sqlite_name = str(sqlite_name)

        self._pool_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._sample_array_cache: "OrderedDict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

        self._read_calls = 0
        self._sample_cache_hits = 0
        self._sample_cache_misses = 0
        self._index_builds = 0
        self._npz_file_opens = 0

    def close_pool_cache(self) -> None:
        self._pool_cache.clear()
        self._sample_array_cache.clear()

    def debug_stats(self) -> Dict[str, int]:
        return {
            "indexed_pools": int(len(self._pool_cache)),
            "cached_samples": int(len(self._sample_array_cache)),
            "sample_cache_hits": int(self._sample_cache_hits),
            "sample_cache_misses": int(self._sample_cache_misses),
            "pool_read_calls": int(self._read_calls),
            "pool_index_builds": int(self._index_builds),
            "pool_npz_file_opens": int(self._npz_file_opens),
        }

    def _enforce_pool_cache_limit(self) -> None:
        while len(self._pool_cache) > self.max_open_pools:
            pool_path, _cache = self._pool_cache.popitem(last=False)
            keys_to_drop = [key for key in self._sample_array_cache.keys() if key[0] == pool_path]
            for key in keys_to_drop:
                self._sample_array_cache.pop(key, None)

    def _enforce_sample_cache_limit(self) -> None:
        if self.sample_cache_size <= 0:
            self._sample_array_cache.clear()
            return
        while len(self._sample_array_cache) > self.sample_cache_size:
            self._sample_array_cache.popitem(last=False)

    def _resolve_pool_dir(self, Df: float, MAS: float) -> str:
        dirname = _format_pool_dirname(Df, MAS, self.pool_suffix)
        candidate = os.path.join(self.pool_dir, dirname)
        sqlite_path = os.path.join(candidate, self.sqlite_name)
        if os.path.isdir(candidate) and os.path.isfile(sqlite_path):
            return candidate

        direct_sqlite = os.path.join(self.pool_dir, self.sqlite_name)
        if os.path.isdir(self.pool_dir) and os.path.isfile(direct_sqlite) and Path(self.pool_dir).name == dirname:
            return self.pool_dir

        raise FileNotFoundError(
            f"[AggPool] Pool directory not found for Df={Df}, MAS={MAS}. Checked: {[candidate, self.pool_dir]}"
        )

    def _build_index(self, pool_path: str) -> Dict[str, Any]:
        self._index_builds += 1
        sqlite_path = os.path.join(pool_path, self.sqlite_name)
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            group_rows = conn.execute(
                "SELECT group_name, np_target, frac_a_target FROM groups ORDER BY group_name"
            ).fetchall()
            sample_rows = conn.execute(
                "SELECT group_name, sample_name, sample_index, npz_relpath FROM samples ORDER BY group_name, sample_index, sample_name"
            ).fetchall()
        finally:
            conn.close()

        sample_records: Dict[Tuple[str, str], Dict[str, str | int]] = {}
        samples_by_group: Dict[str, List[str]] = {}
        for row in sample_rows:
            gname = str(row["group_name"])
            sname = str(row["sample_name"])
            sample_records[(gname, sname)] = {
                "npz_relpath": str(row["npz_relpath"]),
                "sample_index": int(row["sample_index"]),
            }
            samples_by_group.setdefault(gname, []).append(sname)

        groups: List[Dict[str, Any]] = []
        for row in group_rows:
            gname = str(row["group_name"])
            samples = tuple(samples_by_group.get(gname, []))
            Np_t = float(row["np_target"])
            fracA_t = float(row["frac_a_target"])
            if len(samples) == 0 or (not np.isfinite(Np_t)) or (not np.isfinite(fracA_t)):
                continue
            groups.append(
                {
                    "gname": gname,
                    "Np_target": Np_t,
                    "frac_A_target": fracA_t,
                    "samples": samples,
                }
            )

        if len(groups) == 0:
            raise RuntimeError(f"[AggPool] No valid groups in pool file: {pool_path}")

        Np_vals = np.array(sorted({g["Np_target"] for g in groups}), dtype=float)
        XA_vals = np.array(sorted({g["frac_A_target"] for g in groups}), dtype=float)
        logNp = np.array([math.log(max(g["Np_target"], 1e-9)) for g in groups], dtype=float)
        fracA = np.array([g["frac_A_target"] for g in groups], dtype=float)

        return {
            "pool_path": pool_path,
            "groups": groups,
            "sample_records": sample_records,
            "Np_vals": Np_vals,
            "XA_vals": XA_vals,
            "logNp": logNp,
            "fracA": fracA,
            "logNp_range": float(logNp.max() - logNp.min()) if logNp.size > 1 else 1.0,
            "fracA_range": float(fracA.max() - fracA.min()) if fracA.size > 1 else 1.0,
        }

    def _get_cache(self, Df: float, MAS: float) -> Dict[str, Any]:
        pool_path = self._resolve_pool_dir(Df, MAS)
        cache = self._pool_cache.get(pool_path, None)
        if cache is not None:
            self._pool_cache.move_to_end(pool_path)
            return cache

        cache = self._build_index(pool_path)
        self._pool_cache[pool_path] = cache
        self._pool_cache.move_to_end(pool_path)
        self._enforce_pool_cache_limit()
        return cache

    def _read_sample_triplet(self, cache: Dict[str, Any], gname: str, subname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._read_calls += 1
        pool_path = str(cache["pool_path"])
        sample_key = (pool_path, str(gname), str(subname))

        cached = self._sample_array_cache.get(sample_key, None)
        if cached is not None:
            self._sample_cache_hits += 1
            self._sample_array_cache.move_to_end(sample_key)
            M, Hbond, Vbond = cached
            return M.copy(), Hbond.copy(), Vbond.copy()

        self._sample_cache_misses += 1
        record = cache["sample_records"].get((str(gname), str(subname)), None)
        if record is None:
            raise KeyError(f"[AggPool] Sample record not found: group={gname}, sample={subname}")

        self._npz_file_opens += 1
        npz_path = str(Path(pool_path) / str(record["npz_relpath"]))
        with np.load(npz_path, allow_pickle=False) as data:
            M = np.asarray(data["M"])
            Hbond = np.asarray(data["Hbond"])
            Vbond = np.asarray(data["Vbond"])

        if self.sample_cache_size > 0:
            self._sample_array_cache[sample_key] = (M, Hbond, Vbond)
            self._sample_array_cache.move_to_end(sample_key)
            self._enforce_sample_cache_limit()
            return M.copy(), Hbond.copy(), Vbond.copy()

        return M, Hbond, Vbond

    @staticmethod
    def _find_bracketing(vals: np.ndarray, x: float) -> Tuple[float, float]:
        if x <= vals[0]:
            return float(vals[0]), float(vals[0])
        if x >= vals[-1]:
            return float(vals[-1]), float(vals[-1])
        idx = int(np.searchsorted(vals, x))
        return float(vals[idx - 1]), float(vals[idx])

    def _pick_group_knn(
        self,
        cache: Dict[str, Any],
        A_norm: float,
        X1: float,
        rng: np.random.Generator,
        KNN: int,
        sigma: float,
    ) -> Dict[str, Any]:
        logA = math.log(max(A_norm, 1e-9))
        d_logNp = (logA - cache["logNp"]) / cache["logNp_range"]
        d_fracA = (X1 - cache["fracA"]) / cache["fracA_range"]
        dist2 = d_logNp * d_logNp + d_fracA * d_fracA

        K = min(int(KNN), dist2.size)
        nn_idx = np.argpartition(dist2, K - 1)[:K]
        nn_dist2 = dist2[nn_idx]

        w = np.exp(-nn_dist2 / (2.0 * sigma * sigma))
        w_sum = float(w.sum())
        probs = (w / w_sum) if (np.isfinite(w_sum) and w_sum > 0) else np.ones(K) / K
        return cache["groups"][int(nn_idx[int(rng.choice(K, p=probs))])]

    def _pick_group_bilinear(
        self,
        cache: Dict[str, Any],
        A_norm: float,
        X1: float,
        *,
        log_bilinear: bool = False,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        Np_vals = cache["Np_vals"]
        XA_vals = cache["XA_vals"]

        if log_bilinear:
            logNp_vals = np.log(np.maximum(Np_vals, 1e-9))
            logA = math.log(max(A_norm, 1e-9))
            logNp_lo, logNp_hi = self._find_bracketing(logNp_vals, logA)
            Np_lo = float(np.exp(logNp_lo))
            Np_hi = float(np.exp(logNp_hi))
            tx = 0.0 if logNp_hi == logNp_lo else (logA - logNp_lo) / (logNp_hi - logNp_lo)
        else:
            Np_lo, Np_hi = self._find_bracketing(Np_vals, A_norm)
            tx = 0.0 if Np_hi == Np_lo else (A_norm - Np_lo) / (Np_hi - Np_lo)

        XA_lo, XA_hi = self._find_bracketing(XA_vals, X1)
        ty = 0.0 if XA_hi == XA_lo else (X1 - XA_lo) / (XA_hi - XA_lo)

        corners = [
            (Np_lo, XA_lo, (1 - tx) * (1 - ty)),
            (Np_lo, XA_hi, (1 - tx) * ty),
            (Np_hi, XA_lo, tx * (1 - ty)),
            (Np_hi, XA_hi, tx * ty),
        ]

        cand_groups: List[Dict[str, Any]] = []
        cand_w: List[float] = []
        for Np_c, XA_c, w in corners:
            if w <= 0:
                continue
            group = next(
                (
                    gg for gg in cache["groups"]
                    if float(gg["Np_target"]) == float(Np_c)
                    and float(gg["frac_A_target"]) == float(XA_c)
                ),
                None,
            )
            if group is not None:
                cand_groups.append(group)
                cand_w.append(float(w))

        if not cand_groups:
            group = self._pick_group_knn(cache, A_norm, X1, np.random.default_rng(), KNN=1, sigma=1.0)
            return [group], np.array([1.0], dtype=float)

        w_arr = np.array(cand_w, dtype=float)
        w_arr /= w_arr.sum()
        return cand_groups, w_arr

    def sample_grid(
        self,
        Df: float,
        MAS: float,
        A_norm: float,
        X1: float,
        rng: np.random.Generator,
        *,
        interp: str = "knn",
        KNN: int = 4,
        sigma: float = 0.35,
        max_draws: int = 15,
        tau_A: float | None = None,
        tau_X: float | None = None,
        log_bilinear: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache = self._get_cache(Df, MAS)

        if interp != "bilinear":
            group = self._pick_group_knn(cache, A_norm, X1, rng, KNN=KNN, sigma=sigma)
            sample_name = group["samples"][int(rng.integers(0, len(group["samples"]))) ]
            return self._read_sample_triplet(cache, group["gname"], sample_name)

        cand_groups, cand_probs = self._pick_group_bilinear(cache, A_norm, X1, log_bilinear=log_bilinear)

        if tau_A is None or tau_X is None:
            group = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
            sample_name = group["samples"][int(rng.integers(0, len(group["samples"]))) ]
            return self._read_sample_triplet(cache, group["gname"], sample_name)

        best_dist2 = float("inf")
        best_triplet: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for _ in range(int(max_draws)):
            group = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
            sample_name = group["samples"][int(rng.integers(0, len(group["samples"]))) ]
            M, Hbond, Vbond = self._read_sample_triplet(cache, group["gname"], sample_name)

            n1 = int((M == 1).sum())
            n2 = int((M == 2).sum())
            occ = n1 + n2
            if occ <= 0:
                continue

            dA = abs(float(occ) - A_norm) / max(A_norm, 1e-9)
            dX = abs(float(n1) / float(occ) - X1)
            dist2 = dA * dA + dX * dX
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_triplet = (M, Hbond, Vbond)

            if dA <= tau_A and dX <= tau_X:
                return M, Hbond, Vbond

        if best_triplet is not None:
            return best_triplet

        group = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
        sample_name = group["samples"][int(rng.integers(0, len(group["samples"]))) ]
        return self._read_sample_triplet(cache, group["gname"], sample_name)


