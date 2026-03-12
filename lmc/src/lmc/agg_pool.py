# -*- coding: utf-8 -*-
import os
import math
from collections import OrderedDict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import h5py


def _format_pool_filename(Df: float, MAS: float) -> str:
    df_str = str(Df).replace(".", "p")
    mas_str = f"{MAS:.2f}".replace(".", "p")
    return f"aggregate_pool_Df{df_str}_MAS{mas_str}.h5"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


class AggPool:
    """
    Manage offline aggregate pools with bounded caching and optional transient HDF5 reads.

    Why two read modes exist
    ------------------------
    `mc_breakage_from_pool()` differs from `mc_breakage_repeat()` mainly because it
    repeatedly reads small arrays from HDF5. In long MCPBE-LMC runs, that can drive
    native-memory growth inside the HDF5 stack even when Python-level caches stay flat.

    To make this observable and controllable, the pool supports:
    - persistent mode: keep the HDF5 file handle open across reads;
    - transient mode: open the file for each miss, read arrays, then close immediately.

    The default is transient mode, which is safer for long-running jobs.
    """

    def __init__(
        self,
        pool_dir: str,
        *,
        max_open_pools: int = 2,
        sample_cache_size: int = 16,
        keep_h5_open: Optional[bool] = None,
    ):
        self.pool_dir = pool_dir
        self.max_open_pools = max(1, int(max_open_pools))
        self.sample_cache_size = max(0, int(sample_cache_size))
        self.keep_h5_open = _env_flag("LMC_POOL_KEEP_H5_OPEN", False) if keep_h5_open is None else bool(keep_h5_open)

        self._pool_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._sample_array_cache: "OrderedDict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
        self._pool_last_key: Optional[str] = None

        self._read_calls = 0
        self._sample_cache_hits = 0
        self._sample_cache_misses = 0
        self._index_builds = 0
        self._transient_file_opens = 0

    def _close_cache_entry(self, cache: Dict[str, Any]) -> None:
        h5 = cache.get("h5", None)
        try:
            if h5 is not None:
                h5.close()
        except Exception:
            pass
        cache["h5"] = None

    def _drop_sample_cache_for_pool(self, pool_path: str) -> None:
        keys_to_drop = [key for key in self._sample_array_cache.keys() if key[0] == pool_path]
        for key in keys_to_drop:
            self._sample_array_cache.pop(key, None)

    def _enforce_pool_cache_limit(self) -> None:
        while len(self._pool_cache) > self.max_open_pools:
            pool_path, cache = self._pool_cache.popitem(last=False)
            self._drop_sample_cache_for_pool(pool_path)
            self._close_cache_entry(cache)
            if self._pool_last_key == pool_path:
                self._pool_last_key = next(reversed(self._pool_cache), None) if self._pool_cache else None

    def _enforce_sample_cache_limit(self) -> None:
        if self.sample_cache_size <= 0:
            self._sample_array_cache.clear()
            return
        while len(self._sample_array_cache) > self.sample_cache_size:
            self._sample_array_cache.popitem(last=False)

    def close_pool_cache(self) -> None:
        for cache in self._pool_cache.values():
            self._close_cache_entry(cache)
        self._pool_cache.clear()
        self._sample_array_cache.clear()
        self._pool_last_key = None

    def _h5_object_counts(self) -> Dict[str, int]:
        global_total = 0
        global_files = 0
        global_groups = 0
        global_datasets = 0
        global_datatypes = 0
        global_attrs = 0
        try:
            global_total = int(h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_ALL))
            global_files = int(h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE))
            global_groups = int(h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_GROUP))
            global_datasets = int(h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_DATASET))
            global_datatypes = int(h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_DATATYPE))
            global_attrs = int(h5py.h5f.get_obj_count(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_ATTR))
        except Exception:
            pass

        total = 0
        files = 0
        groups = 0
        datasets = 0
        datatypes = 0
        attrs = 0

        for cache in self._pool_cache.values():
            h5 = cache.get("h5", None)
            if h5 is None:
                continue
            try:
                fid = h5.id
                total += int(h5py.h5f.get_obj_count(fid, h5py.h5f.OBJ_ALL))
                files += int(h5py.h5f.get_obj_count(fid, h5py.h5f.OBJ_FILE))
                groups += int(h5py.h5f.get_obj_count(fid, h5py.h5f.OBJ_GROUP))
                datasets += int(h5py.h5f.get_obj_count(fid, h5py.h5f.OBJ_DATASET))
                datatypes += int(h5py.h5f.get_obj_count(fid, h5py.h5f.OBJ_DATATYPE))
                attrs += int(h5py.h5f.get_obj_count(fid, h5py.h5f.OBJ_ATTR))
            except Exception:
                continue

        return {
            "h5_global_obj_total": global_total,
            "h5_global_obj_files": global_files,
            "h5_global_obj_groups": global_groups,
            "h5_global_obj_datasets": global_datasets,
            "h5_global_obj_datatypes": global_datatypes,
            "h5_global_obj_attrs": global_attrs,
            "h5_open_obj_total": total,
            "h5_open_obj_files": files,
            "h5_open_obj_groups": groups,
            "h5_open_obj_datasets": datasets,
            "h5_open_obj_datatypes": datatypes,
            "h5_open_obj_attrs": attrs,
        }

    def cache_stats(self) -> Dict[str, int]:
        stats = {
            "open_pools": int(sum(1 for c in self._pool_cache.values() if c.get("h5", None) is not None)),
            "indexed_pools": int(len(self._pool_cache)),
            "cached_samples": int(len(self._sample_array_cache)),
            "sample_cache_hits": int(self._sample_cache_hits),
            "sample_cache_misses": int(self._sample_cache_misses),
            "pool_read_calls": int(self._read_calls),
            "pool_index_builds": int(self._index_builds),
            "pool_transient_file_opens": int(self._transient_file_opens),
            "pool_keep_h5_open": int(self.keep_h5_open),
        }
        stats.update(self._h5_object_counts())
        return stats

    def _ensure_pool_handle(self, cache: Dict[str, Any]):
        if cache.get("h5", None) is None:
            cache["h5"] = h5py.File(str(cache["pool_path"]), "r")
            self._pool_cache.move_to_end(str(cache["pool_path"]))
            self._pool_last_key = str(cache["pool_path"])

    # ---------- internal: open & index ----------
    def _build_index_from_file(self, pool_path: str) -> Dict[str, Any]:
        self._index_builds += 1
        with h5py.File(pool_path, "r") as h5:
            groups: List[Dict[str, Any]] = []
            for gname in h5.keys():
                grp = h5[gname]
                Np_t = float(grp.attrs.get("Np_target", np.nan))
                fracA_t = float(grp.attrs.get("frac_A_target", np.nan))
                samples = tuple(s for s in grp.keys() if s.startswith("sample_"))
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
        logNp_range = float(logNp.max() - logNp.min()) if logNp.size > 1 else 1.0
        fracA_range = float(fracA.max() - fracA.min()) if fracA.size > 1 else 1.0

        cache = dict(
            h5=None,
            groups=groups,
            pool_path=pool_path,
            Np_vals=Np_vals,
            XA_vals=XA_vals,
            logNp=logNp,
            fracA=fracA,
            logNp_range=logNp_range,
            fracA_range=fracA_range,
        )
        return cache

    def _get_cache(self, Df: float, MAS: float) -> Dict[str, Any]:
        fname = _format_pool_filename(Df, MAS)
        pool_path = os.path.join(self.pool_dir, fname)
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"[AggPool] Pool file not found: {pool_path}")

        cache = self._pool_cache.get(pool_path, None)
        if cache is not None:
            if self.keep_h5_open:
                self._ensure_pool_handle(cache)
            self._pool_cache.move_to_end(pool_path)
            self._pool_last_key = pool_path
            return cache

        cache = self._build_index_from_file(pool_path)
        if self.keep_h5_open:
            cache["h5"] = h5py.File(pool_path, "r")
        self._pool_cache[pool_path] = cache
        self._pool_cache.move_to_end(pool_path)
        self._pool_last_key = pool_path
        self._enforce_pool_cache_limit()
        return cache

    def _read_triplet_from_file(self, pool_path: str, gname: str, subname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._transient_file_opens += 1
        with h5py.File(pool_path, "r") as h5:
            prefix = f"{gname}/{subname}"
            M = h5[f"{prefix}/M"][()]
            Hbond = h5[f"{prefix}/Hbond"][()]
            Vbond = h5[f"{prefix}/Vbond"][()]
        return M, Hbond, Vbond

    def _read_sample_triplet(
        self,
        cache: Dict[str, Any],
        gname: str,
        subname: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if self.keep_h5_open:
            self._ensure_pool_handle(cache)
            h5 = cache["h5"]
            assert h5 is not None
            prefix = f"{gname}/{subname}"
            M = h5[f"{prefix}/M"][()]
            Hbond = h5[f"{prefix}/Hbond"][()]
            Vbond = h5[f"{prefix}/Vbond"][()]
        else:
            M, Hbond, Vbond = self._read_triplet_from_file(pool_path, gname, subname)

        if self.sample_cache_size > 0:
            self._sample_array_cache[sample_key] = (M, Hbond, Vbond)
            self._sample_array_cache.move_to_end(sample_key)
            self._enforce_sample_cache_limit()
            return M.copy(), Hbond.copy(), Vbond.copy()

        return M, Hbond, Vbond

    # ---------- selection helpers ----------
    @staticmethod
    def _find_bracketing(vals: np.ndarray, x: float) -> Tuple[float, float]:
        if x <= vals[0]:
            return float(vals[0]), float(vals[0])
        if x >= vals[-1]:
            return float(vals[-1]), float(vals[-1])
        idx = int(np.searchsorted(vals, x))
        lo = float(vals[idx - 1])
        hi = float(vals[idx])
        return lo, hi

    def _pick_group_knn(
        self, cache: Dict[str, Any], A_norm: float, X1: float,
        rng: np.random.Generator, KNN: int, sigma: float
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

        k_pick = int(rng.choice(K, p=probs))
        return cache["groups"][int(nn_idx[k_pick])]

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

            def t_log(x_log: float, x0_log: float, x1_log: float) -> float:
                if x1_log == x0_log:
                    return 0.0
                return (x_log - x0_log) / (x1_log - x0_log)

            tx = t_log(logA, logNp_lo, logNp_hi)
        else:
            Np_lo, Np_hi = self._find_bracketing(Np_vals, A_norm)

            def t_lin(x: float, x0: float, x1: float) -> float:
                if x1 == x0:
                    return 0.0
                return (x - x0) / (x1 - x0)

            tx = t_lin(A_norm, Np_lo, Np_hi)

        XA_lo, XA_hi = self._find_bracketing(XA_vals, X1)

        def t_lin(x: float, x0: float, x1: float) -> float:
            if x1 == x0:
                return 0.0
            return (x - x0) / (x1 - x0)

        ty = t_lin(X1, XA_lo, XA_hi)

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
            g = next(
                (
                    gg
                    for gg in cache["groups"]
                    if float(gg["Np_target"]) == float(Np_c)
                    and float(gg["frac_A_target"]) == float(XA_c)
                ),
                None,
            )
            if g is not None:
                cand_groups.append(g)
                cand_w.append(float(w))

        if not cand_groups:
            g_nn = self._pick_group_knn(
                cache,
                A_norm,
                X1,
                np.random.default_rng(),
                KNN=1,
                sigma=1.0,
            )
            return [g_nn], np.array([1.0], dtype=float)

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
            g_pick = self._pick_group_knn(cache, A_norm, X1, rng, KNN=KNN, sigma=sigma)
            subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"]))) ]
            return self._read_sample_triplet(cache, g_pick["gname"], subname)

        cand_groups, cand_probs = self._pick_group_bilinear(
            cache, A_norm, X1, log_bilinear=log_bilinear
        )

        if tau_A is None or tau_X is None:
            g_pick = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
            subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"]))) ]
            return self._read_sample_triplet(cache, g_pick["gname"], subname)

        best_dist2 = float("inf")
        best_triplet: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for _ in range(int(max_draws)):
            g_pick = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
            subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"]))) ]
            M, Hbond, Vbond = self._read_sample_triplet(cache, g_pick["gname"], subname)

            n1 = int((M == 1).sum())
            n2 = int((M == 2).sum())
            occ = n1 + n2
            if occ <= 0:
                continue
            A_norm_pool = float(occ)
            X1_pool = float(n1) / float(occ)

            dA = abs(A_norm_pool - A_norm) / max(A_norm, 1e-9)
            dX = abs(X1_pool - X1)

            dist2 = dA * dA + dX * dX
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_triplet = (M, Hbond, Vbond)

            if dA <= tau_A and dX <= tau_X:
                return M, Hbond, Vbond

        if best_triplet is not None:
            return best_triplet

        g_pick = cand_groups[int(rng.choice(len(cand_groups), p=cand_probs))]
        subname = g_pick["samples"][int(rng.integers(0, len(g_pick["samples"]))) ]
        return self._read_sample_triplet(cache, g_pick["gname"], subname)
