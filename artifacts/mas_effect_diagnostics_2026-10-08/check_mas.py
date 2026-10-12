"""Read-only diagnostics for the supplied 20 s / 40 s scans."""
from itertools import combinations, product
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

STRICT_CHECKS = True
OUTPUT = Path(__file__).parent
METRICS = ("number_gain_pct", "mu20_loss_pct", "mu02_loss_pct", "mu11_loss_pct",
           "D50_loss_pct", "total_m2_loss_pct", "fine_volume_pct")
FACTORS = ("MAS", "X1", "STR_combination", "gamma")


def sensitivity(means):
    """Exact variance decomposition on the uniformly weighted discrete grid."""
    variance = np.var(means, axis=(0, 1, 2, 3))
    first, total = [], []
    for axis in range(4):
        other = tuple(i for i in range(4) if i != axis)
        first.append(np.var(np.mean(means, axis=other), axis=0))
        total.append(np.mean(np.var(means, axis=axis), axis=tuple(range(3))))
    return variance, np.stack(first), np.stack(total)


def self_check():
    a, b, c, d = np.indices((3, 3, 2, 3))
    y = (a + 2 * b).astype(float)[..., None, None]
    v, s, st = sensitivity(y)
    np.testing.assert_allclose(s.sum(axis=0), v)
    np.testing.assert_allclose(s, st)
    interaction = ((a - 1) * (b - 1)).astype(float)[..., None, None]
    v, s, st = sensitivity(interaction)
    np.testing.assert_allclose(s, 0.0, atol=1e-15)
    np.testing.assert_allclose(st[:2], np.broadcast_to(v, st[:2].shape))


def read_scan(phase):
    path = Path(f"D:/Codex_tem/LMC/pbe_parameter_scan_phase{phase}/phase1_results.h5")
    with h5py.File(path, "r") as f:
        config = json.loads(f["config_json"][()].decode())
        t = f["t_vec"][()]
        grid = f["psd_x_grid"][()]
        levels = (config["mas_values"], config["x1_values"],
                  [v for v in product(config["str_values"], repeat=3) if v[0] <= v[2]],
                  config["gamma_values"])
        shape = tuple(len(v) for v in levels)
        assert shape == (3, 3, 40, 3)
        n = config["n_repeats"]
        data = np.empty((*shape, n, len(t), len(METRICS)))
        seeds = np.empty((*shape, n), dtype=np.uint64)
        seen = set()
        max_conservation = 0.0
        for g in f["conditions"].values():
            assert bool(g.attrs["complete"])
            key = (float(g.attrs["MAS"]), float(g.attrs["X1"]),
                   tuple(g.attrs["STR"]), float(g.attrs["gamma"]))
            idx = tuple(levels[i].index(key[i]) for i in range(4))
            assert idx not in seen
            seen.add(idx)
            m, x, q = g["moments"][()], g["x50"][()], g["psd_Q"][()]
            if STRICT_CHECKS:
                assert m.shape == (n, 3, 3, len(t))
                assert x.shape == (n, len(t)) and q.shape == (n, len(t), len(grid))
                assert np.all(np.isfinite(m)) and np.all(np.isfinite(x[:, 1:])) and np.all(np.isfinite(q))
                assert np.all(np.isnan(x[:, 0]))
                assert np.all(m[..., 0] > 0)
                np.testing.assert_allclose(m.mean(axis=0), g["moments_mean"][()], rtol=1e-12)
                assert np.all(np.diff(q, axis=-1) >= -1e-12)
                assert np.min(q) >= -1e-12 and np.max(q) <= 1 + 1e-12
            total_m2 = m[:, 2, 0] + 2 * m[:, 1, 1] + m[:, 0, 2]
            d0 = (6 * config["initial_particle_volume"] / np.pi) ** (1 / 3)
            # Stored t=0 medians are NaN; monodisperse initial diameter is known analytically.
            diameter_loss = np.concatenate((np.zeros((n, 1)), 100 * (1 - x[:, 1:] / d0)), axis=1)
            k = np.searchsorted(grid, 0.9 * d0)
            w = (0.9 * d0 - grid[k - 1]) / (grid[k] - grid[k - 1])
            fine = 100 * ((1 - w) * q[..., k - 1] + w * q[..., k])
            data[idx] = np.stack([
                100 * (m[:, 0, 0] / m[:, 0, 0, :1] - 1),
                100 * (1 - m[:, 2, 0] / m[:, 2, 0, :1]),
                100 * (1 - m[:, 0, 2] / m[:, 0, 2, :1]),
                100 * (1 - m[:, 1, 1] / m[:, 1, 1, :1]),
                diameter_loss,
                100 * (1 - total_m2 / total_m2[:, :1]), fine], axis=-1)
            seeds[idx] = g["seed"][()]
            error = float(np.max(g["phase_volume_relative_error"][()]))
            max_conservation = max(max_conservation, error)
        assert len(seen) == np.prod(shape)
    print(f"Loaded phase {phase}: {len(seen)} complete conditions, {n} repeats, T={t[-1]} s")
    return path, config, t, levels, data, seeds, max_conservation


def analyze(phase):
    path, cfg, t, levels, data, seeds, conservation = read_scan(phase)
    means = data.mean(axis=4)
    sd = data.std(axis=4, ddof=1)
    rows, baseline_rows = [], []
    n = cfg["n_repeats"]
    for xi, si, gi in product(range(3), range(40), range(3)):
        for mi, metric in enumerate(METRICS):
            yy = data[:, xi, si, gi, :, -1, mi]
            mm = yy.mean(axis=1)
            # Center each sample before variance to preserve exact constant samples.
            variances = np.var(yy - yy[:, :1], axis=1, ddof=1)
            row = dict(phase=phase, end_time=t[-1], X1=levels[1][xi],
                       STR0=levels[2][si][0], STR1=levels[2][si][1], STR2=levels[2][si][2],
                       gamma=levels[3][gi], metric=metric, span_pp=np.ptp(mm),
                       delta_09_minus_01_pp=mm[2]-mm[0],
                       pooled_repeat_sd_pp=np.sqrt(np.mean(variances)))
            for a in range(3):
                row[f"mean_MAS_{levels[0][a]}"] = mm[a]
            for a, b in combinations(range(3), 2):
                # Different case seeds: these are independent, not paired repeats.
                if variances[a] == 0 and variances[b] == 0:
                    p = 1.0 if mm[a] == mm[b] else np.nan
                else:
                    p = float(stats.ttest_ind_from_stats(mm[a], np.sqrt(variances[a]), n,
                              mm[b], np.sqrt(variances[b]), n, equal_var=False).pvalue)
                row[f"p_{a}{b}"] = p
            rows.append(row)
    table = pd.DataFrame(rows)
    for metric in METRICS:
        mask = table.metric == metric
        p = table.loc[mask, ["p_01", "p_02", "p_12"]].to_numpy()
        valid = np.isfinite(p)
        q = np.full_like(p, np.nan)
        q[valid] = stats.false_discovery_control(p[valid], method="bh")
        table.loc[mask, ["q_01", "q_02", "q_12"]] = q
    table.to_csv(OUTPUT / f"phase{phase}_matched_MAS.csv", index=False)

    # Baseline STR/gamma, all three X1, each repeat's own initial value.
    si, gi = levels[2].index((1., 1., 1.)), levels[3].index(1.)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for xi in range(3):
        for panel, mi in enumerate((0, 2, 3)):
            ax = axes[xi, panel]
            for a, mas in enumerate(levels[0]):
                yy = data[a, xi, si, gi, :, :, mi]
                mean = yy.mean(axis=0)
                half = stats.t.ppf(.975, n - 1) * yy.std(axis=0, ddof=1) / np.sqrt(n)
                ax.plot(t, mean, label=f"MAS={mas}")
                ax.fill_between(t, mean-half, mean+half, alpha=.15)
                baseline_rows.append(dict(phase=phase, X1=levels[1][xi], MAS=mas,
                                          metric=METRICS[mi], final_pct=mean[-1],
                                          CI95_low=mean[-1]-half[-1], CI95_high=mean[-1]+half[-1]))
            ax.set_title(f"X1={levels[1][xi]} | {METRICS[mi]}")
            ax.set_xlabel("time [s]"); ax.set_ylabel("change [% of own initial value]")
            ax.grid(alpha=.3); ax.legend()
    fig.suptitle(f"Phase {phase}: STR=(1,1,1), gamma=1; mean and pointwise 95% CI")
    fig.savefig(OUTPUT / f"phase{phase}_baseline_MAS.png", dpi=160); plt.close(fig)
    pd.DataFrame(baseline_rows).to_csv(OUTPUT / f"phase{phase}_baseline_values.csv", index=False)

    v, first, total = sensitivity(means)
    noise = np.mean(sd ** 2 / n, axis=(0, 1, 2, 3))
    sensitivity_rows = []
    for ti in range(1, len(t)):
        for mi, metric in enumerate(METRICS):
            for a, factor in enumerate(FACTORS):
                sensitivity_rows.append(dict(phase=phase, time=t[ti], metric=metric, factor=factor,
                    first=first[a, ti, mi]/v[ti, mi] if v[ti, mi] > 0 else np.nan,
                    total=total[a, ti, mi]/v[ti, mi] if v[ti, mi] > 0 else np.nan,
                    noise_of_cell_mean_fraction=noise[ti, mi]/v[ti, mi] if v[ti, mi] > 0 else np.nan))
    pd.DataFrame(sensitivity_rows).to_csv(OUTPUT / f"phase{phase}_variance_sensitivity_time.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, mi in zip(axes, (0, 2, 3)):
        for a, factor in enumerate(FACTORS):
            ax.plot(t[1:], total[a, 1:, mi]/v[1:, mi], label=factor)
        ax.set_title(METRICS[mi]); ax.set_xlabel("time [s]"); ax.set_ylabel("total discrete variance fraction")
        ax.grid(alpha=.3); ax.legend()
    fig.savefig(OUTPUT / f"phase{phase}_dynamic_sensitivity.png", dpi=160); plt.close(fig)

    summary = dict(source=str(path), end_time=t[-1], repeats=n, max_conservation_error=conservation,
                   metrics={}, X1_summaries={})
    for mi, metric in enumerate(METRICS):
        tab = table[table.metric == metric]
        iv = np.trapezoid(v[:, mi], t)
        summary["metrics"][metric] = dict(
            median_span_pp=float(tab.span_pp.median()), max_span_pp=float(tab.span_pp.max()),
            median_evolution_pct=float(np.median(means[..., -1, mi])),
            fraction_with_any_FDR05=float((tab[["q_01", "q_02", "q_12"]].min(axis=1) < .05).mean()),
            endpoint_first=dict(zip(FACTORS, (first[:, -1, mi]/v[-1, mi]).tolist())),
            endpoint_total=dict(zip(FACTORS, (total[:, -1, mi]/v[-1, mi]).tolist())),
            functional_first=dict(zip(FACTORS, (np.trapezoid(first[..., mi], t, axis=1)/iv).tolist())),
            functional_total=dict(zip(FACTORS, (np.trapezoid(total[..., mi], t, axis=1)/iv).tolist())),
            noise_mean_fraction=float(noise[-1, mi]/v[-1, mi]))
        for x in levels[1]:
            sub = tab[tab.X1 == x]
            summary["X1_summaries"][f"{x}:{metric}"] = dict(
                median_span_pp=float(sub.span_pp.median()), max_span_pp=float(sub.span_pp.max()),
                fraction_any_FDR05=float((sub[["q_01", "q_02", "q_12"]].min(axis=1)<.05).mean()),
                mean_positive_endpoint_fraction=float((sub.delta_09_minus_01_pp > 0).mean()))
    (OUTPUT / f"phase{phase}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Phase {phase}: summary and plots saved to {OUTPUT}")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for xi, x in enumerate(levels[1]):
        for col, gamma in enumerate((1., 1000.)):
            candidates = table[(table.metric == 'mu02_loss_pct') & (table.X1 == x) & (table.gamma == gamma)]
            row = candidates.loc[candidates.span_pp.idxmax()]
            strength = tuple(float(v) for v in (row.STR0, row.STR1, row.STR2))
            si, gi = levels[2].index(strength), levels[3].index(gamma)
            ax = axes[xi, col]
            for a, mas in enumerate(levels[0]):
                yy = data[a, xi, si, gi, :, :, 2]
                mean = yy.mean(axis=0)
                half = stats.t.ppf(.975, n - 1) * yy.std(axis=0, ddof=1) / np.sqrt(n)
                ax.plot(t, mean, label=f"MAS={mas}")
                ax.fill_between(t, mean-half, mean+half, alpha=.15)
            str_label = ', '.join(f'{v:g}' for v in strength)
            ax.set_title(f"X1={x}, gamma={gamma:g}\nSTR=({str_label})")
            ax.set_xlabel("time [s]"); ax.set_ylabel("mu02 loss [% of initial]")
            ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Exploratory examples: maximum MAS span within each X1/gamma stratum")
    fig.savefig(OUTPUT / f"phase{phase}_strong_MAS_examples.png", dpi=160); plt.close(fig)
    return cfg, t, data, seeds


if __name__ == "__main__":
    self_check()
    c2, t2, y2, s2 = analyze(2)
    c3, t3, y3, s3 = analyze(3)
    assert np.array_equal(s2, s3)
    comparison = dict(same_seeds=True,
        max_abs_common_prefix_difference_per_metric=dict(zip(METRICS,
            np.max(np.abs(y2-y3[..., :len(t2), :]), axis=(0,1,2,3,4,5)).tolist())),
        config_changes={k:[c2[k],c3[k]] for k in c2 if c2[k] != c3[k]})
    (OUTPUT / "phase_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("PHASE COMPARISON", json.dumps(comparison, indent=2))
