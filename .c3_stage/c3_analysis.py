"""Read-only C3 graph accounting; existing initial specimen edges only."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
from regular_aggregate.geometry import require
from regular_aggregate.pipeline import read_json, read_csv, write_json, file_hash
from regular_loading.fragment_history import labels_at, write_csv
from regular_loading.binary_activity import read_activity
from regular_loading.scene import interface
from .first_split import grid_edges
from .c2_report import summary, reach_interval


def partition(n, edges, pre_active, post_active, parent_id, coordinates, transverse_count, strict):
    """Partition broken initial edges using PRE-event parent / POST-event children."""
    edges = np.asarray(edges, dtype=int)
    pre_active, post_active = np.asarray(pre_active, dtype=bool), np.asarray(post_active, dtype=bool)
    require(edges.shape == (len(pre_active), 2) and pre_active.shape == post_active.shape,
            "Graph dimensions differ")
    require(np.all(~post_active | pre_active), "A broken edge was restored during the event")
    before = labels_at(n, edges, pre_active); after = labels_at(n, edges, post_active)
    parent = before == parent_id
    require(parent.any(), "Absent mother")
    children, sizes = np.unique(after[parent], return_counts=True)
    require(len(children) >= 2, "Selected mother did not split")
    order = np.argsort(-sizes, kind="stable"); children, sizes = children[order], sizes[order]
    a, b = edges.T
    same_parent = parent[a] & parent[b]
    same_before = before[a] == before[b]
    different_after = after[a] != after[b]
    pre_broken = ~pre_active
    mother_damage = pre_broken & same_parent
    old_internal = pre_broken & same_before & ~same_parent
    old_cut = pre_broken & ~same_before
    mother_old_cut = old_cut & (parent[a] | parent[b])
    other_old_cut = old_cut & ~mother_old_cut
    cut = same_parent & different_after
    future_cut_damage = mother_damage & cut
    child_internal_damage = mother_damage & ~different_after
    new = pre_active & ~post_active
    new_cut = new & cut
    delta = coordinates[edges[cut, 1]] - coordinates[edges[cut, 0]]
    normals = delta / np.linalg.norm(delta, axis=1)[:, None]
    moment = normals.T @ normals / len(normals)
    midpoint = (coordinates[edges[cut, 0]] + coordinates[edges[cut, 1]]) / 2
    parent_den = int(same_parent.sum())
    categories = np.full(len(edges), "intact_before", dtype="<U32")
    categories[future_cut_damage] = "mother_future_cut_damage"
    categories[child_internal_damage] = "mother_child_internal_damage"
    categories[old_internal] = "old_fragment_internal_damage"
    categories[mother_old_cut] = "mother_old_fragment_cut"
    categories[other_old_cut] = "old_old_fragment_cut"
    result = {
        "n": n, "initial_edges": len(edges), "parent_size": int(parent.sum()),
        "child_sizes": sizes.tolist(), "alpha": float(parent.mean()),
        "f2_initial": float(sizes[1] / n), "f2_conditional": float(sizes[1] / parent.sum()),
        "pre_broken": int(pre_broken.sum()), "pre_broken_fraction": float(pre_broken.mean()),
        "parent_initial_internal_edges": parent_den, "parent_internal_broken": int(mother_damage.sum()),
        "parent_internal_damage_fraction": float(mother_damage.sum() / parent_den),
        "old_fragment_internal_broken": int(old_internal.sum()),
        "pre_existing_fragment_cut": int(old_cut.sum()),
        "mother_old_fragment_cut": int(mother_old_cut.sum()),
        "old_old_fragment_cut": int(other_old_cut.sum()),
        "mother_future_cut_damage": int(future_cut_damage.sum()),
        "mother_child_internal_damage": int(child_internal_damage.sum()),
        "mother_child_internal_damage_fraction": float(child_internal_damage.sum() / parent_den),
        "cut_bonds": int(cut.sum()), "eta_cut": float(cut.sum() / transverse_count),
        "cut_already_broken": int((cut & pre_broken).sum()), "cut_newly_failed": int(new_cut.sum()),
        "simultaneous_new_failures": int(new.sum()), "new_failures_outside_cut": int((new & ~cut).sum()),
        "cut_direction_second_moment": moment.tolist(), "cut_qxx": float(moment[0, 0]),
        "cut_midpoint": midpoint.mean(axis=0).tolist(),
        "pre_component_count": int(len(np.unique(before))), "post_component_count": int(len(np.unique(after))),
        "children": [],
    }
    bounds = np.column_stack((coordinates.min(axis=0), coordinates.max(axis=0)))
    for ident, size in zip(children, sizes):
        cc = coordinates[after == ident]
        box = np.column_stack((cc.min(axis=0), cc.max(axis=0)))
        result["children"].append({"size": int(size), "centroid": cc.mean(axis=0).tolist(),
            "bbox": box.tolist(), "touches_external_faces": np.isclose(box, bounds, rtol=0, atol=0).tolist()})
    if strict:
        require(np.isfinite(coordinates).all() and len(np.unique(edges, axis=0)) == len(edges), "Invalid geometry")
        require(np.all(~same_before <= pre_broken), "Pre-event components joined by an intact edge")
        require(np.all(~post_active[cut]), "New-child cut contains intact edges")
        require(result["pre_broken"] == result["parent_internal_broken"] + result["old_fragment_internal_broken"] + result["pre_existing_fragment_cut"], "Pre-damage partition is not exhaustive")
        require(result["parent_internal_broken"] == result["mother_future_cut_damage"] + result["mother_child_internal_damage"], "Mother damage partition failed")
        require(result["cut_bonds"] == result["cut_already_broken"] + result["cut_newly_failed"], "Cut partition failed")
        require(int(sizes.sum()) == result["parent_size"] and np.bincount(after).sum() == n, "Volume conservation failed")
        # Independent union-find implementation, insensitive to label numbering.
        from fuchs_aggreagte.fuchs_aggregate import legacy
        for mask, labels in ((pre_active, before), (post_active, after)):
            independent = legacy().components(n, edges, mask)
            pairs = np.unique(np.column_stack((independent, labels)), axis=0)
            require(len(pairs) == len(np.unique(labels)) == len(np.unique(independent)), "Connectivity implementations differ")
    return result, {"before": before, "after": after, "parent": parent, "cut": cut,
        "mother_damage": mother_damage, "future_cut_damage": future_cut_damage,
        "child_internal_damage": child_internal_damage, "categories": categories}


def self_check():
    # A cycle damaged before a 4+1 mother split, following prior one-node spall.
    edges = np.array([[0,1],[1,2],[2,3],[3,0],[3,4],[4,5]])
    xy = np.column_stack((np.arange(6), np.zeros(6)))
    pre = np.array([False, True, True, True, True, False]); post = pre.copy(); post[3] = False
    r, _ = partition(6, edges, pre, post, 0, xy, 1, True)
    assert r["child_sizes"] == [4,1]
    assert r["pre_broken"] == 2 and r["parent_internal_broken"] == 1 and r["mother_old_fragment_cut"] == 1
    assert r["cut_bonds"] == 2 and r["cut_already_broken"] == r["cut_newly_failed"] == 1
    # Two simultaneous cuts must be one event, with no fabricated intermediate state.
    square = np.array([[0,1],[1,2],[2,3],[3,0]])
    coords = np.array([[0,0],[1,0],[1,1],[0,1]])
    r, _ = partition(4, square, np.ones(4, bool), np.array([False,True,False,True]), 0, coords, 2, True)
    assert r["child_sizes"] == [2,2] and r["simultaneous_new_failures"] == r["cut_newly_failed"] == 2
    fast, _ = partition(4, square, np.ones(4, bool), np.array([False,True,False,True]), 0, coords, 2, False)
    assert r == fast
    assert pareto(np.array([[1.,2.],[2.,1.],[3.,3.],[1.,2.]])).tolist() == [True,True,False,True]
    try:
        partition(4, square, np.array([False,True,True,True]), np.ones(4, bool), 0, coords, 2, True)
    except ValueError:
        pass
    else:
        raise AssertionError("Restored edge accepted")
    return {"status": "passed", "checks": ["microfragment_and_cycle_partition", "simultaneous_failure", "strict_fast_identity", "pareto_ties", "edge_restoration_rejected"], "new_simulations": 0}


def pareto(values):
    return np.array([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1)) for row in values])


def dem_analysis(c, output):
    source = Path(c["dem_case"]); history_root = Path(c["dem_history"])
    metadata = read_json(source / "metadata.json"); params = read_json(source / "parameters.json")
    require(params["dt_s"] == c["reference_dt_s"] and params["speed_m_s"] == c["reference_speed_m_s"], "Reference condition changed")
    require(params["normal_axis"] == "x", "Expected x-axis reference loading")
    events = read_csv(history_root / "events.csv")
    chosen = None
    for ev in events:
        qualified = [p for p in json.loads(ev["parent_splits"]) if p["child_sizes"][1] / 8000 >= c["primary_threshold"]]
        if qualified:
            require(len(qualified) == 1, "C3 reference requires one qualifying mother")
            chosen = ev; expected = qualified[0]; break
    require(chosen is not None, "Reference has no primary event")
    t = float(chosen["time_s"]); ids = metadata["bulk"]["sphere_ids"]
    lookup = {ident: index for index, ident in enumerate(ids)}
    edges = np.array([[lookup[a], lookup[b]] for a,b in metadata["bulk"]["edges"]], dtype=int)
    activity = read_activity(source / "result.mdem")
    ends = np.array([activity[i][3] for i in metadata["bulk"]["bond_ids"]])
    require(len(ids) == 8000 and len(edges) == 22800, "Reference geometry changed")
    objects, _ = interface().parse_text(source / "source_current.txt")
    xyz_m = np.array([objects[i]["states"][0][12] for i in ids])
    xyz = xyz_m / metadata["bulk"]["diameter_m"]
    before = labels_at(len(ids), edges, ends >= t); after = labels_at(len(ids), edges, ends > t)
    qualifying = []
    for parent_id in np.unique(before):
        sizes = np.sort(np.unique(after[before == parent_id], return_counts=True)[1])[::-1]
        if len(sizes) > 1 and sizes[1] / len(ids) >= c["primary_threshold"]:
            qualifying.append(int(parent_id))
    require(len(qualifying) == 1, "Binary primary event inconsistent")
    result, masks = partition(len(ids), edges, ends >= t, ends > t, qualifying[0], xyz, 400, c["strict_checks"])
    require(result["parent_size"] == expected["parent_size"] and result["child_sizes"] == expected["child_sizes"], "Primary mother/children differ from saved history")
    require(result["pre_broken"] + result["simultaneous_new_failures"] == int(chosen["cumulative_broken_bonds"]), "Failure times differ from history")
    result.update(time_s=t, dt_s=params["dt_s"], speed_m_s=params["speed_m_s"], backend=params["backend"], source=str(source), history_source=str(history_root), new_simulations=0)
    write_json(output / "dem_reference.json", result)
    np.savez_compressed(output / "dem_event_graph.npz", coordinates_d0=xyz, sphere_ids=ids,
        edges=edges, failure_times_s=ends, before=masks["before"], after=masks["after"],
        mother=masks["parent"], event_cut=masks["cut"], pre_internal_damage=masks["mother_damage"],
        pre_child_internal_damage=masks["child_internal_damage"])
    mids = (xyz[edges[:,0]] + xyz[edges[:,1]]) / 2
    axes = np.argmax(np.abs(xyz[edges[:,1]] - xyz[edges[:,0]]), axis=1)
    edge_rows = [{"bond_id": ident, "sphere0": ids[a], "sphere1": ids[b],
        "axis": "xyz"[axis], "mid_x_d0": mid[0], "mid_y_d0": mid[1], "mid_z_d0": mid[2],
        "before_category": category, "event_cut": bool(cut), "fails_at_event": bool(end == t),
        "failure_time_s": float(end) if end <= t else None}
        for ident, (a,b), axis, mid, category, cut, end in zip(metadata["bulk"]["bond_ids"], edges, axes, mids, masks["categories"], masks["cut"], ends)]
    write_csv(output / "dem_edges.csv", edge_rows)
    # Only activity metadata is read from the large binary. Hash exact analyzed arrays.
    import hashlib
    write_json(output / "dem_input_evidence.json", {"result_mdem": str(source / "result.mdem"),
        "binary_size_bytes": (source / "result.mdem").stat().st_size,
        "bulk_activity_sha256": hashlib.sha256(ends.astype("<f8").tobytes()).hexdigest(),
        "metadata_sha256": file_hash(source / "metadata.json"), "parameters_sha256": file_hash(source / "parameters.json"),
        "history_sha256": file_hash(history_root / "events.csv"),
        "source_geometry_sha256": file_hash(source / "source_current.txt"),
        "full_result_binary_hashed": False, "reason": "Only native static activity and the saved initial geometry are analyzed; no trajectory re-export"})
    return result, (xyz, edges, masks)


def lmc_analysis(c, output, reference):
    source = Path(c["c2_root"]); rows = []
    manifests = read_json(source / "output_hashes.json") if c["strict_checks"] else None
    source_rows = [("coarse_or_loop", source, r) for r in read_csv(source / "realizations.csv")]
    source_rows += [("confirmation", source / "confirmation", r) for r in read_csv(source / "confirmation/realizations.csv")]
    for stage, base, row in source_rows:
        group = row["group_id"]; index = int(row["replicate_index"])
        name = f"replicate_{index:04d}" if stage != "confirmation" else f"replicate_{index:05d}"
        path = base / group / (name + ".json")
        if c["strict_checks"]:
            require(file_hash(path) == manifests[str(path.relative_to(source))], "Saved C2 JSON hash changed")
        result = read_json(path)
        record = {"stage": "confirmation" if stage == "confirmation" else row["stage"], "group_id": group,
            "gamma": float(row["gamma"]), "ell0": int(row["ell0"]), "int_bre": float(row["int_bre"]),
            "allow_loops": row["allow_loops"] == "True", "replicate_index": index, "status": row["status"],
            "primary_reached": result["primary_reached"], "source_json": str(path),
            "alpha": None, "f2_initial": None, "f2_conditional": None, "eta_cut": None,
            "pre_broken_fraction": None, "parent_internal_damage_fraction": None,
            "mother_child_internal_damage_fraction": None, "parent_internal_broken": None,
            "mother_future_cut_damage": None, "mother_child_internal_damage": None,
            "pre_existing_fragment_cut": None, "old_fragment_internal_broken": None,
            "cut_already_broken": None, "cut_newly_failed": None, "cut_qxx": None,
            "trigger_prefix_revisits_junction": None, "micro_events_before_primary": None,
            "near_case_resolution": False}
        if result["primary_reached"]:
            primary = result["threshold_events"][1]["event"]
            mother = primary["parent_splits"][0]
            npz = path.with_suffix(".npz")
            if c["strict_checks"]:
                require(file_hash(npz) == manifests[str(npz.relative_to(source))], "Saved C2 NPZ hash changed")
            with np.load(npz, allow_pickle=False) as f:
                mat, hb, vb, saved_before, saved_after = [f[k].copy() for k in ("M", "Hbond", "Vbond", "labels_before", "labels")]
            _, edges = grid_edges(mat, np.full((20,19), 11), np.full((19,20), 11))
            _, remaining = grid_edges(mat, hb, vb)
            active_set = set(map(tuple, remaining.tolist()))
            post = np.array([tuple(e) in active_set for e in edges]); pre = post.copy()
            ax, i, j = primary["path_prefix"][-1]; a = i * 20 + j; b = a + (1 if ax == 0 else 20)
            trigger = np.all(edges == [a,b], axis=1)
            require(trigger.sum() == 1 and not post[trigger].any(), "Absent primary triggering edge")
            pre[trigger] = True
            xy = np.column_stack((np.arange(400) % 20, -(np.arange(400) // 20))).astype(float)
            before = labels_at(400, edges, pre)
            members = saved_before == mother["parent_id"]
            parent_id = int(before[np.flatnonzero(members)[0]])
            metric, masks = partition(400, edges, pre, post, parent_id, xy, 20, c["strict_checks"])
            require(metric["child_sizes"] == mother["child_sizes"] and metric["f2_initial"] >= c["primary_threshold"], "Saved primary child sizes differ")
            if c["strict_checks"]:
                require(np.all(mat == 1) and np.all(masks["parent"] == members), "Mother membership or material changed")
                for actual, stored in ((masks["before"], saved_before), (masks["after"], saved_after)):
                    pairs = np.unique(np.column_stack((actual, stored)), axis=0)
                    require(len(pairs) == len(np.unique(actual)) == len(np.unique(stored)), "Saved partition differs")
                np.testing.assert_allclose(metric["parent_internal_damage_fraction"], float(row["pre_event_parent_internal_broken_fraction"]), rtol=1e-14, atol=0)
                require(metric["cut_bonds"] == int(float(row["cut_bonds"])), "Saved cut count differs")
            for key in record:
                if key in metric: record[key] = metric[key]
            attempt = result["attempt_diagnostics"][primary["crack_index"] - 1]
            points = attempt["tip_points_rc"][:primary["step_number"] + 1]
            record["trigger_prefix_revisits_junction"] = len(set(map(tuple, points))) < len(points)
            record["micro_events_before_primary"] = primary["event_index"]
            record["near_case_resolution"] = all(abs(metric[k] - reference[k]) <= v for k, v in c["resolution_windows"].items())
            if not record["allow_loops"]:
                require(not record["trigger_prefix_revisits_junction"], "No-loop primary revisits junction")
        rows.append(record)
        if len(rows) % 600 == 0: print(f"C3 C2 readback: {len(rows)}/{len(source_rows)}", flush=True)
    write_csv(output / "lmc_realizations.csv", rows)
    metrics = ("alpha", "f2_initial", "f2_conditional", "eta_cut", "pre_broken_fraction", "parent_internal_damage_fraction",
        "mother_child_internal_damage_fraction", "cut_qxx", "mother_future_cut_damage", "mother_child_internal_damage", "pre_existing_fragment_cut")
    groups = []
    for stage, group in sorted({(r["stage"], r["group_id"]) for r in rows}):
        rr = [r for r in rows if r["stage"] == stage and r["group_id"] == group]; seen = [r for r in rr if r["primary_reached"]]
        near = sum(r["near_case_resolution"] for r in rr)
        g = {k: rr[0][k] for k in ("stage", "group_id", "gamma", "ell0", "int_bre", "allow_loops")}
        g.update(total=len(rr), reached=len(seen), near_case_count=near, near_case_fraction=near / len(rr),
            near_case_ci95_wilson=reach_interval(near, len(rr)),
            primary_prefix_revisit_fraction=sum(r["trigger_prefix_revisits_junction"] for r in seen) / len(seen),
            metrics={key: summary([r[key] for r in seen]) for key in metrics})
        g["mean_absolute_deviations"] = {key: abs(g["metrics"][key]["mean"] - reference[key]) for key in c["resolution_windows"]}
        groups.append(g)
    coarse = [g for g in groups if g["stage"] == "coarse"]
    front = pareto(np.array([list(g["mean_absolute_deviations"].values()) for g in coarse]))
    for g, flag in zip(coarse, front): g["shape_mean_pareto"] = bool(flag)
    write_json(output / "lmc_groups.json", groups)
    flat = []
    for g in groups:
        f = {k: g[k] for k in ("stage", "group_id", "gamma", "ell0", "int_bre", "total", "reached", "near_case_count", "near_case_fraction", "primary_prefix_revisit_fraction")}
        f["shape_mean_pareto"] = g["shape_mean_pareto"] if g["stage"] == "coarse" else None
        for k, stat in g["metrics"].items():
            for key in ("mean", "median", "p10", "p90"): f[k + "_" + key] = stat[key]
        flat.append(f)
    write_csv(output / "lmc_groups.csv", flat)
    paired = []
    for g in [g for g in groups if g["stage"] == "no_loop"]:
        false = [r for r in rows if r["group_id"] == g["group_id"] and r["stage"] == "no_loop"]
        true = {r["replicate_index"]: r for r in rows if r["group_id"] == g["group_id"].replace("no_loop", "coarse") and r["stage"] == "coarse"}
        pair = {"gamma": g["gamma"], "ell0": g["ell0"], "n": len(false), "false_minus_true": {}}
        for k in ("f2_initial", "alpha", "eta_cut", "parent_internal_damage_fraction", "mother_child_internal_damage_fraction"):
            pair["false_minus_true"][k] = summary([r[k] - true[r["replicate_index"]][k] for r in false])
        pair["near_true"] = sum(r["near_case_resolution"] for r in true.values())
        pair["near_false"] = sum(r["near_case_resolution"] for r in false)
        paired.append(pair)
    write_json(output / "loop_paired.json", paired)
    return rows, groups


def figures(output, reference, graph, rows, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz, edges, masks = graph
    fig, axes = plt.subplots(1, 3, figsize=(14,4.4))
    mids = (xyz[edges[:,0]] + xyz[edges[:,1]]) / 2
    for ax, (label, mask) in zip(axes[:2], (("Mother pre-damage", masks["mother_damage"]), ("New-child interface", masks["cut"]))):
        cc = mids[mask]
        ax.scatter(cc[:,0], cc[:,1], s=7, c=cc[:,2], cmap="viridis", vmin=-9.5, vmax=9.5, alpha=.35)
        ax.set(xlabel="x / d0", ylabel="y / d0", title=f"{label}: {mask.sum()} edges"); ax.set_aspect("equal")
    for label, mask in (("all mother damage", masks["mother_damage"]), ("future cut damage", masks["future_cut_damage"]), ("child-interior damage", masks["child_internal_damage"])):
        hist, bounds = np.histogram(mids[mask,0], bins=np.arange(-10,10.5,.5))
        axes[2].plot((bounds[1:]+bounds[:-1])/2, hist, label=label)
    axes[2].set(xlabel="x / d0", ylabel="Broken-edge count per slab", title="Damage location before effective split")
    axes[2].legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "dem_damage.png", dpi=180); plt.close(fig)
    coarse = [g for g in groups if g["stage"] == "coarse"]
    gamma = sorted({g["gamma"] for g in coarse}); ell = sorted({g["ell0"] for g in coarse})
    keys = ("f2_initial", "alpha", "eta_cut", "parent_internal_damage_fraction", "mother_child_internal_damage_fraction", "cut_qxx")
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for ax,k in zip(axes.ravel(),keys):
        array=np.array([[next(g for g in coarse if g["gamma"]==ga and g["ell0"]==el)["metrics"][k]["mean"] for ga in gamma] for el in ell])
        im=ax.imshow(array, aspect="auto",origin="lower")
        ax.set(xticks=range(len(gamma)),xticklabels=gamma,yticks=range(len(ell)),yticklabels=ell,xlabel="gamma",ylabel="ell0",title=f"{k}\nDEM reference = {reference[k]:.4f}")
        for i in range(len(ell)):
            for j in range(len(gamma)):ax.text(j,i,f"{array[i,j]:.3f}",ha="center",va="center",fontsize=7,color="white" if array[i,j] < (array.min()+array.max())/2 else "black")
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.tight_layout();fig.savefig(output / "c2_comparison_maps.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key in zip(axes,("parent_internal_damage_fraction","mother_child_internal_damage_fraction")):
        data=[r[key] for r in rows if r["stage"]=="coarse" and r["primary_reached"]]
        ax.hist(data,bins=40,alpha=.65,label="3599 reached coarse runs (parameter mixture)")
        ax.axvline(reference[key],color="red",label=f"DEM = {reference[key]:.4f}")
        ax.set(xlabel=key,ylabel="Count",title="Damage distributions are descriptive, not physical ensembles");ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output / "damage_gap.png",dpi=180);plt.close(fig)


def run(c):
    require(c["primary_threshold"]==.05, "C3 primary threshold must be 5%")
    require(c["resolution_windows"]=={"f2_initial":1/400,"alpha":1/400,"eta_cut":1/20}, "Register a new protocol before changing comparison windows")
    output = Path(c["output"]); output.mkdir(parents=True,exist_ok=False)
    # Freeze the descriptive comparison protocol before reading candidate values.
    write_json(output / "config.json", c)
    write_json(output / "completion.json", {"status":"running", "new_simulations":0})
    snapshots=output / "source_snapshot"; snapshots.mkdir()
    scripts=Path(__file__).resolve().parents[1]
    for relative in ("run_lmc_calibration_c3.py", "lmc_mod/c3_analysis.py", "lmc_mod/first_split.py", "lmc_mod/c2_report.py", "regular_loading/binary_activity.py", "regular_loading/fragment_history.py"):
        target=snapshots/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(scripts/relative,target)
    started=time.perf_counter()
    write_json(output / "smoke.json",self_check())
    reference, graph=dem_analysis(c,output)
    print(f"C3 DEM pre-damage: {reference['parent_internal_broken']}/{reference['parent_initial_internal_edges']}; cut {reference['cut_bonds']}",flush=True)
    rows,groups=lmc_analysis(c,output,reference)
    figures(output,reference,graph,rows,groups)
    write_json(output / "source_hashes.json",{str(p.relative_to(snapshots)):file_hash(p) for p in snapshots.rglob("*.py")})
    write_json(output / "acceptance.json",{"status":"passed", "strict_checks":c["strict_checks"], "dem_binary_activity_and_saved_history_agree":True,
        "lmc_realizations":len(rows),"lmc_reached_states":sum(r["primary_reached"] for r in rows),
        "c2_json_npz_manifest_checked":c["strict_checks"],"damage_partitions_and_volume_conservation":True,
        "independent_connectivity_checked":c["strict_checks"],"new_simulations":0,"physical_validation":"not established by single-case descriptive comparison"})
    write_json(output / "completion.json",{"status":"analysis_completed", "elapsed_s":time.perf_counter()-started,"new_simulations":0})
    return output
