"""Offline audit of the frozen seed-0 pilot; never runs or modifies SLAM."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "docs/artifacts/retention_v1_diagnosis"


def load(path):
    return json.loads(path.read_text())


def aligned_errors(data):
    est = np.asarray(data["trj_est"])[:, :3, 3]
    gt = np.asarray(data["trj_gt"])[:, :3, 3]
    x, y = est - est.mean(0), gt - gt.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    aligned = x @ u @ correction @ vt + gt.mean(0)
    return np.linalg.norm(aligned - gt, axis=1)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = item
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = load(ROOT / "results/retention_pilot_2026-10-01-02-41-28/first_comparison.json")
    report = {"runs": {}, "input_sha256": {}}
    errors, configs, cells = {}, {}, {}
    ground_truth = None
    all_events = []
    for summary in summaries:
        policy, directory = summary["policy"], ROOT / summary["run_dir"]
        manifest = load(directory / "run_manifest.json")
        config = yaml.safe_load((directory / "config.yml").read_text())
        configs[policy] = flatten(config)
        trajectory = load(directory / "plot/trj_fixed_final.json")
        assert trajectory["trj_id"] == list(range(100))
        if ground_truth is None:
            ground_truth = trajectory["trj_gt"]
        assert trajectory["trj_gt"] == ground_truth
        errors[policy] = aligned_errors(trajectory)
        rmse = float(np.sqrt(np.mean(errors[policy] ** 2)))
        assert abs(rmse - summary["fixed_ate_rmse"]) < 1e-10
        render_ids = load(directory / "psnr/before_opt/final_result.json")["view_ids"]
        assert render_ids == summaries[0]["rendering_ids"]
        files = [directory / "run_manifest.json", directory / "config.yml",
                 directory / "plot/trj_fixed_final.json", directory / "plot/trj_final.json",
                 directory / "point_cloud/final/point_cloud.ply"] + list(directory.glob("telemetry_*.csv"))
        for path in files:
            report["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = list(csv.DictReader(next(directory.glob("telemetry_backend*.csv")).open()))
        raw_errors = np.linalg.norm(np.asarray(trajectory["trj_est"])[:, :3, 3] - np.asarray(trajectory["trj_gt"])[:, :3, 3], axis=1)
        plans = [r for r in rows if r["event"].startswith("retention_")]
        events = []
        for r in plans:
            n, kept, p = (int(r[k]) for k in ("count_before", "count_after", "protected_rows"))
            event = {"policy": policy, "frame": int(r["frame_idx"]),
                     "iteration": int(r["mapping_iteration"]), "event": r["event"],
                     "considered": n, "protected": p, "unprotected": n-p,
                     "rank_selected_unprotected": kept-p, "removed": n-kept,
                     "growth": int(r["attempted_growth"]), "rejected": int(r["rejected_growth"]),
                     "wall_ms": float(r["policy_wall_ms"])}
            events.append(event)
            all_events.append(event)
        pressure = [e for e in events if e["removed"] > 0]
        timings = {}
        for name in ("tracking_feedback", "retention_insert_plan", "retention_densify_plan"):
            vals = [float(r["policy_wall_ms"]) for r in rows if r["event"] == name]
            timings[name] = {"count": len(vals), "total_s": sum(vals)/1000,
                             "median_ms": float(np.median(vals)) if vals else 0}
        net_maintenance = {}
        for name in ("upstream_init_densify_prune", "upstream_map_densify_prune"):
            subset = [r for r in rows if r["event"] == name]
            net_maintenance[name] = sum(int(r["count_after"])-int(r["count_before"]) for r in subset)
        ply = PlyData.read(directory / "point_cloud/final/point_cloud.ply")["vertex"].data
        xyz = np.column_stack([ply[k] for k in ("x", "y", "z")]).astype(float)
        opacity = 1 / (1 + np.exp(-ply["opacity"].astype(float)))
        spatial = {}
        for width in (.25, .5, 1.):
            grid = np.floor(xyz / width).astype(np.int64)
            unique, counts = np.unique(grid, axis=0, return_counts=True)
            distribution = counts / counts.sum()
            spatial[str(width)] = {"occupied": len(counts), "effective_cells": float(np.exp(-np.sum(distribution*np.log(distribution)))),
                                   "top10_mass": float(np.sort(counts)[-10:].sum()/len(xyz)),
                                   "singleton_cells": int((counts == 1).sum())}
            cells[policy, width] = set(map(tuple, unique.tolist()))
        info = {"run_dir": summary["run_dir"], "revision": manifest["git_revision"],
                "clean_diff": manifest["tracked_diff_sha256"] == hashlib.sha256(b"").hexdigest(),
                "seed": manifest["seed"], "complete": load(directory/"completion.json"),
                "keyframes": load(directory/"plot/trj_final.json")["trj_id"],
                "first_sync_rows": int(next(r["gaussian_count"] for r in rows if r["event"] == "sync_begin")),
                "raw_first8_errors_mm": (raw_errors[:8]*1000).tolist(),
                "mapping_iterations": sum(r["event"] == "map_iteration" for r in rows),
                "rmse_m": rmse, "timings": timings, "maintenance_net": net_maintenance,
                "plan_events": len(events), "pressure_events": len(pressure),
                "considered_row_events": sum(e["considered"] for e in events),
                "protected_row_events": sum(e["protected"] for e in events),
                "removed_rows": sum(e["removed"] for e in events),
                "pressure_protected_fraction_range": [min(e["protected"]/e["considered"] for e in pressure), max(e["protected"]/e["considered"] for e in pressure)] if pressure else [],
                "pressure_rank_selected_fraction": sum(e["rank_selected_unprotected"] for e in pressure)/sum(e["considered"]-e["removed"] for e in pressure) if pressure else None,
                "spatial": spatial, "opacity_quantiles": np.quantile(opacity,[0,.1,.25,.5,.75,.9,1]).tolist(),
                "segment_rmse_mm": [float(np.sqrt(np.mean(errors[policy][a:b]**2))*1000) for a,b in [(0,20),(20,40),(40,60),(60,80),(80,100)]]}
        report["runs"][policy] = info
    keys = set().union(*(set(c) for c in configs.values()))
    report["config_differences"] = {k:{p:c.get(k) for p,c in configs.items()} for k in sorted(keys) if len({json.dumps(c.get(k),sort_keys=True) for c in configs.values()}) > 1}
    random, v1 = errors["random"], errors["tracking_support"]
    delta = v1**2 - random**2
    report["frame_comparison"] = {"v1_larger_count": int((v1>random).sum()),
        "largest_v1_frames": np.argsort(v1)[-10:][::-1].tolist(),
        "largest_excess_squared_frames": np.argsort(delta)[-10:][::-1].tolist(),
        "top10_positive_share_net_gap": float(np.sort(delta)[-10:].sum()/delta.sum()),
        "cumulative_first_v1_worse": int(np.flatnonzero(np.cumsum(delta)>0)[0]),
        "cumulative_last_v1_better": int(np.flatnonzero(np.cumsum(delta)<0)[-1]) if np.any(np.cumsum(delta)<0) else None,
        "paired_error_difference_median_mm": float(np.median(v1-random)*1000)}
    report["frame_comparison"]["segment_shares_net_squared_gap"] = [float(delta[a:b].sum()/delta.sum()) for a,b in [(0,20),(20,40),(40,60),(60,80),(80,100)]]
    gt = np.asarray(trajectory["trj_gt"])
    positions, rotations = gt[:,:3,3], gt[:,:3,:3]
    pairs = []
    for i in range(100):
        for j in range(i+20,100):
            angle = float(np.degrees(np.arccos(np.clip((np.trace(rotations[i].T@rotations[j])-1)/2,-1,1))))
            pairs.append((i,j,float(np.linalg.norm(positions[i]-positions[j])),angle))
    report["sequence"] = {"path_length_m": float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
        "endpoint_distance_m": float(np.linalg.norm(positions[0]-positions[-1])),
        "endpoint_rotation_deg": next(p[3] for p in pairs if p[:2]==(0,99)),
        "closest_separated_pairs": sorted(pairs,key=lambda p:p[2])[:10],
        "revisit_pair_counts": {f"gap{gap}_distance{distance}_angle30": sum(j-i>=gap and d<distance and a<30 for i,j,d,a in pairs) for gap in (20,40,60) for distance in (.1,.25,.5)},
        "largest_adjacent_turns": sorted([(i,float(np.degrees(np.arccos(np.clip((np.trace(rotations[i-1].T@rotations[i])-1)/2,-1,1))))) for i in range(1,100)],key=lambda p:p[1],reverse=True)[:10]}
    report["spatial_overlap_random_v1"] = {str(w):{"jaccard": len(cells["random",w]&cells["tracking_support",w])/len(cells["random",w]|cells["tracking_support",w]),
        "random_only":len(cells["random",w]-cells["tracking_support",w]), "v1_only":len(cells["tracking_support",w]-cells["random",w])} for w in (.25,.5,1.)}
    with (OUT/"frame_errors.csv").open("w",newline="") as f:
        writer=csv.writer(f); writer.writerow(["frame"]+list(errors)+["v1_minus_random_squared_m2"])
        writer.writerows([[i]+[float(e[i]) for e in errors.values()]+[float(delta[i])] for i in range(100)])
    with (OUT/"policy_events.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_events[0])); writer.writeheader(); writer.writerows(all_events)
    (OUT/"diagnosis.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for policy in ("reference","random","tracking_support","tracking_support_no_T"):
        axes[0].plot(errors[policy]*1000,label=policy,linewidth=1)
    axes[0].set_ylabel("Aligned translation error (mm)"); axes[0].legend(fontsize=8)
    axes[1].plot(np.cumsum(delta)*1e6,color="black")
    axes[1].axhline(0,color="gray",linewidth=.7)
    axes[1].set_ylabel("Cumulative V1 − random SSE (mm²)"); axes[1].set_xlabel("Frame")
    fig.tight_layout(); fig.savefig(OUT/"frame_errors.png",dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!="input_sha256"},indent=2))


if __name__ == "__main__":
    main()
