#!/usr/bin/env python3
"""
Two-step optimisation with COPT.

Step 1: max quality_norm - photo_norm
Step 2: min energy ATSP on selected viewpoints

Joint-compatible fields are filled when energy_norm_denom is provided
so batch comparison uses the same scalarised obj.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import coptpy as cp
from coptpy import COPT

from common import (
    MIN_COVER_PER_SAMPLE,
    MIP_GAP,
    TIME_LIMIT_STEP1,
    TIME_LIMIT_STEP2,
    COPT_THREADS,
    COPT_MEMLIMIT_MB,
    COPT_LOGLEVEL,
    make_json_serializable,
)


def _configure_copt(model: cp.Model, time_limit: float) -> None:
    model.setParam(COPT.Param.TimeLimit, float(time_limit))
    model.setParam(COPT.Param.RelGap, float(MIP_GAP))
    model.setParam(COPT.Param.Threads, int(COPT_THREADS))
    model.setParam(COPT.Param.MemLimit, float(COPT_MEMLIMIT_MB))
    model.setParam(COPT.Param.LogLevel, int(COPT_LOGLEVEL))


def _mip_gap_and_bound(model: cp.Model) -> tuple[Optional[float], Optional[float]]:
    bound = None
    gap = None
    try:
        bound = float(model.getAttr(COPT.Attr.BestBnd))
    except Exception:
        bound = None
    try:
        obj = float(model.objval) if model.hasmipsol else None
    except Exception:
        obj = None
    if obj is not None and bound is not None:
        gap = abs(obj - bound) / max(abs(obj), 1e-10)
    return gap, bound


def _has_solution(model: cp.Model) -> bool:
    try:
        return bool(model.hasmipsol)
    except Exception:
        return bool(model.getAttr(COPT.Attr.HasMipSol))


def solve_step1_coverage(pre: Dict[str, Any], time_limit: int = TIME_LIMIT_STEP1,
                         *, log_output: bool = True) -> Dict[str, Any]:
    print("\n========== Step 1: Coverage Selection (COPT) ==========")
    n_vp = pre["n_vp"]
    n_sp = pre["n_sp"]
    n_hp = pre["n_hp"]
    n_pairs = pre["n_pairs"]
    depot = pre["depot"]
    valid_pairs = pre["valid_pairs"]
    hp_cover = pre["hp_cover"]
    sp_cover = pre["sp_cover"]
    gamma = pre["gamma"]
    total_omega = pre["total_omega"]
    N_h = pre["N_h"]
    vp_to_dirs = pre["vp_to_dirs"]

    env = cp.Envr()
    m = None
    try:
        m = env.createModel("step1_coverage")
        _configure_copt(m, time_limit)
        if not log_output:
            m.setParam(COPT.Param.Logging, 0)

        y = {k: m.addVar(vtype=COPT.BINARY, name=f"y_{k}") for k in range(n_vp)}
        z = {(k, a): m.addVar(vtype=COPT.BINARY, name=f"z_{k}_{a}") for (k, a) in valid_pairs}
        s = {h: m.addVar(vtype=COPT.BINARY, name=f"s_{h}") for h in range(n_hp)}

        m.addConstr(y[depot] == 1, name="depot")

        for k in range(n_vp):
            dirs = vp_to_dirs.get(k, [])
            if not dirs:
                m.addConstr(y[k] == 0, name=f"no_dir_{k}")
                continue
            m.addConstr(cp.quicksum(z[k, a] for a in dirs) <= len(dirs) * y[k])
            m.addConstr(y[k] <= cp.quicksum(z[k, a] for a in dirs))

        # Distinct-viewpoint coverage: each sample needs >= 3 different viewpoints
        from collections import defaultdict
        vp_cover_dirs = [defaultdict(list) for _ in range(n_sp)]
        for ell in range(n_sp):
            for (k, a) in sp_cover[ell]:
                vp_cover_dirs[ell][k].append(a)

        u = {}
        for ell in range(n_sp):
            covering_vps = list(vp_cover_dirs[ell].keys())
            if not covering_vps:
                print(f"WARNING: sample {ell} has no covering viewpoints")
                continue
            if len(covering_vps) < MIN_COVER_PER_SAMPLE:
                print(
                    f"WARNING: sample {ell} only has {len(covering_vps)} covering "
                    f"viewpoints (< {MIN_COVER_PER_SAMPLE}); model may be infeasible"
                )
            for k in covering_vps:
                u[k, ell] = m.addVar(vtype=COPT.BINARY, name=f"u_{k}_{ell}")
                dirs_k = vp_cover_dirs[ell][k]
                m.addConstr(u[k, ell] <= cp.quicksum(z[k, a] for a in dirs_k))
            m.addConstr(
                cp.quicksum(u[k, ell] for k in covering_vps) >= MIN_COVER_PER_SAMPLE,
                name=f"cover_3vp_{ell}",
            )

        for h in range(n_hp):
            if N_h[h] == 0:
                m.addConstr(s[h] == 0)
                continue
            m.addConstr(s[h] <= cp.quicksum(z[k, a] for (k, a) in hp_cover[h]))
            m.addConstr(cp.quicksum(z[k, a] for (k, a) in hp_cover[h]) <= N_h[h] * s[h])

        quality = cp.quicksum(float(gamma[h]) * s[h] for h in range(n_hp)) / total_omega
        photos = cp.quicksum(z[k, a] for (k, a) in valid_pairs) / n_pairs
        m.setObjective(quality - photos, sense=COPT.MAXIMIZE)
        m.solve()

        if not _has_solution(m):
            status = int(m.status)
            raise RuntimeError(f"Step 1 no solution (status={status})")

        selected_vps = [k for k in range(n_vp) if y[k].x > 0.5]
        selected_pairs = [[k, a] for (k, a) in valid_pairs if z[k, a].x > 0.5]
        photos_val = len(selected_pairs)
        weighted_quality = sum(float(gamma[h]) * s[h].x for h in range(n_hp))
        quality_val = weighted_quality / total_omega
        mip_gap, obj_bound = _mip_gap_and_bound(m)
        try:
            runtime = float(m.getAttr(COPT.Attr.SolvingTime))
        except Exception:
            runtime = 0.0
        status = int(m.status)
        obj_val = float(m.objval)

        selected_photos_readable = [[pre["vp_ids"][k], a] for k, a in selected_pairs]
        photos_by_vp: dict[str, list[str]] = {}
        for vp_id, a in selected_photos_readable:
            photos_by_vp.setdefault(vp_id, []).append(a)

        print(
            f"Step 1 done. VPs={len(selected_vps)}, Photos={photos_val}, "
            f"quality_norm={quality_val:.4f}, gap={mip_gap}, status={status}"
        )


        return {
            "selected_vps": selected_vps,
            "selected_vp_ids": [pre["vp_ids"][k] for k in selected_vps],
            "selected_pairs": selected_pairs,
            "selected_photos_readable": selected_photos_readable,
            "photos_by_viewpoint": photos_by_vp,
            "quality_norm": quality_val,
            "photos": photos_val,
            "weighted_quality": weighted_quality,
            "obj": obj_val,
            "runtime": runtime,
            "mip_gap": mip_gap,
            "status": status,
            "obj_bound": obj_bound,
        }

    finally:
        try:
            if m is not None:
                m.dispose()
        finally:
            env.dispose()


def solve_step2_tsp(
    pre: Dict[str, Any],
    selected_vps: List[int],
    time_limit: int = TIME_LIMIT_STEP2,
    *, log_output: bool = True,
) -> Dict[str, Any]:
    print("\n========== Step 2: ATSP (COPT) ==========")
    if pre["depot"] not in selected_vps:
        selected_vps = [pre["depot"]] + list(selected_vps)
    selected_vps = sorted(set(selected_vps))
    n_sel = len(selected_vps)
    print(f"Solving ATSP on {n_sel} viewpoints...")

    edges_sel = []
    cost = {}
    for i in selected_vps:
        for j in selected_vps:
            if i == j:
                continue
            if (i, j) in pre["energy"]:
                edges_sel.append((i, j))
                cost[(i, j)] = pre["energy"][(i, j)]

    env = cp.Envr()
    m = None
    try:
        m = env.createModel("step2_atsp")
        _configure_copt(m, time_limit)
        if not log_output:
            m.setParam(COPT.Param.Logging, 0)

        x = {(i, j): m.addVar(vtype=COPT.BINARY, name=f"x_{i}_{j}") for (i, j) in edges_sel}
        u = {k: m.addVar(lb=0.0, ub=float(n_sel - 1), name=f"u_{k}") for k in selected_vps}
        m.addConstr(u[pre["depot"]] == 0)

        for k in selected_vps:
            out_e = [j for j in selected_vps if (k, j) in cost]
            in_e = [i for i in selected_vps if (i, k) in cost]
            m.addConstr(cp.quicksum(x[k, j] for j in out_e) == 1)
            m.addConstr(cp.quicksum(x[i, k] for i in in_e) == 1)

        for i in selected_vps:
            if i == pre["depot"]:
                continue
            for j in selected_vps:
                if i == j or j == pre["depot"]:
                    continue
                if (i, j) in cost:
                    m.addConstr(u[i] - u[j] + (n_sel - 1) * x[i, j] <= n_sel - 2)

        m.setObjective(cp.quicksum(cost[i, j] * x[i, j] for i, j in edges_sel), sense=COPT.MINIMIZE)
        m.solve()

        if not _has_solution(m):
            status = int(m.status)
            raise RuntimeError(f"Step 2 no solution (status={status})")

        succ = {i: j for (i, j) in edges_sel if x[i, j].x > 0.5}
        tour = [pre["depot"]]
        cur = pre["depot"]
        visited = {cur}
        while len(tour) < n_sel:
            if cur not in succ:
                break
            cur = succ[cur]
            tour.append(cur)
            if cur in visited:
                break
            visited.add(cur)
        tour.append(pre["depot"])

        total_energy = float(m.objval)
        mip_gap, obj_bound = _mip_gap_and_bound(m)
        try:
            runtime = float(m.getAttr(COPT.Attr.SolvingTime))
        except Exception:
            runtime = 0.0
        status = int(m.status)

        print(f"Step 2 done. energy={total_energy:.4f}, gap={mip_gap}, status={status}")


        return {
            "tour": tour,
            "tour_ids": [pre["vp_ids"][i] for i in tour],
            "energy": total_energy,
            "n_selected": n_sel,
            "runtime": runtime,
            "mip_gap": mip_gap,
            "status": status,
            "obj_bound": obj_bound,
        }

    finally:
        try:
            if m is not None:
                m.dispose()
        finally:
            env.dispose()


def run_two_step(
    pre: Dict[str, Any],
    time_limit_step1: int = TIME_LIMIT_STEP1,
    time_limit_step2: int = TIME_LIMIT_STEP2,
    energy_norm_denom: Optional[float] = None,
    *, log_output: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    step1 = solve_step1_coverage(pre, time_limit=time_limit_step1, log_output=log_output)
    step2 = solve_step2_tsp(pre, step1["selected_vps"], time_limit=time_limit_step2, log_output=log_output)
    total_time = time.perf_counter() - t0

    energy_val = step2["energy"]
    quality_norm = step1["quality_norm"]
    photos_val = step1["photos"]
    photo_norm = photos_val / pre["n_pairs"]

    result = {
        "method": "two_step",
        "selected_vps": step1["selected_vps"],
        "selected_vp_ids": step1["selected_vp_ids"],
        "selected_pairs": step1["selected_pairs"],
        "selected_photos_readable": step1["selected_photos_readable"],
        "photos_by_viewpoint": step1["photos_by_viewpoint"],
        "tour": step2["tour"],
        "tour_ids": step2["tour_ids"],
        "energy": float(energy_val),
        "weighted_quality": float(step1["weighted_quality"]),
        "photos": int(photos_val),
        "n_viewpoints": len(step1["selected_vps"]),
        "quality_norm": float(quality_norm),
        "photo_norm": float(photo_norm),
        "runtime_step1": step1["runtime"],
        "runtime_step2": step2["runtime"],
        "runtime": float(total_time),
        "mip_gap_step1": step1.get("mip_gap"),
        "mip_gap_step2": step2.get("mip_gap"),
        "status_step1": step1.get("status"),
        "status_step2": step2.get("status"),
        "obj_bound_step1": step1.get("obj_bound"),
        "obj_bound_step2": step2.get("obj_bound"),
    }

    if energy_norm_denom is not None and energy_norm_denom > 0:
        result["energy_norm_denom"] = float(energy_norm_denom)
        result["energy_norm"] = float(energy_val / energy_norm_denom)
        result["obj"] = float(
            result["energy_norm"] - quality_norm + photo_norm
        )
        result["atsp_runtime"] = 0.0
        result["mip_gap"] = step2.get("mip_gap")
        result["status"] = step2.get("status")
        result["obj_bound"] = None
    else:
        result["energy_norm"] = None
        result["obj"] = None

    return result


def save_two_step_result(result: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(result), f, indent=2, ensure_ascii=False)
    print(f"Two-step result saved: {out_path}")


if __name__ == "__main__":
    import sys
    from common import load_data, preprocess, nn_tour_energy

    json_path = sys.argv[1] if len(sys.argv) > 1 else "data/building1.json"
    print(f"Loading {json_path} ...")
    data = load_data(json_path)
    pre = preprocess(data)
    denom = nn_tour_energy(pre)
    result = run_two_step(pre, energy_norm_denom=denom)
    stem = Path(json_path).stem
    save_two_step_result(result, Path("results") / f"two_step_{stem}.json")
