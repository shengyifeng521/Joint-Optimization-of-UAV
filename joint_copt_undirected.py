#!/usr/bin/env python3
"""
Independent undirected Branch-and-Cut experiment with COPT.

Objective: min energy_norm - quality_norm + photo_norm

Result dict retains every RESULT_FORMAT.md field (method="joint").
As requested, runtime counts the two-step-plus-joint pipeline; the original
joint-solver-only timing is available as joint_solver_runtime.

Routing uses one binary per undirected edge, degree=2*y, integer lazy SEC,
and throttled fractional min-cut user cuts. Symmetry and the closed-route
energy potential identity are audited; original normalization is preserved.
The directed baseline is shipped unchanged as joint_copt.py.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import coptpy as cp
from coptpy import COPT

from common import (
    MIN_COVER_PER_SAMPLE,
    MIP_GAP,
    TIME_LIMIT_JOINT,
    TIME_LIMIT_STEP1,
    TIME_LIMIT_STEP2,
    COPT_THREADS,
    COPT_MEMLIMIT_MB,
    COPT_LOGLEVEL,
    make_json_serializable,
    nn_tour_energy,
)
from joint_mip_start import ValidatedStart, validate_initial_solution
from undirected_routing import (
    FractionalConfig, UndirectedTourCallback, audit_raw_data, symmetric_costs,
    connected_tour, submit_undirected_start,
)
from two_step_copt import run_two_step


REQUIRED_RESULT_FIELDS = (
    "method", "selected_vps", "selected_vp_ids", "selected_pairs",
    "selected_photos_readable", "photos_by_viewpoint", "tour", "tour_ids",
    "energy", "weighted_quality", "photos", "n_viewpoints", "energy_norm",
    "quality_norm", "photo_norm", "energy_norm_denom", "runtime", "atsp_runtime",
    "obj", "mip_gap", "status", "obj_bound",
)


def _configure_copt(model: cp.Model, time_limit: float) -> None:
    model.setParam(COPT.Param.TimeLimit, float(time_limit))
    model.setParam(COPT.Param.RelGap, float(MIP_GAP))
    model.setParam(COPT.Param.Threads, int(COPT_THREADS))
    # Per-process solver budget; Python data/model objects need extra RAM.
    model.setParam(COPT.Param.MemLimit, float(COPT_MEMLIMIT_MB))
    model.setParam(COPT.Param.LogLevel, int(COPT_LOGLEVEL))


def _mip_gap_and_bound(model: cp.Model) -> tuple[Optional[float], Optional[float]]:
    """Return (rel_gap, best_bound)."""
    bound = None
    gap = None
    try:
        bound = float(model.getAttr(COPT.Attr.BestBnd))
    except Exception:
        try:
            bound = float(getattr(model, "BestBnd", None) or model.getInfo(COPT.Info.BestBnd))
        except Exception:
            bound = None
    try:
        obj = float(model.objval) if model.hasmipsol else None
    except Exception:
        obj = None
    if obj is not None and bound is not None:
        denom = max(abs(obj), abs(bound), 1e-10)
        gap = abs(obj - bound) / denom
    return gap, bound


def _has_solution(model: cp.Model) -> bool:
    # A memory/time limit alone does not imply an incumbent exists. In
    # particular, do not read x values when the budget was reached without one.
    try:
        return bool(model.hasmipsol)
    except Exception:
        return bool(model.getAttr(COPT.Attr.HasMipSol))


def solve_joint(
    pre: Dict[str, Any],
    energy_norm_denom: Optional[float] = None,
    time_limit: int = TIME_LIMIT_JOINT,
    log_output: bool = True,
    autosave_path: Optional[Path | str] = None,
    *,
    accelerate: bool = True,
    use_two_step_start: bool = True,
    time_limit_step1: int = TIME_LIMIT_STEP1,
    time_limit_step2: int = TIME_LIMIT_STEP2,
    initial_solution: Optional[Dict[str, Any]] = None,
    relative_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    fractional_config: Optional[FractionalConfig] = None,
) -> Dict[str, Any]:
    """Generate a two-step start, then optimize the unchanged joint problem.

    runtime = runtime_two_step + runtime_joint (wall-clock seconds through
    result construction). joint_solver_runtime is COPT's joint solve time only.
    Failed/infeasible two-step starts are reported and the joint solve continues
    without a start. No variables are fixed to the two-step decisions.
    """
    started_at = time.perf_counter()
    # Fail before spending time constructing a two-step seed on unsupported data.
    symmetric_costs(pre)
    if energy_norm_denom is None:
        energy_norm_denom = nn_tour_energy(pre)
    if not math.isfinite(energy_norm_denom) or energy_norm_denom <= 0:
        raise ValueError("Energy normalization denominator must be finite and positive")
    two_step_runtime = 0.0
    step1_runtime = step2_runtime = 0.0
    start = None
    status = "disabled"
    error = None
    source = "provided" if initial_solution is not None else (
        "two_step" if use_two_step_start else None)
    candidate = initial_solution
    if candidate is None and use_two_step_start:
        seed_started_at = time.perf_counter()
        try:
            candidate = run_two_step(
                pre, time_limit_step1=time_limit_step1,
                time_limit_step2=time_limit_step2,
                energy_norm_denom=energy_norm_denom, log_output=log_output,
            )
            step1_runtime = float(candidate.get("runtime_step1", 0.0))
            step2_runtime = float(candidate.get("runtime_step2", 0.0))
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            print(f"[WARNING] Two-step start failed; continuing without a start: {error}")
        finally:
            two_step_runtime = time.perf_counter() - seed_started_at
    if candidate is not None:
        try:
            start = validate_initial_solution(
                pre, candidate, float(energy_norm_denom), MIN_COVER_PER_SAMPLE,
            )
            status = "submitted"
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            print(f"[WARNING] Initial solution rejected; continuing without a start: {error}")
    timing = {
        "started_at": started_at, "runtime_two_step": two_step_runtime,
        "runtime_step1": step1_runtime, "runtime_step2": step2_runtime,
        "warm_start_source": source, "warm_start_status": status,
        "warm_start_error": error,
        "warm_start_obj": start.objective if start is not None else None,
    }
    # Do not retain the original two-step result while building the joint model.
    candidate = None
    result = _solve_joint_model(
        pre, energy_norm_denom, time_limit, log_output, autosave_path,
        accelerate=accelerate, initial_solution=start, runtime_context=timing,
        relative_gap=relative_gap, absolute_gap=absolute_gap,
        fractional_config=fractional_config,
    )
    # The core checkpoints before native cleanup. Finalize again AFTER it
    # returns so the reported pipeline runtime also includes that cleanup.
    total_runtime = time.perf_counter() - started_at
    result["runtime"] = float(total_runtime)
    result["runtime_joint"] = float(total_runtime - two_step_runtime)
    print(f"Total runtime: {total_runtime:.3f}s = two-step {two_step_runtime:.3f}s "
          f"+ joint {result['runtime_joint']:.3f}s "
          f"(joint COPT solve {result['joint_solver_runtime']:.3f}s); "
          f"warm start: {result['warm_start_status']}")
    if autosave_path is not None:
        try:
            save_joint_result(result, Path(autosave_path))
        except Exception as exc:
            print(f"[WARNING] final autosave failed: {exc}")
    return result


def _solve_joint_model(
    pre: Dict[str, Any],
    energy_norm_denom: Optional[float] = None,
    time_limit: int = TIME_LIMIT_JOINT,
    log_output: bool = True,
    autosave_path: Optional[Path | str] = None,
    *,
    accelerate: bool = True,
    initial_solution: Optional[ValidatedStart] = None,
    runtime_context: Optional[Dict[str, Any]] = None,
    relative_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    fractional_config: Optional[FractionalConfig] = None,
) -> Dict[str, Any]:
    """Solve the equivalent undirected model; keep original coverage and scalars."""
    atsp_runtime = 0.0
    if energy_norm_denom is None:
        energy_norm_denom = nn_tour_energy(pre)
        print(f"\nNN tour energy (denom) = {energy_norm_denom:.4f}")

    print("\n========== Joint Optimisation (COPT) ==========")
    print(
        f"n_vp={pre['n_vp']}, n_sp={pre['n_sp']}, n_hp={pre['n_hp']}, "
        f"n_pairs={pre['n_pairs']}, n_edges={len(pre['edges'])}"
    )
    print(
        f"energy_norm_denom={energy_norm_denom:.4f}, "
        f"Threads={COPT_THREADS}, MemLimit={COPT_MEMLIMIT_MB} MB (~{COPT_MEMLIMIT_MB/1024:.0f} GB)"
    )

    n_vp = pre["n_vp"]
    n_sp = pre["n_sp"]
    n_hp = pre["n_hp"]
    n_pairs = pre["n_pairs"]
    depot = pre["depot"]
    undirected_cost, routing_audit = symmetric_costs(pre)
    edges = list(undirected_cost)
    energy = pre["energy"]
    fractional_config = fractional_config or FractionalConfig()
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
        m = env.createModel("joint")
        if not log_output:
            m.setParam(COPT.Param.Logging, 0)
        _configure_copt(m, time_limit)
        if relative_gap is not None:
            m.setParam(COPT.Param.RelGap, float(relative_gap))
        if absolute_gap is not None:
            m.setParam(COPT.Param.AbsGap, float(absolute_gap))
        print(f"Joint stopping settings: TimeLimit={m.getParam(COPT.Param.TimeLimit):g}s, "
              f"RelGap={m.getParam(COPT.Param.RelGap):g}, "
              f"AbsGap={m.getParam(COPT.Param.AbsGap):g}")

        y = {k: m.addVar(vtype=COPT.BINARY, name=f"y_{k}") for k in range(n_vp)}
        z = {pair: m.addVar(vtype=COPT.BINARY, name=f"z_{pair[0]}_{pair[1]}")
             for pair in valid_pairs}
        # Merge only identical Boolean coverage functions. Keep binary s and BOTH
        # OR inequalities: this preserves quality for every integer photo decision,
        # even with negative weights and even before optimality is proved.
        hp_representative = list(range(n_hp))
        if accelerate:
            representatives = {}
            for h in range(n_hp):
                if N_h[h] != len(hp_cover[h]):
                    raise ValueError("Acceleration requires N_h[h] == len(hp_cover[h])")
                key = frozenset(hp_cover[h])
                hp_representative[h] = representatives.setdefault(key, h)
            # This lookup can be large; it is only needed while assigning groups.
            representatives.clear()
            del representatives
        # Dense integer indices need only a list; retain original per-h indexing.
        s = [None] * n_hp
        for h, representative in enumerate(hp_representative):
            if representative == h:
                s[h] = m.addVar(vtype=COPT.BINARY, name=f"s_{h}")
            else:
                s[h] = s[representative]
        x = {edge: m.addVar(vtype=COPT.BINARY, name=f"x_{edge[0]}_{edge[1]}")
             for edge in edges}

        m.addConstr(y[depot] == 1, name="depot")

        for k in range(n_vp):
            dirs = vp_to_dirs.get(k, [])
            if not dirs:
                m.addConstr(y[k] == 0, name=f"no_dir_{k}")
                continue
            if accelerate:
                # Same integer decisions, stronger LP relaxation than sum(z)<=M*y.
                for a in dirs:
                    m.addConstr(z[k, a] <= y[k], name=f"z_le_y_{k}_{a}")
            else:
                m.addConstr(cp.quicksum(z[k, a] for a in dirs) <= len(dirs) * y[k], name=f"z_le_y_{k}")
            m.addConstr(y[k] <= cp.quicksum(z[k, a] for a in dirs), name=f"y_le_z_{k}")

        # Distinct-viewpoint coverage: each sample needs >= 3 different viewpoints
        # u is a bounded coverage witness, not an iff indicator. With binary z,
        # a continuous u in [0,1] can contribute only from an actual covering VP.
        # Thus sum(u)>=3 is feasible exactly when >=3 distinct VPs cover the sample.
        from collections import defaultdict
        u = {}
        for ell in range(n_sp):
            # Build one sample's temporary grouping at a time, not all n_sp groups.
            vp_cover_dirs = defaultdict(list)
            for (k, a) in sp_cover[ell]:
                vp_cover_dirs[k].append(a)
            covering_vps = list(vp_cover_dirs.keys())
            if not covering_vps:
                print(f"WARNING: sample {ell} has no covering viewpoints")
                continue
            if len(covering_vps) < MIN_COVER_PER_SAMPLE:
                print(
                    f"WARNING: sample {ell} only has {len(covering_vps)} covering "
                    f"viewpoints (< {MIN_COVER_PER_SAMPLE}); model may be infeasible"
                )
            coverage_terms = []
            for k in covering_vps:
                dirs_k = vp_cover_dirs[k]
                if accelerate:
                    dirs_k = list(dict.fromkeys(dirs_k))
                    if len(dirs_k) == 1:
                        # Eliminating a witness bounded by this single binary z is
                        # exact for an existence constraint sum(witnesses)>=3.
                        coverage_terms.append(z[k, dirs_k[0]])
                        continue
                    u[k, ell] = m.addVar(
                        lb=0.0, ub=1.0, vtype=COPT.CONTINUOUS,
                        name=f"u_{k}_{ell}",
                    )
                    m.addConstr(u[k, ell] <= y[k], name=f"u_le_y_{k}_{ell}")
                else:
                    u[k, ell] = m.addVar(vtype=COPT.BINARY, name=f"u_{k}_{ell}")
                m.addConstr(
                    u[k, ell] <= cp.quicksum(z[k, a] for a in dirs_k),
                    name=f"u_le_z_{k}_{ell}",
                )
                coverage_terms.append(u[k, ell])
            m.addConstr(
                cp.quicksum(coverage_terms) >= MIN_COVER_PER_SAMPLE,
                name=f"cover_3vp_{ell}",
            )

        for h in range(n_hp):
            if hp_representative[h] != h:
                continue
            if N_h[h] == 0:
                m.addConstr(s[h] == 0, name=f"s0_{h}")
                continue
            cov = hp_cover[h]
            m.addConstr(s[h] <= cp.quicksum(z[k, a] for (k, a) in cov), name=f"s_ub_{h}")
            m.addConstr(
                cp.quicksum(z[k, a] for (k, a) in cov) <= N_h[h] * s[h],
                name=f"s_lb_{h}",
            )

        incident = {k: [] for k in range(n_vp)}
        for i, j in edges:
            incident[i].append((i, j))
            incident[j].append((i, j))
        for k in range(n_vp):
            m.addConstr(cp.quicksum(x[e] for e in incident[k]) == 2 * y[k],
                        name=f"degree_{k}")
        del incident

        t = {}
        m.setParam(COPT.Param.LazyConstraints, 1)
        subtour_callback = UndirectedTourCallback(y, x, depot, fractional_config)
        contexts = COPT.CBCONTEXT_MIPSOL
        if fractional_config.enabled:
            contexts |= COPT.CBCONTEXT_MIPRELAX
        m.setCallback(subtour_callback, contexts)
        print("Tour: undirected integer lazy SEC; fractional min-cut "
              + ("enabled" if fractional_config.enabled else "disabled"))
        print(f"Routing variables: {routing_audit['directed_edge_count']} directed -> "
              f"{len(edges)} undirected; original normalization retained")

        energy_expr = cp.quicksum(undirected_cost[e] * x[e] for e in edges)
        quality_expr = cp.quicksum(float(gamma[h]) * s[h] for h in range(n_hp))
        photos_expr = cp.quicksum(z[k, a] for (k, a) in valid_pairs)

        obj = (
            energy_expr / energy_norm_denom
            - quality_expr / total_omega
            + photos_expr / n_pairs
        )
        m.setObjective(obj, sense=COPT.MINIMIZE)
        if initial_solution is not None:
            submit_undirected_start(m, initial_solution, y, z, s, x, u, hp_representative)
        m.solve()
        if subtour_callback is not None and subtour_callback.error is not None:
            # COPT 8 can report INVALID from addLazyConstr when that valid
            # cut immediately proves a presolved/fixed model infeasible.
            # Preserve its infeasible status; every other callback failure
            # remains a hard error, never a saved solution.
            if int(m.status) == COPT.INFEASIBLE and not _has_solution(m):
                raise RuntimeError(
                    f"Joint model has no feasible solution (status={int(m.status)}; "
                    f"lazy separation: {subtour_callback.error})"
                )
            raise RuntimeError(f"Lazy subtour callback failed: {subtour_callback.error}")

        if not _has_solution(m):
            status = int(m.status)
            raise RuntimeError(f"Joint model has no feasible solution (status={status})")

        selected_vps = [k for k in range(n_vp) if (y[k].x or 0) > 0.5]
        selected_pairs = [[k, a] for (k, a) in valid_pairs if (z[k, a].x or 0) > 0.5]
        symmetric_energy_val = sum(undirected_cost[e] * (x[e].x or 0) for e in edges)
        quality_val = sum(float(gamma[h]) * (s[h].x or 0) for h in range(n_hp))
        photos_val = len(selected_pairs)

        tour = connected_tour(
            selected_vps, [edge for edge in edges if (x[edge].x or 0) > 0.5], depot,
        )

        energy_val = math.fsum(energy[i, j] for i, j in zip(tour, tour[1:]))
        # A final independent reconstruction must pass BEFORE any result is saved.
        final_witness = validate_initial_solution(pre, {
            "selected_vps": selected_vps, "selected_pairs": selected_pairs, "tour": tour,
        }, float(energy_norm_denom), MIN_COVER_PER_SAMPLE)
        actual_quality = math.fsum(float(gamma[h]) for h, covered in
                                  enumerate(final_witness.hemisphere_covered) if covered)
        if not math.isclose(energy_val, symmetric_energy_val, rel_tol=1e-8, abs_tol=1e-7):
            raise RuntimeError("Directed and undirected final route energy disagree")
        if not math.isclose(quality_val, actual_quality, rel_tol=1e-8, abs_tol=1e-7):
            raise RuntimeError("Reported hemisphere quality disagrees with selected photos")
        if not math.isclose(float(m.objval), final_witness.objective, rel_tol=1e-8, abs_tol=1e-7):
            raise RuntimeError("Solver objective disagrees with reconstructed original objective")
        quality_val = actual_quality

        quality_norm_val = quality_val / total_omega
        photo_norm_val = photos_val / n_pairs
        energy_norm_val = energy_val / energy_norm_denom
        mip_gap, obj_bound = _mip_gap_and_bound(m)
        try:
            joint_runtime = float(m.getAttr(COPT.Attr.SolvingTime))
        except Exception:
            joint_runtime = float(getattr(m, "SolvingTime", 0.0) or 0.0)
        status = int(m.status)
        obj_val = float(m.objval)

        selected_photos_readable = [
            [pre["vp_ids"][k], a] for (k, a) in selected_pairs
        ]
        # selected_pairs already list of lists
        photos_by_vp: dict[str, list[str]] = {}
        for vp_id, a in selected_photos_readable:
            photos_by_vp.setdefault(vp_id, []).append(a)

        print(
            f"Joint done. Energy: {energy_val:.2f} (norm {energy_norm_val:.4f}), "
            f"Quality: {quality_val:.2f} (norm {quality_norm_val:.4f}), "
            f"Photos: {photos_val}, VPs: {len(selected_vps)}, "
            f"Runtime: {joint_runtime:.1f}s, gap: {mip_gap}, status: {status}"
        )

        # Build pure-Python result FIRST so a native crash on cleanup cannot lose the solution
        result = {
            "method": "joint",
            "selected_vps": list(selected_vps),
            "selected_vp_ids": [pre["vp_ids"][k] for k in selected_vps],
            "selected_pairs": selected_pairs,
            "selected_photos_readable": selected_photos_readable,
            "photos_by_viewpoint": photos_by_vp,
            "tour": list(tour),
            "tour_ids": [pre["vp_ids"][i] for i in tour],
            "energy": float(energy_val),
            "weighted_quality": float(quality_val),
            "photos": int(photos_val),
            "n_viewpoints": int(len(selected_vps)),
            "energy_norm": float(energy_norm_val),
            "quality_norm": float(quality_norm_val),
            "photo_norm": float(photo_norm_val),
            "energy_norm_denom": float(energy_norm_denom),
            "runtime": float(joint_runtime),
            "atsp_runtime": float(atsp_runtime),
            "obj": float(obj_val),
            "mip_gap": float(mip_gap) if mip_gap is not None else None,
            "status": int(status),
            "obj_bound": float(obj_bound) if obj_bound is not None else None,
            "solver": "COPT",
            "is_optimal": status == COPT.OPTIMAL,
            "relative_gap_limit": float(m.getParam(COPT.Param.RelGap)),
            "absolute_gap_limit": float(m.getParam(COPT.Param.AbsGap)),
            "time_limit_joint": float(m.getParam(COPT.Param.TimeLimit)),
            "threads": int(m.getParam(COPT.Param.Threads)),
            "memory_limit_mb": float(m.getParam(COPT.Param.MemLimit)),
            "absolute_gap": abs(obj_val - obj_bound) if obj_bound is not None else None,
            "subtour_method": "undirected_branch_and_cut" if fractional_config.enabled
                              else "undirected_lazy_sec",
            "routing_formulation": "undirected",
            "symmetric_energy": float(symmetric_energy_val),
            "energy_identity_error": float(energy_val - symmetric_energy_val),
            "fractional_cuts_enabled": fractional_config.enabled,
            "fractional_config": dict(vars(fractional_config)),
            "search_nodes": int(m.getAttr(COPT.Attr.NodeCnt)),
            **routing_audit,
            **subtour_callback.statistics(),
        }

        if runtime_context is not None:
            total_runtime = time.perf_counter() - runtime_context["started_at"]
            two_step_runtime = runtime_context["runtime_two_step"]
            result.update({
                "runtime": float(total_runtime),
                "runtime_two_step": float(two_step_runtime),
                "runtime_joint": float(total_runtime - two_step_runtime),
                "joint_solver_runtime": float(joint_runtime),
                **{key: value for key, value in runtime_context.items()
                   if key not in {"started_at", "runtime_two_step"}},
            })

        # Save to disk BEFORE native cleanup (segfault risk on some COPT builds)
        if autosave_path is not None:
            try:
                save_joint_result(result, Path(autosave_path))
            except Exception as e:
                print(f"[WARNING] autosave failed: {e}")

        return result

    finally:
        # COPT 8: release the model before its environment on the same thread.
        try:
            if m is not None:
                m.dispose()
        finally:
            env.dispose()


def save_joint_result(result: Dict[str, Any], out_path: Path) -> None:
    missing = set(REQUIRED_RESULT_FIELDS) - result.keys()
    if missing:
        raise ValueError(f"Missing RESULT_FORMAT.md fields: {sorted(missing)}")
    if result["method"] != "joint":
        raise ValueError('Joint result method must be "joint"')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(result), f, indent=2, ensure_ascii=False)
    print(f"Joint result saved: {out_path}")


if __name__ == "__main__":
    import argparse
    from common import load_data, preprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", default="data/building1.json")
    parser.add_argument(
        "--no-accelerate", action="store_true",
        help="Use original coverage formulation; add --use-mtz for original routing",
    )
    parser.add_argument("--output", type=Path, help="Save to a separate result file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing result file")
    parser.add_argument("--no-two-step-start", action="store_true",
                        help="Skip two-step generation and solve without a MIP start")
    parser.add_argument("--step1-time-limit", type=int, default=TIME_LIMIT_STEP1)
    parser.add_argument("--step2-time-limit", type=int, default=TIME_LIMIT_STEP2)
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT_JOINT,
                        help="Joint COPT time limit, in addition to the two-step budgets")
    parser.add_argument("--no-time-limit", action="store_true",
                        help="Use COPT's effectively unlimited 1e20-second limit for all three stages")
    parser.add_argument("--exact", action="store_true",
                        help="Set joint RelGap=AbsGap=0; two-step initialization retains its usual gap")
    parser.add_argument("--no-fractional-cuts", action="store_true",
                        help="Undirected integer lazy SEC only (ablation)")
    parser.add_argument("--fractional-initial-rounds", type=int, default=20)
    parser.add_argument("--fractional-interval", type=int, default=20)
    parser.add_argument("--fractional-max-flows", type=int, default=32)
    parser.add_argument("--fractional-max-cuts", type=int, default=16)
    parser.add_argument("--fractional-budget-seconds", type=float, default=0.25)
    parser.add_argument("--initial-solution", type=Path,
                        help="Provide a saved feasible seed; generation time is not included")
    parser.add_argument("--threads", type=int,
                        help="COPT threads for all three stages (default: common.py)")
    parser.add_argument("--mem-limit-gb", type=float,
                        help="COPT memory limit for all three stages, converted to MB using 1024")
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be a positive integer")
    if args.mem_limit_gb is not None and (
            not math.isfinite(args.mem_limit_gb) or args.mem_limit_gb <= 0
            or args.mem_limit_gb > 1e20 / 1024):
        parser.error("--mem-limit-gb must be finite, positive and at most 1e20/1024")
    # CLI-only overrides are local to this Python process. Both imported
    # two-step stages must receive the same budget as the final joint model.
    import two_step_copt as two_step_module
    if args.threads is not None:
        COPT_THREADS = two_step_module.COPT_THREADS = args.threads
    if args.mem_limit_gb is not None:
        COPT_MEMLIMIT_MB = two_step_module.COPT_MEMLIMIT_MB = args.mem_limit_gb * 1024
    if args.no_time_limit:
        args.step1_time_limit = args.step2_time_limit = args.time_limit = 1e20
    try:
        fractional_config = FractionalConfig(
            enabled=not args.no_fractional_cuts,
            initial_rounds=args.fractional_initial_rounds, interval=args.fractional_interval,
            max_flows=args.fractional_max_flows, max_cuts=args.fractional_max_cuts,
            budget_seconds=args.fractional_budget_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    json_path = args.json_path
    output_path = args.output or Path("results_undirected") / f"joint_{Path(json_path).stem}.json"
    protected = {Path(json_path).resolve()}
    if args.initial_solution:
        protected.add(args.initial_solution.resolve())
    if output_path.resolve() in protected:
        parser.error("Output must not overwrite an input or an initial solution")
    if output_path.exists() and not args.overwrite:
        parser.error("Output exists; use a new --output path or explicit --overwrite")
    print(f"Loading {json_path} ...")
    data = load_data(json_path)
    audit_raw_data(data)
    pre = preprocess(data)
    # pre owns all required values; the geometry-rich raw JSON is no longer
    # needed. This affects only the CLI, never a caller's input dictionary.
    del data
    gc.collect()
    result = solve_joint(
        pre, accelerate=not args.no_accelerate,
        use_two_step_start=not args.no_two_step_start,
        time_limit_step1=args.step1_time_limit,
        time_limit_step2=args.step2_time_limit, time_limit=args.time_limit,
        relative_gap=0.0 if args.exact else None,
        absolute_gap=0.0 if args.exact else None,
        fractional_config=fractional_config,
        initial_solution=load_data(args.initial_solution) if args.initial_solution else None,
    )
    stem = Path(json_path).stem
    save_joint_result(result, output_path)
