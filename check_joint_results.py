#!/usr/bin/env python3
"""Independently validate saved joint results; Python standard library only.

Single: python check_joint_results.py --data data/115.json --result result/joint_115.json
Batch:  python check_joint_results.py --data-dir data --results-dir result --report checks.json

This checks saved decisions and reported metrics, not a new global optimization.
No solver, numpy, common.py, or optimization callbacks are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

REQUIRED = {
    "method", "selected_vps", "selected_vp_ids", "selected_pairs",
    "selected_photos_readable", "photos_by_viewpoint", "tour", "tour_ids",
    "energy", "weighted_quality", "photos", "n_viewpoints", "energy_norm",
    "quality_norm", "photo_norm", "energy_norm_denom", "runtime", "atsp_runtime",
    "obj", "mip_gap", "status", "obj_bound",
}


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def safe_json(value):
    """Retain invalid numeric diagnostics without writing non-standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def invalid_constant(token):
    raise ValueError(f"Non-standard JSON numeric constant: {token}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return data, digest.hexdigest()


def nearest_neighbor(coords, collision, depot, edge_cost):
    """Reproduce common.py's NN scalar, explicitly reporting missing-edge fallbacks."""
    remaining = set(range(len(coords))) - {depot}
    current, total, maximum = depot, 0.0, None
    route, fallbacks = [depot], []

    def fallback_cost():
        nonlocal maximum
        if maximum is None:
            maximum = max((edge_cost(i, j) for i in range(len(coords))
                           for j in range(len(coords))
                           if i != j and collision[i][j] == 1), default=1.0)
        return maximum

    while remaining:
        best, cost = None, math.inf
        for j in remaining:
            if collision[current][j] == 1:
                candidate = edge_cost(current, j)
                if candidate < cost:
                    best, cost = j, candidate
        if best is None:
            best, cost = next(iter(remaining)), fallback_cost()
            fallbacks.append([current, best])
        total += cost
        current = best
        remaining.remove(best)
        route.append(best)
    if current != depot and collision[current][depot] == 1:
        total += edge_cost(current, depot)
    else:
        total += fallback_cost()
        fallbacks.append([current, depot])
    route.append(depot)
    return {"value": total, "tour": route, "fallback_edges": fallbacks,
            "is_legal_full_tour": not fallbacks}


def validate_result(data, result, *, atol=1e-6, rtol=1e-8, min_cover=3,
                    coefficients=(0.5, 0.9, 0.3), normalization="nn",
                    require_zero_gap=False):
    report = {"passed": False, "constraints_satisfied": None,
              "errors": [], "warnings": [], "recomputed": {}, "coverage": {},
              "optimality": {"independently_proven": False}}

    def error(category, code, message, **details):
        report["errors"].append({"category": category, "code": code,
                                 "message": message, **details})

    def warning(code, message, **details):
        report["warnings"].append({"code": code, "message": message, **details})

    def close(a, b):
        return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)

    def metric(key, expected):
        report["recomputed"][key] = expected
        stored = result.get(key)
        if not number(stored) or not close(stored, expected):
            error("metrics", "metric_mismatch", f"{key} 与独立复算值不一致",
                  field=key, stored=stored, expected=expected)

    if not isinstance(data, dict) or not isinstance(result, dict):
        error("input", "not_object", "输入数据和结果必须是 JSON 对象")
        return report
    missing = sorted(REQUIRED - result.keys())
    if missing:
        error("schema", "missing_fields", "缺少 RESULT_FORMAT.md 必需字段", fields=missing)
    if result.get("method") != "joint":
        error("schema", "method", "本工具默认校验 joint 标准结果，method 应为 joint")
    vp = data.get("view_points") or data.get("viewpoints")
    samples, hemispheres = data.get("sample_points"), data.get("hemisphere_points")
    collision = data.get("collision_matrix")
    if not isinstance(vp, list) or not vp or not all(isinstance(v, dict) for v in vp):
        error("input", "viewpoints", "原始数据缺少有效视点列表")
        return report
    n = len(vp)
    if not isinstance(samples, list) or not isinstance(hemispheres, list):
        error("input", "coverage_lists", "原始数据缺少采样点或半球点列表")
        return report
    if (not isinstance(collision, list) or len(collision) != n or
            any(not isinstance(row, list) or len(row) != n for row in collision)):
        error("input", "collision_shape", "碰撞矩阵必须为视点数×视点数")
        return report
    ids = [v.get("id", f"v{k+1}") for k, v in enumerate(vp)]
    if not all(isinstance(vid, str) for vid in ids) or len(set(ids)) != n:
        error("input", "viewpoint_ids", "视点 ID 必须是唯一字符串")
        return report
    index = {vid: k for k, vid in enumerate(ids)}
    coords = []
    for k, point in enumerate(vp):
        coordinate = point.get("coordinate") or point.get("coordinates")
        if not isinstance(coordinate, dict) or not all(number(coordinate.get(a)) for a in ("x", "y", "z")):
            error("input", "coordinates", "视点坐标必须为有限数值", viewpoint=k)
            return report
        coords.append(tuple(float(coordinate[a]) for a in ("x", "y", "z")))
    mandatory = {k for k, point in enumerate(vp)
                 if point.get("is_mandatory") is True or point.get("is_depot") is True}
    # Same depot choice as the shipped common.py, but enforce ALL mandatory points.
    depot = max(mandatory) if mandatory else 0
    if len(mandatory) > 1:
        warning("multiple_mandatory", "按数学要求检查所有强制视点；现有 common.py 仅将最后一个标记点作为起点")

    def coverage_list(points, label):
        rows = []
        for position, point in enumerate(points):
            pairs = point.get("covering_pairs") if isinstance(point, dict) else None
            if not isinstance(pairs, list):
                error("input", "covering_pairs", f"{label}覆盖列表缺失", index=position)
                rows.append(set())
                continue
            parsed = set()
            for pair in pairs:
                if (not isinstance(pair, list) or len(pair) != 2 or
                        not isinstance(pair[0], str) or not isinstance(pair[1], str) or pair[0] not in index):
                    error("input", "covering_pair", f"{label}含无效覆盖对", index=position, pair=pair)
                    continue
                parsed.add((index[pair[0]], pair[1]))
            rows.append(parsed)
        return rows

    sp_cover = coverage_list(samples, "采样点")
    hp_cover = coverage_list(hemispheres, "半球点")
    if any(e["category"] == "input" for e in report["errors"]):
        return report
    if any(not isinstance(h, dict) or not number(h.get("weight")) for h in hemispheres):
        error("input", "weights", "半球权重必须为有限数值")
        return report
    weights = [float(h["weight"]) for h in hemispheres]
    omega = math.fsum(weights)
    valid_pairs = set().union(*sp_cover, *hp_cover) if sp_cover or hp_cover else set()
    if not valid_pairs or not math.isfinite(omega) or omega <= 0:
        error("input", "normalization", "有效拍照对数和总半球权重必须为正")
        return report
    report["dimensions"] = {"viewpoints": n, "samples": len(samples),
                            "hemispheres": len(hemispheres), "photo_pairs": len(valid_pairs),
                            "depot_index": depot, "depot_id": ids[depot]}

    def indices(key):
        value = result.get(key)
        if (not isinstance(value, list) or any(type(k) is not int or not 0 <= k < n for k in value)):
            error("schema", "indices", f"{key} 必须为有效视点索引列表")
            return None
        return value

    selected_list, tour = indices("selected_vps"), indices("tour")
    raw_photos = result.get("selected_pairs")
    if (not isinstance(raw_photos, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not int
            or not 0 <= pair[0] < n or not isinstance(pair[1], str) for pair in raw_photos)):
        error("schema", "photos", "selected_pairs 必须为 [有效整数索引, 方向字符串] 的数组")
        raw_photos = None
    if selected_list is None or tour is None or raw_photos is None:
        return report
    selected, photos = set(selected_list), set(map(tuple, raw_photos))
    if len(selected) != len(selected_list):
        error("constraints", "duplicate_viewpoint", "selected_vps 含重复视点")
    if len(photos) != len(raw_photos):
        error("constraints", "duplicate_photo", "同一视点方向重复拍照，不能作为两个二进制决策计数")
    illegal_photos = sorted(photos - valid_pairs)
    if illegal_photos:
        error("constraints", "invalid_photo", "存在无效视点方向对", photos=illegal_photos)
    photo_vertices = {k for k, a in photos}
    if photo_vertices != selected:
        error("constraints", "photo_visit_link", "访问视点必须拍照，拍照视点必须被访问",
              visited_without_photo=sorted(selected - photo_vertices),
              photos_without_visit=sorted(photo_vertices - selected))
    if (mandatory | {depot}) - selected:
        error("constraints", "mandatory", "缺少起点或强制访问视点", missing=sorted((mandatory | {depot}) - selected))

    readable = [[ids[k], a] for k, a in raw_photos]
    expected_map = defaultdict(list)
    for k, a in raw_photos:
        expected_map[ids[k]].append(a)
    expected_id_fields = {"selected_vp_ids": [ids[k] for k in selected_list],
                          "tour_ids": [ids[k] for k in tour], "selected_photos_readable": readable}
    for key, expected in expected_id_fields.items():
        if result.get(key) != expected:
            error("schema", "readable_mismatch", f"{key} 与索引决策不一致", field=key)
    actual_map = result.get("photos_by_viewpoint")
    if (not isinstance(actual_map, dict) or set(actual_map) != set(expected_map)
            or any(not isinstance(v, list) or not all(isinstance(a, str) for a in v) for v in actual_map.values())
            or any(Counter(actual_map.get(k, [])) != Counter(v) for k, v in expected_map.items())):
        error("schema", "photo_map", "photos_by_viewpoint 与 selected_pairs 不一致")

    counts, failed_samples = [], []
    for ell, pairs in enumerate(sp_cover):
        covering = sorted({k for k, a in pairs if (k, a) in photos})
        counts.append(len(covering))
        if len(covering) < min_cover:
            failed_samples.append({"sample_index": ell, "sample_id": samples[ell].get("id"),
                                   "count": len(covering), "viewpoints": covering,
                                   "viewpoint_ids": [ids[k] for k in covering]})
    report["coverage"] = {"required": min_cover, "minimum": min(counts) if counts else None,
                          "histogram": dict(sorted(Counter(counts).items())),
                          "counts_per_sample": counts, "failed_samples": failed_samples}
    if failed_samples:
        error("constraints", "distinct_coverage", "采样点未达到不同视点覆盖要求；空覆盖点也会失败",
              failed_count=len(failed_samples))

    closed = len(tour) >= 3 and tour[0] == tour[-1]
    if not closed:
        error("constraints", "tour_not_closed", "路径应为至少含两个视点的闭合回路")
    if not tour or tour[0] != depot or tour[-1] != depot:
        error("schema", "tour_start", "标准输出应从指定起点出发并回到起点")
    if closed and (len(tour) != len(selected) + 1 or set(tour[:-1]) != selected
                   or len(set(tour[:-1])) != len(tour) - 1):
        error("constraints", "tour_vertices", "回路必须恰好访问每个已选视点一次，不得遗漏、重复或含额外视点")
    edges = list(zip(tour, tour[1:]))
    outgoing, incoming = Counter(i for i, j in edges), Counter(j for i, j in edges)
    wrong_degree = [k for k in range(n) if outgoing[k] != int(k in selected) or incoming[k] != int(k in selected)]
    if wrong_degree:
        error("constraints", "degrees", "回路的入度/出度必须等于访问变量 y", viewpoints=wrong_degree)
    illegal_edges = [[i, j] for i, j in edges if i == j or collision[i][j] != 1]
    if illegal_edges:
        error("constraints", "illegal_edges", "路径含自环或不合法的有向飞行边", edges=illegal_edges)
    report["route"] = {"closed": closed, "positions": len(tour),
                       "unique_visited": len(set(tour[:-1])) if closed else len(set(tour)),
                       "illegal_edges": illegal_edges, "wrong_degree_vertices": wrong_degree}
    # The JSON omits auxiliary t/u/s values. Construct witnesses instead of
    # trusting unsaved solver variables; check the original selective MTZ rows.
    route_valid = (closed and not illegal_edges and not wrong_degree
                   and len(tour) == len(selected) + 1 and set(tour[:-1]) == selected
                   and len(set(tour[:-1])) == len(selected))
    witness = {"mtz_extension_verified": False, "mtz_rows_checked": 0}
    if route_valid:
        offset = tour[:-1].index(depot) if depot in selected else None
        if offset is not None:
            cycle = tour[:-1]
            ordered = cycle[offset:] + cycle[:offset]
            rank = dict.fromkeys(range(n), 0)
            rank.update({k: position for position, k in enumerate(ordered)})
            edge_set, big_m, bad_mtz = set(edges), n - 1, []
            for i in range(n):
                for j in range(n):
                    if i == j or depot in (i, j) or collision[i][j] != 1:
                        continue
                    witness["mtz_rows_checked"] += 1
                    lhs = rank[i] - rank[j] + big_m * int((i, j) in edge_set)
                    rhs = big_m - 1 + big_m * int(i not in selected) + big_m * int(j not in selected)
                    if lhs > rhs:
                        bad_mtz.append([i, j])
            if bad_mtz:
                error("constraints", "mtz_extension", "重构序号不能满足原 MTZ 约束", edges=bad_mtz)
            else:
                witness["mtz_extension_verified"] = True
    report["auxiliary_witnesses"] = witness

    pd, pc, ps = coefficients

    def edge_cost(i, j):
        a, b = coords[i], coords[j]
        distance = math.sqrt(sum((a[q] - b[q]) ** 2 for q in range(3)))
        return pd * distance + pc * max(0.0, b[2] - a[2]) + ps * max(0.0, a[2] - b[2])

    quality = math.fsum(weight for weight, pairs in zip(weights, hp_cover) if photos.intersection(pairs))
    energy = math.fsum(edge_cost(i, j) for i, j in edges)
    metric("energy", energy)
    metric("weighted_quality", quality)
    for key, value in (("photos", len(photos)), ("n_viewpoints", len(selected))):
        report["recomputed"][key] = value
        if type(result.get(key)) is not int or result[key] != value:
            error("metrics", "count_mismatch", f"{key} 必须为与决策一致的整数", field=key, expected=value, stored=result.get(key))
    denom = result.get("energy_norm_denom")
    if not number(denom) or denom <= 0:
        error("metrics", "energy_denom", "energy_norm_denom 必须为有限正数")
    else:
        norm_denom = denom
        if normalization == "nn":
            nn = nearest_neighbor(coords, collision, depot, edge_cost)
            report["normalization"] = {"mode": "nn", **nn}
            metric("energy_norm_denom", nn["value"])
            norm_denom = nn["value"]
            if nn["fallback_edges"]:
                warning("nn_fallback", "原 NN 分母使用了缺边回退；数值可复现，但不是合法完整回路的能耗",
                        edges=nn["fallback_edges"])
        else:
            report["normalization"] = {"mode": "reported", "value": denom}
            warning("reported_denominator", "使用结果中提供的分母，未独立验证其计算来源")
        if not number(norm_denom) or norm_denom <= 0:
            error("metrics", "recomputed_denom", "独立复算的分母不是有限正数，无法定义归一化目标")
        else:
            metric("energy_norm", energy / norm_denom)
            metric("quality_norm", quality / omega)
            metric("photo_norm", len(photos) / len(valid_pairs))
            metric("obj", energy / norm_denom - quality / omega + len(photos) / len(valid_pairs))
    report["recomputed"]["total_omega"] = omega
    report["recomputed"]["covered_hemispheres"] = sum(bool(photos.intersection(pairs)) for pairs in hp_cover)

    for key in ("runtime", "atsp_runtime", "runtime_two_step", "runtime_joint",
                "joint_solver_runtime", "runtime_step1", "runtime_step2"):
        if key in result and (not number(result[key]) or result[key] < 0):
            error("metrics", "runtime", f"{key} 必须为有限非负数", field=key)
    if all(number(result.get(k)) for k in ("runtime", "runtime_two_step", "runtime_joint")):
        if not close(result["runtime"], result["runtime_two_step"] + result["runtime_joint"]):
            error("metrics", "runtime_sum", "runtime 不等于 runtime_two_step + runtime_joint")
    if all(number(result.get(k)) for k in ("runtime_joint", "joint_solver_runtime")):
        if result["joint_solver_runtime"] > result["runtime_joint"] + atol:
            error("metrics", "solver_runtime", "联合求解器时间超过联合阶段墙钟时间")

    obj, bound, gap = (result.get(k) for k in ("obj", "obj_bound", "mip_gap"))
    if not isinstance(result.get("status"), (int, str)) or isinstance(result.get("status"), bool):
        error("schema", "status_type", "status 必须为整数或字符串")
    for key in ("obj_bound", "mip_gap"):
        value = result.get(key)
        if value is not None and (not number(value) or (key == "mip_gap" and value < 0)):
            error("metrics", "certificate_number", f"{key} 必须为有限数值或 null", field=key)
    bounds_match = False
    if number(obj) and number(bound):
        if bound > obj and not close(bound, obj):
            error("certificate", "bound_direction", "最小化问题的最优界不应大于可行解目标值")
        denominator = max(abs(obj), abs(bound))
        expected_gap = abs(obj - bound) / denominator if denominator else 0.0
        report["optimality"]["recomputed_relative_gap"] = expected_gap
        report["optimality"]["absolute_gap"] = abs(obj - bound)
        if number(gap) and not close(gap, expected_gap):
            error("certificate", "gap_mismatch", "保存的 mip_gap 与 obj/obj_bound 不一致", expected=expected_gap, stored=gap)
        # Relative tolerance alone could accept an absolute difference of 1
        # at a very large objective; zero-gap checks use absolute tolerance.
        bounds_match = math.isclose(obj, bound, rel_tol=0.0, abs_tol=atol)
    if "absolute_gap" in result and number(obj) and number(bound):
        metric("absolute_gap", abs(obj - bound))
    copt = result.get("solver") == "COPT"
    reported_optimal = copt and type(result.get("status")) is int and result["status"] == 1
    if "is_optimal" in result and copt:
        if type(result["is_optimal"]) is not bool or result["is_optimal"] != reported_optimal:
            error("certificate", "optimal_flag", "is_optimal 与 COPT 状态不一致")
    certificate_zero = reported_optimal and bounds_match and number(gap) and gap == 0
    report["optimality"].update({"solver": result.get("solver"), "status": result.get("status"),
        "reported_copt_optimal": reported_optimal, "zero_gap_record_consistent": certificate_zero,
        "note": "仅核对保存的状态、界和 gap；不重新求解，不能独立证明全局最优。"})
    if require_zero_gap and not certificate_zero:
        error("certificate", "zero_gap_required", "要求零 gap，但当前记录不能确认 COPT 最优状态与零 gap")
    if not reported_optimal:
        warning("not_certified_optimal", "可行性与最优性分别判断；当前状态未确认 COPT 最优")
    report["constraints_satisfied"] = not any(e["category"] in ("input", "constraints") for e in report["errors"])
    report["schema_valid"] = not any(e["category"] == "schema" for e in report["errors"])
    report["metrics_consistent"] = not any(e["category"] == "metrics" for e in report["errors"])
    report["passed"] = not report["errors"]
    return report


def check_file(data_path, result_path, **options):
    base = {"data_file": str(data_path), "result_file": str(result_path)}
    try:
        data, base["data_sha256"] = read_json(data_path)
        result, base["result_sha256"] = read_json(result_path)
        return {**base, **validate_result(data, result, **options)}
    except (OSError, UnicodeError, ValueError, TypeError, OverflowError) as exc:
        return {**base, "passed": False, "constraints_satisfied": None,
                "errors": [{"category": "input", "code": "read_or_validate_error", "message": str(exc)}], "warnings": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, help="单个原始输入 JSON")
    inputs.add_argument("--data-dir", type=Path, help="原始输入目录（递归匹配）")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--result", type=Path, help="单个 joint 结果 JSON")
    outputs.add_argument("--results-dir", type=Path, help="结果目录（递归查找 joint_*.json）")
    parser.add_argument("--report", type=Path, default=Path("constraint_check_report.json"))
    parser.add_argument("--normalization", choices=("nn", "reported"), default="nn")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--min-cover", type=int, default=3)
    parser.add_argument("--p-dist", type=float, default=0.5)
    parser.add_argument("--p-climb", type=float, default=0.9)
    parser.add_argument("--p-descend", type=float, default=0.3)
    parser.add_argument("--require-zero-gap", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.data) != bool(args.result):
        parser.error("--data 应配合 --result；--data-dir 应配合 --results-dir")
    if args.min_cover < 1 or any(not number(v) or v < 0 for v in (args.atol, args.rtol, args.p_dist, args.p_climb, args.p_descend)):
        parser.error("容差/能耗系数应为有限非负数，min-cover 应为正整数")
    options = {"atol": args.atol, "rtol": args.rtol, "min_cover": args.min_cover,
               "coefficients": (args.p_dist, args.p_climb, args.p_descend),
               "normalization": args.normalization, "require_zero_gap": args.require_zero_gap}
    jobs = []
    if args.data:
        jobs.append((args.data, args.result, None))
    else:
        if not args.data_dir.is_dir() or not args.results_dir.is_dir():
            parser.error("输入目录和结果目录必须存在")
        catalogue = defaultdict(list)
        for path in sorted(args.data_dir.rglob("*.json")):
            catalogue[path.stem].append(path)
        for result_path in sorted(args.results_dir.rglob("joint_*.json")):
            stem = result_path.stem[len("joint_"):]
            # Exact names first, then the longest instance stem + run suffix.
            matches = catalogue.get(stem, [])
            if not matches:
                keys = [key for key in catalogue if stem.startswith(key + "_")]
                if keys:
                    longest = max(map(len, keys))
                    matches = [p for key in keys if len(key) == longest for p in catalogue[key]]
            if len(matches) == 1:
                jobs.append((matches[0], result_path, None))
            else:
                jobs.append((None, result_path, "没有唯一匹配的输入数据；请用 --data/--result 显式指定"))
        if not jobs:
            parser.error("未找到 joint_*.json；没有文件被检查")
    # Never overwrite either an input dataset or a result with the check report.
    protected = {p.resolve() for data_path, result_path, _ in jobs for p in (data_path, result_path) if p}
    if args.data_dir:
        protected.update(path.resolve() for paths in catalogue.values() for path in paths)
    if args.report.resolve() in protected:
        parser.error("报告路径不能与原始数据或求解结果相同")
    records = []
    for data_path, result_path, issue in jobs:
        record = ({"data_file": None, "result_file": str(result_path), "passed": False,
                   "constraints_satisfied": None, "errors": [{"category": "input", "code": "data_matching", "message": issue}], "warnings": []}
                  if issue else check_file(data_path, result_path, **options))
        records.append(record)
        state = "PASS" if record["passed"] else "FAIL"
        print(f"[{state}] {result_path.name}: errors={len(record['errors'])}, warnings={len(record['warnings'])}")
        for item in record["errors"][:8]:
            print(f"  {item['code']}: {item['message']}")
    passed = sum(record["passed"] for record in records)
    document = {"format_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "settings": options, "summary": {"total": len(records), "passed": passed,
                    "failed": len(records) - passed, "all_passed": passed == len(records)},
                "note": "校验保存结果的约束、指标和证书字段一致性，不重新求解或独立证明全局最优。",
                "results": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(safe_json(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"合计 {len(records)}，通过 {passed}，失败 {len(records)-passed}。报告：{args.report}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
