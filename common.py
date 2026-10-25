#!/usr/bin/env python3
"""Shared utilities for UAV viewpoint selection & path planning (COPT package)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ====================== Default Parameters ======================
P_DIST = 0.5
P_CLIMB = 0.9
P_DESCEND = 0.3
MIN_COVER_PER_SAMPLE = 3
MIP_GAP = 0.0001  # Relative gap: 0.01% (0.0001 = 0.01 / 100)

TIME_LIMIT_STEP1 = 1800
TIME_LIMIT_STEP2 = 1800
TIME_LIMIT_JOINT = 3600

# Per-process budget for TWO concurrent jobs on the 16-core / 125-GiB server.
# Leave room for Python data, model construction, the OS and other services.
COPT_THREADS = 8
# COPT MemLimit is in MB; this limits solver memory, not total process RSS.
COPT_MEMLIMIT_MB = 40 * 1024  # 40960 MB per solver; 81920 MB for two solvers
COPT_LOGLEVEL = 3  # Include solver memory information in MIP logs.


def load_data(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess(data: dict) -> Dict[str, Any]:
    view_points = data.get("view_points") or data.get("viewpoints")
    if not view_points:
        raise KeyError("Missing view_points / viewpoints")

    sample_points = data["sample_points"]
    hemisphere_points = data["hemisphere_points"]
    collision = data["collision_matrix"]

    n_vp = len(view_points)
    n_sp = len(sample_points)
    n_hp = len(hemisphere_points)

    vp_ids = []
    vp_id_to_idx = {}
    coords = []
    depot = 0
    depot_id = None
    for i, vp in enumerate(view_points):
        vid = vp.get("id", f"v{i+1}")
        vp_ids.append(vid)
        vp_id_to_idx[vid] = i
        c = vp.get("coordinate") or vp.get("coordinates")
        coords.append((float(c["x"]), float(c["y"]), float(c["z"])))
        if vp.get("is_mandatory") is True or vp.get("is_depot") is True:
            depot = i
            depot_id = vid
    if depot_id is None:
        depot_id = vp_ids[depot]
    print(f"  Depot / mandatory viewpoint: {depot_id} (index {depot})")

    edges = []
    energy = {}
    for i in range(n_vp):
        xi, yi, zi = coords[i]
        for j in range(n_vp):
            if i == j:
                continue
            if collision[i][j] != 1:
                continue
            xj, yj, zj = coords[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            climb = max(0.0, zj - zi)
            descend = max(0.0, zi - zj)
            e = P_DIST * dist + P_CLIMB * climb + P_DESCEND * descend
            # Share this immutable key between edges and energy instead of
            # retaining two separate tuples for every directed edge.
            edge = (i, j)
            edges.append(edge)
            energy[edge] = e
    print(f"  Number of directed edges: {len(edges)}")

    valid_pairs = set()
    for hp in hemisphere_points:
        for v_id, a in hp["covering_pairs"]:
            if v_id in vp_id_to_idx:
                valid_pairs.add((vp_id_to_idx[v_id], a))
    for sp in sample_points:
        for v_id, a in sp["covering_pairs"]:
            if v_id in vp_id_to_idx:
                valid_pairs.add((vp_id_to_idx[v_id], a))
    valid_pairs = sorted(list(valid_pairs))
    valid_pair_set = set(valid_pairs)
    # Reuse canonical immutable pairs throughout hp_cover and sp_cover. The
    # lists stay independent and all ordering/duplicate records are retained.
    canonical_pairs = {pair: pair for pair in valid_pairs}
    n_pairs = len(valid_pairs)
    print(f"  Number of valid (viewpoint, direction) pairs: {n_pairs}")

    vp_to_dirs = defaultdict(list)
    for (k, a) in valid_pairs:
        vp_to_dirs[k].append(a)

    hp_cover = [[] for _ in range(n_hp)]
    gamma = np.zeros(n_hp)
    for h, hp in enumerate(hemisphere_points):
        gamma[h] = hp["weight"]
        for v_id, a in hp["covering_pairs"]:
            if v_id in vp_id_to_idx:
                ka = canonical_pairs.get((vp_id_to_idx[v_id], a))
                if ka is not None:
                    hp_cover[h].append(ka)

    total_omega = float(gamma.sum())
    print(f"  total_omega (sum of hemisphere weights): {total_omega:.4f}")

    sp_cover = [[] for _ in range(n_sp)]
    for ell, sp in enumerate(sample_points):
        for v_id, a in sp["covering_pairs"]:
            if v_id in vp_id_to_idx:
                ka = canonical_pairs.get((vp_id_to_idx[v_id], a))
                if ka is not None:
                    sp_cover[ell].append(ka)

    N_h = np.array([len(cov) for cov in hp_cover], dtype=int)

    return {
        "n_vp": n_vp,
        "n_sp": n_sp,
        "n_hp": n_hp,
        "n_pairs": n_pairs,
        "vp_ids": vp_ids,
        "vp_id_to_idx": vp_id_to_idx,
        "coords": coords,
        "depot": depot,
        "depot_id": depot_id,
        "edges": edges,
        "energy": energy,
        "valid_pairs": valid_pairs,
        "valid_pair_set": valid_pair_set,
        "vp_to_dirs": vp_to_dirs,
        "hp_cover": hp_cover,
        "gamma": gamma,
        "total_omega": total_omega,
        "sp_cover": sp_cover,
        "N_h": N_h,
        "total_covering_pairs": n_pairs,
    }


def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k:
                make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    return obj


def nn_tour_energy(pre: Dict[str, Any]) -> float:
    """Nearest-neighbour closed tour energy from depot over ALL viewpoints."""
    n_vp = pre["n_vp"]
    depot = pre["depot"]
    energy = pre["energy"]
    unvisited = set(range(n_vp))
    unvisited.remove(depot)
    cur = depot
    total = 0.0
    while unvisited:
        best_j, best_e = None, float("inf")
        for j in unvisited:
            e = energy.get((cur, j))
            if e is not None and e < best_e:
                best_e, best_j = e, j
        if best_j is None:
            for j in list(unvisited):
                if (cur, j) in energy:
                    best_j, best_e = j, energy[(cur, j)]
                    break
            if best_j is None:
                best_j = next(iter(unvisited))
                best_e = max(energy.values()) if energy else 1.0
        total += best_e
        cur = best_j
        unvisited.remove(best_j)
    back = energy.get((cur, depot))
    if back is None:
        back = max(energy.values()) if energy else 1.0
    total += back
    return float(total)
