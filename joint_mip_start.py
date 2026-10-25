"""Validate a real feasible solution and provide a COMPLETE COPT MIP start.

This module never changes variable bounds, constraints, or objective coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any

from coptpy import COPT


@dataclass(frozen=True)
class ValidatedStart:
    visited: frozenset
    photos: frozenset
    tour: tuple
    edges: frozenset
    ranks: tuple
    covered_viewpoints: tuple
    hemisphere_covered: tuple
    objective: float


def _index(value: Any, size: int) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a viewpoint index")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"Viewpoint index must be an integer: {value!r}") from error
    if not 0 <= value < size:
        raise ValueError(f"Viewpoint index out of range: {value}")
    return value


def validate_initial_solution(pre: dict, solution: dict,
                              energy_norm_denom: float,
                              min_cover: int) -> ValidatedStart:
    """Recompute all feasibility/quality data instead of trusting saved metrics."""
    n = pre["n_vp"]
    if not math.isfinite(energy_norm_denom) or energy_norm_denom <= 0:
        raise ValueError("Initial solution requires a positive finite energy denominator")
    viewpoints = [_index(k, n) for k in solution["selected_vps"]]
    visited = frozenset(viewpoints)
    if len(visited) != len(viewpoints):
        raise ValueError("Initial solution repeats a selected viewpoint")
    if pre["depot"] not in visited:
        raise ValueError("Initial solution does not visit the depot")
    photos = []
    for pair in solution["selected_pairs"]:
        if len(pair) != 2:
            raise ValueError("Each photo must contain viewpoint and direction")
        k, a = pair
        photo = (_index(k, n), a)
        if photo not in pre["valid_pair_set"]:
            raise ValueError(f"Initial solution contains an invalid photo: {photo}")
        photos.append(photo)
    photo_set = frozenset(photos)
    if len(photos) != len(photo_set):
        raise ValueError("Initial solution repeats a photo")
    if {k for k, a in photo_set} != visited:
        raise ValueError("Visited viewpoints must match viewpoints taking photos")
    tour = tuple(_index(k, n) for k in solution["tour"])
    if len(visited) < 2 or len(tour) != len(visited) + 1:
        raise ValueError("Initial route must visit each selected viewpoint exactly once")
    if tour[0] != pre["depot"] or tour[-1] != pre["depot"]:
        raise ValueError("Initial route must start and end at the depot")
    if len(set(tour[:-1])) != len(visited) or set(tour[:-1]) != visited:
        raise ValueError("Initial route has a subtour, repeated or missing viewpoint")
    route_edges = frozenset(zip(tour, tour[1:]))
    for i, j in route_edges:
        if i == j or (i, j) not in pre["energy"]:
            raise ValueError(f"Initial route contains an illegal directed edge: {(i, j)}")
    if not route_edges.issubset(pre["edges"]):
        raise ValueError("An initial route edge has no joint-model variable")
    if "selected_vp_ids" in solution:
        expected_ids = [pre["vp_ids"][k] for k in viewpoints]
        if list(solution["selected_vp_ids"]) != expected_ids:
            raise ValueError("Initial viewpoint IDs do not match this dataset")
    if "tour_ids" in solution and list(solution["tour_ids"]) != [pre["vp_ids"][k] for k in tour]:
        raise ValueError("Initial route IDs do not match this dataset")
    covered_viewpoints = []
    for ell, covering in enumerate(pre["sp_cover"]):
        covered = frozenset(k for k, a in covering if (k, a) in photo_set)
        if len(covered) < min_cover:
            raise ValueError(f"Initial solution covers sample {ell} from only {len(covered)} viewpoints")
        covered_viewpoints.append(covered)
    hp_covered = tuple(bool(photo_set.intersection(covering)) for covering in pre["hp_cover"])
    ranks = [0.] * n
    for position, k in enumerate(tour[:-1]):
        ranks[k] = float(position)
    energy = sum(pre["energy"][i, j] for i, j in zip(tour, tour[1:]))
    quality = sum(float(pre["gamma"][h]) * hp_covered[h] for h in range(pre["n_hp"]))
    objective = energy / energy_norm_denom - quality / pre["total_omega"] \
        + len(photo_set) / pre["n_pairs"]
    if not math.isfinite(objective):
        raise ValueError("Initial solution objective is not finite")
    return ValidatedStart(visited, photo_set, tour, route_edges, tuple(ranks),
                          tuple(covered_viewpoints), hp_covered, float(objective))


def submit_mip_start(model, start: ValidatedStart, y, z, s, x, t, u,
                     hp_representative) -> int:
    """Assign every model variable, with merged hemisphere variables only once."""
    variables, values = [], []
    def add(variable, value):
        variables.append(variable)
        values.append(float(value))
    for k, variable in y.items():
        add(variable, k in start.visited)
    for pair, variable in z.items():
        add(variable, pair in start.photos)
    for h, representative in enumerate(hp_representative):
        if representative == h:
            add(s[h], start.hemisphere_covered[h])
    for edge, variable in x.items():
        add(variable, edge in start.edges)
    for k, variable in t.items():
        add(variable, start.ranks[k])
    for (k, ell), variable in u.items():
        add(variable, k in start.covered_viewpoints[ell])
    if len(variables) != int(model.getAttr(COPT.Attr.Cols)):
        raise RuntimeError("MIP start is incomplete; refusing to submit partial values")
    model.setParam(COPT.Param.MipStartMode, 1)
    model.setMipStart(variables, values)
    model.loadMipStart()
    return len(variables)
