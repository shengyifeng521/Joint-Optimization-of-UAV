"""Exact optional-node undirected routing and separated connectivity cuts.

No external graph package is required. All cuts are valid for the FULL tour
problem; every integer candidate is checked independently of cut throttling.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import threading
import time

import coptpy as cp
from coptpy import COPT

from common import P_CLIMB, P_DESCEND, MIN_COVER_PER_SAMPLE
from joint_mip_start import submit_mip_start


@dataclass(frozen=True)
class FractionalConfig:
    enabled: bool = True
    initial_rounds: int = 20
    interval: int = 20
    max_flows: int = 32
    max_cuts: int = 16
    budget_seconds: float = 0.25  # soft per-round limit; 0 means unlimited
    violation_tolerance: float = 1e-6
    flow_tolerance: float = 1e-10

    def __post_init__(self):
        for name in ("initial_rounds", "interval", "max_flows", "max_cuts"):
            value = getattr(self, name)
            minimum = 0 if name == "initial_rounds" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("budget_seconds", "violation_tolerance", "flow_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "budget_seconds" and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        if self.flow_tolerance >= self.violation_tolerance:
            raise ValueError("Flow tolerance must be smaller than cut violation tolerance")


def audit_raw_data(data):
    """Reject unsupported mandatory-point semantics rather than change them."""
    vps = data.get("view_points") or data.get("viewpoints") or []
    mandatory = [i for i, vp in enumerate(vps)
                 if vp.get("is_mandatory") is True or vp.get("is_depot") is True]
    if len(mandatory) > 1:
        raise ValueError("Multiple mandatory viewpoints: old code only enforces its final depot; "
                         "refusing to silently change that model")


def symmetric_costs(pre):
    """Audit exact transformation e_ij = c_ij + phi_j - phi_i.

    Reject empty/under-covered samples, asymmetric arcs, missing energy keys,
    unsupported energy functions, and models permitting a two-vertex tour.
    Numerical identities use tight floating-point tolerances, reported below.
    """
    n = pre["n_vp"]
    if MIN_COVER_PER_SAMPLE != 3:
        raise ValueError("This equivalence experiment is defined for MIN_COVER_PER_SAMPLE=3")
    if n < 3 or not pre["sp_cover"]:
        raise ValueError("Undirected experiment requires a nonempty 3-viewpoint coverage problem")
    for ell, covering in enumerate(pre["sp_cover"]):
        if len({k for k, a in covering}) < 3:
            raise ValueError(f"Sample {ell} has fewer than 3 candidate viewpoints; "
                             "empty samples are never silently skipped")
    if pre["n_pairs"] <= 0 or not math.isfinite(pre["total_omega"]) or pre["total_omega"] <= 0:
        raise ValueError("Normalization requires photos and positive finite total hemisphere weight")
    edges, energy = pre["edges"], pre["energy"]
    if len(edges) != len(energy) or any(edge not in energy for edge in edges):
        raise ValueError("Edge variables and energy keys do not match")
    if len(set(edges)) != len(edges):
        raise ValueError("Duplicate directed arcs")
    phi = [0.5 * (P_CLIMB - P_DESCEND) * float(c[2]) for c in pre["coords"]]
    if len(phi) != n or not all(map(math.isfinite, phi)):
        raise ValueError("Invalid viewpoint heights")
    costs, residual = {}, 0.0
    for i, j in edges:
        if i == j or not (0 <= i < n and 0 <= j < n):
            raise ValueError("Invalid directed arc endpoint or self-loop")
        if (j, i) not in energy:
            raise ValueError(f"Asymmetric allowed graph: missing {(j, i)}")
        value, reverse = float(energy[i, j]), float(energy[j, i])
        if not math.isfinite(value) or value < 0 or not math.isfinite(reverse) or reverse < 0:
            raise ValueError("Energy coefficients must be finite and nonnegative")
        if i < j:
            c = (value + reverse) / 2.0
            error = abs((value - c) - (phi[j] - phi[i]))
            residual = max(residual, error)
            if error > 1e-9 + 1e-12 * max(abs(value), abs(reverse)):
                raise ValueError(f"Energy has no required height potential decomposition at {(i, j)}")
            costs[i, j] = c
    return costs, {"directed_edge_count": len(edges), "undirected_edge_count": len(costs),
                   "max_energy_potential_residual": residual,
                   "energy_identity_atol": 1e-9, "energy_identity_rtol": 1e-12}


def selected_components(vertices, edges, depot):
    vertices = set(vertices)
    if depot not in vertices or len(vertices) < 3:
        raise ValueError("Selected vertices must include depot and at least 3 viewpoints")
    adjacency = {i: [] for i in vertices}
    seen = set()
    for i, j in edges:
        edge = (min(i, j), max(i, j))
        if i == j or i not in vertices or j not in vertices or edge in seen:
            raise ValueError("Invalid, repeated, or unselected undirected edge")
        seen.add(edge)
        adjacency[i].append(j)
        adjacency[j].append(i)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("Integer candidate violates undirected degree=2*y")
    unseen, components = set(vertices), []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        component = []
        while stack:
            i = stack.pop()
            component.append(i)
            for j in adjacency[i]:
                if j in unseen:
                    unseen.remove(j)
                    stack.append(j)
        components.append(tuple(sorted(component)))
    return components, adjacency


def connected_tour(vertices, edges, depot):
    components, adjacency = selected_components(vertices, edges, depot)
    if len(components) != 1:
        raise ValueError(f"Final solution has {len(components)} disconnected undirected cycles")
    tour, previous, current = [depot], None, depot
    while True:
        neighbors = sorted(adjacency[current])
        following = neighbors[0] if previous is None or neighbors[0] != previous else neighbors[1]
        tour.append(following)
        if following == depot:
            break
        if following in tour[:-1]:
            raise ValueError("Invalid repeated viewpoint in reconstructed tour")
        previous, current = current, following
    if len(tour) != len(adjacency) + 1:
        raise ValueError("Final route omits selected viewpoints")
    return tour


def minimum_cut(n, weighted_edges, source, sink, *, tolerance=1e-10, cutoff=None):
    """Undirected s-t min-cut using iterative Dinic (no recursion depth limit).

    Each undirected edge is a pair of mutually reverse residual arcs, both with
    initial capacity c. A cutoff may stop once a sufficient flow is proved;
    in that case the returned shore is None. Tiny capacities may be ignored by
    the flow search, so callers MUST re-evaluate a proposed cut on FULL values.
    """
    if source == sink or not (0 <= source < n and 0 <= sink < n):
        raise ValueError("Invalid source/sink")
    graph = [[] for _ in range(n)]
    for i, j, capacity in weighted_edges:
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError("Min-cut requires finite nonnegative capacities")
        if capacity <= tolerance:
            continue
        graph[i].append([j, len(graph[j]), float(capacity)])
        graph[j].append([i, len(graph[i]) - 1, float(capacity)])
    flow = 0.0
    while True:
        level = [-1] * n
        level[source] = 0
        queue = deque([source])
        while queue:
            i = queue.popleft()
            for j, reverse, cap in graph[i]:
                if cap > tolerance and level[j] < 0:
                    level[j] = level[i] + 1
                    queue.append(j)
        if level[sink] < 0:
            return flow, frozenset(i for i in range(n) if level[i] >= 0)
        pointer = [0] * n
        while True:
            nodes, path, bottlenecks = [source], [], [math.inf]
            while nodes and nodes[-1] != sink:
                i = nodes[-1]
                while pointer[i] < len(graph[i]):
                    arc = graph[i][pointer[i]]
                    if arc[2] > tolerance and level[arc[0]] == level[i] + 1:
                        path.append((i, pointer[i]))
                        nodes.append(arc[0])
                        bottlenecks.append(min(bottlenecks[-1], arc[2]))
                        break
                    pointer[i] += 1
                else:
                    level[i] = -1
                    nodes.pop()
                    bottlenecks.pop()
                    if path:
                        previous, index = path.pop()
                        pointer[previous] += 1
            if not nodes:
                break
            amount = bottlenecks[-1]
            for i, index in path:
                j, reverse, capacity = graph[i][index]
                graph[i][index][2] -= amount
                graph[j][reverse][2] += amount
            flow += amount
            if cutoff is not None and flow >= cutoff:
                return flow, None


def submit_undirected_start(model, start, y, z, s, x, u, representatives):
    undirected_edges = frozenset((min(i, j), max(i, j)) for i, j in start.edges)
    if len(undirected_edges) != len(start.visited):
        raise ValueError("MIP start cannot be mapped to a simple undirected tour")
    return submit_mip_start(model, replace(start, edges=undirected_edges),
                            y, z, s, x, {}, u, representatives)


class UndirectedTourCallback(cp.CallbackBase):
    def __init__(self, y, x, depot, config=None):
        super().__init__()
        self.y, self.x, self.depot = y, x, depot
        self.config = config or FractionalConfig()
        self.vertices, self.edges = list(y), list(x)
        self.y_variables, self.x_variables = list(y.values()), list(x.values())
        self.incident = {i: [] for i in y}
        for index, (i, j) in enumerate(self.edges):
            self.incident[i].append(index)
            self.incident[j].append(index)
        self.calls = self.cut_submissions = self.relax_calls = 0
        self.fractional_rounds = self.user_cuts = self.flow_calls = 0
        self.fractional_seconds = 0.0
        self.anchor_offset = 0
        self.error = None
        self.lock = threading.RLock()

    def cut_indices(self, members):
        internal, boundary = [], []
        for i in members:
            for index in self.incident[i]:
                a, b = self.edges[index]
                other = b if a == i else a
                if other not in members:
                    boundary.append(index)
                elif i == a:
                    internal.append(index)
        return internal, boundary

    def inequality(self, members, anchor, internal, boundary):
        # Under degree=2*y, these are exactly equivalent. Choose fewer entries.
        if len(internal) + len(members) - 1 <= len(boundary) + 1:
            return cp.quicksum(self.x[self.edges[e]] for e in internal) <= cp.quicksum(
                self.y[i] for i in members if i != anchor)
        return cp.quicksum(self.x[self.edges[e]] for e in boundary) >= 2 * self.y[anchor]

    def _integer(self):
        self.calls += 1
        yv, xv = self.getSolution(self.y_variables), self.getSolution(self.x_variables)
        if len(yv) != len(self.y) or len(xv) != len(self.x):
            raise ValueError("Incomplete integer candidate")
        if any(not math.isfinite(v) or abs(v - round(v)) > 1e-5 for v in (*yv, *xv)):
            raise ValueError("Noninteger value in MIPSOL candidate")
        vertices = [i for i, v in zip(self.vertices, yv) if v > 0.5]
        edges = [e for e, v in zip(self.edges, xv) if v > 0.5]
        components, adjacency = selected_components(vertices, edges, self.depot)
        for component in components:
            if self.depot in component:
                continue
            members = frozenset(component)
            internal, boundary = self.cut_indices(members)
            # Never throttle or permanently deduplicate integer feasibility checks.
            self.addLazyConstr(self.inequality(members, min(members), internal, boundary))
            self.cut_submissions += 1

    def _fractional(self):
        self.relax_calls += 1
        cfg = self.config
        if not cfg.enabled or (self.relax_calls > cfg.initial_rounds and
                               (self.relax_calls - cfg.initial_rounds) % cfg.interval):
            return
        started = time.perf_counter()
        self.fractional_rounds += 1
        try:
            yv, xv = self.getRelaxSol(self.y_variables), self.getRelaxSol(self.x_variables)
            if len(yv) != len(self.y) or len(xv) != len(self.x):
                raise ValueError("Incomplete LP relaxation")
            if any(not math.isfinite(v) or v < -1e-5 or v > 1 + 1e-5 for v in (*yv, *xv)):
                raise ValueError("Invalid LP relaxation values")
            values = dict(zip(self.vertices, yv))
            support = [(i, j, max(0.0, float(v))) for (i, j), v in zip(self.edges, xv)
                       if v > cfg.flow_tolerance]
            candidates = [i for i in self.vertices
                          if i != self.depot and values[i] > cfg.violation_tolerance]
            candidates.sort(key=lambda i: (-values[i], i))
            if not candidates:
                return
            offset = self.anchor_offset % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
            generated, cuts, flows = set(), 0, 0
            for source in candidates:
                if flows >= cfg.max_flows or cuts >= cfg.max_cuts:
                    break
                if flows and cfg.budget_seconds and time.perf_counter() - started >= cfg.budget_seconds:
                    break
                self.flow_calls += 1
                flows += 1
                flow, members = minimum_cut(len(self.y), support, source, self.depot,
                    tolerance=cfg.flow_tolerance,
                    cutoff=max(0.0, 2 * values[source] - cfg.violation_tolerance))
                if members is None:
                    continue
                if source not in members or self.depot in members:
                    raise ValueError("Min-cut returned an invalid shore")
                anchor = max(members, key=lambda i: (values[i], -i))
                key = (members, anchor)
                if key in generated:
                    continue
                generated.add(key)
                internal, boundary = self.cut_indices(members)
                # Evaluate FULL original LP values. Edges omitted from positive
                # flow support remain in both this check and the added row.
                capacity = math.fsum(xv[e] for e in boundary)
                if capacity < 2 * values[anchor] - cfg.violation_tolerance:
                    self.addUserCut(self.inequality(members, anchor, internal, boundary))
                    self.user_cuts += 1
                    cuts += 1
            self.anchor_offset += max(1, flows)
        finally:
            self.fractional_seconds += time.perf_counter() - started

    def callback(self):
        with self.lock:
            if self.error is not None:
                self.interrupt()
                return
            try:
                if self.where() == COPT.CBCONTEXT_MIPSOL:
                    self._integer()
                elif self.where() == COPT.CBCONTEXT_MIPRELAX:
                    self._fractional()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.interrupt()

    def statistics(self):
        return {"lazy_callback_calls": self.calls, "lazy_cut_submissions": self.cut_submissions,
                "fractional_callback_calls": self.relax_calls,
                "fractional_separation_rounds": self.fractional_rounds,
                "fractional_user_cut_submissions": self.user_cuts,
                "fractional_mincut_calls": self.flow_calls,
                "fractional_separation_seconds": self.fractional_seconds}
