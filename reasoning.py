"""Graph retrieval + multi-hop reasoning.
Everything here is deterministic Python. The LLM only *explains* the facts we return."""
import re
from collections import deque
import networkx as nx
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# ---------- small helpers ----------
def _edges(G, node, rel, direction):
    it = G.out_edges(node, keys=True, data=True) if direction == "out" else G.in_edges(node, keys=True, data=True)
    return [(u, v, d) for u, v, _, d in it if d["relation"] == rel]

def _sym_edges(G, node, rel):
    return _edges(G, node, rel, "out") + _edges(G, node, rel, "in")

def fmt(u, v, d):
    c = f"[{d['constraint']}]" if d["constraint"] else ""
    return f"{u} --{d['relation']}{c}--> {v}"

def satisfies(version, spec):
    return Version(version) in SpecifierSet(spec or "")

# ---------- 1. entity linking (question -> graph nodes) ----------
def find_mentions(G, question):
    """Return [(lib, version_or_None)] in order of appearance. Longest alias wins."""
    table = {}
    for n, a in G.nodes(data=True):
        for name in {n, n.replace("-", " "), *a["aliases"]}:
            table[name.lower()] = n
    ql, taken, found = question.lower(), [], []
    for alias in sorted(table, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", ql):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append(m.span())
            # a version right after the name: "numpy 2.0", "numpy to 2.0", "numpy v2"
            vm = re.match(r"\s*(?:to|version|v|==|@)?\s*v?(\d+(?:\.\d+)*)", ql[m.end():])
            found.append((m.start(), table[alias], vm.group(1) if vm else None))
    out = {}
    for _, lib, ver in sorted(found):
        if lib not in out or (out[lib] is None and ver):
            out[lib] = ver
    return list(out.items())

def find_features(G, question):
    ql = question.lower()
    return [(f, v["libs"]) for f, v in G.graph["features"].items()
            if f in ql or any(k in ql for k in v["keywords"])]

# ---------- 2. multi-hop: "what breaks if I upgrade X to Y?" ----------
def _dep_graph(G):
    """Plain directed graph containing only depends_on edges."""
    return nx.DiGraph([(u, v) for u, v, k in G.edges(keys=True) if k == "depends_on"])

def _chain(D, a, b):
    """Shortest depends_on chain a -> ... -> b as a node list, or None."""
    try:
        return nx.shortest_path(D, a, b) if a in D and b in D else None
    except nx.NetworkXNoPath:
        return None

def _context(G, lib, other):
    """How `other` relates to `lib` (used when the question mentions several libraries)."""
    facts, edges, D = [], [], _dep_graph(G)
    found = False
    c = _chain(D, other, lib)
    if c:
        found = True
        facts.append(f"CONTEXT: {other} depends on {lib}" + (" directly" if len(c) == 2 else f" transitively via {' -> '.join(c)}")
                     + f". If any link in that chain appears under BREAKS above, {other} is affected.")
        edges += [(a, b, "depends_on") for a, b in zip(c, c[1:])]
    c = _chain(D, lib, other)
    if c:
        found = True
        facts.append(f"CONTEXT: {lib} depends on {other}" + (" directly" if len(c) == 2 else f" transitively via {' -> '.join(c)}")
                     + f". Upgrading {lib} does not change the {other} version, but {lib} still needs {other} to stay inside its range.")
        edges += [(a, b, "depends_on") for a, b in zip(c, c[1:])]
    for rel in ("recommended_with", "compatible_with", "conflicts_with"):
        for u, v, d in _sym_edges(G, lib, rel):
            if other in (u, v):
                found = True
                facts.append(f"CONTEXT: {fmt(u, v, d)} ({d['note']})")
                edges.append((u, v, rel))
    # 'python' is a hub that nearly everything shares, so it is not an informative link
    shared = sorted(((nx.descendants(D, lib) if lib in D else set())
                     & (nx.descendants(D, other) if other in D else set())) - {"python"})
    for dep in shared:
        found = True
        r = []
        for x in (lib, other):
            if D.has_edge(x, dep):
                r.append(f"{x} needs {G.edges[x, dep, 'depends_on']['constraint'] or 'any version'}")
        facts.append(f"CONTEXT: {lib} and {other} both rely on {dep}" + (f" ({'; '.join(r)})" if r else " (indirectly)")
                     + f", so {dep} must stay inside both ranges.")
        edges += [(x, dep, "depends_on") for x in (lib, other) if D.has_edge(x, dep)]
    if not found:
        facts.append(f"CONTEXT: the graph has no link between {lib} and {other} (no dependency in either direction, no shared dependency, no soft relation).")
    return facts, edges


def upgrade_impact(G, lib, version, others=()):
    facts, edges = [], []
    known = G.nodes[lib]["versions"]
    facts.append(f"TARGET: upgrade {lib} to {version} (known versions: {known})")
    if version not in known:
        facts.append(f"NOTE: {version} is not in the graph's known versions for {lib}.")
    broken, ok = [], []
    for dep, _, d in _edges(G, lib, "depends_on", "in"):          # hop 1: direct dependents
        (ok if satisfies(version, d["constraint"]) else broken).append((dep, d))
    for dep, d in ok:
        facts.append(f"OK: {fmt(dep, lib, d)} accepts {version}")
        edges.append((dep, lib, "depends_on"))
    seen = set()
    for dep, d in broken:
        facts.append(f"BREAKS (direct): {fmt(dep, lib, d)} rejects {version}")
        edges.append((dep, lib, "depends_on"))
        queue = deque([(dep, [dep])])                               # hop 2+: who depends on the broken lib
        while queue:
            node, path = queue.popleft()
            for up, _, d2 in _edges(G, node, "depends_on", "in"):
                if up in seen:
                    continue
                seen.add(up)
                facts.append(f"AT RISK (transitive): {fmt(up, node, d2)}  (chain: {' <- '.join([lib] + path + [up])})")
                edges.append((up, node, "depends_on"))
                queue.append((up, path + [up]))
    if not broken:
        facts.append(f"RESULT: no declared dependent in the graph rejects {lib} {version}.")
    for u, v, d in _sym_edges(G, lib, "conflicts_with"):
        facts.append(f"WARNING: {fmt(u, v, d)} ({d['note']})")
        edges.append((u, v, "conflicts_with"))
    for other in others:                                            # other libraries named in the question
        f2, e2 = _context(G, lib, other)
        facts += f2
        edges += e2
    facts.append("LIMITATION: the graph stores one constraint per edge, not per-version changelogs.")
    return facts, edges

# ---------- 3. multi-hop: closure + constraint solving ----------
def resolve(G, libs, pins=None):
    """Walk depends_on transitively from `libs`, intersect every constraint per package,
    and pick the newest known version satisfying all. Also reports explicit conflicts."""
    pins = pins or {}
    needed = {l: [] for l in libs}                 # node -> [(requirer, spec)]
    edges, seen, queue = [], set(libs), deque(libs)
    while queue:
        n = queue.popleft()
        for u, v, d in _edges(G, n, "depends_on", "out"):
            needed.setdefault(v, []).append((u, d["constraint"]))
            edges.append((u, v, "depends_on"))
            if v not in seen:
                seen.add(v); queue.append(v)
    for lib, ver in pins.items():
        needed.setdefault(lib, []).append(("user pin", f"=={ver}"))

    facts = [f"REQUESTED: {', '.join(libs)}" + (f" | user pins: {pins}" if pins else "")]
    install, conflict = {}, False
    for node, reqs in needed.items():
        specs = [s for _, s in reqs if s]
        who = "; ".join(f"{s} (from {r})" for r, s in reqs if s) or "no constraints"
        cands = [v for v in G.nodes[node]["versions"] if satisfies(v, ",".join(specs))]
        if cands:
            best = max(cands, key=Version)
            install[node] = best
            if specs:
                facts.append(f"RESOLVED {node}: {who} -> viable {cands} -> pick {best}")
        else:
            conflict = True
            facts.append(f"CONFLICT {node}: {who} -> NO known version satisfies all of these")
    for u, v, rel in dict.fromkeys(edges):
        facts.append("PATH: " + fmt(u, v, G.edges[u, v, rel]))
    for a in needed:                               # explicit conflicts_with inside the set
        for u, v, d in _edges(G, a, "conflicts_with", "out"):
            if v in needed:
                facts.append(f"EXPLICIT CONFLICT: {fmt(u, v, d)} ({d['note']})")
                edges.append((u, v, "conflicts_with"))
    facts.append("SUGGESTED INSTALL SET: " + (", ".join(f"{k}=={v}" for k, v in install.items())
                 if not conflict else "none (hard conflict above)"))
    return facts, edges, conflict

def related(G, libs):
    """Soft, informational relations (recommended/compatible/conflicts) around the requested libs."""
    facts, edges = [], []
    for lib in libs:
        for rel in ("recommended_with", "compatible_with", "conflicts_with"):
            for u, v, d in _sym_edges(G, lib, rel):
                facts.append(f"INFO: {fmt(u, v, d)} ({d['note']})")
                edges.append((u, v, rel))
    return list(dict.fromkeys(facts)), edges

# ---------- 4. router: question -> intent -> facts ----------
def analyze(G, question, parsed=None):
    """parsed = output of llm.understand_question (or None -> keyword rules)."""
    if parsed:                                              # LLM-based understanding
        mentions = parsed["mentions"]
        feats = [(f, G.graph["features"][f]["libs"]) for f in parsed["features"]]
        wants_upgrade = parsed["intent"] == "upgrade_impact"
        wants_compat = parsed["intent"] == "compatibility"
        source = "llm"
    else:                                                   # keyword rules (fallback)
        mentions, feats = find_mentions(G, question), find_features(G, question)
        ql = question.lower()
        wants_upgrade = bool(re.search(r"upgrad|updat|bump|break|migrat|downgrad", ql))
        wants_compat = bool(re.search(r"compatib|conflict|work with|together|alongside|clash", ql))
        source = "rules"

    libs = list(dict.fromkeys([m[0] for m in mentions] + [l for _, ls in feats for l in ls]))
    pins = {l: v for l, v in mentions if v}
    target = next(((l, v) for l, v in mentions if v), None)

    if wants_upgrade and target:
        intent = "upgrade_impact"
        facts, edges = upgrade_impact(G, *target, others=[l for l in libs if l != target[0]])
    elif libs:
        intent = "compatibility" if wants_compat else "requirements"
        facts, edges, _ = resolve(G, libs, pins)
        f2, e2 = related(G, libs)
        facts, edges = facts + f2, edges + e2
    else:
        intent, facts, edges = "none", [], []
    return {"intent": intent, "understood_by": source, "mentions": mentions,
            "features": [f for f, _ in feats], "facts": facts,
            "edges": list(dict.fromkeys(edges))}