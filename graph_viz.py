"""Matplotlib drawing helpers for subgraphs."""
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

COLORS = {"depends_on": "#4a6fa5", "conflicts_with": "#d64545",
          "recommended_with": "#3a9d5d", "compatible_with": "#e0952b"}
HOP_COLORS = ["#ffd966", "#cfe0f5", "#e6edf7", "#f3f5f9"]   # center, hop1, hop2, hop3
MAX_LABELS = 14   # skip version labels when there would be too many to read


def ego_edges(G, node, radius=1, relations=None, hide=()):
    """Edges within `radius` hops of `node`.
    - relations: only follow these relation types (None = all)
    - hide: node ids to drop entirely (e.g. {"python"}, a hub that clutters everything)
    Edges between two outermost nodes are dropped, so extra hops expand outward cleanly."""
    U = nx.Graph()
    U.add_nodes_from(n for n in G.nodes if n not in hide or n == node)
    for u, v, k in G.edges(keys=True):
        if u in U and v in U and (relations is None or k in relations):
            U.add_edge(u, v)
    dist = nx.single_source_shortest_path_length(U, node, cutoff=radius)
    return [(u, v, k) for u, v, k in G.edges(keys=True)
            if u in dist and v in dist and U.has_edge(u, v)
            and (relations is None or k in relations)
            and min(dist[u], dist[v]) < radius]


def _layout(H, center):
    """Concentric rings around `center` (ring = number of hops). Falls back to spring layout."""
    if center in H:
        dist = nx.single_source_shortest_path_length(H.to_undirected(), center)
        rings = {}
        for n, d in dist.items():
            rings.setdefault(d, []).append(n)
        shells = [sorted(rings[d]) for d in sorted(rings)]
        rest = sorted(set(H.nodes) - set(dist))
        if rest:
            shells.append(rest)
        return nx.shell_layout(H, nlist=shells), dist
    return nx.spring_layout(H, seed=42, k=1.4, iterations=100), {}


def draw_graph(G, edges, figsize=(6, 5), center=None, show_constraints=False):
    """edges: list of (u, v, relation). center: node to put in the middle."""
    fig, ax = plt.subplots(figsize=figsize)
    H = nx.DiGraph()
    for u, v, rel in edges:
        H.add_edge(u, v, rel=rel)
    pos, dist = _layout(H, center)
    size = 1500 if len(H) <= 8 else 1000

    colors = [HOP_COLORS[min(dist.get(n, 3), 3)] if dist else "#eef2f7" for n in H.nodes]
    nx.draw_networkx_nodes(H, pos, node_color=colors, edgecolors="#555", node_size=size, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)

    for rel, color in COLORS.items():
        sub = [(u, v) for u, v, d in H.edges(data=True) if d["rel"] == rel]
        if sub:
            nx.draw_networkx_edges(H, pos, edgelist=sub, edge_color=color, arrows=True,
                                   arrowsize=12, node_size=size, width=1.4, alpha=0.85, ax=ax,
                                   connectionstyle="arc3,rad=0.1")

    if show_constraints:
        labels = {(u, v): G.edges[u, v, "depends_on"]["constraint"]
                  for u, v, d in H.edges(data=True)
                  if d["rel"] == "depends_on" and G.edges[u, v, "depends_on"]["constraint"]}
        if 0 < len(labels) <= MAX_LABELS:
            nx.draw_networkx_edge_labels(H, pos, edge_labels=labels, font_size=7, label_pos=0.35,
                                         rotate=False, ax=ax,
                                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    used = {d["rel"] for _, _, d in H.edges(data=True)}
    handles = [Line2D([0], [0], color=c, lw=2, label=r) for r, c in COLORS.items() if r in used]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    return fig