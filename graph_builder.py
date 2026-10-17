"""Loads the hand-curated JSON into a NetworkX MultiDiGraph."""
import json
from pathlib import Path
import networkx as nx

DATA_PATH = Path(__file__).parent / "data" / "graph_data.json"
# Relations with no direction (A conflicts_with B == B conflicts_with A)
SYMMETRIC = {"conflicts_with", "compatible_with", "recommended_with"}


def build_graph(path: Path = DATA_PATH) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text())
    G = nx.MultiDiGraph()  # multi: two libs can have several relations
    for n in data["nodes"]:
        G.add_node(n["id"], versions=n["versions"], category=n["category"],
                   aliases=n.get("aliases", []))
    for src, rel, dst, constraint, note in data["edges"]:
        # `constraint` is a PEP 440 specifier applied to the *target* version
        G.add_edge(src, dst, key=rel, relation=rel, constraint=constraint, note=note)
    G.graph["features"] = data["features"]
    return G
