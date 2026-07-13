"""Knowledge graph construction from the extracted triples.

We use networkx.DiGraph (a DIRECTED graph) because the relations have
semantic direction: "symptom indicates_cause cause" is not the same as the
reverse. Direction is what lets us walk the causal chain in the right order
during a query.

No external graph database in this version: the graph fits in memory and
the goal is to learn the mechanics, not to operate at scale.
"""

import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
TRIPLES_FILE = ROOT / "data" / "processed" / "triples.json"
GRAPH_FILE = ROOT / "data" / "processed" / "graph.graphml"


def build_graph(triples: list[dict]) -> nx.DiGraph:
    """Build the DiGraph: node = entity (with type), edge = relation (with source)."""
    graph = nx.DiGraph()
    for t in triples:
        # add_node is idempotent: the same entity mentioned across several
        # documents becomes ONE single node — this is how knowledge from
        # different sources connects, something isolated chunks in a plain
        # RAG never do.
        graph.add_node(t["origin"], type=t["origin_type"])
        graph.add_node(t["destination"], type=t["destination_type"])
        graph.add_edge(
            t["origin"],
            t["destination"],
            relation_type=t["relation"],
            source=t["source"],
        )
    return graph


def save_graph(graph: nx.DiGraph, path: Path = GRAPH_FILE) -> None:
    """Persist as GraphML — a text-based, readable format, openable by other
    tools (Gephi, yEd), which helps inspect/debug the graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path, encoding="utf-8")


def load_graph(path: Path = GRAPH_FILE) -> nx.DiGraph:
    return nx.read_graphml(path)


def run_graph_construction() -> nx.DiGraph:
    data = json.loads(TRIPLES_FILE.read_text(encoding="utf-8"))
    graph = build_graph(data["triples"])
    save_graph(graph)
    print(
        f"Graph built: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges. Saved to {GRAPH_FILE}"
    )
    return graph


if __name__ == "__main__":
    run_graph_construction()
