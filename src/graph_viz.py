"""Graph visualization, highlighting the path used in the last answer.

We use pyvis (an interactive HTML network) instead of a static matplotlib
image because, for a portfolio, being able to drag/zoom and SEE the
highlighted reasoning communicates far more than a frozen picture. The
generated HTML is embedded in Streamlit via st.components.v1.html.

This module also generates the TRACE ANIMATION: while the agent is
answering, we show the subgraph "building itself" node by node, in the
order the traversal (BFS from the starting entity) reaches them —
visually simulating the algorithm in graph_query.py exploring the
knowledge base. This uses a minimal custom HTML/JS page (not the pyvis
template), because we need to progressively add nodes/edges to vis-network
via setTimeout, and pyvis only generates complete, static datasets.
"""

import json
from collections import deque

import networkx as nx
from pyvis.network import Network

# Colors per entity type — the visual legend of the causal chain.
TYPE_COLORS = {
    "equipment": "#4e79a7",
    "component": "#59a14f",
    "symptom": "#f28e2b",
    "cause": "#e15759",
    "action": "#b07aa1",
}
DEFAULT_COLOR = "#9c9c9c"
HIGHLIGHT_COLOR = "#ffd700"  # gold: nodes/edges on the path used in the answer


def subgraph_from_paths(graph: nx.DiGraph, paths: list[list[dict]]) -> nx.DiGraph:
    """Build a graph containing ONLY the nodes/edges walked for the answer.

    Why a subgraph instead of the full graph with highlights? For the end
    user, what matters is seeing the points the query connected in the
    knowledge base — showing all ~100 entities would just be visual noise.
    The subgraph is the isolated "reasoning trail".
    """
    sub = nx.DiGraph()
    for path in paths:
        for t in path:
            for node in (t["origin"], t["destination"]):
                # Copy the "type" attribute from the original graph to keep the colors.
                sub.add_node(node, type=graph.nodes[node].get("type", "?") if node in graph else "?")
            sub.add_edge(t["origin"], t["destination"], relation_type=t["relation"], source=t.get("source", ""))
    return sub


def generate_graph_html(graph: nx.DiGraph, paths: list[list[dict]] | None = None) -> str:
    """Generate the network's HTML, highlighting the paths (if provided)."""
    # Sets of nodes/edges to highlight, derived from the paths' triples.
    highlighted_nodes: set[str] = set()
    highlighted_edges: set[tuple[str, str]] = set()
    for path in paths or []:
        for t in path:
            highlighted_nodes.update([t["origin"], t["destination"]])
            highlighted_edges.add((t["origin"], t["destination"]))

    net = Network(height="550px", width="100%", directed=True, bgcolor="#ffffff")
    net.barnes_hut()  # reasonable default physics for small graphs

    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "?")
        highlighted = node in highlighted_nodes
        net.add_node(
            node,
            label=node,
            title=f"{node} ({node_type})",
            color=TYPE_COLORS.get(node_type, DEFAULT_COLOR),
            # Gold border + larger size signal the reasoning path without
            # losing the type color (which carries the chain's semantics).
            borderWidth=4 if highlighted else 1,
            size=28 if highlighted else 16,
            **({"borderWidthSelected": 4, "shapeProperties": {"borderDashes": False}} if highlighted else {}),
        )
        if highlighted:
            net.get_node(node)["color"] = {
                "background": TYPE_COLORS.get(node_type, DEFAULT_COLOR),
                "border": HIGHLIGHT_COLOR,
            }

    for u, v, data in graph.edges(data=True):
        is_highlighted = (u, v) in highlighted_edges
        net.add_edge(
            u,
            v,
            label=data.get("relation_type", ""),
            color=HIGHLIGHT_COLOR if is_highlighted else "#c0c0c0",
            width=4 if is_highlighted else 1,
            font={"size": 10, "align": "middle"},
        )

    # generate_html returns the full page as a string — this avoids writing
    # a temporary file to disk just to read it back.
    return net.generate_html(notebook=False)


def generate_trace_steps(
    subgraph: nx.DiGraph, root: str, max_nodes: int = 8
) -> list[dict]:
    """Generate the ordered sequence of steps (node, edge, node, edge...) the
    animation will reveal, exploring the subgraph in BFS order from the root.

    BFS (closest first) instead of the raw triple order because that's what
    an "inside-out" exploration from the starting entity would really look
    like — and it reuses incoming and outgoing edges (reverse context +
    causal chain) in the order they'd be discovered from the root, not in
    the order they happen to appear in the triples.
    """
    if root not in subgraph:
        return []

    visited = {root}
    queue = deque([root])
    steps = [
        {
            "type": "node",
            "id": root,
            "label": root,
            "color": TYPE_COLORS.get(subgraph.nodes[root].get("type"), DEFAULT_COLOR),
        }
    ]

    while queue and len(visited) < max_nodes:
        current = queue.popleft()
        neighbors = list(subgraph.successors(current)) + list(subgraph.predecessors(current))
        for v in neighbors:
            if v in visited or len(visited) >= max_nodes:
                continue
            visited.add(v)
            queue.append(v)
            # The edge may exist in either direction between current and v.
            src, dst = (current, v) if subgraph.has_edge(current, v) else (v, current)
            steps.append(
                {
                    "type": "edge",
                    "id": f"{src}=>{dst}",
                    "from": src,
                    "to": dst,
                    "label": subgraph.edges[src, dst]["relation_type"],
                }
            )
            steps.append(
                {
                    "type": "node",
                    "id": v,
                    "label": v,
                    "color": TYPE_COLORS.get(subgraph.nodes[v].get("type"), DEFAULT_COLOR),
                }
            )

    return steps


def generate_trace_html(
    graph: nx.DiGraph,
    paths: list[list[dict]],
    root: str,
    final_message: str = "Path found.",
    step_duration_ms: int = 450,
    max_nodes: int = 8,
) -> tuple[str, int]:
    """Generate the trace animation and return (html, number of steps).

    The step count is returned so the caller can estimate how long the
    animation takes (and, if needed, wait that long before swapping it for
    the final answer — see app.py).
    """
    sub = subgraph_from_paths(graph, paths)
    steps = generate_trace_steps(sub, root, max_nodes=max_nodes)
    steps_json = json.dumps(steps, ensure_ascii=False)
    total_duration_ms = len(steps) * step_duration_ms

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; font-family: -apple-system, Segoe UI, sans-serif; background: #ffffff; }}
  #network {{ width: 100%; height: 400px; }}
  #status {{
    text-align: center; font-size: 13px; color: #888; padding: 8px 0 2px 0;
    transition: color .3s ease;
  }}
  #status.done {{ color: #2a9d5c; font-weight: 600; }}
</style>
</head>
<body>
<div id="network"></div>
<div id="status">🔎 Walking the knowledge graph...</div>
<script>
  const steps = {steps_json};
  const DELAY = {step_duration_ms};

  const nodes = new vis.DataSet([]);
  const edges = new vis.DataSet([]);
  const container = document.getElementById('network');
  const network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, {{
    physics: {{ stabilization: {{ iterations: 80 }}, barnesHut: {{ springLength: 120 }} }},
    layout: {{ improvedLayout: true }},
    interaction: {{ dragNodes: true, zoomView: true }},
    edges: {{ arrows: 'to', font: {{ size: 11, align: 'middle' }}, smooth: {{ type: 'continuous' }} }},
    nodes: {{ shape: 'dot', font: {{ size: 13 }} }},
  }});

  steps.forEach((step, i) => {{
    setTimeout(() => {{
      if (step.type === 'node') {{
        // Gold flash when the node is "discovered", then settles into its
        // type color — visually communicates "just arrived here" -> "known node".
        nodes.add({{ id: step.id, label: step.label, color: '#ffd700', borderWidth: 4, size: 24 }});
        setTimeout(() => {{
          try {{ nodes.update({{ id: step.id, color: step.color, borderWidth: 2, size: 16 }}); }} catch (e) {{}}
        }}, Math.max(DELAY - 120, 100));
      }} else {{
        edges.add({{ id: step.id, from: step.from, to: step.to, label: step.label, color: {{ color: '#ffd700' }}, width: 3 }});
      }}
    }}, i * DELAY);
  }});

  setTimeout(() => {{
    const st = document.getElementById('status');
    if (st) {{
      st.textContent = {json.dumps("✅ " + final_message)};
      st.classList.add('done');
    }}
  }}, Math.max(steps.length, 1) * DELAY + 150);
</script>
</body>
</html>"""

    return html, len(steps)
