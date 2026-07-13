"""Graph query: starting-entity identification + multi-hop reasoning.

This is the heart of the difference from a regular RAG:
- RAG: embed the question, search for similar chunks, return loose text.
- KAG: (1) the LLM only does the "linking" — mapping the question to ONE
  node in the graph; (2) the TRAVERSAL is done by a deterministic algorithm
  (DFS), not by the LLM. Every hop follows an edge that really exists, so
  the chain symptom -> cause -> action is guaranteed to be backed by data.

On top of that, the system asks the user for more information when the
question isn't specific enough — for example, asking about a COMPONENT
without stating the SYMPTOM. That check is also deterministic: we look at
how many distinct "next steps" exist in the graph from the identified
entity. If there's more than one, the question is ambiguous and we ask the
user to choose, instead of guessing.
"""

import json

import networkx as nx

from src.llm_client import call_llm

# Hop limit: 3 covers the chain symptom -> cause -> action (with one hop to
# spare when starting from a component). With no limit, the traversal would
# bring back the whole graph and drown the final prompt in irrelevant context.
MAX_HOPS = 3

ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        # anyOf with null: the LLM needs to be able to say "no entity
        # matches" — that's what lets the system admit it doesn't know,
        # instead of forcing a bad match and hallucinating an answer.
        "entity": {"anyOf": [{"type": "string"}, {"type": "null"}]}
    },
    "required": ["entity"],
    "additionalProperties": False,
}


def identify_starting_entity(question: str, graph: nx.DiGraph) -> str | None:
    """Use the LLM to map the question to an existing node in the graph.

    We pass the full list of nodes (with type) in the prompt — feasible
    because the graph is small. The LLM picks among REAL options, never
    creates new names; we still validate the answer is an actual node.
    """
    lines = [
        f"- {node} (type: {data.get('type', '?')})" for node, data in graph.nodes(data=True)
    ]
    response = call_llm(
        system=(
            "You map questions about maintenance of mining assets "
            "(vibrating screens, belt conveyors, off-road haul trucks, "
            "jaw crushers) to entities in a knowledge graph. Given the "
            "question and the list of entities, return the EXACT name of "
            "the MOST SPECIFIC entity mentioned in the question as the "
            "starting point of the reasoning: prefer a symptom if one is "
            "mentioned; if there's no symptom but a component is "
            "mentioned, return the component; if only the equipment is "
            "mentioned, return the equipment.\n\n"
            "CRITICAL RULE — never switch equipment: if the question "
            "explicitly mentions a type of equipment (screen, conveyor, "
            "truck or crusher), the chosen entity MUST belong to THAT "
            "equipment. Never pick an entity from a different piece of "
            "equipment just because the name sounds like a match (e.g. if "
            "the question is about the crusher, never return an entity "
            "that only exists on the screen, even if the symptom word is "
            "similar). If no entity from the correct equipment matches the "
            "described symptom, prefer returning the correct component or "
            "equipment (even without a matching symptom) over returning an "
            "entity from another piece of equipment; if there's nothing "
            "like that either, return null.\n\n"
            "If the question spans more than one conversation turn "
            "(previous context + the user's new answer), combine that "
            "information to find the most specific entity possible. If "
            "NO entity in the list matches the subject of the question, "
            "return null. Never invent a name outside the list."
        ),
        user=f"Question: {question}\n\nGraph entities:\n" + "\n".join(lines),
        json_schema=ENTITY_SCHEMA,
    )
    entity = json.loads(response)["entity"]
    # Belt and suspenders: only accept it if it's actually a node in the graph.
    if entity is not None and entity in graph:
        return entity
    return None


def _successors_by_relation(graph: nx.DiGraph, node: str, relation: str) -> list[str]:
    """Outgoing neighbors of `node` connected specifically by `relation`."""
    return sorted(
        v for v in graph.successors(node) if graph.edges[node, v]["relation_type"] == relation
    )


def check_specificity(entity: str, graph: nx.DiGraph) -> dict:
    """Deterministically check whether the entity is specific enough.

    Core idea: if the next step in the causal chain from the entity has
    MORE THAN ONE option (e.g. a component with several possible symptoms,
    or a piece of equipment with several possible components), the question
    didn't give enough information to choose which path to follow — so we
    stop here and return the options for the user to choose, instead of
    letting the LLM guess (and possibly cite a cause that isn't the user's
    actual problem).

    Symptoms with multiple causes, or causes with multiple actions, do NOT
    count as ambiguity: at that level it's normal and useful to list every
    possibility in the final answer (answer_builder already does this well).
    """
    node_type = graph.nodes[entity].get("type")

    if node_type == "equipment":
        candidates = _successors_by_relation(graph, entity, "has_component")
        if len(candidates) > 1:
            return {
                "needs_clarification": True,
                "ambiguous_entity": entity,
                "ambiguous_relation": "has_component",
                "candidates": candidates,
                "clarifying_question": (
                    f"The **{entity}** has several components that could be "
                    f"involved: {', '.join(candidates)}. Which component is "
                    f"showing the problem?"
                ),
            }
        if len(candidates) == 1:
            # Only one possible component: resolve automatically and go one
            # level deeper to check whether the COMPONENT, in turn, is specific.
            return check_specificity(candidates[0], graph)
        return {"needs_clarification": False, "resolved_entity": entity}

    if node_type == "component":
        candidates = _successors_by_relation(graph, entity, "has_symptom")
        if len(candidates) > 1:
            return {
                "needs_clarification": True,
                "ambiguous_entity": entity,
                "ambiguous_relation": "has_symptom",
                "candidates": candidates,
                "clarifying_question": (
                    f"The **{entity}** can present different symptoms: "
                    f"{', '.join(candidates)}. Which of these symptoms are "
                    f"you observing?"
                ),
            }
        return {"needs_clarification": False, "resolved_entity": entity}

    # symptom, cause or action: already the most specific level possible —
    # nothing to clarify, proceed to normal traversal.
    return {"needs_clarification": False, "resolved_entity": entity}


def collect_paths(graph: nx.DiGraph, start: str, max_hops: int = MAX_HOPS) -> list[list[dict]]:
    """Walk the graph from the entity and return paths as lists of triples.

    Strategy: depth-limited DFS following OUTGOING edges (the causal
    direction: symptom -> cause -> action). Each returned path is an
    ordered sequence of triples — exactly the "evidence" the LLM will use
    to answer. We also include 1 hop of INCOMING context (who points to the
    start), to give context: e.g. which component shows the symptom.
    """
    paths: list[list[dict]] = []

    def dfs(node: str, current_path: list[dict], visited: set[str]):
        successors = [v for v in graph.successors(node) if v not in visited]
        # Leaf node or hop limit reached: close the accumulated path.
        if not successors or len(current_path) >= max_hops:
            if current_path:
                paths.append(list(current_path))
            return
        for v in successors:
            edge = graph.edges[node, v]
            triple = {
                "origin": node,
                "relation": edge["relation_type"],
                "destination": v,
                "source": edge.get("source", ""),
            }
            current_path.append(triple)
            dfs(v, current_path, visited | {v})
            current_path.pop()

    dfs(start, [], {start})

    # Reverse context (1 hop): e.g. "crusher drive motor has_symptom
    # overcurrent" when the start is the symptom. Helps the answer mention
    # the affected component.
    for u in graph.predecessors(start):
        edge = graph.edges[u, start]
        paths.append([
            {
                "origin": u,
                "relation": edge["relation_type"],
                "destination": start,
                "source": edge.get("source", ""),
            }
        ])

    return paths


def _paths_from_options(graph: nx.DiGraph, check: dict) -> list[list[dict]]:
    """Build 1-hop 'paths' representing the candidate options.

    Used only to visually display, in the chat subgraph, the points the
    user needs to choose between — this doesn't count as an answer.
    """
    node = check["ambiguous_entity"]
    relation = check["ambiguous_relation"]
    return [
        [
            {
                "origin": node,
                "relation": relation,
                "destination": c,
                "source": graph.edges[node, c].get("source", ""),
            }
        ]
        for c in check["candidates"]
    ]


def query(question: str, graph: nx.DiGraph, previous_context: str | None = None) -> dict:
    """Orchestrate the query: linking + specificity check + traversal.

    previous_context: when the assistant's previous reply was a
    clarification request, we pass the user's original question along with
    their new answer, so the LLM identifies the entity by combining both
    pieces of information (e.g. "crusher drive motor" + "overcurrent").
    """
    effective_question = f"{previous_context}\n{question}" if previous_context else question
    start = identify_starting_entity(effective_question, graph)

    if start is None:
        return {
            "start_entity": None,
            "paths": [],
            "needs_clarification": False,
            "clarifying_question": None,
        }

    check = check_specificity(start, graph)

    if check["needs_clarification"]:
        return {
            "start_entity": check["ambiguous_entity"],
            "paths": _paths_from_options(graph, check),
            "needs_clarification": True,
            "clarifying_question": check["clarifying_question"],
        }

    resolved_entity = check["resolved_entity"]
    return {
        "start_entity": resolved_entity,
        "paths": collect_paths(graph, resolved_entity),
        "needs_clarification": False,
        "clarifying_question": None,
    }
