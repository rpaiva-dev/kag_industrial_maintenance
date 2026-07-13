"""Final answer generation from the paths found in the graph.

The final prompt is deliberately RESTRICTIVE: the LLM can only use the
relations provided (the triples in the paths), never its own knowledge.
This is the opposite of a regular RAG, where the LLM receives loose text
and can "fill in" with what it knows. Here, if the graph doesn't have the
information, the answer is "I don't know" — groundedness comes from the
structure, not from the model's good will.
"""

from src.llm_client import call_llm

NO_INFORMATION_MESSAGE = (
    "I couldn't find that information in the knowledge base. "
    "The graph has no entity related to your question, so I'd rather "
    "admit I don't know than make something up."
)

SYSTEM_PROMPT = """You are an industrial maintenance assistant that answers EXCLUSIVELY based on a knowledge graph.

You will receive the user's question and the paths walked in the graph (sequences of origin -relation-> destination triples, with the source document of each relation).

MANDATORY rules:
1. Use ONLY the relations provided. Do not add causes, actions or facts that aren't in the triples, even if you happen to know them.
2. Make the graph PATH that supports the conclusion explicit in the answer, in the format: entity -[relation]-> entity -[relation]-> entity.
3. If the provided paths aren't enough to answer, say clearly that the knowledge base doesn't cover that point.
4. Cite the source document ("source" field) of the relations used.
5. Answer in English, clearly and directly: probable cause(s) first, then the corrective action(s)."""


def format_paths(paths: list[list[dict]]) -> str:
    """Serialize the paths into readable text for the prompt.

    "a -[rel]-> b -[rel]-> c" format instead of raw JSON: more compact, and
    the model mirrors this format in the answer when it makes the path explicit.
    """
    lines = []
    for i, path in enumerate(paths, 1):
        steps = " ".join(
            f"{t['origin']} -[{t['relation']}]-> {t['destination']}" if j == 0
            else f"-[{t['relation']}]-> {t['destination']}"
            for j, t in enumerate(path)
        )
        sources = sorted({t["source"] for t in path if t.get("source")})
        lines.append(f"Path {i}: {steps}\n  Sources: {'; '.join(sources)}")
    return "\n".join(lines)


def generate_answer(question: str, query_result: dict) -> str:
    """Build the final prompt and generate the graph-grounded answer."""
    if query_result["start_entity"] is None:
        # Short-circuit WITHOUT calling the LLM: if there's no starting
        # entity, there's no evidence at all — any answer would be made up.
        return NO_INFORMATION_MESSAGE

    paths_text = format_paths(query_result["paths"])
    user = (
        f"User question: {question}\n\n"
        f"Starting entity in the graph: {query_result['start_entity']}\n\n"
        f"Paths found in the graph:\n{paths_text}"
    )
    return call_llm(system=SYSTEM_PROMPT, user=user)
