"""Entity and relation extraction (triples) from the raw source documents.

Fundamental difference from a regular RAG: instead of slicing text into
chunks and indexing by similarity, the LLM READS each document and turns it
into STRUCTURED knowledge — triples (origin, relation, destination) with
entity types. That structure is what later enables multi-hop reasoning:
following the chain equipment -> component -> symptom -> cause -> action,
something similarity search never does, because the chain rarely lives
entirely inside a single passage.
"""

import json
from pathlib import Path

from src.llm_client import call_llm

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "maintenance_manuals.json"
TRIPLES_FILE = ROOT / "data" / "processed" / "triples.json"

# Closed vocabulary of relations and types. An open vocabulary would let the
# LLM invent variations ("has-component", "has_part"...) and the graph would
# end up fragmented — nodes that should connect wouldn't connect.
VALID_RELATIONS = {"has_component", "has_symptom", "indicates_cause", "resolved_by"}
VALID_TYPES = {"equipment", "component", "symptom", "cause", "action"}

# Semantic coherence: each relation only makes sense between certain types.
# Validating this blocks structural hallucinations (e.g. symptom -has_component-> cause).
RELATION_DOMAIN = {
    "has_component": ("equipment", "component"),
    "has_symptom": ("component", "symptom"),
    "indicates_cause": ("symptom", "cause"),
    "resolved_by": ("cause", "action"),
}

# Schema for structured outputs: the API guarantees valid JSON in this
# shape, eliminating the "the LLM returned broken JSON" class of error.
TRIPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "origin_type": {"type": "string", "enum": sorted(VALID_TYPES)},
                    "relation": {"type": "string", "enum": sorted(VALID_RELATIONS)},
                    "destination": {"type": "string"},
                    "destination_type": {"type": "string", "enum": sorted(VALID_TYPES)},
                },
                "required": ["origin", "origin_type", "relation", "destination", "destination_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a knowledge extractor for industrial maintenance manuals.

Your task: read the document and extract ALL triples (origin, relation, destination) present in the text, using ONLY these relations:

- has_component: equipment -> component
- has_symptom: component -> symptom
- indicates_cause: symptom -> cause
- resolved_by: cause -> action

Rules:
1. Extract only what is EXPLICIT in the text. Never invent a relation.
2. Use lowercase names for entities. For symptoms, causes and actions, PRESERVE any reference to the component/equipment present in the text (e.g. "metallic noise in the screen eccentric", NEVER shorten it to just "metallic noise"). This is critical: a name that's too generic and short makes symptoms from DIFFERENT PIECES OF EQUIPMENT collide into the SAME graph node by mistake — which would make the system answer with data from the wrong machine. For components and equipment, use exactly the (already qualified) name as it appears in the text.
3. A symptom may indicate more than one cause, and a cause may have more than one action — extract all of them.
4. Respect the direction of the relation (origin -> destination) as defined above."""


def extract_triples_from_document(text: str, source: str) -> list[dict]:
    """Call the LLM for one document and return the validated triples."""
    response = call_llm(
        system=SYSTEM_PROMPT,
        user=f"Document (source: {source}):\n\n{text}",
        json_schema=TRIPLES_SCHEMA,
    )
    data = json.loads(response)

    valid_triples = []
    for t in data["triples"]:
        # Even with the schema guaranteeing the shape, we validate the
        # SEMANTICS: the relation needs to connect the right types.
        expected = RELATION_DOMAIN.get(t["relation"])
        if expected != (t["origin_type"], t["destination_type"]):
            print(f"  [discarded] incoherent triple: {t}")
            continue
        t["origin"] = t["origin"].strip().lower()
        t["destination"] = t["destination"].strip().lower()
        t["source"] = source  # traceability: which document the relation came from
        valid_triples.append(t)
    return valid_triples


def run_extraction() -> list[dict]:
    """Full pipeline: read data/raw, extract from each document, save the JSON."""
    data = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    all_triples: list[dict] = []
    for doc in data["documents"]:
        print(f"Extracting triples from: {doc['source']}")
        triples = extract_triples_from_document(doc["text"], doc["source"])
        print(f"  {len(triples)} triples extracted")
        all_triples.extend(triples)

    TRIPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRIPLES_FILE.write_text(
        json.dumps({"triples": all_triples}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTotal: {len(all_triples)} triples saved to {TRIPLES_FILE}")
    return all_triples


if __name__ == "__main__":
    run_extraction()
