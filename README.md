# KAG Agent Mining

A **Knowledge Augmented Generation** system built from scratch (no LangChain GraphRAG, LlamaIndex KG or similar) over a mining-maintenance knowledge base: equipment, components, symptoms, causes and corrective actions.

> **All data used in this project is synthetic.** The four equipment manuals, and every component, symptom, cause and corrective action inside them, were written to simulate realistic industrial-maintenance documentation — none of it was extracted from a real manufacturer's manual or a real mining operation.

Unlike a plain RAG (which searches for similar-sounding text passages), KAG answers questions that require walking a **chain of cause-effect relations** that doesn't live entirely in one passage:

> "The crusher's eccentric shaft is vibrating — what's the most likely cause and the corrective action?"
>
> Requires walking: Equipment → Component → Symptom → Cause → Corrective Action.

## Architecture

```
data/raw/  ──►  src/extraction.py  ──►  data/processed/triples.json
                (LLM extracts triples)        │
                                              ▼
                                    src/graph_builder.py  ──►  graph.graphml (nx.DiGraph)
                                              │
Question ──► src/graph_query.py ──► paths (traversal ≤ 3 hops)
             (LLM only identifies             │
              the starting entity;            ▼
              deterministic specificity  src/answer_builder.py ──► graph-grounded answer
              check asks for detail            │
              when the question is    src/graph_viz.py ──► pyvis graph with the path highlighted
              ambiguous)                        │
                                             app.py (Streamlit)
```

Graph relations: `has_component`, `has_symptom`, `indicates_cause`, `resolved_by`.

If the question doesn't map to any entity in the graph, the system **admits it doesn't have the information** — it never invents a relation. If the question is ambiguous (e.g. it names a component but not the symptom), the agent **asks for the missing detail** instead of guessing.

## How to run

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -q

# set up the API key
copy .env.example .env   # then edit it with your OPENAI_API_KEY

# offline pipeline (uses the LLM to extract the triples)
.venv/Scripts/python -m src.extraction
.venv/Scripts/python -m src.graph_builder

# app
.venv/Scripts/python -m streamlit run app.py
```

## Example questions

- Multi-hop: *"The crusher's eccentric shaft is vibrating — what's the probable cause and the corrective action?"*
- Ambiguous (triggers a clarifying question): *"The crusher's drive motor has a problem."*
- Out of scope: *"How do I change a car tire?"* → the system answers that it doesn't have that information.
