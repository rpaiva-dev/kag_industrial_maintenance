# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: KAG Agent Mining

A Knowledge Augmented Generation (KAG) system built from SCRATCH, with no ready-made knowledge-graph framework (no LangChain GraphRAG, no LlamaIndex KG, etc). The goal is to learn entity/relation extraction, knowledge graph construction and multi-hop reasoning by implementing every piece by hand.

Domain: maintenance of **mining** assets — currently 4 asset types: vibrating screen, belt conveyor, off-road haul truck and jaw crusher. **All data in this project is synthetic** — the manuals, components, symptoms, causes and corrective actions were written to simulate realistic maintenance documentation; none of it comes from a real manufacturer manual or a real mining operation. Unlike plain RAG (text-similarity search), the system answers questions that require walking a CHAIN of cause-effect relations: Equipment → Component → Symptom → Cause → Corrective Action.

When the user's question isn't specific enough to pick a single path (e.g. it names a component but not the symptom, or the equipment but not the component), the agent ASKS for more detail instead of guessing — that check is deterministic, based on the graph's structure (`src/graph_query.py::check_specificity`), not a judgment call made by the LLM.

## Commands

```powershell
# Setup (once)
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip -q
.venv/Scripts/python -m pip install -r requirements.txt -q

# Offline pipeline: extract triples from the documents and build the graph
.venv/Scripts/python -m src.extraction     # data/raw -> data/processed/triples.json
.venv/Scripts/python -m src.graph_builder  # triples.json -> data/processed/graph.graphml

# Run the app
.venv/Scripts/python -m streamlit run app.py --server.headless true
```

Always use the `.venv` interpreter (`.venv/Scripts/python` on Windows), never the system Python.

The API key goes in a local `.env` (`OPENAI_API_KEY=...`, see `.env.example`) — never hardcoded. In Streamlit production, use `st.secrets`.

## Architecture

Two-phase pipeline:

**Offline phase (indexing):**
1. `data/raw/maintenance_manuals.json` — synthetic mining-maintenance manuals (JSON with source metadata)
2. `src/extraction.py` — an LLM (via the `openai` lib) extracts `(origin, relation, destination)` triples with entity types; allowed relations: `has_component`, `has_symptom`, `indicates_cause`, `resolved_by`. Output validated into `data/processed/triples.json`
3. `src/graph_builder.py` — builds an `nx.DiGraph` (nodes with a `type` attribute, edges with `relation_type` and `source`) and persists it to `.graphml`

**Online phase (query, orchestrated by `app.py`):**
4. `src/graph_query.py` — the LLM identifies the starting entity in the question; a deterministic specificity check asks for clarification when the entity has more than one possible next step; a limited traversal (up to 3 hops) collects the relevant paths as ordered lists of triples
5. `src/answer_builder.py` — final prompt built from the graph paths, instructing the model to answer ONLY based on the graph's relations, making the path explicit. If there's no starting entity, the system admits it doesn't have the information — it never invents a relation
6. `src/graph_viz.py` — pyvis visualization of the graph highlighting the path used in the last answer, plus a step-by-step trace animation played while the answer is being generated
7. `app.py` — Streamlit front-end: chat-style UI with per-conversation history in `st.session_state`, an About page explaining the pipeline, and the live graph-trace animation

`src/llm_client.py` centralizes every call to the OpenAI API (model `gpt-4o`).

## Conventions

- All code (comments, docstrings, identifiers, prompts) is written in English
- No KG frameworks — only `openai`, `networkx`, `pyvis`, `streamlit`
- The `index.html` file must only be created/modified via the `format_index_to_landing_page` skill (`/format_index_to_landing_page` command), and only when explicitly requested
