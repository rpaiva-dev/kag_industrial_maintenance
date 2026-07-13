"""Streamlit front-end for KAG Agent Mining (all mining data is synthetic).

LLM-style layout (ChatGPT/Claude):
- Sidebar with the "KAG Agent Mining" logo, a new-chat button and the list
  of session conversations (each conversation keeps its own history).
- Config menu with an "About" option: explains the pipeline step by step,
  lists the monitored assets (with description and components) and shows
  the FULL knowledge graph.
- In the chat, each answer only shows the SUBGRAPH of the points the query
  connected in the knowledge base — the reasoning trail, without the noise
  of the full graph.
- When the question isn't specific enough (e.g. it names a component but
  not the symptom), the agent ASKS for more detail instead of guessing, and
  the user's next message is interpreted together with the original
  question (clarification context).

Flow for every question:
1. graph_query.query          -> the LLM identifies the starting entity;
   deterministic specificity check; traversal (up to 3 hops).
2. If clarification is needed -> show the question and the options, without
   calling the answer LLM (there isn't enough "evidence" yet to answer).
3. Otherwise, answer_builder.generate_answer -> the LLM answers based
   SOLELY on the paths found.
"""

import json
import time
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.answer_builder import generate_answer
from src.graph_builder import GRAPH_FILE, load_graph, run_graph_construction
from src.graph_query import query
from src.graph_viz import generate_graph_html, generate_trace_html, subgraph_from_paths

st.set_page_config(page_title="KAG Agent Mining", page_icon="🛠️", layout="wide")

RAW_FILE = Path(__file__).resolve().parent / "data" / "raw" / "maintenance_manuals.json"

# Duration (s) of each trace-animation step, mirroring the default value
# used inside generate_trace_html — we need it here to estimate how long to
# wait before swapping the animation for the final answer.
ANIMATION_STEP_DURATION_S = 0.45


@st.cache_resource
def get_graph():
    """Load the graph from disk once per server session.

    cache_resource (not cache_data) because the DiGraph is a mutable shared
    object, not serializable data to be copied on every rerun.
    """
    if not GRAPH_FILE.exists():
        from src.graph_builder import TRIPLES_FILE

        if TRIPLES_FILE.exists():
            return run_graph_construction()
        st.error(
            "Graph not found. Run the offline pipeline first:\n\n"
            "`python -m src.extraction` and then `python -m src.graph_builder`"
        )
        st.stop()
    return load_graph()


@st.cache_data
def get_monitored_assets() -> list[dict]:
    """Read the display metadata (description + components) straight from the raw data.

    We don't use the graph's nodes to build this list because the exact
    name the LLM extracts can vary slightly from the source text — this
    metadata is the stable, reliable source for the machine descriptions.
    """
    data = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    return [
        {
            "source": doc["source"],
            "category": doc.get("category", ""),
            "description": doc.get("machine_description", ""),
            "components": doc.get("main_components", []),
        }
        for doc in data["documents"]
    ]


graph = get_graph()

# ---------------------------------------------------------------------------
# Session state: multiple conversations, like LLM interfaces.
# Shape: {id: {"title": str, "messages": [{question, answer, paths, clarification}]}}
# ---------------------------------------------------------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None
if "page" not in st.session_state:
    st.session_state.page = "chat"  # "chat" | "about"


def new_conversation():
    """Create an empty conversation and make it active (like 'New chat')."""
    cid = str(uuid.uuid4())[:8]
    st.session_state.conversations[cid] = {"title": "New chat", "messages": []}
    st.session_state.active_conversation = cid
    st.session_state.page = "chat"


# Make sure there's always at least one open conversation.
if not st.session_state.conversations:
    new_conversation()

# ---------------------------------------------------------------------------
# SIDEBAR — logo, new chat, conversation list and Config.
# ---------------------------------------------------------------------------
with st.sidebar:
    # "Logo" in the top-left corner, like LLM interfaces.
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
          <div style="font-size:30px;">🛠️</div>
          <div>
            <div style="font-size:19px;font-weight:700;line-height:1.1;">KAG Agent Mining</div>
            <div style="font-size:12px;opacity:.65;">Knowledge Augmented Generation · synthetic data</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("➕ New chat", use_container_width=True):
        new_conversation()
        st.rerun()

    st.markdown("**Conversations**")
    # List of conversations (most recent first), clickable like LLM UIs.
    # Each one has a three-dot (⋮) button with rename and delete options —
    # the popover acts as the context menu.
    for cid, conv in reversed(list(st.session_state.conversations.items())):
        is_active = cid == st.session_state.active_conversation and st.session_state.page == "chat"
        label = ("🟢 " if is_active else "") + conv["title"][:32]

        col_name, col_menu = st.columns([5, 1])
        with col_name:
            if st.button(label, key=f"open_{cid}", use_container_width=True):
                st.session_state.active_conversation = cid
                st.session_state.page = "chat"
                st.rerun()
        with col_menu:
            with st.popover("⋮", use_container_width=True, key=f"menu_{cid}"):
                new_title = st.text_input(
                    "Rename chat", value=conv["title"], key=f"rename_input_{cid}"
                )
                if st.button("✏️ Rename", key=f"rename_btn_{cid}", use_container_width=True):
                    title = new_title.strip()
                    if title:
                        conv["title"] = title
                    st.rerun()
                if st.button("🗑️ Delete", key=f"delete_btn_{cid}", use_container_width=True):
                    del st.session_state.conversations[cid]
                    # If it was the active conversation, pick another
                    # existing one (or leave None — the guard above creates
                    # a new empty one).
                    if st.session_state.active_conversation == cid:
                        remaining = list(st.session_state.conversations.keys())
                        st.session_state.active_conversation = remaining[-1] if remaining else None
                    st.rerun()

    st.divider()

    # Config menu: today it only has "About", but the popover leaves room
    # for future options (model, hop count, etc.) without changing the layout.
    with st.popover("⚙️ Config", use_container_width=True):
        if st.button("ℹ️ About — how the app works", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()

# ---------------------------------------------------------------------------
# "ABOUT" PAGE — the KAG pipeline step by step, the monitored assets and the full graph.
# ---------------------------------------------------------------------------
if st.session_state.page == "about":
    col_title, col_back = st.columns([5, 1])
    with col_title:
        st.title("ℹ️ About — how this KAG works")
    with col_back:
        if st.button("← Back to chat"):
            st.session_state.page = "chat"
            st.rerun()

    st.markdown(
        """
A **KAG (Knowledge Augmented Generation)** grounds the LLM's answers in a
**knowledge graph**, instead of loose text passages (as in RAG). This lets
it answer questions that require **chaining relations** together — and
prove the path used, or ask for more detail when the question isn't
specific enough.

> ⚠️ **All data in this app is synthetic.** The four equipment manuals,
> their components, symptoms, causes and corrective actions were written
> to simulate realistic industrial-maintenance documentation — none of it
> was extracted from a real manufacturer's manual or any real mining
> operation.

### Offline phase — building the knowledge base (runs once)

| Step | What happens | Module |
|---|---|---|
| **1. Documents** | Synthetic maintenance manuals for 4 mining asset types live in `data/raw/`, each with a source label | `data/raw/` |
| **2. Triple extraction** | The LLM reads each manual and extracts structured relations `(origin, relation, destination)` with types. Only 4 relations are allowed: `has_component`, `has_symptom`, `indicates_cause`, `resolved_by`. Each triple is validated (the relation must connect the right types) | `src/extraction.py` |
| **3. Graph construction** | The triples become a directed graph: each entity is a **node** (with a type), each relation is an **edge** (with its source document). The same entity mentioned across manuals becomes ONE shared node | `src/graph_builder.py` |

### Online phase — what happens on every question

| Step | What happens | Module |
|---|---|---|
| **4. Entity identification** | The LLM maps your question to a **real node in the graph** (e.g. "overcurrent"), preferring the most specific match possible. If no node matches, the system **admits it doesn't know** — it never makes one up | `src/graph_query.py` |
| **5. Specificity check** | If the identified entity is a **component** or **equipment** with more than one possible next step (several symptoms, or several components), the agent **asks for more detail** instead of guessing — exactly like a technician would | `src/graph_query.py` |
| **6. Multi-hop traversal** | With a specific entity, a **deterministic** algorithm (not the LLM!) walks the graph up to 3 hops following the causal chain: symptom → cause → action | `src/graph_query.py` |
| **7. Grounded answer** | The LLM receives ONLY the paths found and must answer based on them, making the path explicit and citing the sources | `src/answer_builder.py` |
| **8. Visualization** | The chat shows the subgraph of the points that were connected — the reasoning trail, or the candidate options when the agent asks for clarification | `src/graph_viz.py` |

**The key difference from RAG:** in RAG, the LLM receives similar-sounding
text and can "fill in the gaps" with what it knows. In KAG, the *evidence*
is structural: if the edge doesn't exist in the graph, the relation doesn't
make it into the answer — and if information is missing to choose between
edges, the agent asks.

### The domain's causal chain

`Equipment` →`has_component`→ `Component` →`has_symptom`→ `Symptom` →`indicates_cause`→ `Cause` →`resolved_by`→ `Corrective Action`

### Example of the clarification flow

> **You:** "The crusher's drive motor has a problem."
>
> **Agent:** "The crusher drive motor can present different symptoms:
> overcurrent in the crusher drive motor. Which of these symptoms are you
> observing?" *(if the mentioned component had more than one registered
> symptom, every option would appear for you to choose from)*
>
> **You:** "overcurrent"
>
> **Agent:** answers with the cause and the corrective action, citing the source.
        """
    )

    st.divider()
    st.subheader("🏭 Monitored assets")
    st.caption("The 4 mining asset types covered by the current knowledge base — all data is synthetic.")
    for asset in get_monitored_assets():
        with st.container(border=True):
            st.markdown(f"**{asset['source']}**  \n*{asset['category']}*")
            st.write(asset["description"])
            st.markdown(
                "Main components: "
                + ", ".join(f"`{c}`" for c in asset["components"])
            )

    st.divider()
    st.subheader("Full knowledge graph")
    st.caption(
        f"{graph.number_of_nodes()} entities · {graph.number_of_edges()} relations — "
        "🔵 equipment · 🟢 component · 🟠 symptom · 🔴 cause · 🟣 action"
    )
    components.html(generate_graph_html(graph), height=570, scrolling=False)
    st.stop()

# ---------------------------------------------------------------------------
# CHAT PAGE
# ---------------------------------------------------------------------------
conversation = st.session_state.conversations[st.session_state.active_conversation]

if not conversation["messages"]:
    # Empty-conversation welcome screen, like LLM interfaces.
    st.markdown(
        """
        <div style="text-align:center;padding:60px 0 20px 0;">
          <div style="font-size:52px;">🛠️</div>
          <h2 style="margin:4px 0;">KAG Agent Mining</h2>
          <p style="opacity:.7;">Describe the problem on a mining asset (vibrating screen, belt
          conveyor, haul truck or jaw crusher) and I'll walk the knowledge graph to find the
          probable cause and the corrective action. All data is synthetic.</p>
          <p style="opacity:.55;font-size:13px;">E.g.: <i>"The crusher's eccentric shaft has
          excessive vibration — what's the cause and the fix?"</i></p>
          <p style="opacity:.55;font-size:13px;">If the question doesn't have enough detail
          (e.g. <i>"the crusher's motor has a problem"</i>), I'll ask which symptom you're
          observing before answering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render the active conversation's history, including each answer's subgraph.
for i, msg in enumerate(conversation["messages"]):
    with st.chat_message("user"):
        st.write(msg["question"])
    with st.chat_message("assistant"):
        st.write(msg["answer"])
        if msg["paths"]:
            expander_title = (
                "🔎 Options found in the knowledge base — pick one for me to continue"
                if msg.get("clarification")
                else "🔎 Points connected in the knowledge base"
            )
            with st.expander(expander_title, expanded=False):
                sub = subgraph_from_paths(graph, msg["paths"])
                components.html(
                    generate_graph_html(sub, msg["paths"]),
                    height=420,
                    scrolling=False,
                )

question = st.chat_input("Describe the problem with the asset...")

if question:
    with st.chat_message("user"):
        st.write(question)

    # If the agent's last reply was a clarification request, the original
    # question becomes context so the LLM combines both pieces of
    # information (e.g. "crusher's motor" + "overcurrent").
    last_msg = conversation["messages"][-1] if conversation["messages"] else None
    previous_context = last_msg["question"] if last_msg and last_msg.get("clarification") else None

    with st.chat_message("assistant"):
        with st.spinner("Identifying the starting entity..."):
            result = query(question, graph, previous_context=previous_context)

        # --- Trace animation: shows the agent "walking" through the graph
        # nodes the query connected, while the answer is being prepared.
        # Since components.html() only EMBEDS the iframe and returns right
        # away (the JS animation runs in the browser, independent of
        # Python), it really does play in parallel with the LLM call below.
        animation_placeholder = st.empty()
        if result["paths"]:
            final_animation_message = (
                "Options found — pick one for me to continue."
                if result["needs_clarification"]
                else "Path found — generating the answer..."
            )
            trace_html, n_steps = generate_trace_html(
                graph,
                result["paths"],
                result["start_entity"],
                final_message=final_animation_message,
                step_duration_ms=int(ANIMATION_STEP_DURATION_S * 1000),
            )
            with animation_placeholder.container():
                components.html(trace_html, height=450, scrolling=False)

        if result["needs_clarification"]:
            # We don't call the answer LLM: there isn't enough evidence (a
            # complete path) to answer without guessing. Since there's no
            # network call here to "fill" the animation's time, we wait
            # manually so the user sees the full trace.
            estimated_duration = min(n_steps * ANIMATION_STEP_DURATION_S + 0.4, 4.0)
            time.sleep(estimated_duration)
            answer = result["clarifying_question"]
        else:
            # The LLM call (network) takes long enough for the animation to
            # play in full, in parallel, in the browser.
            answer = generate_answer(question, result)

        animation_placeholder.empty()
        st.write(answer)

        # Show the connected points (answer) or the candidate options
        # (clarification) — never the full graph, that stays in About.
        if result["paths"]:
            expander_title = (
                "🔎 Options found in the knowledge base — pick one for me to continue"
                if result["needs_clarification"]
                else "🔎 Points connected in the knowledge base"
            )
            with st.expander(expander_title, expanded=False):
                sub = subgraph_from_paths(graph, result["paths"])
                components.html(
                    generate_graph_html(sub, result["paths"]),
                    height=420,
                    scrolling=False,
                )

    conversation["messages"].append(
        {
            "question": question,
            "answer": answer,
            "paths": result["paths"],
            "clarification": result["needs_clarification"],
        }
    )
    # The first question becomes the conversation's title in the sidebar (like LLM UIs).
    if conversation["title"] == "New chat":
        conversation["title"] = question[:45]
    st.rerun()
