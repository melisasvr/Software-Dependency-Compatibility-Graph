"""Streamlit UI: chat (LLM understanding -> graph retrieval -> Groq) + graph explorer."""
import os
import matplotlib.pyplot as plt
import streamlit as st
from graph_builder import build_graph
from reasoning import analyze
from llm import ask_groq, list_models, understand_question
from graph_viz import draw_graph, ego_edges, COLORS

# Must be the FIRST Streamlit call
st.set_page_config(page_title="Dependency Graph KAG", page_icon="🕸️", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_data(ttl=3600, show_spinner=False)
def get_models(key):
    return list_models(key)


G = get_graph()

EXAMPLES = ["What breaks if I upgrade numpy to 2.0?",
            "What breaks if I upgrade starlette to 0.40?",
            "I want a web api and a dashboard together. What do I need?",
            "Is streamlit compatible with tensorflow?",
            "Are fastapi and pydantic compatible if I also use langchain?",
            "Will bumping sklearn to 1.5 wreck my pandas setup?"]

st.session_state.setdefault("messages", [])

with st.sidebar:
    st.title("🕸️ Dependency KAG")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    # Model list comes live from Groq, so it never goes stale
    models = get_models(api_key) if api_key else ["llama-3.1-8b-instant"]
    PREFERRED = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
    default = next((models.index(p) for p in PREFERRED if p in models), 0)
    model = st.selectbox("Model", models, index=default)

    smart = st.toggle("AI question understanding", value=True,
                      help="Let the LLM extract intent and libraries from your question. "
                           "Falls back to keyword rules if it fails.")
    show_reasoning = st.toggle("Show reasoning path", value=True)
    st.caption(f"{G.number_of_nodes()} libraries · {G.number_of_edges()} relations")

    # Clear chat history
    if st.button("🗑️ Clear chat", use_container_width=True,
                 disabled=not st.session_state["messages"]):
        st.session_state["messages"] = []
        st.session_state.pop("pending", None)
        st.rerun()

    st.markdown("**Try:**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

tab_chat, tab_graph = st.tabs(["💬 Ask", "🕸️ Explore graph"])


def render(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        t = m.get("trace")
        if t and show_reasoning:
            with st.expander(f"🔍 Reasoning path · intent: {t['intent']}"):
                st.caption(f"Question understood by: {t.get('understood_by', 'rules')} · "
                           f"libraries found: {t['mentions'] or 'none'}"
                           + (f" · features: {t['features']}" if t["features"] else ""))
                st.code("\n".join(t["facts"]) or "no facts", language=None)
                if t["edges"]:
                    fig = draw_graph(G, t["edges"])
                    st.pyplot(fig)
                    plt.close(fig)


with tab_chat:
    for m in st.session_state["messages"]:
        render(m)

question = st.chat_input("Ask about dependencies or compatibility...") or st.session_state.pop("pending", None)
if question:
    st.session_state["messages"].append({"role": "user", "content": question})

    # 0) understand the question (LLM first, keyword rules as fallback)
    parsed = None
    if api_key and smart:
        with st.spinner("Understanding your question..."):
            parsed = understand_question(question, G, api_key, model)
    # 1) graph retrieval + multi-hop reasoning
    trace = analyze(G, question, parsed)

    if trace["intent"] == "none":
        answer = "I couldn't find a known library in that question. Known: " + ", ".join(sorted(G.nodes))
    elif not api_key:
        answer = "⚠️ Add a Groq API key in the sidebar for a written answer. Raw graph facts are below."
    else:
        try:                                           # 2) LLM explains the retrieved facts
            with st.spinner("Writing the answer..."):
                answer = ask_groq(question, trace["facts"], api_key, model)
        except Exception as e:
            answer = f"⚠️ Groq error: {e}"
    st.session_state["messages"].append({"role": "assistant", "content": answer, "trace": trace})
    st.rerun()

with tab_graph:
    c1, c2 = st.columns([2, 1])
    lib = c1.selectbox("Library", sorted(G.nodes))
    radius = c2.slider("Hops", 1, 3, 1)
    rels = st.multiselect("Relations to show", list(COLORS), default=list(COLORS))
    hide_py = st.checkbox("Hide 'python' (it connects to almost everything)", value=True)
    hide = {"python"} if hide_py and lib != "python" else set()

    edges = ego_edges(G, lib, radius, relations=set(rels), hide=hide)
    if not edges:
        st.info("No relations to show with these filters.")
    else:
        st.caption("Center = selected library · inner ring = 1 hop · outer rings = 2–3 hops")
        fig = draw_graph(G, edges, figsize=(8, 7), center=lib, show_constraints=True)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe([{"from": u, "relation": k, "to": v,
                       "constraint": G.edges[u, v, k]["constraint"],
                       "note": G.edges[u, v, k]["note"]} for u, v, k in edges],
                     use_container_width=True)