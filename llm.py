"""Groq calls: (1) understand the question -> JSON, (2) explain the graph facts."""
import json
import re
from groq import Groq

# ---------------------------------------------------------------- explanation
SYSTEM_PROMPT = """You are a software dependency & compatibility assistant.
You receive GRAPH FACTS retrieved from a curated knowledge graph. They were computed
by deterministic code (multi-hop traversal + version-constraint solving).

How to read the facts:
- `a --depends_on[RANGE]--> b` means: a requires b to be within RANGE.
  RANGE restricts the version of b (the target), NOT the version of a.
- BREAKS = the new version violates a RANGE. AT RISK = depends on something that breaks.
- CONFLICT = no version of a package satisfies all requirers at once.

Rules:
- Use ONLY the graph facts. Never invent versions, packages or constraints.
- Never state versions of a package unless they appear in the facts.
- Only recommend actions the facts support (e.g. keep b inside RANGE, or pick the
  version listed after "pick").
- If the facts don't answer the question, say what is missing.
- Cite relationships in the form `a --depends_on[RANGE]--> b`.
- CONTEXT lines describe how the OTHER libraries in the question relate to the library being
  changed. Address each of them in your answer. If a CONTEXT line says there is no link, say so
  plainly.
- Never hedge with phrases like "if it does" about a relationship. State only what the facts say.
- Format: **Answer** (1-2 sentences), **Why** (short reasoning chain), **Do this** (concrete steps).
- Be concise."""

# Used only if the live model list can't be fetched
FALLBACK_MODELS = ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]

# Skip non-chat models (speech, safety classifiers, etc.)
_SKIP = ("whisper", "tts", "orpheus", "guard", "embed", "playai", "compound")


def list_models(api_key):
    """Ask Groq which models this key can actually use right now."""
    try:
        ids = [m.id for m in Groq(api_key=api_key).models.list().data]
        chat = sorted(i for i in ids if not any(s in i.lower() for s in _SKIP))
        return chat or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


def ask_groq(question, facts, api_key, model):
    context = "\n".join(f"- {f}" for f in facts) or "- (no graph facts found)"
    resp = Groq(api_key=api_key).chat.completions.create(
        model=model, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"QUESTION:\n{question}\n\nGRAPH FACTS:\n{context}"}])
    return resp.choices[0].message.content


# ------------------------------------------------------- question understanding
UNDERSTAND_PROMPT = """You convert a user's question about Python packages into JSON for a graph lookup.

Known libraries (alternative names in brackets): __LIBS__
Known features: __FEATURES__

Return ONLY a JSON object, no explanation, in exactly this shape:
{"intent": "upgrade_impact" | "compatibility" | "requirements" | "none",
 "libraries": [{"name": "<known library>", "version": "<version or null>"}],
 "features": ["<known feature>"]}

Intent:
- upgrade_impact: asks what breaks / is affected if a library is upgraded, downgraded or bumped.
  The version is the TARGET version the user wants to move to.
- compatibility: asks whether libraries work or conflict together.
- requirements: asks what to install or what is needed for libraries or features.
- none: not about these libraries.

Rules:
- Use ONLY names from the known lists. Map synonyms (sklearn -> scikit-learn, pytorch -> torch).
- Copy versions exactly as the user wrote them (e.g. "2.0", "0.40"); use null if none given.
- Include every library the user mentions, even in passing ("if I also use langchain")."""

_INTENTS = {"upgrade_impact", "compatibility", "requirements", "none"}


def understand_question(question, G, api_key, model):
    """Use the LLM to extract intent + entities. Returns a validated dict, or None on any
    failure (the caller then falls back to keyword rules)."""
    try:
        libs = ", ".join(f"{n} [{', '.join(a['aliases'])}]" if a["aliases"] else n
                         for n, a in G.nodes(data=True))
        features = ", ".join(G.graph["features"])
        system = UNDERSTAND_PROMPT.replace("__LIBS__", libs).replace("__FEATURES__", features)
        resp = Groq(api_key=api_key).chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": question}])
        text = resp.choices[0].message.content
        raw = json.loads(text[text.index("{"): text.rindex("}") + 1])   # tolerate ```json fences
    except Exception:
        return None

    # --- validate everything against the graph: the LLM can't invent nodes ---
    lookup = {}
    for n, a in G.nodes(data=True):
        for name in {n, n.replace("-", " "), *a["aliases"]}:
            lookup[name.lower()] = n
    mentions = {}
    for item in raw.get("libraries", []):
        lib = lookup.get(str(item.get("name", "")).lower())
        ver = str(item.get("version") or "").strip().lstrip("vV")
        ver = ver if re.fullmatch(r"\d+(\.\d+)*", ver) else None
        if lib and (lib not in mentions or (mentions[lib] is None and ver)):
            mentions[lib] = ver
    feats = [f for f in raw.get("features", []) if f in G.graph["features"]]
    intent = raw.get("intent") if raw.get("intent") in _INTENTS else None
    if not intent or (not mentions and not feats):
        return None
    return {"intent": intent, "mentions": list(mentions.items()), "features": feats}