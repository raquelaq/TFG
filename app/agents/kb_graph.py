from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langsmith import traceable
import os
import json

from app.services.KnowledgeBaseFiltering import rebuild_embeddings

class KBState(TypedDict, total=False):
    id: str
    title: str
    description_problem: str
    symptoms: List[str]
    resolution_guide_llm: Dict[str, Any]
    escalation_criteria: str
    keywords_tags: List[str]

    output: str
    __output__: str

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "KnowledgeBase.json"))


@traceable(name="KBManager")
async def kb_manager_node(state: KBState) -> KBState:
    """
    Nodo de entrada del grafo de técnico.
    - Si el rol no es 'tech', corta el flujo.
    - Si detecta 'añadir entrada', pasa al siguiente nodo.
    """
    if state.get("role") != "tech":
        return {
            "__output__": "__end__",
            "output": "❌ No tienes permisos para gestionar la base de conocimiento.",
        }

    msg = state.get("user_message", "").lower()

    if "añadir" in msg or "agregar" in msg or "nueva entrada" in msg:
        return {
            "__output__": "KB_CollectTitle",
            "output": (
                "📝 Vale. Vamos a crear una nueva entrada en la KB.\n\n"
                "Dime el **título** de la incidencia:"
            ),
        }

    return {
        "__output__": "KBManager",
        "output": (
            "📘 Estás en el **panel técnico**.\n\n"
            "Puedes decirme por ejemplo:\n"
            "- `Añadir entrada`\n"
            "(Más adelante: `Editar entrada`, `Eliminar entrada`...)\n"
        ),
    }


@traceable(name="KB_CollectTitle")
async def kb_collect_title_node(state: KBState) -> KBState:
    state["kb_title"] = state.get("user_message", "")
    return {
        "__output__": "KB_CollectDescription",
        "output": "Perfecto. Ahora dime la **descripción del problema**:",
    }


@traceable(name="KB_CollectDescription")
async def kb_collect_description_node(state: KBState) -> KBState:
    state["kb_description"] = state.get("user_message", "")
    return {
        "__output__": "KB_CollectSymptoms",
        "output": "Bien. Ahora dime los **síntomas** separados por comas:",
    }


@traceable(name="KB_CollectSymptoms")
async def kb_collect_symptoms_node(state: KBState) -> KBState:
    raw = state.get("user_message", "")
    symptoms = [s.strip() for s in raw.split(",") if s.strip()]
    state["kb_symptoms"] = symptoms

    return {
        "__output__": "KB_SaveEntry",
        "output": "Perfecto. Por último, dame las **palabras clave** separadas por comas:",
    }


@traceable(name="KB_SaveEntry")
async def kb_save_entry_node(state: KBState) -> KBState:
    """
    Nodo único del grafo KB: guarda una nueva entrada en la KB y regenera embeddings.
    No es conversacional: recibe todos los campos desde Streamlit.
    """

    os.makedirs(os.path.dirname(KB_PATH), exist_ok=True)

    if not os.path.exists(KB_PATH):
        with open(KB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
    except Exception:
        kb_data = []

    new_entry = {
        "id": state["id"],
        "title": state["title"],
        "description_problem": state["description_problem"],
        "symptoms": state.get("symptoms", []),
        "resolution_guide_llm": state.get("resolution_guide_llm", {
            "initial_questions": [],
            "diagnostic_steps": []
        }),
        "escalation_criteria": state.get("escalation_criteria", ""),
        "keywords_tags": state.get("keywords_tags", [])
    }

    kb_data.append(new_entry)

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb_data, f, indent=4, ensure_ascii=False)

    rebuild_embeddings()

    return KBState(
        __output__="__end__",
        output=f"✅ Entrada añadida: **{new_entry['title']}**."
    )
def build_kb_graph():
    graph = StateGraph(KBState)
    graph.add_node("KB_SaveEntry", kb_save_entry_node)
    graph.add_edge("KB_SaveEntry", END)
    graph.set_entry_point("KB_SaveEntry")
    return graph.compile()
