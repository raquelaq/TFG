from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langsmith import traceable
from datetime import datetime
from typing import TypedDict, List, Dict, Any
import json

from app.config import DATA_DIR
from app.services.gemini import call_gemini_llm
from app.services.utils import get_conversation, save_conversation, convert_markdown_for_google_chat
from app.services.KnowledgeBaseFiltering import get_relevant_incidents_weighted_context
from app.services.hybrid_search import buscar_hibrido
from app.services.KnowledgeBaseFiltering import initialize_model_and_kb, rebuild_embeddings
from app.services.hybrid_search import get_kb_item_by_id


MIN_COSINE_SIMILARITY = 0.80
MIN_HYBRID_SCORE = 0.55

class SupportState(TypedDict, total=False):
    user_message: str
    user_email: str
    role: Literal["user", "tech"]
    response_mode: Literal["generative", "hybrid"]

    id: str
    title: str
    description_problem: str
    symptoms: list
    resolution_guide_llm: Dict[str, Any]
    escalation_criteria: str
    keywords_tags: list

    solved: Optional[bool]
    output: Optional[str]
    action: Optional[Literal["ticket", "none"]]

@traceable(name="RouteByRole")
async def route_by_role(state: SupportState) -> Command:
    if state.get("role") == "tech":
        return Command(goto="KB_SaveEntry")
    return Command(goto="RouteByResponseMode")

@traceable(name="RouteByResponseMode")
async def route_by_response_mode(state: SupportState) -> Command:
    state["trace"] = f"routing:{state.get('response_mode')}"

    mode = state.get("response_mode", "generative")
    return Command(goto="HybridResponse" if mode == "hybrid" else "GenerativeResponse")

@traceable(name="GenerativeResponse")
async def generative_agent_node(state: SupportState) -> Command:
    user_message = state["user_message"]
    user_email = state["user_email"]

    relevant_incidents = get_relevant_incidents_weighted_context(
        user_email=user_email,
        query=user_message,
        top_n=10,
        decay_factor=0.9
    )

    if not relevant_incidents:
        return Command(
            goto=END,
            update={
                "output": (
                    "No he encontrado una solución en la base de conocimiento para tu consulta. "
                    "Si lo deseas, puedo ayudarte a crear un ticket para soporte técnico."
                ),
                "solved": False,
                "action": "ticket"
            }
        )

    conversation_total = get_conversation(user_email)
    prev = conversation_total.get("conversation", []) if isinstance(conversation_total, dict) else []

    def _coerce_role(r: str) -> str:
        return "user" if r == "user" else "model"

    prev_clean = [
        {
            "role": _coerce_role(m.get("role", "user")),
            "content": m.get("content", "")
        }
        for m in prev
        if m.get("role") != "system"
    ]

    system_msg = {
        "role": "system",
        "content": f"""Actúa como un asistente de soporte informático que guía a los usuarios de negocio en las consultas más frecuentes.
                1. Saludo y contexto  
                   • Empieza cada conversación con un saludo creativo y una breve frase sobre los problemas que puedes resolver.

                2. Guía de conocimiento  
                   • Responde **exclusivamente** con la información contenida en la guía de soporte.
                   Basado en el contexto de la conversación, la siguiente información de la base de conocimiento es la más pertinente: 
                     \"\"\"{json.dumps(relevant_incidents, indent=4, ensure_ascii=False)}\"\"\"

                3. Estilo y tono  
                   • Habla siempre en **segunda persona del singular**.  
                   • Mantén un tono cercano y amable.
                   • Usa siempre listas con viñetas para que tu respuesta sea fácil de leer. Sin embargo, si me das pasos para resolver algo, envíalos en mensajes separados en lugar de usar viñetas dentro de un mismo mensaje.

                4. Proceso de atención  
                   • Si la guía no ofrece una solución directa o hay varias opciones, formula preguntas aclaratorias antes de proponer una solución.  
                   • Cuando dispongas de la información necesaria, entrega la **solución completa**, paso a paso, sin omitir nada.

                5. Formato de la respuesta  
                   • Presenta los pasos en **viñetas o lista numerada** para mayor claridad.    
                   • No cierres tu respuesta hasta haber incluido **todos** los pasos o las preguntas aclaratorias necesarias.

                6. Casos no cubiertos  
                   • Si el problema no aparece en la guía, responde únicamente:  
                     “Lo siento, no puedo ayudarte a resolver este problema. Sí puedo ayudarte a abrir un ticket con soporte.”
                   • Si el usuario dice específicamente que no ha podido resolver el problema pese a haber seguido los pasos:
                     “Siento no haber podido ayudarte a resolver este problema. Si lo deseas puedo abrir un ticket por ti.”
    
                   • Es posible que el usuario empieze una nueva conversación y te consulte sobre un problema diferente sin borrar caché. actúa de forma natural y ayúdale en ese caso    """,
        "timestamp": datetime.now().isoformat()
    }

    conversation = [system_msg] + prev_clean + [{
        "role": "user",
        "content": user_message
    }]

    parsed = await call_gemini_llm(conversation)

    response_text = convert_markdown_for_google_chat(
        parsed.get("Response", "")
        if isinstance(parsed, dict)
        else str(parsed)
    )

    llm_solved = parsed.get("solved", True)

    save_conversation(
        user_email,
        {
            "conversation": conversation + [
                {"role": "model", "content": response_text}
            ],
            "Incidents": [
                e["id"] for e in relevant_incidents if "id" in e
            ]
        }
    )

    if not llm_solved:
        return Command(
            goto=END,
            update={
                "output": response_text,
                "solved": False,
                "action": "ticket"
            }
        )

    return Command(
        goto=END,
        update={
            "output": response_text,
            "solved": True,
            "action": "none"
        }
    )


@traceable(name="HybridResponse")
async def hybrid_response_node(state: SupportState) -> Command:
    query = state["user_message"]

    resultados = buscar_hibrido(query, alpha=0.25, top_k=1)

    if not resultados:
        msg = (
            "No he encontrado una solución clara en la base de conocimiento para tu consulta. "
            "¿Quieres que creemos un ticket para que soporte técnico lo revise?"
        )
        return Command(
            goto=END,
            update={
                "output": msg,
                "solved": False,
                "action": "ticket"
            }
        )

    mejor = resultados[0]

    if (
        mejor["score_cosine"] < MIN_COSINE_SIMILARITY
        or mejor["score_hybrid"] < MIN_HYBRID_SCORE
    ):
        msg = (
            "No he encontrado una solución clara en la base de conocimiento para tu consulta. "
            "¿Quieres que creemos un ticket para que soporte técnico lo revise?"
        )
        return Command(
            goto=END,
            update={
                "output": msg,
                "solved": False,
                "action": "ticket"
            }
        )

    incidente_id = mejor["id"]

    incidente = get_kb_item_by_id(incidente_id)

    if not incidente:
        return Command(goto="Ticket", update={"output": "He encontrado coincidencia, pero no puedo cargar la guía.", "solved": False})

    texto = f"**{incidente.get('title', '(Sin título)')}**\n\n"

    preguntas = incidente.get("questions_llm", [])
    pasos = incidente.get("resolution_guide_llm", {}).get("diagnostic_steps", [])

    if preguntas:
        texto += "**Preguntas iniciales:**\n"
        for p in preguntas:
            texto += f"- {p}\n"
        texto += "\n"

    if pasos:
        texto += "**Pasos para resolver la incidencia:**\n\n"
        for i, step in enumerate(pasos, 1):
            titulo = step.get("title", "")
            accion = step.get("user_action", "")
            texto += f"**Paso {i}: {titulo}**\n{accion}\n\n"
    else:
        texto += "⚠️ Esta incidencia no tiene pasos detallados.\n\n"


    return Command(goto=END, update={"output": texto, "solved": True})

@traceable(name="Ticket")
async def ticket_node(state: SupportState) -> Command:
    return Command(
        goto=END,
        update={
            "output": "No he podido encontrar una solución clara. Si quieres, puedes crear un ticket con el botón.",
            "solved": False,
            "action": "ticket"
        }
    )

@traceable(name="KB_SaveEntry")
async def kb_save_entry_node(state: SupportState) -> Command:
    from app.services.KnowledgeBaseFiltering import rebuild_embeddings
    import json, os

    KB_PATH = DATA_DIR / "KnowledgeBase.json"
    KB_PATH.parent.mkdir(exist_ok=True)

    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = []

    kb.append({
        "id": state["id"],
        "title": state["title"],
        "description_problem": state["description_problem"],
        "symptoms": state.get("symptoms", []),
        "resolution_guide_llm": state.get("resolution_guide_llm", {}),
        "escalation_criteria": state.get("escalation_criteria", ""),
        "keywords_tags": state.get("keywords_tags", [])
    })

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)

    rebuild_embeddings()

    return Command(
        goto=END,
        update={"output": f"✅ Entrada '{state['title']}' añadida correctamente."}
    )


def build_support_graph():
    initialize_model_and_kb(str(DATA_DIR / "kb_embeddings.json"))

    graph = StateGraph(SupportState)

    graph.add_node("RouteByRole", route_by_role)
    graph.add_node("RouteByResponseMode", route_by_response_mode)
    graph.add_node("GenerativeResponse", generative_agent_node)
    graph.add_node("HybridResponse", hybrid_response_node)
    graph.add_node("Ticket", ticket_node)
    graph.add_node("KB_SaveEntry", kb_save_entry_node)

    graph.set_entry_point("RouteByRole")

    return graph.compile()


