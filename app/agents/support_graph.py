from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langsmith import traceable
from datetime import datetime
import json
import os

from app.services.gemini import call_gemini_llm
from app.services.utils import get_conversation, save_conversation, convert_markdown_for_google_chat
from app.services.KnowledgeBaseFiltering import get_relevant_incidents_weighted_context
from app.services.hybrid_search import buscar_hibrido
from app.services.KnowledgeBaseFiltering import initialize_model_and_kb
from app.services.hybrid_search import get_kb_item_by_id


MIN_COSINE_SIMILARITY = 0.80
MIN_HYBRID_SCORE = 0.55

class SupportState(TypedDict, total=False):
    user_message: str
    user_email: str
    role: Literal["user", "tech"]
    response_mode: Literal["generative", "hybrid"]
    solved: Optional[bool]
    output: Optional[str]
    #kb_answer: str
    action: Optional[Literal["ticket", "none"]]


# @traceable(name="DetectIntent")
# async def detect_intent_node(state: SupportState) -> Command:
#     user_msg = state.get("user_message", "").strip().lower()
#
#     if state.get("solved") is True and user_msg in (
#             "no", "no.", "nop", "nope", "negativo", "no se resolvió", "no funciona"
#     ):
#         return Command(goto="Ticket")
#
#     if state.get("role") == "tech":
#         return Command(goto="KBManager")
#
#     intent = await detect_intent(state["user_message"])
#
#     if intent == "UNKNOWN":
#         return Command(
#             goto=END,
#             update={"output": "¿Podrías darme un poco más de detalle?", "solved": False}
#         )
#
#     return Command(goto="RouteByResponseMode", update={"intent": intent})

@traceable(name="RouteByResponseMode")
async def route_by_response_mode(state: SupportState) -> Command:
    if state.get("role") == "tech":
        return Command(goto="KBManager")

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
        msg = "No he encontrado nada en la base de conocimiento. ¿Quieres que creemos un ticket?"
        return Command(
            goto=END,
            update={
                "output": msg,
                "solved": False,
                "action": "ticket"
            }
        )

    conversation_total = get_conversation(user_email)
    prev = conversation_total.get("conversation", []) if isinstance(conversation_total, dict) else []

    def _coerce_role(r: str) -> str:
        return "user" if r == "user" else "model"

    prev_clean = []
    for m in prev:
        if m.get("role") == "system":
            continue
        prev_clean.append({
            "role": _coerce_role(m.get("role", "user")),
            "content": m.get("content", ""),
            "timestamp": m.get("timestamp")
        })

    if not isinstance(relevant_incidents, list):
        relevant_incidents = []

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
                   • Si la guía no ofrece una solución directa o hay varias opciones, formula preguntas aclaratorias antes de proponer una solución -> en este caso devuelve `solved=true`.  
                   • Cuando dispongas de la información necesaria, entrega la **solución completa**, paso a paso, sin omitir nada.

                5. Formato de la respuesta  
                   • Presenta los pasos en **viñetas o lista numerada** para mayor claridad.  
                   • Al final del mensaje, añade una frase breve indicando que la solución proviene de la base de conocimiento.  
                   • No cierres tu respuesta hasta haber incluido **todos** los pasos o las preguntas aclaratorias necesarias.

                6. Casos no cubiertos  
                   • Si el problema no aparece en la guía, responde únicamente:  
                     “Lo siento, no puedo ayudarte a resolver este problema. Sí puedo ayudarte a abrir un ticket con soporte.”
                   • Si el usuario dice específicamente que no ha podido resolver el problema pese a haber seguido los pasos:
                     “Siento no haber podido ayudarte a resolver este problema. Si lo deseas puedo abrir un ticket por ti.”
                   • En ambos casos devuelve `solved=false`.
                   • Es posible que el usuario empieze una nueva conversación y te consulte sobre un problema diferente sin borrar caché. actúa de forma natural y ayúdale en ese caso

                7. Indicador de estado  
                   • Devuelve `solved=true` cuando proporciones una solución completa o sólo plantees preguntas aclaratorias.  
                   • Devuelve `solved=false` cuando la base de conocimiento te lo indique, o siguiendo el punto 6.
                   • Devuelve `solved=false` cuando hayas conseguido suficiente información en aquellas incidencias que indiquen que necesitan ser escaladas a soporte y responde unicamente:
                    “Esta incidencia debe ser resuelta por soporte, usa el siguiente botón para enviar un ticket a soporte con un resumen de nuestra conversación”    """,
        "timestamp": datetime.now().isoformat()
    }

    conversation = [system_msg] + prev_clean

    conversation.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })

    tools = [{
        "function_declarations": [{
            "name": "return_json_response",
            "description": "Devuelve una respuesta como JSON con 'Response' y 'solved'",
            "parameters": {
                "type": "object",
                "properties": {
                    "Response": {"type": "string"},
                    "solved": {"type": "boolean"}
                },
                "required": ["Response", "solved"]
            }
        }]
    }]

    parsed = await call_gemini_llm(conversation, tools)
    if isinstance(parsed, dict):
        response_text = convert_markdown_for_google_chat(parsed.get("Response", ""))
        solved = bool(parsed.get("solved", False))
    else:
        response_text = convert_markdown_for_google_chat(str(parsed))
        solved = False

    conversation.append({
        "role": "model",
        "content": response_text,
        "timestamp": datetime.now().isoformat()
    })

    incident_ids = []

    if isinstance(relevant_incidents, list):
        for e in relevant_incidents:
            if isinstance(e, dict) and "id" in e:
                incident_ids.append(e["id"])

    save_conversation(
        user_email,
        {"conversation": conversation, "Incidents": incident_ids}
    )

    return Command(goto=END, update={"output": response_text, "solved": True})


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

    texto = f"📘 **{incidente.get('title', '(Sin título)')}**\n\n"

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

    texto += "¿El problema quedó resuelto?"

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

@traceable(name="KBManager")
async def kb_manager_node(state: SupportState) -> Command:
    return Command(
        goto=END,
        update={"output": "📘 Panel técnico.\n\nPuedes añadir/editar/eliminar entradas de la base de conocimiento.", "solved": True}
    )

def build_support_graph():
    initialize_model_and_kb("app/data/kb_embeddings.json")

    graph = StateGraph(SupportState)

    graph.add_node("RouteByResponseMode", route_by_response_mode)
    graph.add_node("GenerativeResponse", generative_agent_node)
    graph.add_node("HybridResponse", hybrid_response_node)
    graph.add_node("Ticket", ticket_node)
    graph.add_node("KBManager", kb_manager_node)

    graph.set_entry_point("RouteByResponseMode")

    return graph.compile()


