from typing import Dict
import json
from langsmith import traceable
import os

from app.services.hybrid_search import buscar_hibrido

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "..", "data", "KnowledgeBase.json")

MIN_COSINE_SIMILARITY = 0.80
MIN_HYBRID_SCORE = 0.55


@traceable(name="HybridResponse")
async def hybrid_response_node(state: Dict) -> Dict:
    """
    Nodo de respuesta basado en el modelo híbrido (BM25 + embeddings).
    Genera una respuesta completa de la guía si existe una coincidencia clara.
    """

    user_message = state["user_message"]

    resultados = buscar_hibrido(
        user_message,
        alpha=0.25,
        top_k=2
    )

    # 1️⃣ No hay resultados
    if not resultados:
        msg = (
            "No he encontrado una solución clara en la base de conocimiento. "
            "¿Quieres que creemos un ticket para que soporte técnico lo revise?"
        )
        return {
            "output": msg,
            "solved": False,
            "__output__": "Ticket"
        }

    mejor = resultados[0]

    # 2️⃣ Ambigüedad (dos resultados muy cercanos)
    if len(resultados) > 1:
        delta = abs(
            resultados[0]["score_hybrid"] - resultados[1]["score_hybrid"]
        )
        if delta < 0.05:
            return {
                "output": (
                    "Tu consulta puede referirse a varios problemas distintos. "
                    "¿Podrías darme un poco más de contexto para ayudarte mejor?"
                ),
                "solved": True,
                "__output__": "__end__"
            }

    # 3️⃣ Umbrales de confianza
    if (
        mejor["score_cosine"] < MIN_COSINE_SIMILARITY
        or mejor["score_hybrid"] < MIN_HYBRID_SCORE
    ):
        return {
            "output": (
                "No he encontrado una solución suficientemente relacionada con tu consulta. "
                "¿Quieres que creemos un ticket para que soporte técnico lo revise?"
            ),
            "solved": False,
            "__output__": "Ticket"
        }

    # 4️⃣ Cargar la incidencia completa
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    incidente = next(
        (x for x in kb if x.get("id") == mejor["id"]),
        None
    )

    if not incidente:
        return {
            "output": (
                "He encontrado una coincidencia, pero no puedo cargar los detalles "
                "de la guía en la base de conocimiento."
            ),
            "solved": False,
            "__output__": "Ticket"
        }

    # 5️⃣ Construir respuesta final (guía completa)
    preguntas = incidente.get("questions_llm", [])
    pasos = incidente.get("resolution_guide_llm", {}).get("diagnostic_steps", [])

    texto = f"**{incidente.get('title', '(Sin título)')}**\n\n"

    if preguntas:
        texto += "**Preguntas iniciales:**\n"
        for p in preguntas:
            texto += f"- {p}\n"
        texto += "\n"

    if pasos:
        texto += "**Pasos para resolver la incidencia:**\n\n"
        for i, step in enumerate(pasos, 1):
            titulo = (step.get("title") or "").strip()
            accion = (step.get("user_action") or "").strip()

            if titulo:
                texto += f"**Paso {i}: {titulo}**\n"
            else:
                texto += f"**Paso {i}:**\n"

            texto += f"{accion}\n\n"
    else:
        texto += "⚠️ Esta incidencia no tiene pasos detallados en la guía.\n\n"

    texto += "¿El problema quedó resuelto?"

    return {
        "output": texto,
        "solved": True,
        "__output__": "__end__"
    }
