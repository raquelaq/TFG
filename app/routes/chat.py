from fastapi import APIRouter, Request, Header, Depends, HTTPException
import urllib3
import json
import os
from datetime import datetime

from ..services.utils import *
from ..agents.ticket_agent import TicketAgent
from app.agents.support_graph import build_support_graph
from ..services.KnowledgeBaseFiltering import *

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

compiled_graph = build_support_graph()
router = APIRouter()

@router.get("/delete_cache")
async def delete_cache(request: Request, api_key_info: dict = Depends(api_key_guard)):
        delete_converation_cache()

@router.get("/delete_cache_user")
async def delete_cache_user(
        request: Request,
        user_id: str,
        api_key_info: dict = Depends(api_key_guard)
):
    try:
        success = delete_conversation_cache_user(user=user_id)
        print(success)

        if success:
            return {"message": f"Cache for user '{user_id}' deleted successfully."}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to delete cache for user '{user_id}'. User might not exist or file issues."
            )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=f"Error deleting cache: {str(e)}")

@router.post("/message")
async def handle_message(request: Request, authorization: str = Header(None)):
    try:
        data = await request.json()
        request_type = data.get("type")

        if request_type == "MESSAGE":
            pregunta = data.get("message", {}).get("text")
            usuario = (
                data.get("message", {})
                .get("sender", {})
                .get("email", "usuario@local.test")
                .split("@")[0]
            )

            modo_ui = data.get("modo_respuesta", "IA Generativa")

            response_mode = (
                "hybrid"
                if modo_ui == "Modelo ML (embeddings)"
                else "generative"
            )

            state = {
                "user_message": pregunta,
                "user_email": usuario,
                "role": "user",
                "response_mode": response_mode
            }

            result = await compiled_graph.ainvoke(state)

            response_text = result.get("output", "")
            solved = result.get("solved", False)

            response_obj = {"text": response_text}
            if not solved:
                response_obj["cardsV2"] = [
                    {
                        "cardId": "helpOptions",
                        "card": {
                            "header": {
                                "title": "¿Quieres que creemos un ticket para soporte?",
                                "subtitle": "Tu incidencia será tratada con prioridad"
                            },
                            "sections": [
                                {
                                    "widgets": [
                                        {
                                            "buttonList": {
                                                "buttons": [
                                                    {
                                                        "text": "Sí",
                                                        "onClick": {
                                                            "action": {
                                                                "function": "createJiraTicket",
                                                                "parameters": [
                                                                    {
                                                                        "key": "messages",
                                                                        "value": json.dumps([
                                                                            {
                                                                                "role": "user",
                                                                                "content": pregunta
                                                                            }
                                                                        ])
                                                                    }
                                                                ]
                                                            }
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]

            return response_obj

        elif request_type == "CARD_CLICKED":
            print("resolving card click")
            response = await onClick(data)
        else:
            return {
                "text": "Lo siento, ha ocurrido un error y no puedo ayudarte ahora mismo."
            }
    except Exception as e:
        print("ERROR: ", str(e))
        return {
            "text": "Lo siento, ha ocurrido un error y no puedo ayudarte ahora mismo."
        }

async def respond_message(data):
    try:
        user_message = data.get("message", {}).get("text")
        chat_id = data.get("space", {}).get("name")
        user_email = data.get("message", {}).get("sender").get("email")

        # --- Semantic Filtering Step ---
        start_filter = time.time()
        print(f"Filtering relevant incidents for query: '{user_message}' (Chat ID: {chat_id}, User Email: {user_email})")
        relevant_incidents = get_relevant_incidents_weighted_context(
            user_email=user_email,
            query=user_message,
            top_n=10,
            decay_factor=0.9
        )
        ids = []
        if isinstance(relevant_incidents, list):
            for e in relevant_incidents:
                if isinstance(e, dict) and "id" in e:
                    ids.append(e["id"])

        print(ids)


        conversation_total = get_conversation(user_email)

        if isinstance(conversation_total, list):
            conversation = conversation_total
            incident_ids = []
        elif isinstance(conversation_total, dict):
            conversation = conversation_total.get("conversation", [])
            incident_ids = conversation_total.get("Incidents", [])
        else:
            conversation = []
            incident_ids = []

# todo especificar que no escriba el llm_action
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

        if not conversation:
            conversation = [system_msg]
        else:
            conversation[0] = system_msg

        conversation.append({"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()})

        tools = [
            {
                "function_declarations": [
                    {
                        "name": "return_json_response",
                        "description": "Devuelve una respuesta como JSON con los campos 'Response' y 'solved'",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "Response": {
                                    "type": "string",
                                    "description": "Respuesta del modelo a la pregunta del usuario basandose en la base de conocimiento dada"
                                },
                                "solved": {
                                    "type": "boolean",
                                    "description": "true si es una posible solución al problema, false si no estás seguro."
                                }
                            },
                            "required": ["Response", "solved"]
                        }
                    }
                ]
            }
        ]
        start_gemini = time.time()
        parsed_reply = await call_gemini_llm(conversation, tools)
        print("Gemini: ", time.time() - start_gemini)
        response_text = convert_markdown_for_google_chat(parsed_reply["Response"])
        solved = parsed_reply["solved"]

        conversation.append({"role": "model", "content": parsed_reply["Response"], "timestamp": datetime.now().isoformat()})
        save_conversation(user_email, {"conversation": conversation, "Incidents": [e["id"] for e in relevant_incidents]})

        response_obj = {"text": response_text}

        if not solved:
            response_obj["cardsV2"] = [
                {
                    "cardId": "helpOptions",
                    "card": {
                        "header": {
                            "title": "¿Quieres que creemos un ticket para soporte?",
                            "subtitle": "Al haber utilizado este canal, tu ticket será tratado de forma prioritaria"
                        },
                        "sections": [
                            {
                                "widgets": [
                                    {
                                        "buttonList": {
                                            "buttons": [
                                                {
                                                    "text": "Sí",
                                                    "onClick": {
                                                        "action": {
                                                            "function": "createJiraTicket",
                                                            "parameters": [
                                                                {
                                                                    "key": "messages",
                                                                    "value": json.dumps(conversation[1:])
                                                                }
                                                            ]
                                                        }
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]

        return response_obj

    except Exception as e:
        print("ERROR: ", str(e))
        return {
            "text": "Lo siento, ha ocurrido un error y no puedo ayudarte ahora mismo."
        }


async def onClick(data):
    try:
        action = data["action"]["actionMethodName"]

        if action == "createJiraTicket":
            params = data["common"]["parameters"]
            user = data["user"]["email"].split("@")[0]
            messages = json.loads(params["messages"])

            agent = TicketAgent(messages, user)
            ticket_result = await agent.create_ticket()

            ticket_url = ticket_result["url"]

            return {
                "cardsV2": [
                    {
                        "cardId": "TicketCreated",
                        "card": {
                            "header": {
                                "title": "✅ Ticket creado correctamente"
                            },
                            "sections": [
                                {
                                    "widgets": [
                                        {
                                            "buttonList": {
                                                "buttons": [
                                                    {
                                                        "text": "Ver ticket",
                                                        "onClick": {
                                                            "openLink": {
                                                                "url": ticket_url
                                                            }
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }

        if action == "markSolved":
            return {
                "text": "¡Perfecto! Me alegra que hayas podido resolver la incidencia 😊.\n\nSi tienes cualquier otra duda o problema, no dudes en preguntarme."
            }

    except Exception as e:
        return {
            "text": f"❌ Error inesperado al procesar la acción: {str(e)}"
        }