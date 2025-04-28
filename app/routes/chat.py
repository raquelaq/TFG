from fastapi import APIRouter, Request, Header, HTTPException
from ..services.gemini import call_gemini_llm, call_gemini_prompt
from ..services.google_docs import read_google_doc
from ..services.jira import create_jira_ticket
from ..services.utils import *
from ..config import *
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from datetime import datetime
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


router = APIRouter()

@router.get("/delete_cache")
async def delete_cache(request: Request, api_key_info: dict = Depends(api_key_guard)):
    # try:

        delete_converation_cache()
        return {"message": "Cache deleted"}
    # except Exception as e:
    #     print("ERROR: ", str(e))
    #     return {"message": "Error deleting cache"}
    
@router.post("/reload_kb")
async def reload_kb(request: Request, api_key_info: dict = Depends(api_key_guard)):
    try:
        data = await request.json()
        new_kb_content = data.get("content", "")


        if not new_kb_content:
            return {"message": "No content provided to update the knowledge base."}
        
        write_kb_file(new_kb_content)

        return {"message": "Knowledge base updated successfully."}
    
    except Exception as e:
        print("ERROR: ", str(e))
        return {"message": "Error updating the knowledge base."}

@router.post("/message")
async def handle_message(request: Request, authorization: str = Header(None)):
    try:
        # Step 1: Validate token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")

        token = authorization.split(" ")[1]
        claims = verify_google_chat_token(token, expected_audience=["974882915974", "478283974773"])
        if not claims:
            raise HTTPException(status_code=401, detail="Invalid Google token")

        data = await request.json()
        request_type = data.get("type")

        if request_type == "MESSAGE":
            response = await respond_message(data)
        elif request_type == "CARD_CLICKED":
            print("resolving card click")
            response = await onClick(data)
        else:
            response = { "text": "Lo siento, ha ocurrido un error y no puedo ayudarte ahora mismo." }

        return response


    except Exception as e:
        print("ERROR: ", str(e))
        return {
            "text": "Lo siento, ha ocurrido un error y no puedo ayudarte ahora mismo."
        }


async def respond_message(data):
    try:
        user_message = data.get("message", {}).get("text")
        chat_id = data.get("space", {}).get("name")

        conversation = get_conversation(chat_id)
        if not conversation:
            # guide_text = read_google_doc(ID_DRIVE_KB)
            guide_text = read_kb_file()
            system_msg = {
                "role": "system",
                "content": f"""Actúa como un guía para atender las peticiones más frecuentes que los usuarios de negocio hacen al departamento de soporte informático. 
                                        Inicia la conversación saludando al usuario con una frase creativa y explicando de forma resumida qué consultas puedes resolver.
                                        Responde a los usuarios utilizando la segunda persona del singular. 
                                        Basa tus respuestas unicamente en esta guía de soporte:
    
                                        \"\"\"{guide_text}\"\"\"
    
                                        Usa un tono amigable y se ameno con los usuarios. Si no tienes claro que solución de la guía dar, hazle preguntas al usuario para aclarar
                                        bien que problema tiene antes de responder con una solución. Si haces esto solved debe ser true, para no ofrecer ticket aún.
    
                                        Formatea perfectamente tus respuestas, utilizando viñetas si son necesarias para mejorar la claridad del mensaje.
    
                                        Si la respuesta no se encuentra en la guía responde que no puedes ayudarle a resolver ese problema, propón que creen un ticket para soporte y devuelve solved=false.
                                        Sin embargo si el usuario no ha preguntado nada y no hay que responder a una pregunta devuelve siempre solved=true.
    
                                        Siempre que devuelvas solved=false, que el mensaje al usuario sea: "Lo siento, no puedo ayudarte a resolver este problema. Si que puedo ayudarte a abrir un ticket con soporte.
    
                                        Una vez respondas una respuesta que crees que es valida, explica que la respuesta está basada en lo encontrado en la base de conocimiento al principio del mensaje.""",
                "timestamp": datetime.now().isoformat()

            }
            conversation = [system_msg]

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

        parsed_reply = await call_gemini_llm(conversation, tools)
        response_text = convert_markdown_for_google_chat(parsed_reply["Response"])
        solved = parsed_reply["solved"]
        print(parsed_reply)

        conversation.append({"role": "model", "content": parsed_reply["Response"], "timestamp": datetime.now().isoformat()})
        save_conversation(chat_id, conversation)

        response_obj = {"text": response_text}

        if not solved:
            # Si no se resolvió, añadimos la card
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
                                                                    # desde el primer user message
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
        if data["action"]["actionMethodName"] == "createJiraTicket":
            params = data["common"]["parameters"]
            user = data["user"]["email"].split("@")[0]
            messages = json.loads(params["messages"])

            ticket_info = await create_ticket_contents(messages)
            title = ticket_info["title"]
            summary = ticket_info["summary"]

            url = "https://soporte.satocan.com/rest/api/2/issue"
            payload = {
                "fields": {
                    "project": {"key": "SDS"},
                    "reporter": {"name": user},
                    "summary": title,
                    "description": summary,
                    "issuetype": {"name": "Incidencia"},
                    "labels": ["Ticketing"]
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Basic YXBpamlyYXVzZXI6TW1QWUVQeTd3bDhQTlRpaHhaZmM=",  # <-- tu token base64
                "Cookie": "JSESSIONID=36C703B6C1A9F43882100C71BAB93960; atlassian.xsrf.token=BEGO-HDDW-ESA5-0LD2|9d03d9ee19bc09144abe7d6e463e4ad162f890a7|lin"
            }
            print(payload)

            response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
            code = response.status_code
            content = response.text

            if 200 <= code < 300:
                json_response = response.json()
                ticket_key = json_response["key"]
                ticket_url = f"https://soporte.satocan.com/browse/{ticket_key}"

                response_obj = {
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
                return response_obj

            else:
                return {
                    "text": f"❌ Ocurrió un error al crear el ticket ({code}): {content}"
                }

    except Exception as e:
        return {
            "text": f"❌ Error inesperado al crear el ticket: {str(e)}"
        }


