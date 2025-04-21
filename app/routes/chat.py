from fastapi import APIRouter, Request
from ..services.gemini import call_gemini_llm
from ..services.google_docs import read_google_doc
from ..services.jira import create_jira_ticket
from ..services.utils import convert_markdown_for_google_chat, get_conversation, save_conversation,delete_converation_cache, read_kb_file, write_kb_file
from ..config import ID_DRIVE_KB, AUDIENCE
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from datetime import datetime 

router = APIRouter()

@router.get("/delete_cache")
async def delete_cache(request: Request):
    try:        
        delete_converation_cache()
        return {"message": "Cache deleted"}
    except Exception as e:
        print("ERROR: ", str(e))
        return {"message": "Error deleting cache"}
    
@router.post("/reload_kb")
async def reload_kb(request: Request):
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
async def handle_message(request: Request):

    # Imprimir request
    # print("1 - Request Body:", await request.json())
    # print("2 - Request Headers:", request.headers)
    # print("3 - Request URL:", request.url)

    # Respuesta rápida estoy pensando

    # Validar el token de autorización para confirmar que se llama desde el proyecto de ticketing
    # validate_token(request)

    try:

        data = await request.json()
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

        conversation.append({"role": "model", "content": parsed_reply["Response"], "timestamp": datetime.now().isoformat()})
        save_conversation(chat_id, conversation)

        return {
            "text": response_text
            # "show_ticket": not parsed_reply.get("solved", True),
            # "conversation": conversation
        }
    except Exception as e:
        print("ERROR: ", str(e))
        return {
            "text": "Lo siento, ha ocurrido un error y no puedo ayudarte ahora mismo."
        }

def validate_token(request):

    # Bearer Tokens received by apps will always specify this issuer.
    CHAT_ISSUER = 'chat@system.gserviceaccount.com'

    try:

        # Obtener el token del header Authorization
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            print('No Bearer token found')
            return False

        bearer = authorization.split("Bearer ")[1]
        print('TOKEN: ' + bearer)

        # Verify valid token, signed by CHAT_ISSUER, intended for a third party.
        request = google_requests.Request()
        token = id_token.verify_oauth2_token(bearer, request, AUDIENCE)
        print('EMAIL: ' + token['email'])

        return token['email'] == CHAT_ISSUER

    except Exception as e:
        print('ERROR: ' + str(e))
        return False