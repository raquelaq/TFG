import json
from ..services.gemini import call_gemini_prompt
from ..services import jira


class TicketAgent:
    def __init__(self, messages, user):
        self.messages = messages
        self.user = user

    async def generate_ticket_contents(self):
        prompt = f"""A partir de los siguientes mensajes entre un chat de soporte y un usuario,
        crea un título corto descriptivo del problema que el usuario está teniendo.
        Además, redacta un resumen claro para que el equipo técnico entienda la situación.

        Devuelve el resultado en formato JSON:
        {{
          "title": "...",
          "summary": "..."
        }}

        Mensajes:
        \"\"\"{json.dumps(self.messages)}\"\"\""""

        tools = [
            {
                "function_declarations": [
                    {
                        "name": "return_json_response",
                        "description": "Devuelve un objeto con 'title' y 'summary'",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "summary": {"type": "string"}
                            },
                            "required": ["title", "summary"]
                        }
                    }
                ]
            }
        ]

        return await call_gemini_prompt(prompt)

    async def create_ticket(self, image_path=None):
        ticket_info = await self.generate_ticket_contents()
        title = ticket_info["title"]
        summary = ticket_info["summary"]

        return await jira.create_jira_ticket(title, summary, image_path=image_path)