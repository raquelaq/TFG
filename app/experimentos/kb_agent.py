from datetime import datetime
from app.services.gemini import call_gemini_llm
from app.services.utils import read_kb_file
from app.services.kb_json_loader import load_kb_from_json

class KBResponseAgent:
    def __init__(self, chat_id, conversation):
        self.chat_id = chat_id
        self.conversation = conversation or []
        self.guide_text = load_kb_from_json()

    def get_system_message(self):
        return {
            "role": "system",
            "content": f"""
Actúa como un asistente de soporte. Utiliza exclusivamente la siguiente base de conocimiento para responder:

\"\"\"{self.guide_text}\"\"\"

Responde en un tono amistoso. Si no puedes ayudar, indica que se cree un ticket. Usa el campo 'solved' para indicar si se resolvió el problema.
""",
            "timestamp": datetime.now().isoformat()
        }

    async def respond(self, user_message):
        if not self.conversation:
            self.conversation.append(self.get_system_message())

        self.conversation.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        tools = [
            {
                "function_declarations": [{
                    "name": "return_json_response",
                    "description": "Respuesta con campos 'Response' y 'solved'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Response": {"type": "string"},
                            "solved": {"type": "boolean"}
                        },
                        "required": ["Response", "solved"]
                    }
                }]
            }
        ]

        result = await call_gemini_llm(self.conversation, tools)
        return result, self.conversation