from app.services.gemini import call_gemini_prompt
import json

INTENT_PROMPT_TEMPLATE = """
Si no, Clasifica la intención del siguiente mensaje en una de estas categorías:

- KB_RESPONSE: si el usuario está preguntando algo técnico que podría estar en una guía de soporte.
- UNKNOWN: si no está claro lo que quiere.

Responde en formato JSON: {{ "intent": "..." }}

Mensaje:
\"\"\"{user_message}\"\"\"
"""

INTENT_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "return_json_response",
                "description": "Clasifica el mensaje en base a su intención",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": ["KB_RESPONSE", "UNKNOWN"]
                        }
                    },
                    "required": ["intent"]
                }
            }
        ]
    }
]

async def detect_intent(user_message: str) -> str:
    prompt = INTENT_PROMPT_TEMPLATE.format(user_message=user_message)
    raw = await call_gemini_prompt(prompt)

    raw = raw.strip()

    try:
        data = json.loads(raw)
        intent = data.get("intent", "").upper()
    except json.JSONDecodeError:
        intent = raw.upper()

    if "KB_RESPONSE" in intent:
        return "KB_RESPONSE"

    return "UNKNOWN"