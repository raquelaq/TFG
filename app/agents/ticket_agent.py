import json
from ..services.gemini import call_gemini_prompt
from ..services import jira


class TicketAgent:
    def __init__(self, messages, user, cached_summary: str | None = None):
        self.messages = messages
        self.user = user
        self.cached_summary = cached_summary

    async def generate_ticket_contents(self):
        if self.cached_summary:
            return {
                "title": "Incidencia reportada por usuario",
                "summary": self.cached_summary
            }

        prompt = f"""
    A partir de los siguientes mensajes entre un chat de soporte y un usuario,
    crea un título corto descriptivo del problema y un resumen claro para el equipo técnico.
    Devuelve el resultado en formato JSON **SIN TEXTO ADICIONAL**:

    {{ "title": "...", "summary": "..." }}

    Mensajes:
    \"\"\"{json.dumps(self.messages, ensure_ascii=False)}\"\"\"
    """

        raw = await call_gemini_prompt(prompt)

        # 🔒 BLINDAJE OBLIGATORIO
        try:
            parsed = json.loads(raw)
            return {
                "title": parsed.get("title", "Incidencia reportada por usuario"),
                "summary": parsed.get("summary", raw)
            }
        except json.JSONDecodeError:
            # Fallback si Gemini no cumple
            return {
                "title": "Incidencia reportada por usuario",
                "summary": raw
            }

    async def create_ticket(self, image_path=None):
        ticket_info = await self.generate_ticket_contents()
        title = ticket_info["title"]
        summary = ticket_info["summary"]

        return await jira.create_jira_ticket(
            title=title,
            summary=summary,
            image_path=image_path
        )
