from ..services import jira


class TicketAgent:
    def __init__(self, messages, user, cached_summary: str | None = None):
        self.messages = messages
        self.user = user
        self.cached_summary = cached_summary

    async def generate_ticket_contents(self):
        user_messages = [
            m["content"]
            for m in self.messages
            if m.get("role") == "user"
        ]

        summary = (
                "\n".join(f"- {msg}" for msg in user_messages)
        )

        return {
            "title": "Incidencia reportada por usuario",
            "summary": summary[:1500]
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
