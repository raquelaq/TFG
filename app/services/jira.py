
import httpx
from ..config import JIRA_AUTH_HEADER

async def create_jira_ticket(title: str, summary: str, reporter: str):
    payload = {
        "fields": {
            "project": { "key": "SDS" },
            "reporter": { "name": reporter },
            "summary": title,
            "description": summary,
            "issuetype": { "name": "Incidencia" },
            "labels": ["Ticketing"]
        }
    }

    headers = {
        "Authorization": f"Basic {JIRA_AUTH_HEADER}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("https://soporte.satocan.com/rest/api/2/issue", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["key"]
