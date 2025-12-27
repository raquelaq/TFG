import os
import httpx
from ..config import JIRA_AUTH_HEADER

BASE_URL = "https://ralmeidaquesada-1765035802101.atlassian.net"

async def attach_file_to_ticket(issue_key: str, file_path: str):
    headers = {
        "Authorization": JIRA_AUTH_HEADER,
        "X-Atlassian-Token": "no-check"
    }

    filename = os.path.basename(file_path)
    files = {
        "file": (filename, open(file_path, "rb"), "application/octet-stream")
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BASE_URL}/rest/api/3/issue/{issue_key}/attachments",
            headers=headers,
            files=files
        )
        resp.raise_for_status()
        return resp.json()

async def create_jira_ticket(
    title: str,
    summary: str,
    image_path: str | None = None
) -> dict:

    payload = {
        "fields": {
            "project": {"key": "KAN"},
            "summary": title,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": summary}
                        ]
                    }
                ]
            },
            "issuetype": {"name": "Task"},  # 👈 CLAVE
            "labels": ["Ticketing"]
        }
    }

    headers = {
        "Authorization": JIRA_AUTH_HEADER,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/rest/api/3/issue",
            json=payload,
            headers=headers
        )
        r.raise_for_status()
        data = r.json()

        issue_key = data["key"]

        if image_path:
            await attach_file_to_ticket(issue_key, image_path)

        return {
            "key": issue_key,
            "url": f"{BASE_URL}/browse/{issue_key}"
        }
