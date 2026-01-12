import os
import httpx
import requests
import mimetypes

from ..config import JIRA_AUTH_HEADER

BASE_URL = "https://ralmeidaquesada-1765035802101.atlassian.net"

SERVICE_DESK_ID = "1"
REQUEST_TYPE_ID = "1"

def attach_file_to_ticket(issue_key: str, file_path: str):
    filename = os.path.basename(file_path)

    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        mime_type = "application/octet-stream"

    headers = {
        "Authorization": JIRA_AUTH_HEADER,
        "X-Atlassian-Token": "no-check"
    }

    with open(file_path, "rb") as f:
        files = {
            "file": (filename, f, mime_type)
        }

        resp = requests.post(
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
        "serviceDeskId": SERVICE_DESK_ID,
        "requestTypeId": REQUEST_TYPE_ID,
        "requestFieldValues": {
            "summary": title,
            "description": summary
        }
    }

    headers = {
        "Authorization": JIRA_AUTH_HEADER,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/rest/servicedeskapi/request",
            json=payload,
            headers=headers
        )

        print("STATUS:", r.status_code)
        print("RESPONSE:", r.text)

        r.raise_for_status()
        data = r.json()

        issue_key = data["issueKey"]

        if image_path:
            attach_file_to_ticket(issue_key, image_path)

        return {
            "key": issue_key,
            "url": f"{BASE_URL}/servicedesk/customer/portal/1/{issue_key}"
        }
