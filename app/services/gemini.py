
import httpx
from ..config import GEMINI_API_KEY

async def call_gemini_llm(conversation_history, tools):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "system_instruction": {
            "parts": [{"text": conversation_history[0]["content"]}]
        },
        "contents": [
            {
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            } for msg in conversation_history[1:]
        ],
        "tools": tools,
        "tool_config": {
            "function_calling_config": { "mode": "ANY" }
        }
    }

    print(payload)

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        print("Response: ", data)
        return data["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]




async def call_gemini_prompt(prompt_text, tools):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ],
        "tools": tools,
        "tool_config": {
            "function_calling_config": {"mode": "ANY"}
        }
    }


    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]