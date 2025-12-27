import asyncio
import httpx
from langsmith import traceable
from ..config import GEMINI_API_KEY
import google.generativeai as genai

MAX_RETRIES = 5
BACKOFF_BASE = 1

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


@traceable(name="Gemini Call")
async def call_gemini_llm(conversation_history, tools=None):

    prompt_parts = []

    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            prompt_parts.append(f"{role.upper()}: {content}")

    prompt = "\n".join(prompt_parts)

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )

        text = response.text.strip() if response.text else ""

        solved = True
        if "no puedo ayudarte" in text.lower() or "ticket" in text.lower():
            solved = False

        return {
            "Response": text,
            "solved": solved
        }


    except Exception as e:

        print("❌ Gemini error:", repr(e))

        return {
            "Response": (
                "Ahora mismo no puedo generar la respuesta automáticamente. "
                "Si lo deseas, puedo ayudarte a crear un ticket de soporte."
            ),
            "solved": False
        }


async def call_gemini_prompt(prompt_text: str) -> str:

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            prompt_text
        )
        return response.text.strip() if response.text else ""

    except Exception:
        return (
            "Ahora mismo no puedo generar la respuesta automáticamente."
        )



# @traceable(name="Gemini Call")
# async def call_gemini_llm(conversation_history, tools):
#     url = (
#         "https://generativelanguage.googleapis.com/v1beta/"
#         f"models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
#     )
#
#     payload = {
#         "system_instruction": {
#             "parts": [{"text": conversation_history[0]["content"]}]
#         },
#         "contents": [
#             {
#                 "role": msg["role"],
#                 "parts": [{"text": msg["content"]}]
#             }
#             for msg in conversation_history[1:]
#         ],
#         "tools": tools,
#         "tool_config": {
#             "function_calling_config": {"mode": "ANY"}
#         }
#     }
#
#     for attempt in range(MAX_RETRIES):
#         try:
#             async with httpx.AsyncClient(timeout=30) as client:
#                 response = await client.post(url, json=payload)
#                 response.raise_for_status()
#                 data = response.json()
#
#                 return data["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
#
#         except httpx.HTTPStatusError as e:
#             if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
#                 wait = BACKOFF_BASE * (2 ** attempt)
#                 print(f"⚠️ Gemini rate limit (429). Reintentando en {wait:.1f}s...")
#                 await asyncio.sleep(wait)
#                 continue
#             raise
#
#
# async def call_gemini_prompt(prompt_text, tools):
#     url = (
#         "https://generativelanguage.googleapis.com/v1beta/"
#         f"models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
#     )
#
#     payload = {
#         "contents": [
#             {
#                 "role": "user",
#                 "parts": [{"text": prompt_text}]
#             }
#         ],
#         "tools": tools,
#         "tool_config": {
#             "function_calling_config": {"mode": "ANY"}
#         }
#     }
#
#     for attempt in range(MAX_RETRIES):
#         try:
#             async with httpx.AsyncClient(timeout=30) as client:
#                 response = await client.post(url, json=payload)
#                 response.raise_for_status()
#                 data = response.json()
#
#                 return data["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]
#
#         except httpx.HTTPStatusError as e:
#             if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
#                 wait = BACKOFF_BASE * (2 ** attempt)
#                 print(f"⚠️ Gemini rate limit (429). Reintentando en {wait:.1f}s...")
#                 await asyncio.sleep(wait)
#                 continue
#             raise
