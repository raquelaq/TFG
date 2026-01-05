import csv
import re
import json
import time
import asyncio
from statistics import mean

from app.agents.support_graph import build_support_graph

GRAPH = build_support_graph()

MODES = ["generative", "hybrid"]
RESULTS = []

def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None



async def run_query(query: str, mode: str):
    state = {
        "user_message": query,
        "user_email": "benchmark@test.com",
        "role": "user",
        "response_mode": mode
    }

    t0 = time.perf_counter()
    try:
        raw_result = await GRAPH.ainvoke(state)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000

        # Caso 1: el grafo ya devuelve dict estructurado (ideal)
        if isinstance(raw_result, dict):
            solved = raw_result.get("solved", False)
            ticket = raw_result.get("action") == "ticket"
            output = raw_result.get("output", "")

            return {
                "latency_ms": latency_ms,
                "solved": solved,
                "ticket": ticket,
                "error": False,
                "parse_error": False,
                "response": output
            }

        # Caso 2: el grafo devuelve texto (modo generativo)
        parsed = extract_json(str(raw_result))

        if parsed:
            solved = parsed.get("solved", False)
            ticket = parsed.get("action") == "ticket"
        else:
            solved = False
            ticket = False

        return {
            "latency_ms": latency_ms,
            "solved": solved,
            "ticket": ticket,
            "error": False,          # ← CLAVE
            "parse_error": parsed is None,
            "response": str(raw_result)
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "latency_ms": (t1 - t0) * 1000,
            "solved": False,
            "ticket": False,
            "error": True,
            "parse_error": False,
            "response": str(e)
        }



def main():
    with open("benchmark_queries.csv", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    for mode in MODES:
        print(f"\n▶ Ejecutando benchmark: {mode.upper()}")

        for row in queries:
            result = asyncio.run(run_query(row["query"], mode))

            RESULTS.append({
                "mode": mode,
                "id": row["id"],
                "query": row["query"],
                "expected": row["expected"],
                **result
            })

            print(
                f"  ✓ [{row['id']}] {row['query'][:35]}... "
                f"({result['latency_ms']:.1f} ms)"
            )

    with open("benchmark_results.jsonl", "w", encoding="utf-8") as f:
        for r in RESULTS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n✔ Benchmark completado. Resultados guardados.")


if __name__ == "__main__":
    main()
