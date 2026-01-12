import json
from statistics import mean
from collections import defaultdict

results = []

with open("benchmark_results.jsonl", encoding="utf-8") as f:
    for line in f:
        results.append(json.loads(line))


def analyze(mode):
    subset = [r for r in results if r["mode"] == mode]
    total = len(subset)

    latency = mean(r["latency_ms"] for r in subset)
    error_rate = sum(r["error"] for r in subset) / total
    ticket_rate = sum(r["ticket"] for r in subset) / total

    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for r in subset:
        expected = r["expected"]
        predicted = "solved" if r["solved"] else "ticket"

        if expected == "solved" and predicted == "solved":
            cm["TP"] += 1
        elif expected == "ticket" and predicted == "solved":
            cm["FP"] += 1
        elif expected == "solved" and predicted == "ticket":
            cm["FN"] += 1
        elif expected == "ticket" and predicted == "ticket":
            cm["TN"] += 1

    accuracy = (cm["TP"] + cm["TN"]) / total

    fpr = cm["FP"] / (cm["FP"] + cm["TN"]) if (cm["FP"] + cm["TN"]) > 0 else 0
    fnr = cm["FN"] / (cm["FN"] + cm["TP"]) if (cm["FN"] + cm["TP"]) > 0 else 0

    return {
        "latency_ms": latency,
        "accuracy": accuracy,
        "ticket_rate": ticket_rate,
        "error_rate": error_rate,
        "confusion_matrix": cm,
        "fpr": fpr,
        "fnr": fnr
    }


for mode in ["generative", "hybrid"]:
    stats = analyze(mode)

    print(f"\nRESULTADOS {mode.upper()}")
    print(f"- Latencia media: {stats['latency_ms']:.2f} ms")
    print(f"- Precisión (expected vs output): {stats['accuracy']*100:.1f}%")
    print(f"- Ticket rate: {stats['ticket_rate']*100:.1f}%")
    print(f"- Error rate: {stats['error_rate']*100:.1f}%")

    cm = stats["confusion_matrix"]
    print("Matriz de confusión:")
    print(f"  TP: {cm['TP']} | FP: {cm['FP']} | FN: {cm['FN']} | TN: {cm['TN']}")

    print(f"- False Positive Rate (FPR): {stats['fpr']*100:.1f}%")
    print(f"- False Negative Rate (FNR): {stats['fnr']*100:.1f}%")
