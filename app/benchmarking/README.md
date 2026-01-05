# Benchmark de modelos – TFG

Este directorio contiene los experimentos realizados para comparar:

- Modelo de IA generativa (Gemini)
- Modelo híbrido (BM25 + embeddings)

## Metodología

- Las consultas se definen manualmente en `benchmark_queries.csv`
- Se invoca directamente el grafo de LangGraph
- No se utiliza FastAPI para evitar latencias artificiales
- Se miden latencia, precisión, tasa de tickets y errores

## Ejecución

```bash
python benchmark_run.py
python analyze_results.py
