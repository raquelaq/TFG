# app/services/embeddings_index.py
import os, json, re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

KB_JSON_PATH = os.getenv("KB_JSON_PATH", "app/data/KnowledgeBase.json")
EMB_CACHE_PATH = os.getenv("KB_EMBEDDINGS_CACHE", "app/data/kb_embeddings.json")
EMB_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\sáéíóúñü]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _compose_entry_text(entry: Dict[str, Any],
                        field_weights: Dict[str, float]) -> Tuple[str, Dict[str, str]]:
    fields_raw = {
        "title": entry.get("title", ""),
        "description": entry.get("description_problem", ""),
        "symptoms": " ".join(entry.get("symptoms", [])),
        "tags": " ".join(entry.get("keywords_tags", [])),
        "diagnostics": " ".join(
            f"{step.get('title','')} {step.get('user_action','')}"
            for step in entry.get("resolution_guide_llm", {}).get("diagnostic_steps", [])
        ),
        "escalation": entry.get("escalation_criteria", ""),
    }
    fields_norm = {k: _norm(v) for k, v in fields_raw.items()}
    text_weighted = []
    for k, v in fields_norm.items():
        w = field_weights.get(k, 1.0)
        if not v:
            continue
        reps = max(1, int(round(w)))
        text_weighted.extend([v] * reps)
    return ". ".join(text_weighted), fields_norm

class EmbeddingsIndex:
    def __init__(self,
                 kb_path: str = KB_JSON_PATH,
                 cache_path: str = EMB_CACHE_PATH,
                 model_path: str = EMB_MODEL_PATH,
                 field_weights: Optional[Dict[str, float]] = None):
        self.kb_path = kb_path
        self.cache_path = cache_path
        self.model_path = model_path
        self.field_weights = field_weights or {
            "title": 2.0,
            "description": 1.5,
            "symptoms": 1.2,
            "tags": 1.2,
            "diagnostics": 1.5,
            "escalation": 1.0
        }
        self._model: Optional[SentenceTransformer] = None
        self._kb: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._corpus_texts: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_path, device=device)

    def _load_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        self._kb = data
        self._ids = [str(e.get("id")) for e in data]

    def _build_corpus(self):
        self._corpus_texts = []
        for e in self._kb:
            txt, _ = _compose_entry_text(e, self.field_weights)
            self._corpus_texts.append(txt)

    def _save_cache(self):
        obj = {
            "ids": self._ids,
            "embeddings": self._embeddings.tolist()
        }
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def _load_cache(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if obj.get("ids") != self._ids:
                # La KB cambió; no usamos caché antiguo
                return False
            self._embeddings = np.array(obj["embeddings"], dtype=np.float32)
            return True
        except Exception:
            return False

    def initialize(self, force_rebuild: bool = False):
        self._load_model()
        self._load_kb()
        self._build_corpus()

        if not force_rebuild and self._load_cache():
            return

        emb = self._model.encode(self._corpus_texts, convert_to_tensor=False, normalize_embeddings=True)
        self._embeddings = np.asarray(emb, dtype=np.float32)
        self._save_cache()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self._embeddings is None:
            raise RuntimeError("Index not initialized. Call initialize() first.")
        q = _norm(query)
        q_vec = self._model.encode([q], convert_to_tensor=False, normalize_embeddings=True)
        sims = cosine_similarity(q_vec, self._embeddings)[0]  # shape: (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_idx:
            item = self._kb[i].copy()
            item["_similarity"] = float(sims[i])
            results.append(item)
        return results
