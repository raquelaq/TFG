import json
import os
from langchain_core.runnables import Runnable

KB_PATH = "app/data/KnowledgeBase.json"
KB_EMBEDDINGS_PATH = "app/data/kb_embeddings.json"


def load_kb():
    if not os.path.exists(KB_PATH):
        return []

    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print("❌ KB corrupta, devolviendo lista vacía")
        return []
    except Exception as e:
        print(f"❌ Error cargando KB: {e}")
        return []


def save_kb(data):
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def regenerate_embeddings():
    from app.services.KnowledgeBaseFiltering import rebuild_embeddings
    rebuild_embeddings()


class KBManagerNode(Runnable):

    def invoke(self, state, config=None):
        """
        Nodo principal: dirige al técnico al menú correcto
        """
        role = state.get("role")
        user_msg = state.get("user_message", "").lower()

        if role != "tech":
            return {
                "output": "❌ No tienes permisos para gestionar la base de conocimiento.",
                "next": "END"
            }

        # Primer paso: menú de opciones
        if "añadir" in user_msg or "agregar" in user_msg or "nueva entrada" in user_msg:
            state["kb_action"] = "add"
            return {
                "output": "Perfecto. Vamos a **añadir una nueva entrada a la KB**.\n\n"
                          "Dime el **título de la entrada**:",
                "next": "KB_COLLECT_TITLE"
            }

        elif "editar" in user_msg:
            state["kb_action"] = "edit"
            kb = load_kb()
            titles = "\n".join([f"- {item['title']}" for item in kb])
            return {
                "output": f"Perfecto, ¿qué entrada deseas editar?\n\nEntradas disponibles:\n{titles}",
                "next": "KB_SELECT_ENTRY"
            }

        elif "eliminar" in user_msg or "borrar" in user_msg:
            state["kb_action"] = "delete"
            kb = load_kb()
            titles = "\n".join([f"- {item['title']}" for item in kb])
            return {
                "output": f"¿Qué entrada deseas **eliminar**?\n\nEntradas disponibles:\n{titles}",
                "next": "KB_DELETE_ENTRY"
            }

        return {
            "output": "👋 Estás en el **panel técnico**.\n\n"
                      "Puedes decirme:\n"
                      "- “Añadir entrada”\n"
                      "- “Editar entrada”\n"
                      "- “Eliminar entrada”\n",
            "next": "KBManager"
        }


# ---------- SUB-NODOS -------------

class KBCollectTitle(Runnable):
    def invoke(self, state, config=None):
        state["kb_title"] = state["user_message"]
        return {
            "output": "Perfecto. Ahora dime la **descripción del problema**:",
            "next": "KB_COLLECT_DESCRIPTION"
        }


class KBCollectDescription(Runnable):
    def invoke(self, state, config=None):
        state["kb_description"] = state["user_message"]
        return {
            "output": "Muy bien. Ahora dime los **síntomas** (separados por comas):",
            "next": "KB_COLLECT_SYMPTOMS"
        }


class KBCollectSymptoms(Runnable):
    def invoke(self, state, config=None):
        symptoms = [s.strip() for s in state["user_message"].split(",")]
        state["kb_symptoms"] = symptoms

        return {
            "output": "Por último, dime las **palabras clave** (separadas por comas):",
            "next": "KB_SAVE_ENTRY"
        }


class KBSaveEntry(Runnable):
    def invoke(self, state, config=None):
        kb = load_kb()

        # Crear nueva entrada
        new_entry = {
            "id": len(kb) + 1,  # simple autoincrement
            "title": state["kb_title"],
            "description_problem": state["kb_description"],
            "symptoms": state["kb_symptoms"],
            "keywords_tags": [k.strip() for k in state["user_message"].split(",")],
            "steps": []
        }

        kb.append(new_entry)
        save_kb(kb)
        regenerate_embeddings()

        return {
            "output": f"✅ Entrada añadida correctamente:\n\n"
                      f"**{new_entry['title']}**\n\n"
                      "Los embeddings han sido regenerados.",
            "next": "END"
        }
