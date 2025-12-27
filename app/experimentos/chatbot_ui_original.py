import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import asyncio
import nest_asyncio
nest_asyncio.apply()
import requests
from langchain_core.runnables import RunnableConfig

from app.agents.support_graph import build_support_graph
from app.agents.kb_graph import build_kb_graph
from app.services.utils import delete_converation_cache
from app.services.KnowledgeBaseFiltering import initialize_model_and_kb
from app.services.auth import authenticate
from app.agents.ticket_agent import TicketAgent

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

initialize_model_and_kb("app/data/kb_embeddings.json")
compiled_graph = build_support_graph()

SUPPORT_GRAPH = build_support_graph()
KB_GRAPH = build_kb_graph()

async def process_message(user_message: str, prev_state: dict, active_graph) -> dict:

    state = {
        **prev_state,
        "user_message": user_message,
        "role": st.session_state.role,
        "user_email": st.session_state.user_email
    }

    result = await active_graph.ainvoke(
        state,
        config=RunnableConfig(
            run_name="Chat soporte",
            metadata={"usuario": st.session_state.user_email}
        )
    )

    return result

if "inicio" not in st.session_state:
    st.session_state.inicio = True
    st.session_state.logged_in = False
    st.session_state.selected_role = None
    st.session_state.user_email = None
    st.session_state.role = None

    st.session_state.graph_state = {}
    st.session_state.chat_history = []
    st.session_state.active_graph = None

    st.session_state.last_hybrid_query = ""
    st.session_state.pendiente_crear_ticket = None
    st.session_state.esperando_confirmacion = False
    st.session_state.ticket_summary = ""

st.set_page_config(page_title="Asistente de soporte", page_icon="⚙️", layout="centered")

if not st.session_state.logged_in:

    st.title("🔐 Acceso al sistema")
    st.subheader("Selecciona tu tipo de acceso")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Acceder como Usuario"):
            st.session_state.selected_role = "user"

    with col2:
        if st.button("🛠 Acceder como Técnico"):
            st.session_state.selected_role = "tech"

    if st.session_state.get("selected_role"):
        st.write("---")
        st.subheader("Introduce tus credenciales")

        email = st.text_input("Correo corporativo")
        password = st.text_input("Contraseña", type="password")

        if st.button("Iniciar sesión"):

            user = authenticate(email, password)

            if user is None:
                st.error("❌ Credenciales incorrectas.")
                st.stop()

            if user["role"] != st.session_state.selected_role:
                st.error("❌ No tienes permisos para este tipo de acceso.")
                st.stop()

            st.session_state.logged_in = True
            st.session_state.user_email = user["email"]
            st.session_state.role = user["role"]

            if user["role"] == "user":
                st.session_state.active_graph = SUPPORT_GRAPH
            else:
                st.session_state.active_graph = KB_GRAPH

            st.rerun()

    st.stop()

if st.session_state.role == "tech":
    st.title("🛠 Panel Técnico – Gestión de la Base de Conocimiento")
    st.markdown(
        "Aquí puedes añadir nuevas entradas a la base de conocimiento.\n\n"
        "Cuando guardes, los datos se enviarán al **grafo KB de LangGraph**, "
        "que se encarga de actualizar el fichero JSON y regenerar los embeddings."
    )

    st.write("---")
    st.write("### ➕ Nueva entrada en la KB")

    # Campos principales
    new_id = st.text_input("ID único de la incidencia", value=f"INC_{st.session_state.user_email}_1")
    title = st.text_input("Título de la incidencia")
    description = st.text_area("Descripción del problema")

    sintomas_raw = st.text_area("Síntomas (separados por comas)")
    symptoms = [s.strip() for s in sintomas_raw.split(",") if s.strip()]

    keywords_raw = st.text_input("Palabras clave (separadas por comas)")
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    st.markdown("#### Guía generada para el LLM")

    preguntas_raw = st.text_area("Preguntas iniciales (una por línea)")
    initial_questions = [p.strip() for p in preguntas_raw.split("\n") if p.strip()]

    pasos_raw = st.text_area("Pasos de resolución (uno por línea, en tono natural para el usuario)")
    pasos_list = [p.strip() for p in pasos_raw.split("\n") if p.strip()]

    # Construimos diagnostic_steps con una estructura razonable
    diagnostic_steps = []
    for i, texto_paso in enumerate(pasos_list, start=1):
        diagnostic_steps.append({
            "step_number": i,
            "title": f"Paso {i}",
            "llm_instruction": "",
            "user_action": texto_paso
        })

    escalation_criteria = st.text_area(
        "Criterios de escalado (cuándo devolver solved=false para crear ticket)",
        value="Si el usuario no consigue resolver la incidencia tras seguir todos los pasos, devuelve solved=false para generar un ticket a soporte técnico."
    )

    if st.button("💾 Guardar entrada en la KB"):
        if not new_id or not title or not description:
            st.error("⚠️ ID, título y descripción son obligatorios.")
            st.stop()

        # Estado que se envía al grafo KB (no hay conversación, solo datos)
        kb_state = {
            "id": new_id,
            "title": title,
            "description_problem": description,
            "symptoms": symptoms,
            "resolution_guide_llm": {
                "initial_questions": initial_questions,
                "diagnostic_steps": diagnostic_steps
            },
            "escalation_criteria": escalation_criteria,
            "keywords_tags": keywords
        }

        try:
            out = asyncio.run(
                KB_GRAPH.ainvoke(
                    kb_state,
                    config=RunnableConfig(
                        run_name="KB Manager",
                        metadata={"usuario": st.session_state.user_email}
                    )
                )
            )
            st.success(out.get("output", "✅ Entrada procesada correctamente."))

        except Exception as e:
            st.error(f"❌ Error guardando la entrada: {e}")

    st.write("---")
    if st.button("⏪ Cerrar sesión"):
        st.session_state.clear()
        st.rerun()

    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;900&display=swap');

body {
    background-color: #1e1e1e;
}

.snowchat-title {
   font-size: 48px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(-45deg, #f3ec78, #af4261, #66ffff, #cc66ff);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 6s ease infinite;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.message {
    display: flex;
    margin: 10px 0;
    gap: 10px;
    align-items: flex-start;
}

.message.user {
    justify-content: flex-end;
}

.message.bot {
    justify-content: flex-start;
}

.bubble {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 1rem;
    line-height: 1.5;
    box-shadow: 0px 0px 5px rgba(0,0,0,0.3);
}

.bubble.user {
    background-color: #1e88e5;
    color: white;
    border-bottom-right-radius: 4px;
}

.bubble.bot {
    background-color: #3c3f41;
    color: white;
    border-bottom-left-radius: 4px;
}

.avatar {
    font-size: 1.8em;
}
</style>

<div class="snowchat-title">Asistente de Soporte</div>
""", unsafe_allow_html=True)

st.caption(f"Sesión iniciada como: **{st.session_state.user_email}**")

if st.sidebar.button("Cerrar sesión"):
    st.session_state.clear()
    st.rerun()

st.caption("Cuéntame tu incidencia y trataré de ayudarte")

st.sidebar.title("Instrucciones")
st.sidebar.markdown("""
- Describe tu problema con claridad.
- Este asistente es automático.
- Puedes usar modo IA generativa o modo modelo ML (híbrido).
""")

if "modo_respuesta" not in st.session_state:
    st.session_state.modo_respuesta = "IA Generativa"

modo = st.radio(
    "Selecciona el tipo de modelo a usar:",
    ["IA Generativa", "Modelo ML (embeddings)"],
    index=0
)
st.session_state.modo_respuesta = modo

if st.sidebar.button("Reiniciar conversación"):
    st.session_state.graph_state = {}
    st.session_state.chat_history.append({
        "role": "bot",
        "content": "Ok, empezamos de cero. ¿Qué incidencia tienes ahora?"
    })
    st.session_state.esperando_confirmacion = False
    st.session_state.pendiente_crear_ticket = None

user_input = st.chat_input("Escribe tu consulta aquí...")

if user_input:
    st.session_state.esperando_confirmacion = False
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if st.session_state.modo_respuesta == "IA Generativa":
        try:

            bot_state = asyncio.run(
                process_message(
                    user_input,
                    st.session_state.graph_state,
                    st.session_state.active_graph
                )
            )

            st.session_state.graph_state = bot_state

            bot_response = bot_state.get("output", "Sin respuesta")
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})

            out_flag = (bot_state.get("__output__") or "").lower()

            if out_flag == "ticket" or "ticket" in bot_response.lower():
                try:
                    from app.agents.ticket_agent import TicketAgent

                    messages = st.session_state.chat_history
                    agent = TicketAgent(messages, user="ralmeidaquesada")

                    ticket_contents = asyncio.run(agent.generate_ticket_contents())
                    st.session_state.ticket_summary = ticket_contents["summary"]
                    st.session_state.pendiente_crear_ticket = True

                except Exception as e:
                    st.session_state.ticket_summary = "No se pudo generar el resumen automáticamente."
                    st.session_state.pendiente_crear_ticket = True

            if any(x in bot_response.lower() for x in [
                "hemos completado todos los pasos",
                "incidencia resuelta",
                "problema resuelto"
            ]):
                st.session_state.graph_state = {}
                st.session_state.pendiente_crear_ticket = None

        except Exception as e:
            bot_response = f"Error de conexión: {e}"
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})

    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/message",
                json={
                    "type": "MESSAGE",
                    "modo_respuesta": st.session_state.modo_respuesta,
                    "message": {
                        "text": user_input,
                        "sender": {
                            "email": "usuario_streamlit@local.test"
                        }
                    },
                    "space": {"name": "chat-streamlit"}
                }
            )

            resp_json = response.json()
            bot_response = resp_json.get("text", "Error procesando la respuesta.")
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})

            if resp_json.get("need_feedback"):
                st.session_state.esperando_confirmacion = True
                st.session_state.last_hybrid_query = user_input
            else:
                if (
                    resp_json.get("ticket_suggested")
                    or "no se encontró una solución" in bot_response.lower()
                    or "no he encontrado una solución clara" in bot_response.lower()
                    or "crear un ticket" in bot_response.lower()
                    or "abrir un ticket" in bot_response.lower()
                ):
                    st.session_state.pendiente_crear_ticket = user_input

        except Exception as e:
            bot_response = f"Error de conexión: {e}"
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})


for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="message user">
            <div class="bubble user">{msg["content"]}</div>
            <div class="avatar">👤</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message bot">
            <div class="avatar">🤖</div>
            <div class="bubble bot">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)

        if "url" in msg:
            st.markdown(
                f'<a href="{msg["url"]}" target="_blank">'
                '<button style="margin:8px 0; padding:8px 16px; border:none; border-radius:8px; background-color:#1e88e5; color:white; cursor:pointer;">🔗 Abrir ticket en JIRA</button></a>',
                unsafe_allow_html=True
            )

        if (
            "pendiente_crear_ticket" in st.session_state
            and st.session_state.pendiente_crear_ticket
            and i == len(st.session_state.chat_history) - 1
            and any(x in msg["content"].lower() for x in [
                "ticket", "no puedo ayudarte", "no se encontró una solución"
            ])
        ):
            with st.expander("Crear ticket de soporte"):
                with st.form("ticket_form"):
                    descripcion = st.text_area(
                        "Descripción del problema",
                        value=st.session_state.get("ticket_summary", "")
                    )
                    imagen = st.file_uploader(
                        "Adjuntar imagen (opcional)",
                        type=["png", "jpg", "jpeg"]
                    )
                    submitted = st.form_submit_button("Enviar ticket")

                    if submitted:
                        image_path = None
                        if imagen is not None:
                            image_path = f"temp_{imagen.name}"
                            with open(image_path, "wb") as f:
                                f.write(imagen.getbuffer())

                        try:
                            from app.agents.ticket_agent import TicketAgent
                            import asyncio
                            if sys.platform.startswith("win"):
                                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

                            messages = [{"role": "user", "content": descripcion}]
                            agent = TicketAgent(messages, user="ralmeidaquesada")
                            ticket_result = asyncio.run(agent.create_ticket(image_path=image_path))

                            ticket_url = ticket_result["url"]
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "content": f"Ticket creado en JIRA",
                                "url": ticket_result["url"]
                            })
                            st.link_button("🔗 Abrir ticket en JIRA", ticket_url)
                            st.session_state.pendiente_crear_ticket = None
                            st.success(f"Ticket enviado a JIRA correctamente. ID: {ticket_result['key']}")
                            st.session_state.graph_state = {}

                        except Exception as e:
                            st.error(f"❌ Error enviando ticket a JIRA: {e}")


if st.session_state.get("esperando_confirmacion"):
    st.markdown("**¿El problema quedó resuelto?**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sí, se ha solucionado", key="btn_solved_hybrid"):
            st.session_state.chat_history.append({
                "role": "bot",
                "content": "¡Perfecto! Me alegra que hayas podido resolver la incidencia 😊. "
                           "Si tienes cualquier otra duda, aquí estaré."
            })
            st.session_state.esperando_confirmacion = False

    with col2:
        if st.button("No, quiero crear un ticket", key="btn_not_solved_hybrid"):
            st.session_state.chat_history.append({
                "role": "bot",
                "content": "De acuerdo, vamos a crear un ticket para que soporte técnico pueda revisarlo en detalle."
            })
            st.session_state.pendiente_crear_ticket = st.session_state.get("last_hybrid_query", "")
            st.session_state.esperando_confirmacion = False
