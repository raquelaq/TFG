import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import asyncio
import nest_asyncio
nest_asyncio.apply()
from langchain_core.runnables import RunnableConfig

from app.config import DATA_DIR
from app.agents.support_graph import build_support_graph
#from app.agents.kb_graph import build_kb_graph
from app.services.KnowledgeBaseFiltering import initialize_model_and_kb
from app.services.auth import authenticate
from app.agents.ticket_agent import TicketAgent

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@st.cache_resource
def load_support_graph():
    return build_support_graph()

SUPPORT_GRAPH = load_support_graph()

async def process_message(user_message: str, prev_state: dict) -> dict:
    state = {
        **prev_state,
        "user_message": user_message,
        "role": st.session_state.role,
        "user_email": st.session_state.user_email,
        "response_mode": (
            "generative"
            if st.session_state.modo_respuesta == "IA Generativa"
            else "hybrid"
        )
    }

    result = await SUPPORT_GRAPH.ainvoke(
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

    st.session_state.pendiente_crear_ticket = None
    st.session_state.ticket_summary = ""

st.set_page_config(page_title="Asistente de soporte", page_icon="⚙️", layout="centered")

if not st.session_state.logged_in:

    st.title("Acceso al sistema")
    st.subheader("Selecciona tu tipo de acceso")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Acceder como Usuario"):
            st.session_state.selected_role = "user"

    with col2:
        if st.button("Acceder como Técnico"):
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

            if user["role"] == "tech":
                initialize_model_and_kb(
                    str(DATA_DIR / "kb_embeddings.json"),
                    force_reload=True
                )

            st.rerun()

    st.stop()

if st.session_state.role == "tech":
    st.title("Panel Técnico")
    st.markdown(
        "Aquí puedes añadir nuevas entradas a la base de conocimiento.\n\n"
    )

    st.write("---")

    st.write("### Nueva entrada en la KB")

    new_id = st.text_input("ID único de la incidencia. Escribir sin espacios, con guiones bajos", value=f"nombre_de_la_inicidenica")
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

    if st.button("Guardar entrada"):
        if not new_id or not title or not description:
            st.error("⚠️ ID, título y descripción son obligatorios.")
            st.stop()

        kb_state = {
            "role": "tech",
            "user_email": st.session_state.user_email,

            "user_message": "__kb_operation__",

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
                SUPPORT_GRAPH.ainvoke(
                    kb_state,
                    config=RunnableConfig(
                        run_name="KB Manager",
                        metadata={"usuario": st.session_state.user_email}
                    )
                )
            )
            st.success(out.get("output", "✅ Entrada procesada correctamente."))
            initialize_model_and_kb(str(DATA_DIR / "kb_embeddings.json"), force_reload=True)

        except Exception as e:
            st.error(f"❌ Error guardando la entrada: {e}")

    if st.button("Ver base de conocimiento"):
        from app.services.KnowledgeBaseFiltering import (
            KB_CORPUS_DATA,
            initialize_model_and_kb
        )

        initialize_model_and_kb(str(DATA_DIR / "kb_embeddings.json"))

        if not KB_CORPUS_DATA:
            st.warning("La base de conocimiento está vacía.")
        else:
            st.subheader("Base de conocimiento")
            st.caption(f"{len(KB_CORPUS_DATA)} incidencias registradas")

            for incident in KB_CORPUS_DATA:
                with st.expander(f"{incident.get('id')} – {incident.get('title')}"):
                    st.markdown(f"**Descripción:** {incident.get('description_problem', '')}")

                    keywords = incident.get("keywords_tags", [])
                    if keywords:
                        st.markdown(f"**Palabras clave:** {', '.join(keywords)}")

                    pasos = incident.get("resolution_guide_llm", {}).get("diagnostic_steps", [])
                    if pasos:
                        st.markdown("**Pasos de resolución:**")
                        for i, step in enumerate(pasos, 1):
                            st.markdown(f"- **Paso {i}:** {step.get('user_action', '')}")

    st.write("---")
    if st.button("Cerrar sesión"):
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

st.caption("Cuéntame tu incidencia y trataré de ayudarte")

st.sidebar.title("Instrucciones")
st.sidebar.markdown("""
- Describe tu problema con claridad.
- Puedes elegir entre IA generativa o modelo ML (híbrido).
- Sigue las indicaciones del asistente paso a paso.
- Si no se resuelve, podrás crear un ticket de soporte.
""")

if st.sidebar.button("Cerrar sesión"):
    st.session_state.clear()
    st.rerun()

if st.sidebar.button("Reiniciar conversación"):
    st.session_state.graph_state = {}
    st.session_state.chat_history = []
    st.session_state.esperando_confirmacion = False
    st.session_state.pendiente_crear_ticket = None

    from app.services.utils import delete_conversation_cache_user
    delete_conversation_cache_user(user=st.session_state.user_email)

    st.rerun()

if "modo_respuesta" not in st.session_state:
    st.session_state.modo_respuesta = "IA Generativa"

modo = st.radio(
    "Selecciona el tipo de modelo a usar:",
    ["IA Generativa", "Modelo Híbrido (BM25 + embeddings)"],
    index=0
)
st.session_state.modo_respuesta = modo


user_input = st.chat_input("Escribe tu consulta aquí...")

if user_input:
    st.session_state.esperando_confirmacion = False
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        bot_state = asyncio.run(
            process_message(
                user_input,
                st.session_state.graph_state
            )
        )

        st.session_state.graph_state = bot_state

        bot_response = bot_state.get("output", "Sin respuesta")
        st.session_state.chat_history.append(
            {"role": "bot", "content": bot_response}
        )

        if bot_state.get("action") == "ticket":
            try:
                messages = st.session_state.chat_history
                agent = TicketAgent(messages, user="ralmeidaquesada")

                ticket_contents = asyncio.run(
                    agent.generate_ticket_contents()
                )

                st.session_state.ticket_summary = ticket_contents["summary"]
                st.session_state.pendiente_crear_ticket = True

            except Exception:
                st.session_state.ticket_summary = (
                    "No se pudo generar el resumen automáticamente."
                )
                st.session_state.pendiente_crear_ticket = True

        if bot_state.get("solved") is True:
            st.session_state.graph_state = {}

    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "bot", "content": f"❌ Error: {e}"}
        )


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

if st.session_state.get("pendiente_crear_ticket"):

    with st.expander("Crear ticket de soporte", expanded=True):
        with st.form(key="ticket_form_unico"):

            descripcion = st.text_area(
                "Descripción del problema (puedes editarlo si lo consideras necesario)",
                value=st.session_state.get("ticket_summary", "")
            )

            imagen = st.file_uploader(
                "Adjuntar imagen (opcional)",
                type=["png", "jpg", "jpeg"]
            )

            submitted = st.form_submit_button("Enviar ticket")

            if submitted:
                try:
                    image_path = None

                    if imagen is not None:
                        temp_dir = "tmp_uploads"
                        os.makedirs(temp_dir, exist_ok=True)

                        image_path = os.path.join(temp_dir, imagen.name)

                        with open(image_path, "wb") as f:
                            f.write(imagen.getbuffer())

                    agent = TicketAgent(
                        [{"role": "user", "content": descripcion}],
                        user=st.session_state.user_email
                    )

                    ticket_result = asyncio.run(
                        agent.create_ticket(image_path=image_path)
                    )

                    st.session_state.chat_history.append({
                        "role": "bot",
                        "content": "Ticket creado correctamente.",
                        "url": ticket_result["url"]
                    })

                    st.session_state.pendiente_crear_ticket = False
                    st.session_state.graph_state = {}

                    st.success(f"Ticket enviado a JIRA. ID: {ticket_result['key']}")

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error enviando ticket a JIRA: {e}")