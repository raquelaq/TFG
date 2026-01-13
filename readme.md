# Asistente Corporativo de Soporte Técnico  
## Arquitectura multiagente con LangGraph, IA generativa y modelo híbrido

Este repositorio contiene un asistente de soporte técnico corporativo desarrollado en Python, basado en una arquitectura multiagente implementada con LangGraph. El sistema permite atender incidencias de usuarios finales y gestionar una base de conocimiento interna por parte de técnicos, integrando distintos enfoques de recuperación y generación de respuestas.

El proyecto ha sido desarrollado como Trabajo de Fin de Grado y prioriza la modularidad, la trazabilidad del flujo de decisión y la capacidad de evolución del sistema.

---

## Descripción general del sistema

El núcleo del sistema se implementa como un grafo de ejecución único, en el que el flujo de interacción se decide dinámicamente en función del estado de la conversación y del rol del usuario. Dentro de este grafo se integran los siguientes procesos:

- Atención a usuarios finales
- Selección dinámica del modo de respuesta (IA generativa o modelo híbrido)
- Escalado automático a ticket de soporte técnico
- Gestión de la base de conocimiento por parte de técnicos (alta de nuevas incidencias)

La interacción principal con el sistema se realiza mediante una interfaz web desarrollada con Streamlit, que actúa como cliente directo del grafo multiagente.

---

## Tecnologías utilizadas

- Python 3.11
- Streamlit (interfaz de usuario)
- LangGraph (modelado del flujo multiagente)
- LangChain (abstracciones para LLM y herramientas)
- Google Gemini (modelo de lenguaje generativo)
- Sentence Transformers (embeddings semánticos)
- BM25 (recuperación léxica)
- FastAPI (capa API opcional, preparada para despliegue futuro)
- JIRA API (creación de tickets de soporte)

---

## Requisitos del sistema

### Requisitos generales

- Python 3.11 o superior
- Entorno virtual recomendado (venv o conda)
- Clave de API válida para Google Gemini
- Credenciales de JIRA para la creación de tickets

---

## Instalación

1. Clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```
 2. Crear y activar un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```
3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Variables de entorno
El proyecto utiliza un archivo .env para la configuración local. Debe incluir, al menos, las siguientes variables:

```ini
[GENERAL]
GEMINI_API_KEY=tu_clave_api_gemini
JIRA_AUTH_HEADER=tu_token_jira
```

Otros parámetros internos (rutas de datos, configuración de la base de conocimiento, etc.) se gestionan desde el módulo `app/config.py`.

## Ejecución del sistema

### Interfaz Streamlit (modo principal)

Este es el modo de ejecución principal del proyecto y el utilizado durante el desarrollo y la evaluación del TFG.

```bash
streamlit run app/chatbot_ui.py
```

Desde la interfaz es posible:

- Acceder como usuario o técnico 
- Consultar incidencias de soporte 
- Seleccionar el modo de respuesta (IA generativa o modelo híbrido)
- Escalar incidencias a ticket de soporte 
- Añadir nuevas entradas a la base de conocimiento

## Ejecución como API (opcional)

El proyecto incluye una capa FastAPI preparada para una posible separación futura entre frontend y backend o para su despliegue en entornos productivos.

Esta capa no es necesaria para el funcionamiento actual del sistema vía Streamlit.

Para ejecutar la API localmente:

```bash
streamlit run app/chatbot_ui.py
```
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Arquitectura del grafo

El sistema se modela como un único grafo de estados, con nodos especializados que representan las distintas responsabilidades del asistente. Entre los nodos principales se incluyen:

- RouteByRole 
- RouteByResponseMode 
- GenerativeResponse 
- HybridResponse 
- Ticket 
- KB_SaveEntry

Las transiciones entre nodos no son estáticas, sino que se determinan dinámicamente en función del estado de la conversación (rol del usuario, modo de respuesta seleccionado, resolución de la incidencia, etc.).

## Estructura del proyecto

```graphql
app/
├── agents/            # Nodos del grafo y lógica de agentes
├── services/          # Base de conocimiento, embeddings, autenticación y utilidades
├── data/              # Datos y persistencia local
├── chatbot_ui.py      # Interfaz Streamlit
├── main.py            # API FastAPI (opcional)
```

## Anotaciones

El sistema ha sido diseñado con un enfoque modular y extensible, permitiendo la sustitución de modelos, interfaces o mecanismos de recuperación sin modificar el núcleo de la arquitectura. El uso de una arquitectura multiagente basada en grafos facilita la trazabilidad del comportamiento del asistente y el control del flujo de decisión, aspectos especialmente relevantes en entornos corporativos.
