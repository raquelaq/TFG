# API Corporativa Genia Soportia

API backend desarrollada con FastAPI que proporciona asistencia de soporte impulsada por el modelo Gemini de Google.

## Prerrequisitos

- Python 3.11+
- Archivos de certificado SSL (cert.pem y key.pem) Creados solo para ejecutar el proyecto desde local
- Archivo de configuración (config.ini) con las claves API y configuraciones necesarias (Usado sólopara el proyecto en local)
- Credenciales válidas de Google Cloud

## Instalación

1. Clona el repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Crea el archivo config.ini con las configuraciones requeridas:

```ini
[GENERAL]
DATA_STORE = app/data/conversation_store.json
GOOGLE_CLIENT_EMAIL = tu-email@tu-proyecto.iam.gserviceaccount.com
GOOGLE_PRIVATE_KEY = tu-clave-privada
AUDIENCE = tu-audience-id
GEMINI_API_KEY = tu-clave-api-gemini
JIRA_AUTH_HEADER = tu-auth-jira
ID_DRIVE_KB = tu-id-google-docs
```

## Ejecución Local

### Usando VS Code

El repositorio incluye una configuración de lanzamiento para VS Code. Para ejecutar el servidor:

1. Abre el proyecto en VS Code
2. Presiona F5 o usa el menú Ejecutar y Depurar
3. Selecciona la configuración "FastAPI Debug"

La configuración de lanzamiento está configurada como:

```json
{
    "name": "FastAPI Debug",
    "type": "debugpy",
    "request": "launch", 
    "module": "uvicorn",
    "args": [
        "app.main:app",
        "--ssl-keyfile", "key.pem",
        "--ssl-certfile", "cert.pem",
        "--reload",
        "--host", "0.0.0.0",
        "--port","8000"
    ],
    "jinja": true,
    "justMyCode": true
}
```
