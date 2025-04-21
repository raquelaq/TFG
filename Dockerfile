# Usa la imagen oficial de Python como base
FROM python:3.11


# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias especificadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación al directorio de trabajo
COPY . .

# Ejecuta la aplicación cuando se inicie el contenedor
#CMD ["python", "streamlit run app.py"]

# HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]