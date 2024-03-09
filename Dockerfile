# Utilizar una imagen base de Python
FROM python:3.11-slim-bookworm

# Establece variables de entorno para Python
#Quitar? ENV PYTHONDONTWRITEBYTECODE 1
#Quitar? ENV PYTHONUNBUFFERED 1

# Instala dependencias del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código del proyecto
COPY ./app /app
WORKDIR /app

# Run app.py when the container launches
#PARA PROD: CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]




