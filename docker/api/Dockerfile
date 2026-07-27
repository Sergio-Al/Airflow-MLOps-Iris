# API de inferencia.
#
# Hereda de la imagen 'entrenador' a propósito: el pipeline que guarda PyCaret
# contiene clases de PyCaret dentro del pickle, así que para deserializarlo
# hay que tener PyCaret instalado también aquí. Heredando reutilizamos las capas
# ya construidas en vez de descargar ~2 GB de dependencias por segunda vez.
#
# Consecuencia: la imagen 'entrenador' debe construirse ANTES que esta.
# El script scripts/levantar.sh se encarga del orden.
FROM mlops-iris/entrenador:latest

RUN pip install --no-cache-dir \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.29"

WORKDIR /app
COPY src/api/main.py /app/main.py
COPY src/comun /app/comun

ENTRYPOINT []
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
