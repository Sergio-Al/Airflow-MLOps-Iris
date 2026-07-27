# Imagen de entrenamiento: aquí sí van las dependencias pesadas.
# Airflow nunca importa PyCaret; solo lanza este contenedor y espera a que termine.
# Así los pines estrictos de PyCaret no chocan con los de Airflow.
FROM python:3.11-slim

# libgomp1 -> runtime de OpenMP, requerido por XGBoost y LightGBM
# build-essential -> por si alguna dependencia no trae wheel para arm64
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Sin s3fs a propósito: fija fsspec a una versión exacta y choca con el árbol
# de dependencias de PyCaret. Los objetos de MinIO se bajan con boto3.
RUN pip install --no-cache-dir \
      "pycaret>=3.3,<3.4" \
      "xgboost>=2.0,<3" \
      "mlflow>=2.17,<3" \
      "boto3>=1.34" \
      "pyarrow>=15.0"

WORKDIR /app
COPY src/comun /app/comun
COPY src/entrenador/entrenar.py /app/entrenar.py

ENTRYPOINT ["python", "-u", "/app/entrenar.py"]
