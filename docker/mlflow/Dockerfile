# Servidor de MLflow (tracking + model registry).
# Se construye desde python-slim en vez de usar la imagen oficial porque
# necesitamos añadirle el driver de Postgres y el cliente S3.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
      "mlflow>=2.17,<3" \
      "psycopg2-binary>=2.9" \
      "boto3>=1.34"

EXPOSE 5000
