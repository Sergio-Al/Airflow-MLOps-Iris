#!/bin/bash
# Se ejecuta una sola vez, cuando Postgres inicializa su volumen de datos.
# Crea la base de datos que usa MLflow como backend store (Airflow usa la suya propia).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
EOSQL

echo "Base de datos 'mlflow' creada."
