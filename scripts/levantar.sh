#!/usr/bin/env bash
# Levanta el stack completo en el orden correcto.
set -euo pipefail

cd "$(dirname "$0")/.."

azul() { printf '\033[1;34m==> %s\033[0m\n' "$1"; }
rojo() { printf '\033[1;31mERROR: %s\033[0m\n' "$1"; }

# ------------------------------------------------------------------ 0. .env
# El .env real está en .gitignore, así que en un clon nuevo no existe.
if [ ! -f .env ]; then
  azul "No hay .env; copiando desde .env.example..."
  cp .env.example .env
fi

# --------------------------------------------------------------- 1. Docker
azul "Verificando que Docker esté corriendo..."
if ! docker ps >/dev/null 2>&1; then
  rojo "El daemon de Docker no responde."
  echo "   Abre Docker Desktop y espera a que el icono de la ballena deje de animarse."
  echo "   Si sigue fallando: docker context use desktop-linux"
  exit 1
fi
echo "    Docker OK."

# --------------------------------------------------- 2. carpetas y permisos
azul "Preparando carpetas montadas..."
mkdir -p dags logs plugins config src
# Compose da prioridad a las variables del shell sobre las del .env,
# así que el UID real del host siempre gana.
export AIRFLOW_UID="$(id -u)"
echo "    AIRFLOW_UID=${AIRFLOW_UID}"

# ------------------------------------------- 3. imagen de entrenamiento
# Se construye con 'docker build' y no con compose porque NO es un servicio:
# es una imagen que Airflow arranca a demanda vía DockerOperator.
# La imagen de la API hereda de esta, por eso va primero.
azul "Construyendo imagen 'entrenador' (PyCaret + XGBoost)..."
echo "    La primera vez tarda entre 5 y 12 minutos. Es normal."
docker build -t mlops-iris/entrenador:latest -f docker/entrenador/Dockerfile .

# ------------------------------------------------- 4. resto de imágenes
azul "Construyendo airflow, mlflow y api..."
docker compose build

# ------------------------------------------------------- 5. arrancar todo
azul "Arrancando servicios..."
docker compose up -d

# -------------------------------------------------------- 6. esperar UI
azul "Esperando a que la UI de Airflow responda (puede tardar ~1 min)..."
for intento in $(seq 1 60); do
  if curl -sf http://localhost:8080/api/v2/monitor/health >/dev/null 2>&1; then
    echo "    Airflow listo."
    break
  fi
  if [ "$intento" -eq 60 ]; then
    rojo "Airflow no respondió a tiempo. Revisa: docker compose logs airflow-apiserver"
    exit 1
  fi
  sleep 5
done

cat <<'FIN'

──────────────────────────────────────────────────────────────
  Todo arriba. Accesos:

    Airflow      http://localhost:8080     (airflow / airflow)
    MLflow       http://localhost:5001
    MinIO        http://localhost:9001     (minioadmin / minioadmin)
    API          http://localhost:8000/docs

  Para ejecutar el pipeline:
    1. Entra a Airflow, busca el DAG 'pipeline_iris'
    2. Pulsa el botón de play (Trigger DAG)

  O desde la terminal:
    docker compose exec airflow-scheduler airflow dags trigger pipeline_iris

  Para apagar todo:
    docker compose down          # conserva los datos
    docker compose down -v       # borra también los volúmenes
──────────────────────────────────────────────────────────────
FIN
