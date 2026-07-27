"""Cliente mínimo de la API REST de MLflow, hecho solo con `requests`.

¿Por qué no el paquete `mlflow`? Porque mlflow 2.x exige `cachetools<6` y los
constraints de Airflow 3.3 fijan `cachetools==7.1.4`: instalarlo degradaría las
dependencias de Airflow (de hecho degrada SQLAlchemy y rompe el servidor web).

El DAG solo necesita cuatro operaciones contra MLflow, y todas son llamadas HTTP
sencillas. El orquestador habla con MLflow por HTTP; no necesita el stack de ML.
La API de inferencia sí usa el cliente `mlflow` completo, porque ahí sí hay que
deserializar el modelo.
"""

from __future__ import annotations

import os
from typing import Any

import requests

BASE = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")
API = f"{BASE}/api/2.0/mlflow"
ESPERA = 60


def _get(ruta: str, params: dict | None = None) -> dict:
    respuesta = requests.get(f"{API}{ruta}", params=params, timeout=ESPERA)
    respuesta.raise_for_status()
    return respuesta.json()


def _post(ruta: str, cuerpo: dict) -> dict:
    respuesta = requests.post(f"{API}{ruta}", json=cuerpo, timeout=ESPERA)
    respuesta.raise_for_status()
    return respuesta.json() or {}


def _o_none(fn, *args, **kwargs):
    """Ejecuta fn y devuelve None si MLflow responde 'no existe' (404/400)."""
    try:
        return fn(*args, **kwargs)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (400, 404):
            return None
        raise


def obtener_experimento(nombre: str) -> dict | None:
    datos = _o_none(_get, "/experiments/get-by-name", {"experiment_name": nombre})
    return datos.get("experiment") if datos else None


def buscar_run_por_nombre(experiment_id: str, nombre_run: str) -> dict | None:
    """Localiza el run que creó el contenedor de entrenamiento, por su nombre."""
    datos = _post(
        "/runs/search",
        {
            "experiment_ids": [experiment_id],
            "filter": f"attributes.run_name = '{nombre_run}'",
            "max_results": 1,
            "order_by": ["attributes.start_time DESC"],
        },
    )
    runs = datos.get("runs", [])
    return runs[0] if runs else None


def obtener_run(run_id: str) -> dict | None:
    datos = _o_none(_get, "/runs/get", {"run_id": run_id})
    return datos.get("run") if datos else None


def metricas_de(run: dict) -> dict[str, float]:
    """Aplana la lista de métricas de un run a un dict {nombre: valor}."""
    return {m["key"]: float(m["value"]) for m in run.get("data", {}).get("metrics", [])}


def buscar_versiones_por_run(nombre_modelo: str, run_id: str) -> list[dict]:
    datos = _o_none(_get, "/model-versions/search", {"filter": f"run_id='{run_id}'"})
    versiones = (datos or {}).get("model_versions", [])
    return [v for v in versiones if v.get("name") == nombre_modelo]


def obtener_version_por_alias(nombre_modelo: str, alias: str) -> dict | None:
    datos = _o_none(
        _get, "/registered-models/alias", {"name": nombre_modelo, "alias": alias}
    )
    return datos.get("model_version") if datos else None


def asignar_alias(nombre_modelo: str, alias: str, version: str | int) -> None:
    _post(
        "/registered-models/alias",
        {"name": nombre_modelo, "alias": alias, "version": str(version)},
    )


def salud() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=10).ok
    except requests.RequestException:
        return False
