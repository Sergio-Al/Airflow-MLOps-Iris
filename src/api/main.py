"""API de inferencia: sirve el modelo marcado como 'campeon' en MLflow.

El modelo NO se empaqueta en la imagen. Se descarga del Model Registry al
arrancar y se puede recargar en caliente con POST /recargar, que es lo que
hace el DAG después de promover una versión nueva.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from comun.features import CLASES, construir_features

NOMBRE_MODELO = os.environ.get("NOMBRE_MODELO", "iris_xgboost")
ALIAS = "campeon"

# Estado del modelo cargado en memoria.
_estado: dict[str, Any] = {"modelo": None, "version": None, "run_id": None, "clases": CLASES}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Arrancar sin modelo es normal: la primera vez, el pipeline aún no ha corrido.
    try:
        _estado.update(_cargar_modelo())
        print(f"[api] Modelo cargado: {NOMBRE_MODELO} v{_estado['version']}")
    except Exception as exc:
        print(f"[api] Aún no hay modelo con alias '{ALIAS}': {exc}")
    yield


app = FastAPI(
    title="API de Inferencia - Iris",
    description="Sirve el modelo PyCaret/XGBoost promovido en MLflow.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


class Medida(BaseModel):
    """Las 4 medidas crudas del dataset Iris, en centímetros."""

    largo_sepalo: float = Field(..., gt=0, examples=[5.1])
    ancho_sepalo: float = Field(..., gt=0, examples=[3.5])
    largo_petalo: float = Field(..., gt=0, examples=[1.4])
    ancho_petalo: float = Field(..., gt=0, examples=[0.2])


class PeticionPredict(BaseModel):
    instancias: list[Medida] = Field(..., min_length=1)


def _cargar_modelo() -> dict[str, Any]:
    """Descarga del registry la versión con alias 'campeon'."""
    cliente = MlflowClient()
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO, ALIAS)
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@{ALIAS}")

    # El mapeo índice -> nombre de clase se guardó como artefacto en el run.
    clases = CLASES
    try:
        ruta = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="clases.json"
        )
        with open(ruta) as fh:
            clases = json.load(fh)["clases"]
    except Exception:
        pass  # nos quedamos con el orden por defecto del módulo compartido

    return {
        "modelo": modelo,
        "version": version.version,
        "run_id": version.run_id,
        "clases": clases,
    }


@app.get("/salud")
def salud() -> dict[str, Any]:
    return {
        "estado": "ok",
        "modelo_cargado": _estado["modelo"] is not None,
        "version": _estado["version"],
    }


@app.get("/modelo")
def info_modelo() -> dict[str, Any]:
    if _estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="No hay modelo cargado.")
    return {
        "nombre": NOMBRE_MODELO,
        "alias": ALIAS,
        "version": _estado["version"],
        "run_id": _estado["run_id"],
        "clases": _estado["clases"],
    }


@app.post("/recargar")
def recargar() -> dict[str, Any]:
    """Recarga el campeón desde MLflow. La llama el DAG tras promover."""
    try:
        _estado.update(_cargar_modelo())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No se pudo cargar el modelo: {exc}")
    return {"recargado": True, "version": _estado["version"], "run_id": _estado["run_id"]}


def _a_etiqueta(valor: Any, clases: list[str]) -> str:
    """Normaliza la salida del modelo a un nombre de especie.

    El Pipeline que guarda PyCaret ya deshace la codificación del target y
    devuelve la etiqueta original ('setosa'). Un estimador de scikit-learn a
    secas devolvería el índice (0). Aceptamos ambas formas para que la API siga
    funcionando si algún día se cambia el entrenador.
    """
    if isinstance(valor, bytes):
        return valor.decode()
    if isinstance(valor, str):
        return valor
    try:
        indice = int(valor)
    except (TypeError, ValueError):
        return str(valor)
    return clases[indice] if 0 <= indice < len(clases) else str(indice)


@app.post("/predict")
def predict(peticion: PeticionPredict) -> dict[str, Any]:
    if _estado["modelo"] is None:
        raise HTTPException(
            status_code=503,
            detail="No hay modelo cargado. Ejecuta el DAG y llama a POST /recargar.",
        )

    crudo = pd.DataFrame([m.model_dump() for m in peticion.instancias])
    # Mismo código de features que usó el pipeline al entrenar.
    features = construir_features(crudo)

    modelo = _estado["modelo"]
    crudas = modelo.predict(features)
    clases = _estado["clases"]

    probabilidades = None
    if hasattr(modelo, "predict_proba"):
        try:
            probabilidades = modelo.predict_proba(features)
        except Exception:
            probabilidades = None

    predicciones = []
    for i, cruda in enumerate(crudas):
        etiqueta = _a_etiqueta(cruda, clases)
        item: dict[str, Any] = {"prediccion": etiqueta}
        if etiqueta in clases:
            item["codigo"] = clases.index(etiqueta)
        if probabilidades is not None:
            item["confianza"] = round(float(max(probabilidades[i])), 4)
        predicciones.append(item)

    return {"version_modelo": _estado["version"], "predicciones": predicciones}
