"""Entrenamiento del modelo con PyCaret + XGBoost.

Este script NO corre dentro de Airflow: corre en su propio contenedor, que
Airflow lanza con DockerOperator. Recibe todo por variables de entorno y
deja su resultado en MLflow (métricas, artefactos y una versión registrada
del modelo). Airflow luego consulta MLflow para decidir si promueve.

Variables de entorno esperadas:
    URI_FEATURES   s3://... al parquet ya featurizado
    NOMBRE_RUN     identificador del run de Airflow (para poder localizarlo)
    EXPERIMENTO    nombre del experimento en MLflow
    NOMBRE_MODELO  nombre bajo el que se registra el modelo
"""

from __future__ import annotations

import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from comun.almacen import leer_parquet

# PyCaret imprime mucho ruido y abre widgets si detecta notebook: lo silenciamos.
os.environ.setdefault("PYCARET_CUSTOM_LOGGING_LEVEL", "CRITICAL")

from pycaret.classification import (  # noqa: E402
    create_model,
    finalize_model,
    predict_model,
    pull,
    save_model,
    setup,
    tune_model,
)

COLUMNA_OBJETIVO = "especie"
CLASES = ["setosa", "versicolor", "virginica"]


def main() -> int:
    uri_features = os.environ["URI_FEATURES"]
    nombre_run = os.environ.get("NOMBRE_RUN", "run_manual")
    experimento = os.environ.get("EXPERIMENTO", "iris_pycaret")
    nombre_modelo = os.environ.get("NOMBRE_MODELO", "iris_xgboost")

    print(f"[entrenador] Leyendo features desde {uri_features}")
    df = leer_parquet(uri_features)
    print(f"[entrenador] Filas={len(df)} columnas={list(df.columns)}")

    # ------------------------------------------------------------- PyCaret
    # setup() arma el pipeline de preprocesado y separa el holdout.
    print("[entrenador] setup() de PyCaret...")
    setup(
        data=df,
        target=COLUMNA_OBJETIVO,
        session_id=42,
        train_size=0.8,
        fold=5,
        verbose=False,
        html=False,
        n_jobs=1,
        # Logueamos a MLflow manualmente más abajo para controlar qué se guarda.
        log_experiment=False,
    )

    print("[entrenador] Entrenando XGBoost...")
    modelo = create_model("xgboost", verbose=False)

    print("[entrenador] Ajustando hiperparámetros...")
    # choose_better=True: si el tuning empeora el modelo, se queda con el base.
    afinado = tune_model(modelo, optimize="F1", n_iter=10, choose_better=True, verbose=False)

    # Métricas sobre el holdout que PyCaret reservó (datos nunca vistos).
    predict_model(afinado, verbose=False)
    metricas = pull().iloc[0]
    resultados = {
        "accuracy": float(metricas["Accuracy"]),
        "f1": float(metricas["F1"]),
        "precision": float(metricas["Prec."]),
        "recall": float(metricas["Recall"]),
        "kappa": float(metricas["Kappa"]),
    }
    print(f"[entrenador] Métricas en holdout: {resultados}")

    # finalize_model reentrena con el 100% de los datos, ya con la config ganadora.
    final = finalize_model(afinado)

    # save_model persiste el Pipeline completo (preprocesado + estimador).
    ruta = "/tmp/modelo_iris"
    save_model(final, ruta, verbose=False)
    objeto = joblib.load(f"{ruta}.pkl")
    # Según la versión, save_model devuelve el Pipeline o una tupla (Pipeline, ruta).
    pipeline = objeto[0] if isinstance(objeto, (list, tuple)) else objeto
    if not hasattr(pipeline, "predict"):
        raise RuntimeError(f"El objeto cargado no es un modelo entrenable: {type(pipeline)}")

    # -------------------------------------------------------------- MLflow
    mlflow.set_experiment(experimento)
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("origen", "airflow")
        mlflow.set_tag("framework", "pycaret+xgboost")
        mlflow.log_params(
            {
                "estimador": "xgboost",
                "train_size": 0.8,
                "fold": 5,
                "n_iter_tuning": 10,
                "n_filas": len(df),
                "n_features": df.shape[1] - 1,
            }
        )
        mlflow.log_metrics(resultados)
        # El Pipeline de PyCaret ya deshace la codificación del target: predice
        # directamente 'setosa'. Guardamos las clases como documentación del
        # modelo y para que la API pueda validar y ordenar la salida.
        mlflow.log_dict({"clases": CLASES}, "clases.json")

        ejemplo = df.drop(columns=[COLUMNA_OBJETIVO]).head(5)
        firma = infer_signature(ejemplo, pipeline.predict(ejemplo))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="modelo",
            signature=firma,
            input_example=ejemplo,
            registered_model_name=nombre_modelo,
        )

        print(f"[entrenador] run_id={run.info.run_id}")
        print(f"[entrenador] Modelo registrado como '{nombre_modelo}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
