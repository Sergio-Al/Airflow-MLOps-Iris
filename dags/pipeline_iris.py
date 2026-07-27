"""Pipeline MLOps end-to-end sobre el dataset Iris.

    ingesta -> validación -> features -> entrenamiento -> evaluación
            -> (si mejora) promoción -> despliegue -> inferencia

Dos ideas de diseño que conviene notar al leerlo:

1. Entre tareas viajan URIs y métricas, nunca DataFrames. XCom guarda su
   contenido en la base de metadatos de Airflow: meter datos ahí la infla y
   la vuelve lenta. Los datos van a MinIO (S3); por XCom va la ruta.

2. El entrenamiento corre en OTRO contenedor (DockerOperator). Airflow no
   tiene PyCaret ni XGBoost instalados y no los necesita: solo orquesta.
"""

from __future__ import annotations

import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import dag, get_current_context, task

from comun import registro_mlflow
from comun.almacen import escribir_parquet, leer_parquet
from comun.features import CLASES, COLUMNA_OBJETIVO, COLUMNAS_CRUDAS, construir_features

# ------------------------------------------------------------------ constantes

BUCKET = "s3://datos"
EXPERIMENTO = os.environ.get("EXPERIMENTO", "iris_pycaret")
NOMBRE_MODELO = os.environ.get("NOMBRE_MODELO", "iris_xgboost")
ALIAS_CAMPEON = "campeon"
URL_API = os.environ.get("URL_API", "http://api:8000")

# Variables que el contenedor de entrenamiento necesita para hablar con MLflow/MinIO.
ENTORNO_ENTRENADOR = {
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    "AWS_DEFAULT_REGION": "us-east-1",
    "EXPERIMENTO": EXPERIMENTO,
    "NOMBRE_MODELO": NOMBRE_MODELO,
    # Se resuelven en tiempo de ejecución con Jinja.
    "URI_FEATURES": "{{ ti.xcom_pull(task_ids='extraer_features') }}",
    "NOMBRE_RUN": "{{ run_id }}",
}


def _sufijo_run() -> str:
    """run_id de Airflow convertido en algo válido como nombre de objeto S3."""
    run_id = get_current_context()["run_id"]
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in run_id)


@dag(
    dag_id="pipeline_iris",
    schedule=None,  # se dispara a mano desde la UI durante el ejercicio
    catchup=False,
    tags=["mlops", "iris", "pycaret"],
    doc_md=__doc__,
    default_args={"retries": 1, "owner": "clase-mlops"},
)
def pipeline_iris():

    # ---------------------------------------------------------- 1. ingesta
    @task
    def ingesta() -> str:
        """Carga Iris y lo deja en la zona cruda de MinIO."""
        from sklearn.datasets import load_iris

        datos = load_iris(as_frame=True)
        df = datos.frame.rename(
            columns={
                "sepal length (cm)": "largo_sepalo",
                "sepal width (cm)": "ancho_sepalo",
                "petal length (cm)": "largo_petalo",
                "petal width (cm)": "ancho_petalo",
            }
        )
        # target viene como 0/1/2; lo pasamos a nombre de especie.
        df[COLUMNA_OBJETIVO] = df.pop("target").map(dict(enumerate(datos.target_names)))

        uri = escribir_parquet(df, f"{BUCKET}/crudo/iris_{_sufijo_run()}.parquet")
        print(f"[ingesta] {len(df)} filas escritas en {uri}")
        return uri

    # -------------------------------------------------------- 2. validación
    @task
    def validar(uri: str) -> str:
        """Puerta de calidad: si los datos no cumplen, el pipeline se detiene aquí."""
        df = leer_parquet(uri)
        problemas: list[str] = []

        faltantes = [c for c in COLUMNAS_CRUDAS + [COLUMNA_OBJETIVO] if c not in df.columns]
        if faltantes:
            problemas.append(f"columnas ausentes: {faltantes}")

        nulos = df.isna().sum()
        if nulos.any():
            problemas.append(f"valores nulos: {nulos[nulos > 0].to_dict()}")

        if len(df) < 100:
            problemas.append(f"muy pocas filas: {len(df)}")

        if not faltantes:
            especies = set(df[COLUMNA_OBJETIVO].unique())
            if especies != set(CLASES):
                problemas.append(f"clases inesperadas: {especies}")

            no_positivos = [c for c in COLUMNAS_CRUDAS if (df[c] <= 0).any()]
            if no_positivos:
                problemas.append(f"medidas no positivas en: {no_positivos}")

        if problemas:
            raise ValueError("Validación fallida -> " + "; ".join(problemas))

        print(f"[validar] OK: {len(df)} filas, {df[COLUMNA_OBJETIVO].nunique()} clases balanceadas")
        return uri

    # ---------------------------------------------------------- 3. features
    @task
    def extraer_features(uri: str) -> str:
        """Aplica la ingeniería de features compartida con la API."""
        features = construir_features(leer_parquet(uri))

        destino = escribir_parquet(features, f"{BUCKET}/features/iris_{_sufijo_run()}.parquet")
        print(f"[features] {features.shape[1] - 1} features -> {destino}")
        return destino

    # ------------------------------------------------------ 4. entrenamiento
    # Contenedor aparte: aísla PyCaret/XGBoost del entorno de Airflow.
    entrenar = DockerOperator(
        task_id="entrenar_modelo",
        image=os.environ.get("IMAGEN_ENTRENADOR", "mlops-iris/entrenador:latest"),
        # Vía el proxy: Airflow no monta el socket de Docker (en macOS no tendría
        # permiso de escritura sobre él sin correr como root).
        docker_url=os.environ.get("DOCKER_URL", "tcp://docker-proxy:2375"),
        # Se une a la red de compose para poder resolver 'mlflow' y 'minio'.
        network_mode=os.environ.get("RED_DOCKER", "mlops_red"),
        auto_remove="force",
        mount_tmp_dir=False,  # evita problemas de montaje de /tmp en macOS
        environment=ENTORNO_ENTRENADOR,
    )

    # -------------------------------------------------- 5. evaluar y promover
    @task
    def evaluar_y_promover() -> dict:
        """Compara el modelo nuevo contra el campeón y promueve solo si mejora."""
        nombre_run = get_current_context()["run_id"]

        experimento = registro_mlflow.obtener_experimento(EXPERIMENTO)
        if experimento is None:
            raise ValueError(f"No existe el experimento '{EXPERIMENTO}' en MLflow.")

        run = registro_mlflow.buscar_run_por_nombre(experimento["experiment_id"], nombre_run)
        if run is None:
            raise ValueError(f"No se encontró el run '{nombre_run}'. ¿Falló el entrenamiento?")

        run_id = run["info"]["run_id"]
        metricas = registro_mlflow.metricas_de(run)
        if "f1" not in metricas:
            raise ValueError(f"El run {run_id} no registró la métrica 'f1'.")
        f1_nuevo = metricas["f1"]

        versiones = registro_mlflow.buscar_versiones_por_run(NOMBRE_MODELO, run_id)
        if not versiones:
            raise ValueError("El run no tiene ninguna versión registrada del modelo.")
        version_nueva = max(versiones, key=lambda v: int(v["version"]))["version"]

        # ¿Hay campeón? La primera ejecución no lo tendrá.
        campeon = registro_mlflow.obtener_version_por_alias(NOMBRE_MODELO, ALIAS_CAMPEON)
        if campeon is None:
            f1_campeon, version_campeon = -1.0, None
        else:
            version_campeon = campeon["version"]
            run_campeon = registro_mlflow.obtener_run(campeon["run_id"])
            f1_campeon = registro_mlflow.metricas_de(run_campeon or {}).get("f1", -1.0)

        promover = f1_nuevo > f1_campeon
        if promover:
            registro_mlflow.asignar_alias(NOMBRE_MODELO, ALIAS_CAMPEON, version_nueva)
            print(f"[evaluar] PROMOVIDO v{version_nueva}: F1 {f1_nuevo:.4f} > {f1_campeon:.4f}")
        else:
            print(f"[evaluar] RECHAZADO v{version_nueva}: F1 {f1_nuevo:.4f} <= {f1_campeon:.4f}")

        return {
            "promovido": promover,
            "version_nueva": version_nueva,
            "version_campeon_anterior": version_campeon,
            "f1_nuevo": f1_nuevo,
            "f1_campeon_anterior": f1_campeon,
            "run_id": run_id,
        }

    @task.branch
    def decidir_despliegue(resultado: dict) -> str:
        """Solo se despliega lo que superó al campeón."""
        return "recargar_api" if resultado["promovido"] else "omitir_despliegue"

    # ---------------------------------------------------------- 6. despliegue
    @task
    def recargar_api() -> dict:
        """Pide a la API que baje del registry la versión recién promovida."""
        import requests

        respuesta = requests.post(f"{URL_API}/recargar", timeout=120)
        respuesta.raise_for_status()
        cuerpo = respuesta.json()
        print(f"[despliegue] API sirviendo la versión {cuerpo.get('version')}")
        return cuerpo

    # --------------------------------------------------------- 7. inferencia
    @task
    def probar_inferencia() -> dict:
        """Prueba de humo: llama al endpoint y verifica que acierta casos conocidos."""
        import requests

        casos = [
            ({"largo_sepalo": 5.1, "ancho_sepalo": 3.5, "largo_petalo": 1.4, "ancho_petalo": 0.2}, "setosa"),
            ({"largo_sepalo": 6.4, "ancho_sepalo": 3.2, "largo_petalo": 4.5, "ancho_petalo": 1.5}, "versicolor"),
            ({"largo_sepalo": 6.9, "ancho_sepalo": 3.1, "largo_petalo": 5.4, "ancho_petalo": 2.1}, "virginica"),
        ]

        respuesta = requests.post(
            f"{URL_API}/predict",
            json={"instancias": [caso for caso, _ in casos]},
            timeout=60,
        )
        respuesta.raise_for_status()
        cuerpo = respuesta.json()

        obtenidas = [p["prediccion"] for p in cuerpo["predicciones"]]
        esperadas = [esperada for _, esperada in casos]
        aciertos = sum(o == e for o, e in zip(obtenidas, esperadas))

        for (caso, esperada), pred in zip(casos, cuerpo["predicciones"]):
            marca = "OK " if pred["prediccion"] == esperada else "MAL"
            print(f"[inferencia] {marca} esperado={esperada} obtenido={pred}")

        if aciertos < len(casos):
            raise ValueError(f"La prueba de humo falló: {aciertos}/{len(casos)} aciertos.")

        print(f"[inferencia] {aciertos}/{len(casos)} correctos con el modelo v{cuerpo['version_modelo']}")
        return {"aciertos": aciertos, "total": len(casos), "predicciones": obtenidas}

    omitir = EmptyOperator(task_id="omitir_despliegue")

    # ------------------------------------------------------------ dependencias
    uri_crudo = ingesta()
    uri_validado = validar(uri_crudo)
    uri_features = extraer_features(uri_validado)

    uri_features >> entrenar

    resultado = evaluar_y_promover()
    entrenar >> resultado

    rama = decidir_despliegue(resultado)
    rama >> recargar_api() >> probar_inferencia()
    rama >> omitir


pipeline_iris()
