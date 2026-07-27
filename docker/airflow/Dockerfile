# Imagen de Airflow: deliberadamente SIN añadir nada.
#
# La imagen oficial ya trae todo lo que el DAG necesita:
#   apache-airflow-providers-docker 4.5.7  (DockerOperator)
#   pandas, numpy, pyarrow, boto3, requests, scikit-learn
#
# Intentar añadir aquí el cliente de MLflow rompe Airflow: mlflow 2.x exige
# `cachetools<6` y los constraints de Airflow 3.3 fijan `cachetools==7.1.4`.
# pip resuelve ese conflicto degradando SQLAlchemy de 2.0.51 a 1.4.54, lo que
# deja inservible a Flask-SQLAlchemy -> proveedor FAB -> servidor web.
#
# Por eso el DAG habla con MLflow por su API REST (src/comun/registro_mlflow.py)
# y con MinIO por boto3 (src/comun/almacen.py). Las librerías pesadas de ML
# viven en la imagen 'entrenador'.
#
# Si necesitas añadir algún paquete, hazlo SIEMPRE con el fichero de constraints:
#
#   USER airflow
#   RUN pip install --no-cache-dir \
#         --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.0/constraints-3.13.txt" \
#         "el-paquete-que-necesites"
#
FROM apache/airflow:3.3.0
