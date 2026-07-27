"""Lectura y escritura de Parquet en MinIO (S3) usando solo boto3.

¿Por qué no s3fs, que sería más corto? Porque s3fs fija `fsspec` a una versión
exacta y eso rompe tanto el árbol de dependencias de PyCaret como el de Airflow.
boto3 ya viene en la imagen base de Airflow y en la del entrenador, así que
usarlo nos deja con CERO dependencias añadidas.
"""

from __future__ import annotations

import io
import os
from urllib.parse import urlparse

import boto3
import pandas as pd


def _cliente():
    """Cliente S3 apuntando a MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(
            "S3_ENDPOINT_URL", os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        ),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )


def _partir_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/carpeta/archivo.parquet' -> ('bucket', 'carpeta/archivo.parquet')"""
    partes = urlparse(uri)
    if partes.scheme != "s3":
        raise ValueError(f"Se esperaba un URI s3://, llegó: {uri}")
    return partes.netloc, partes.path.lstrip("/")


def escribir_parquet(df: pd.DataFrame, uri: str) -> str:
    """Serializa el DataFrame a Parquet en memoria y lo sube. Devuelve el URI."""
    bucket, clave = _partir_uri(uri)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    _cliente().put_object(Bucket=bucket, Key=clave, Body=buffer.getvalue())
    return uri


def leer_parquet(uri: str) -> pd.DataFrame:
    """Descarga el objeto y lo lee como DataFrame."""
    bucket, clave = _partir_uri(uri)
    respuesta = _cliente().get_object(Bucket=bucket, Key=clave)
    return pd.read_parquet(io.BytesIO(respuesta["Body"].read()))
