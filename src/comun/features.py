"""Ingeniería de features compartida entre el pipeline y la API.

Este módulo existe para evitar el *training/serving skew*: si el DAG calculara
las features de una forma y la API de otra, el modelo recibiría en producción
columnas distintas a las que vio al entrenar. Al importar los dos el mismo
código, esa clase de bug desaparece por construcción.
"""

from __future__ import annotations

import pandas as pd

# Las 4 medidas crudas del dataset Iris (en cm).
COLUMNAS_CRUDAS: list[str] = [
    "largo_sepalo",
    "ancho_sepalo",
    "largo_petalo",
    "ancho_petalo",
]

COLUMNA_OBJETIVO: str = "especie"

# Orden de clases que produce el LabelEncoder de scikit-learn (alfabético).
CLASES: list[str] = ["setosa", "versicolor", "virginica"]


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features derivadas a las 4 medidas crudas.

    Devuelve una copia; no modifica el DataFrame de entrada. Si viene la
    columna objetivo, se conserva al final para que el orden de columnas sea
    estable entre entrenamiento e inferencia.
    """
    faltantes = [c for c in COLUMNAS_CRUDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas crudas requeridas: {faltantes}")

    salida = df.copy()

    # Proporciones: capturan la *forma* del pétalo/sépalo, no solo su tamaño.
    salida["razon_sepalo"] = salida["largo_sepalo"] / salida["ancho_sepalo"]
    salida["razon_petalo"] = salida["largo_petalo"] / salida["ancho_petalo"]

    # Áreas aproximadas: proxy del tamaño absoluto.
    salida["area_sepalo"] = salida["largo_sepalo"] * salida["ancho_sepalo"]
    salida["area_petalo"] = salida["largo_petalo"] * salida["ancho_petalo"]

    # Relación entre las dos estructuras: es la señal que mejor separa setosa.
    salida["razon_petalo_sepalo"] = salida["largo_petalo"] / salida["largo_sepalo"]

    columnas = COLUMNAS_CRUDAS + [
        "razon_sepalo",
        "razon_petalo",
        "area_sepalo",
        "area_petalo",
        "razon_petalo_sepalo",
    ]
    if COLUMNA_OBJETIVO in salida.columns:
        columnas = columnas + [COLUMNA_OBJETIVO]

    return salida[columnas]


def columnas_de_entrada() -> list[str]:
    """Nombres y orden exactos de las columnas que espera el modelo."""
    return [
        c
        for c in construir_features(
            pd.DataFrame([{c: 1.0 for c in COLUMNAS_CRUDAS}])
        ).columns
        if c != COLUMNA_OBJETIVO
    ]
