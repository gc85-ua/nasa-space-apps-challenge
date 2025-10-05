import pandas as pd

def clean(df_list):
    """
    Limpia y unifica varios DataFrames con diferentes unidades y nombres de columnas.
    Devuelve un único DataFrame consolidado.

    Parámetros
    ----------
    df_list : list[pd.DataFrame]
        Lista con los 3 DataFrames a unificar.

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio y unificado.
    """

    # -------------------------------------
    # 1️⃣ Normalizar nombres de columnas
    # -------------------------------------
    # Mapa de equivalencias de nombres
    column_map = {
        "pl_orbper": "period",
        "koi_period": "period",

        "pl_rade": "depth",
        "windSpeed_mps": "wind_speed",
        "viento": "wind_speed",

        "pressure_hpa": "pressure",
        "presion_Pa": "pressure"
    }

    # Renombrar columnas en todos los datasets
    normalized = []
    for df in df_list:
        df = df.rename(columns=lambda c: column_map.get(c, c))  # solo cambia si está en el mapa
        normalized.append(df)

    # -------------------------------------
    # 2️⃣ Homogeneizar unidades de medida
    # -------------------------------------
    for df in normalized:
        # Temperatura: pasar todo a Celsius
        if "temperature" in df.columns:
            # si parece estar en Fahrenheit
            if df["temperature"].max() > 60:  # umbral simple
                df["temperature"] = (df["temperature"] - 32) * 5 / 9

        # Velocidad del viento: pasar todo a m/s
        if "wind_speed" in df.columns:
            # si parece estar en km/h
            if df["wind_speed"].mean() > 20:
                df["wind_speed"] = df["wind_speed"] / 3.6

        # Presión: pasar todo a hPa
        if "pressure" in df.columns:
            # si parece estar en Pascales
            if df["pressure"].mean() > 2000:
                df["pressure"] = df["pressure"] / 100.0

    # -------------------------------------
    # 3️⃣ Unificar columnas
    # -------------------------------------
    unified = pd.concat(normalized, ignore_index=True)

    # -------------------------------------
    # 4️⃣ Opcional: limpiar NaN o duplicados
    # -------------------------------------
    unified = unified.drop_duplicates()
    unified = unified.dropna(subset=["temperature", "wind_speed"], how="all")

    # -------------------------------------
    # 5️⃣ Retornar DataFrame limpio
    # -------------------------------------
    return unified


# Ejemplo de uso directo (puedes quitar esto si lo usas como módulo):
if __name__ == "__main__":
    df1 = pd.read_csv("cumulative.csv")
    df2 = pd.read_csv("k2pandc.csv")
    df3 = pd.read_csv("TOI.csv")

    clean_df = clean([df1, df2, df3])
    clean_df.to_csv("dataset_unificado.csv", index=False)
    print("✅ Dataset unificado guardado en dataset_unificado.csv")
