#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import re

def main():
    workdir = Path(".")
    output_file = workdir / "all_fields_photometry_COMPLETE.csv"

    # Buscar solo los archivos individuales COMPLETE de campos CenA##
    pattern = re.compile(r"^CenA\d{2}_photometry_COMPLETE\.csv$")
    files = sorted([f for f in workdir.glob("CenA*_photometry_COMPLETE.csv") if pattern.match(f.name)])

    if not files:
        print("❌ No se encontraron archivos CenA##_photometry_COMPLETE.csv")
        return

    print(f"📁 Encontrados {len(files)} archivos:")
    for f in files:
        print(f"   - {f.name}")

    dfs = []
    total_rows = 0

    for f in files:
        try:
            df = pd.read_csv(f)
            df["SOURCE_FILE"] = f.name  # opcional: trazabilidad
            dfs.append(df)
            total_rows += len(df)
            print(f"✅ {f.name}: {len(df)} filas")
        except Exception as e:
            print(f"⚠️ Error leyendo {f.name}: {e}")

    if not dfs:
        print("❌ No se pudo leer ningún archivo.")
        return

    # Concatenar respetando todas las columnas (aunque no sean idénticas)
    final_df = pd.concat(dfs, ignore_index=True, sort=False)

    # (Opcional) eliminar duplicados exactos por coordenadas si existen
    # descomenta si quieres deduplicar:
    # if {"RAJ2000", "DEJ2000"}.issubset(final_df.columns):
    #     before = len(final_df)
    #     final_df = final_df.drop_duplicates(subset=["RAJ2000", "DEJ2000"])
    #     print(f"🧹 Duplicados removidos por RA/DEC: {before - len(final_df)}")

    final_df.to_csv(output_file, index=False)

    print("\n🎉 Listo")
    print(f"💾 Archivo generado: {output_file}")
    print(f"📊 Filas esperadas (suma): {total_rows}")
    print(f"📊 Filas en output: {len(final_df)}")
    print(f"🧩 Columnas: {len(final_df.columns)}")

if __name__ == "__main__":
    main()
