#!/usr/bin/env python3
"""
plot_FeH_vs_age.py
Gráfico simple: metalicidad [Fe/H] (eje X) vs edad estelar (eje Y)
a partir de los resultados de CIGALE (out/results.fits).
No requiere catálogo adicional.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
import os

plt.rcParams.update({'font.size': 12, 'figure.figsize': (8, 6)})

def plot_FeH_vs_age(results_file='out/results.fits', output_plot='FeH_vs_age.png'):
    if not os.path.exists(results_file):
        print(f"❌ No se encontró el archivo: {results_file}")
        sys.exit(1)

    # Cargar la tabla
    table = Table.read(results_file)
    print(f"✅ Archivo cargado: {results_file}")

    # Buscar columnas de edad y metalicidad (bayes.stellar...)
    age_col = None
    met_col = None
    for col in table.colnames:
        if col.startswith('bayes.stellar.age_m_star') and not col.endswith('_err'):
            age_col = col
        if col.startswith('bayes.stellar.metallicity') and not col.endswith('_err'):
            met_col = col

    # Fallback a best si no hay bayesianas
    if age_col is None:
        for col in table.colnames:
            if col == 'best.stellar.age_m_star':
                age_col = col
                print("⚠️  Usando 'best.stellar.age_m_star'")
                break
    if met_col is None:
        for col in table.colnames:
            if col == 'best.stellar.metallicity':
                met_col = col
                print("⚠️  Usando 'best.stellar.metallicity'")
                break

    if age_col is None or met_col is None:
        print("❌ No se encontraron columnas de edad y metalicidad.")
        print("   Columnas:", table.colnames)
        sys.exit(1)

    print(f"   edad = {age_col}, metalicidad = {met_col}")

    # Extraer datos
    age = table[age_col].data  # Myr
    metallicity = table[met_col].data  # Z

    # Convertir edad a Gyr
    if np.nanmedian(age) > 1000:
        age = age / 1e3
        print("   Edad convertida de Myr a Gyr.")

    # Convertir metalicidad a [Fe/H] (Z_solar = 0.02)
    Z_solar = 0.02
    FeH = np.log10(metallicity / Z_solar)

    # Filtrar valores válidos
    valid = np.isfinite(FeH) & np.isfinite(age)
    feh_clean = FeH[valid]
    age_clean = age[valid]
    print(f"   Objetos con datos válidos: {len(age_clean)}")

    # Gráfico: ahora FeH en X, Age en Y
    fig, ax = plt.subplots()
    ax.scatter(feh_clean, age_clean, c='steelblue', alpha=0.7, s=30, edgecolors='none')
    ax.set_xlabel('[Fe/H] (dex)', fontsize=14)
    ax.set_ylabel('Edad estelar (Gyr)', fontsize=14)
    ax.set_title('Cúmulos globulares en NGC 5128\n(estimaciones bayesianas)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Líneas de referencia
    med_feh = np.median(feh_clean)
    med_age = np.median(age_clean)
    #ax.axvline(med_feh, color='red', linestyle='--', alpha=0.7, label=f'Mediana [Fe/H]: {med_feh:.2f}')
    #ax.axhline(med_age, color='red', linestyle=':', alpha=0.7, label=f'Mediana edad: {med_age:.1f} Gyr')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"✅ Gráfico guardado: {output_plot}")
    plt.show()

if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else 'out/results.fits'
    salida  = sys.argv[2] if len(sys.argv) > 2 else 'FeH_vs_age.png'
    plot_FeH_vs_age(entrada, salida)
