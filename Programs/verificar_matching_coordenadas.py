#!/usr/bin/env python3
# verificar_matching_beasley.py

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# Cargar resultados del matching
coincidencias = pd.read_csv('coincidencias_beasley_phot.csv')

print("="*60)
print("VERIFICACIÓN DE MATCHING BEASLEY-FOTOMETRÍA")
print("="*60)

# 1. Estadísticas básicas
print(f"\n📊 ESTADÍSTICAS DEL MATCHING:")
print(f"   Total coincidencias: {len(coincidencias)}")
print(f"   Separación media: {coincidencias['Sep_arcsec'].mean():.2f} arcsec")
print(f"   Separación máxima: {coincidencias['Sep_arcsec'].max():.2f} arcsec")
print(f"   Separación mínima: {coincidencias['Sep_arcsec'].min():.2f} arcsec")

# 2. Histograma de separaciones
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(coincidencias['Sep_arcsec'].values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Separación (arcsec)')
plt.ylabel('Número')
plt.title('Distribución de separaciones')

# 3. Scatter plot RA vs Dec offsets
plt.subplot(1, 2, 2)
ra_offset = (coincidencias['RA_Beasley'] - coincidencias['RA_phot']) * 3600 * np.cos(np.radians(coincidencias['Dec_Beasley']))
dec_offset = (coincidencias['Dec_Beasley'] - coincidencias['Dec_phot']) * 3600

plt.scatter(ra_offset, dec_offset, alpha=0.6, s=20)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('ΔRA * cos(Dec) (arcsec)')
plt.ylabel('ΔDec (arcsec)')
plt.title('Offsets astrométricos')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('verificacion_matching.png', dpi=150)
print(f"\n📈 Gráfico guardado: verificacion_matching.png")

# 4. Verificar posibles duplicados
print(f"\n🔍 VERIFICACIÓN DE DUPLICADOS:")
# Buscar si algún T17ID aparece más de una vez
t17id_counts = coincidencias['T17ID'].value_counts()
duplicados = t17id_counts[t17id_counts > 1]
if len(duplicados) > 0:
    print(f"   ⚠️  ADVERTENCIA: {len(duplicados)} T17IDs aparecen múltiples veces:")
    for t17id, count in duplicados.head().items():
        print(f"      {t17id}: {count} coincidencias")
else:
    print(f"   ✅ No hay T17IDs duplicados")

# 5. Verificar metalicidades
if 'MH_Beasley' in coincidencias.columns:
    mh_valid = coincidencias['MH_Beasley'].dropna()
    print(f"\n🎯 METALICIDADES ([M/H] de Beasley):")
    print(f"   Objetos con [M/H]: {len(mh_valid)} / {len(coincidencias)}")
    print(f"   Rango: [{mh_valid.min():.2f}, {mh_valid.max():.2f}]")
    
    # Verificar valores extremos
    if (mh_valid < -2.5).any():
        print(f"   ⚠️  Hay objetos con [M/H] < -2.5 (muy pobres en metales)")
    if (mh_valid > 0.5).any():
        print(f"   ⚠️  Hay objetos con [M/H] > 0.5 (muy ricos en metales)")

print("\n" + "="*60)
print("VERIFICACIÓN COMPLETADA")
print("="*60)
