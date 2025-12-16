#!/usr/bin/env python3
# check_cigale_columns.py
# Verifica qué columnas tiene tu archivo FITS

from astropy.io import fits
import pandas as pd
import numpy as np

def check_columns(fits_file):
    """Verifica columnas disponibles en el archivo FITS."""
    
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        df = pd.DataFrame(data)
    
    print(f"\n📋 COLUMNAS DISPONIBLES ({len(df.columns)} total):")
    print("="*60)
    
    # Columnas relacionadas con edad
    print("\n🔭 COLUMNAS DE EDAD:")
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    for col in age_cols:
        unique_vals = len(df[col].dropna().unique())
        total_vals = len(df[col].dropna())
        pct_unique = unique_vals/total_vals*100 if total_vals > 0 else 0
        print(f"  • {col}: {unique_vals}/{total_vals} únicos ({pct_unique:.1f}%)")
    
    # Columnas de metalicidad
    print("\n🔭 COLUMNAS DE METALICIDAD:")
    metal_cols = [c for c in df.columns if 'metal' in c.lower() or 'z' == c.lower()]
    for col in metal_cols:
        unique_vals = len(df[col].dropna().unique())
        total_vals = len(df[col].dropna())
        pct_unique = unique_vals/total_vals*100 if total_vals > 0 else 0
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  • {col}: {unique_vals}/{total_vals} únicos ({pct_unique:.1f}%)")
            print(f"    Rango: [{vals.min():.5f}, {vals.max():.5f}]")
    
    # Columnas bayesianas vs best
    print("\n🔭 COMPARACIÓN BAYES vs BEST:")
    
    # Edad
    bayes_age = [c for c in age_cols if 'bayes' in c]
    best_age = [c for c in age_cols if 'best' in c]
    
    print(f"  • Bayesian age columns: {bayes_age}")
    print(f"  • Best age columns: {best_age}")
    
    # Metalicidad
    bayes_metal = [c for c in metal_cols if 'bayes' in c]
    best_metal = [c for c in metal_cols if 'best' in c]
    
    print(f"  • Bayesian metallicity columns: {bayes_metal}")
    print(f"  • Best metallicity columns: {best_metal}")
    
    # Recomendación
    print("\n💡 RECOMENDACIÓN:")
    if bayes_age and bayes_metal:
        print(f"  ✅ USAR: {bayes_age[0]} y {bayes_metal[0]}")
        print(f"     (Valores continuos bayesianos)")
    elif best_age and best_metal:
        print(f"  ⚠️  USAR: {best_age[0]} y {best_metal[0]}")
        print(f"     (Valores discretos del grid - limitado)")
    else:
        print(f"  ❌ No se encuentran columnas apropiadas")
    
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = check_columns(sys.argv[1])
        
        # Mostrar primeros valores de columnas clave
        print("\n📊 PRIMEROS VALORES DE COLUMNAS CLAVE:")
        key_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['age', 'metal', 'z', 'feh']):
                key_cols.append(col)
        
        if key_cols:
            print(df[key_cols].head(10).to_string())
    else:
        print("Uso: python check_cigale_columns.py archivo.fits")
