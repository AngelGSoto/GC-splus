#!/usr/bin/env python3
# analizar_resultados_cigale_fixed.py
# Analiza los resultados de CIGALE - FIX para problema endianness

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("📊 ANALIZADOR DE RESULTADOS CIGALE - VERSIÓN CORREGIDA")
    print("=" * 70)
    
    # Verificar que existen resultados
    resultados_path = "out/results.fits"
    if not os.path.exists(resultados_path):
        print("❌ No se encontraron resultados (out/results.fits)")
        print("📁 Archivos en directorio 'out/':")
        for f in os.listdir("out/")[:10]:
            print(f"   • {f}")
        return
    
    # 1. Leer resultados CORRECTAMENTE (evitando problema endianness)
    print("\n📖 Leyendo results.fits (con fix para big-endian)...")
    
    # Método 1: Usar astropy Table (más robusto)
    try:
        table = Table.read(resultados_path)
        print(f"✅ {len(table)} objetos analizados")
        print(f"📋 {len(table.colnames)} propiedades calculadas")
        
        # Convertir a DataFrame de manera segura
        df = table.to_pandas()
        
        # Convertir columnas numéricas a little-endian si es necesario
        for col in df.columns:
            if hasattr(df[col], 'values') and hasattr(df[col].values, 'dtype'):
                if df[col].values.dtype.byteorder == '>':
                    df[col] = df[col].values.byteswap().newbyteorder()
        
    except Exception as e:
        print(f"⚠️  Error con Table.read: {e}")
        print("Intentando método alternativo...")
        
        # Método 2: Usar fits.getdata directamente
        try:
            from astropy.io.fits import getdata
            data = getdata(resultados_path, 1)
            
            # Convertir a DataFrame manualmente
            df = pd.DataFrame()
            for name in data.names:
                col_data = data[name]
                # Convertir big-endian a little-endian si es necesario
                if hasattr(col_data, 'dtype') and col_data.dtype.byteorder == '>':
                    col_data = col_data.byteswap().newbyteorder()
                df[name] = col_data
            
            print(f"✅ {len(df)} objetos analizados (método alternativo)")
            print(f"📋 {len(df.columns)} propiedades calculadas")
            
        except Exception as e2:
            print(f"❌ Error grave: {e2}")
            return
    
    # 2. Mostrar información básica
    print("\n📝 Primeras 10 propiedades disponibles:")
    for i, col in enumerate(df.columns[:10], 1):
        print(f"   {i:2}. {col}")
    if len(df.columns) > 10:
        print(f"   ... y {len(df.columns) - 10} más")
    
    # 3. Buscar propiedades clave
    print("\n🔍 Buscando propiedades científicas importantes...")
    
    propiedades_encontradas = []
    categorias = {
        'Edad': ['age', 'sfh.age'],
        'Metalicidad': ['metal', 'xsl.metal', 'metallicity'],
        'SFR': ['sfr', 'sfh.sfr'],
        'Masa': ['mass', 'stellar.mass'],
        'Tau': ['tau', 'sfh.tau']
    }
    
    for categoria, keywords in categorias.items():
        encontradas = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                encontradas.append(col)
        
        if encontradas:
            propiedades_encontradas.extend(encontradas[:2])  # Tomar máximo 2 por categoría
            print(f"✅ {categoria}: {', '.join(encontradas[:3])}")
            if len(encontradas) > 3:
                print(f"   ... y {len(encontradas) - 3} más")
    
    # 4. Estadísticas básicas si hay propiedades
    if propiedades_encontradas:
        print("\n📈 ESTADÍSTICAS BÁSICAS:")
        print("-" * 40)
        
        for prop in propiedades_encontradas[:6]:  # Mostrar máximo 6
            if prop in df.columns:
                valores = df[prop]
                if len(valores) > 0 and np.issubdtype(valores.dtype, np.number):
                    # Filtrar valores extremos/no válidos
                    valores_validos = valores[np.isfinite(valores)]
                    if len(valores_validos) > 0:
                        print(f"\n  • {prop}:")
                        print(f"    Mín: {valores_validos.min():.4e}")
                        print(f"    Máx: {valores_validos.max():.4e}")
                        print(f"    Mediana: {np.median(valores_validos):.4e}")
                        print(f"    Media: {valores_validos.mean():.4e}")
    
    # 5. Generar gráficos SIMPLES (evitando problemas)
    print("\n🎨 Generando gráficos básicos...")
    
    # Buscar columnas para gráficos
    columnas_numericas = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Verificar que tenga suficientes valores válidos
            valores_validos = df[col].dropna()
            if len(valores_validos) > 10:
                columnas_numericas.append(col)
    
    if len(columnas_numericas) >= 2:
        # Tomar dos columnas numéricas para scatter plot
        col1, col2 = columnas_numericas[0], columnas_numericas[1]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df[col1], df[col2], alpha=0.5, s=20)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'{col1} vs {col2}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scatter_plot.png', dpi=150)
        print("✅ Gráfico guardado: scatter_plot.png")
        
        # Histograma de la primera columna
        plt.figure(figsize=(10, 6))
        valores = df[col1].dropna()
        plt.hist(valores, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel(col1)
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de {col1}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('histograma.png', dpi=150)
        print("✅ Gráfico guardado: histograma.png")
    
    # 6. Exportar resultados en formatos alternativos
    print("\n💾 Exportando resultados...")
    
    # Opción 1: CSV simple (solo algunas columnas)
    try:
        columnas_exportar = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'age', 'metal', 'sfr', 'mass', 'flux']):
                columnas_exportar.append(col)
            if len(columnas_exportar) >= 15:  # Limitar a 15 columnas
                break
        
        if columnas_exportar:
            df_simple = df[columnas_exportar].copy()
            
            # Convertir cualquier columna big-endian restante
            for col in df_simple.columns:
                if hasattr(df_simple[col], 'values') and hasattr(df_simple[col].values, 'dtype'):
                    if df_simple[col].values.dtype.byteorder == '>':
                        df_simple[col] = df_simple[col].values.byteswap().newbyteorder()
            
            df_simple.to_csv('resultados_simple.csv', index=False)
            print(f"✅ CSV guardado: resultados_simple.csv ({len(df_simple.columns)} columnas)")
    except Exception as e:
        print(f"⚠️  Error exportando CSV: {e}")
    
    # Opción 2: TXT para columnas específicas
    try:
        with open('resultados_resumen.txt', 'w') as f:
            f.write("# Resumen de resultados CIGALE\n")
            f.write(f"# Total objetos: {len(df)}\n")
            f.write(f"# Total propiedades: {len(df.columns)}\n")
            f.write("#" * 80 + "\n\n")
            
            # Escribir nombres de columnas
            f.write("Columnas disponibles:\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:3}. {col}\n")
                if i >= 50:  # Limitar a 50 columnas
                    f.write(f"... y {len(df.columns) - 50} más\n")
                    break
    except Exception as e:
        print(f"⚠️  Error exportando TXT: {e}")
    
    # 7. Información del directorio out/
    print("\n📁 CONTENIDO DE 'out/':")
    print("-" * 40)
    
    archivos = os.listdir("out/")
    categorias_archivos = {}
    
    for archivo in archivos:
        if archivo.endswith('.fits'):
            if 'best_model' in archivo:
                categorias_archivos.setdefault('best_model', []).append(archivo)
            elif 'SFH' in archivo:
                categorias_archivos.setdefault('SFH', []).append(archivo)
            else:
                categorias_archivos.setdefault('otros_fits', []).append(archivo)
        elif archivo.endswith('.txt'):
            categorias_archivos.setdefault('txt', []).append(archivo)
        else:
            categorias_archivos.setdefault('otros', []).append(archivo)
    
    for categoria, lista in categorias_archivos.items():
        print(f"  {categoria}: {len(lista)} archivos")
        if categoria in ['best_model', 'SFH'] and lista:
            print(f"    Ejemplos: {lista[0]}, {lista[1] if len(lista) > 1 else ''}")
    
    print("\n" + "=" * 70)
    print("🎉 ANÁLISIS COMPLETADO")
    print("=" * 70)
    
    # Comandos útiles
    print("\n🔧 PRÓXIMOS PASOS RECOMENDADOS:")
    print("1. Ver resultados detallados:")
    print("   $ topcat out/results.fits")
    print("2. Generar gráficos con CIGALE:")
    print("   $ pcigale-plots sed")
    print("   $ pcigale-plots pdf")
    print("3. Analizar SEDs individuales:")
    print("   $ python -c \"from astropy.io import fits; import matplotlib.pyplot as plt")
    print("      data = fits.getdata('out/T17-0006_best_model.fits');")
    print("      plt.plot(data['wave'], data['flux']); plt.show()\"")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
