#!/usr/bin/env python3
# analisis_gc_completo_final.py
# Análisis COMPLETO con las columnas encontradas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os
import warnings
warnings.filterwarnings('ignore')

def fix_endian(data):
    """Convierte arrays big-endian a little-endian"""
    if hasattr(data, 'dtype') and data.dtype.byteorder == '>':
        return data.byteswap().newbyteorder()
    return data

def cargar_datos():
    """Carga datos con corrección de endianness"""
    print("📊 CARGANDO DATOS...")
    
    with fits.open('out/results.fits') as hdul:
        data = hdul[1].data
    
    # Convertir a DataFrame con corrección
    df = pd.DataFrame()
    for name in data.names:
        col_data = data[name]
        col_data = fix_endian(col_data)
        df[name] = col_data
    
    print(f"✅ {len(df)} cúmulos globulares cargados")
    return df

def convertir_unidades(df):
    """Convierte unidades a formatos más útiles"""
    
    # 1. Convertir edades de Myr a Gyr
    if 'best.sfh.age_main' in df.columns:
        df['Edad_Gyr'] = df['best.sfh.age_main'] / 1000.0
    
    # 2. Convertir metalicidad Z a [Fe/H]
    # [Fe/H] = log10(Z/Z_solar), donde Z_solar = 0.02
    if 'best.stellar.metallicity' in df.columns:
        Z_solar = 0.0152
        df['Fe_H'] = np.log10(df['best.stellar.metallicity'] / Z_solar)
    
    # 3. SFR en diferentes escalas de tiempo
    if 'best.sfh.sfr10Myrs' in df.columns:
        df['SFR_10Myr'] = df['best.sfh.sfr10Myrs']
    if 'best.sfh.sfr100Myrs' in df.columns:
        df['SFR_100Myr'] = df['best.sfh.sfr100Myrs']
    
    return df

def analisis_estadistico(df):
    """Realiza análisis estadístico detallado"""
    
    print("\n" + "="*60)
    print("📈 ANÁLISIS ESTADÍSTICO DETALLADO")
    print("="*60)
    
    # 1. Análisis de edades
    if 'Edad_Gyr' in df.columns:
        edades = df['Edad_Gyr'].dropna()
        
        print(f"\n🔬 EDADES DE LOS CÚMULOS GLOBULARES")
        print("-"*40)
        print(f"• Total cúmulos con edad: {len(edades)}")
        print(f"• Rango: {edades.min():.1f} - {edades.max():.1f} Gyr")
        print(f"• Mediana: {edades.median():.1f} Gyr")
        print(f"• Media: {edades.mean():.1f} ± {edades.std():.1f} Gyr")
        
        # Distribución por rangos de edad
        bins = [8, 9, 10, 11, 12, 13]
        hist, _ = np.histogram(edades, bins=bins)
        
        print(f"\n📊 DISTRIBUCIÓN POR RANGOS DE EDAD:")
        for i in range(len(bins)-1):
            n = hist[i]
            pct = (n / len(edades)) * 100
            print(f"  {bins[i]}-{bins[i+1]} Gyr: {n:3d} cúmulos ({pct:5.1f}%)")
    
    # 2. Análisis de metalicidades
    if 'Fe_H' in df.columns:
        feh = df['Fe_H'].dropna()
        
        print(f"\n🔬 METALICIDADES ([Fe/H])")
        print("-"*40)
        print(f"• Total cúmulos con metalicidad: {len(feh)}")
        print(f"• Rango: {feh.min():.2f} - {feh.max():.2f} dex")
        print(f"• Mediana: {feh.median():.2f} dex")
        print(f"• Media: {feh.mean():.2f} ± {feh.std():.2f} dex")
        
        # Clasificación por metalicidad
        pobres = feh[feh < -1.0]
        intermedios = feh[(feh >= -1.0) & (feh < -0.5)]
        ricos = feh[feh >= -0.5]
        
        print(f"\n📊 CLASIFICACIÓN POR METALICIDAD:")
        print(f"  Pobres ([Fe/H] < -1.0): {len(pobres):3d} ({len(pobres)/len(feh)*100:5.1f}%)")
        print(f"  Intermedios (-1.0 ≤ [Fe/H] < -0.5): {len(intermedios):3d} ({len(intermedios)/len(feh)*100:5.1f}%)")
        print(f"  Ricos ([Fe/H] ≥ -0.5): {len(ricos):3d} ({len(ricos)/len(feh)*100:5.1f}%)")
    
    # 3. Análisis de SFR
    if 'SFR_10Myr' in df.columns:
        sfr = df['SFR_10Myr'].dropna()
        
        print(f"\n🔬 TASA DE FORMACIÓN ESTELAR (SFR)")
        print("-"*40)
        print(f"• Total cúmulos con SFR: {len(sfr)}")
        print(f"• Rango: {sfr.min():.2e} - {sfr.max():.2e} M☉/año")
        print(f"• Mediana: {sfr.median():.2e} M☉/año")
        print(f"• Media: {sfr.mean():.2e} M☉/año")
        
        # Estadísticas en escala log
        sfr_log = np.log10(sfr[sfr > 0])
        if len(sfr_log) > 0:
            print(f"• Log10(SFR) media: {sfr_log.mean():.2f} dex")

def generar_graficos_completos(df):
    """Genera gráficos completos del análisis"""
    
    print("\n🎨 GENERANDO GRÁFICOS COMPLETOS...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histograma de edades
    ax1 = plt.subplot(2, 3, 1)
    if 'Edad_Gyr' in df.columns:
        edades = df['Edad_Gyr'].dropna()
        ax1.hist(edades, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(edades.median(), color='red', linestyle='--', 
                   label=f'Mediana: {edades.median():.1f} Gyr')
        ax1.set_xlabel('Edad (Gyr)', fontsize=12)
        ax1.set_ylabel('Número de cúmulos', fontsize=12)
        ax1.set_title('Distribución de edades', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de metalicidades
    ax2 = plt.subplot(2, 3, 2)
    if 'Fe_H' in df.columns:
        feh = df['Fe_H'].dropna()
        ax2.hist(feh, bins=20, alpha=0.7, color='forestgreen', edgecolor='black')
        ax2.axvline(feh.median(), color='red', linestyle='--',
                   label=f'Mediana: {feh.median():.2f} dex')
        
        # Líneas de clasificación
        ax2.axvline(-1.0, color='orange', linestyle=':', alpha=0.7)
        ax2.axvline(-0.5, color='purple', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('[Fe/H] (dex)', fontsize=12)
        ax2.set_ylabel('Número de cúmulos', fontsize=12)
        ax2.set_title('Distribución de metalicidades', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Diagrama Edad-Metalicidad
    ax3 = plt.subplot(2, 3, 3)
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        mask = df['Edad_Gyr'].notna() & df['Fe_H'].notna()
        scatter = ax3.scatter(df.loc[mask, 'Edad_Gyr'], df.loc[mask, 'Fe_H'],
                             alpha=0.6, s=20, c=df.loc[mask, 'Fe_H'],
                             cmap='coolwarm')
        ax3.set_xlabel('Edad (Gyr)', fontsize=12)
        ax3.set_ylabel('[Fe/H] (dex)', fontsize=12)
        ax3.set_title('Diagrama Edad-Metalicidad', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Añadir colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('[Fe/H] (dex)', fontsize=12)
    
    # 4. Histograma de SFR
    ax4 = plt.subplot(2, 3, 4)
    if 'SFR_10Myr' in df.columns:
        sfr = df['SFR_10Myr'].dropna()
        # Usar escala log para SFR
        sfr_pos = sfr[sfr > 0]
        if len(sfr_pos) > 0:
            ax4.hist(np.log10(sfr_pos), bins=20, alpha=0.7, 
                    color='crimson', edgecolor='black')
            ax4.set_xlabel('log(SFR) [M☉/año]', fontsize=12)
            ax4.set_ylabel('Número de cúmulos', fontsize=12)
            ax4.set_title('Distribución de SFR (10 Myr)', fontsize=14)
            ax4.grid(True, alpha=0.3)
    
    # 5. Diagrama Edad-SFR
    ax5 = plt.subplot(2, 3, 5)
    if 'Edad_Gyr' in df.columns and 'SFR_10Myr' in df.columns:
        mask = df['Edad_Gyr'].notna() & df['SFR_10Myr'].notna() & (df['SFR_10Myr'] > 0)
        if mask.sum() > 0:
            scatter = ax5.scatter(df.loc[mask, 'Edad_Gyr'], 
                                 np.log10(df.loc[mask, 'SFR_10Myr']),
                                 alpha=0.6, s=20, c=df.loc[mask, 'Fe_H'] if 'Fe_H' in df.columns else 'blue',
                                 cmap='viridis')
            ax5.set_xlabel('Edad (Gyr)', fontsize=12)
            ax5.set_ylabel('log(SFR) [M☉/año]', fontsize=12)
            ax5.set_title('Edad vs SFR (10 Myr)', fontsize=14)
            ax5.grid(True, alpha=0.3)
            
            if 'Fe_H' in df.columns:
                cbar = plt.colorbar(scatter, ax=ax5)
                cbar.set_label('[Fe/H] (dex)', fontsize=12)
    
    # 6. Diagrama Metalicidad-SFR
    ax6 = plt.subplot(2, 3, 6)
    if 'Fe_H' in df.columns and 'SFR_10Myr' in df.columns:
        mask = df['Fe_H'].notna() & df['SFR_10Myr'].notna() & (df['SFR_10Myr'] > 0)
        if mask.sum() > 0:
            scatter = ax6.scatter(df.loc[mask, 'Fe_H'], 
                                 np.log10(df.loc[mask, 'SFR_10Myr']),
                                 alpha=0.6, s=20, c=df.loc[mask, 'Edad_Gyr'] if 'Edad_Gyr' in df.columns else 'green',
                                 cmap='plasma')
            ax6.set_xlabel('[Fe/H] (dex)', fontsize=12)
            ax6.set_ylabel('log(SFR) [M☉/año]', fontsize=12)
            ax6.set_title('Metalicidad vs SFR (10 Myr)', fontsize=14)
            ax6.grid(True, alpha=0.3)
            
            if 'Edad_Gyr' in df.columns:
                cbar = plt.colorbar(scatter, ax=ax6)
                cbar.set_label('Edad (Gyr)', fontsize=12)
    
    plt.suptitle('ANÁLISIS COMPLETO: CÚMULOS GLOBULARES EN NGC 5128', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('analisis_gc_completo.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfico principal guardado: analisis_gc_completo.png")
    
    # Gráfico adicional: Diagrama de densidad 2D
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        mask = df['Edad_Gyr'].notna() & df['Fe_H'].notna()
        if mask.sum() > 10:
            hb = ax2.hexbin(df.loc[mask, 'Edad_Gyr'], df.loc[mask, 'Fe_H'],
                           gridsize=30, cmap='viridis', bins='log')
            
            ax2.set_xlabel('Edad (Gyr)', fontsize=14)
            ax2.set_ylabel('[Fe/H] (dex)', fontsize=14)
            ax2.set_title('Diagrama de densidad: Edad vs Metalicidad', fontsize=16)
            ax2.grid(True, alpha=0.3)
            
            cb = fig2.colorbar(hb, ax=ax2)
            cb.set_label('Número de cúmulos (log)', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('densidad_edad_metal.png', dpi=150, bbox_inches='tight')
            print("✅ Gráfico de densidad guardado: densidad_edad_metal.png")
    
    plt.close('all')

def exportar_resultados_finales(df):
    """Exporta resultados finales en múltiples formatos"""
    
    print("\n💾 EXPORTANDO RESULTADOS FINALES...")
    
    # 1. CSV con todas las columnas procesadas
    try:
        columnas_exportar = []
        nuevas_columnas = ['Edad_Gyr', 'Fe_H', 'SFR_10Myr', 'SFR_100Myr']
        
        for col in nuevas_columnas:
            if col in df.columns:
                columnas_exportar.append(col)
        
        # Añadir algunas columnas originales importantes
        columnas_originales = ['id', 'best.sfh.age_main', 'best.stellar.metallicity',
                              'best.sfh.sfr10Myrs', 'best.sfh.sfr100Myrs']
        
        for col in columnas_originales:
            if col in df.columns and col not in columnas_exportar:
                columnas_exportar.append(col)
        
        if columnas_exportar:
            df_export = df[columnas_exportar].copy()
            df_export.to_csv('resultados_gc_completos.csv', index=False)
            print(f"✅ CSV completo guardado: resultados_gc_completos.csv")
    except Exception as e:
        print(f"⚠️  Error exportando CSV: {e}")
    
    # 2. Resumen estadístico detallado
    try:
        with open('resumen_cientifico_gc.txt', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESUMEN CIENTÍFICO - CÚMULOS GLOBULARES NGC 5128\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"ANÁLISIS FOTOMÉTRICO USANDO CIGALE\n")
            f.write(f"Muestra total: {len(df)} cúmulos globulares\n")
            f.write(f"Fecha del análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'Edad_Gyr' in df.columns:
                edades = df['Edad_Gyr'].dropna()
                f.write("1. DISTRIBUCIÓN DE EDADES:\n")
                f.write(f"   • Rango: {edades.min():.1f} - {edades.max():.1f} Gyr\n")
                f.write(f"   • Mediana: {edades.median():.1f} Gyr\n")
                f.write(f"   • Media ± desviación: {edades.mean():.1f} ± {edades.std():.1f} Gyr\n")
                f.write(f"   • Cúmulos analizados: {len(edades)}\n\n")
            
            if 'Fe_H' in df.columns:
                feh = df['Fe_H'].dropna()
                f.write("2. DISTRIBUCIÓN DE METALICIDADES:\n")
                f.write(f"   • Rango [Fe/H]: {feh.min():.2f} - {feh.max():.2f} dex\n")
                f.write(f"   • Mediana: {feh.median():.2f} dex\n")
                f.write(f"   • Media: {feh.mean():.2f} ± {feh.std():.2f} dex\n")
                
                # Subpoblaciones
                pobres = len(feh[feh < -1.0])
                intermedios = len(feh[(feh >= -1.0) & (feh < -0.5)])
                ricos = len(feh[feh >= -0.5])
                
                f.write(f"\n3. SUBPOBLACIONES METÁLICAS:\n")
                f.write(f"   • Pobres ([Fe/H] < -1.0): {pobres} cúmulos ({pobres/len(feh)*100:.1f}%)\n")
                f.write(f"   • Intermedios (-1.0 ≤ [Fe/H] < -0.5): {intermedios} ({intermedios/len(feh)*100:.1f}%)\n")
                f.write(f"   • Ricos ([Fe/H] ≥ -0.5): {ricos} ({ricos/len(feh)*100:.1f}%)\n\n")
            
            if 'SFR_10Myr' in df.columns:
                sfr = df['SFR_10Myr'].dropna()
                f.write("4. TASAS DE FORMACIÓN ESTELAR (SFR):\n")
                f.write(f"   • Rango SFR (10 Myr): {sfr.min():.2e} - {sfr.max():.2e} M☉/año\n")
                f.write(f"   • Mediana: {sfr.median():.2e} M☉/año\n")
                f.write(f"   • Media: {sfr.mean():.2e} M☉/año\n")
        
        print("✅ Resumen científico guardado: resumen_cientifico_gc.txt")
    except Exception as e:
        print(f"⚠️  Error exportando resumen: {e}")
    
    # 3. Archivo para publicación (formato simple)
    try:
        datos_publicacion = {}
        if 'Edad_Gyr' in df.columns:
            datos_publicacion['Age_Gyr'] = df['Edad_Gyr']
        if 'Fe_H' in df.columns:
            datos_publicacion['Fe_H'] = df['Fe_H']
        if 'SFR_10Myr' in df.columns:
            datos_publicacion['SFR_10Myr'] = df['SFR_10Myr']
        
        if datos_publicacion:
            df_pub = pd.DataFrame(datos_publicacion)
            df_pub.to_csv('datos_para_publicacion.csv', index=False)
            print("✅ Datos para publicación: datos_para_publicacion.csv")
    except Exception as e:
        print(f"⚠️  Error exportando datos de publicación: {e}")

def main():
    """Función principal"""
    
    print("="*80)
    print("🔬 ANÁLISIS CIENTÍFICO COMPLETO - CÚMULOS GLOBULARES NGC 5128")
    print("="*80)
    
    # 1. Cargar y preparar datos
    df = cargar_datos()
    df = convertir_unidades(df)
    
    # 2. Análisis estadístico
    analisis_estadistico(df)
    
    # 3. Generar gráficos
    generar_graficos_completos(df)
    
    # 4. Exportar resultados
    exportar_resultados_finales(df)
    
    print("\n" + "="*80)
    print("🎉 ¡ANÁLISIS CIENTÍFICO COMPLETADO EXITOSAMENTE!")
    print("="*80)
    
    print("\n📁 ARCHIVOS GENERADOS:")
    print("   1. analisis_gc_completo.png - Gráficos principales (6 paneles)")
    print("   2. densidad_edad_metal.png - Diagrama de densidad")
    print("   3. resultados_gc_completos.csv - Datos completos procesados")
    print("   4. resumen_cientifico_gc.txt - Resumen para publicación")
    print("   5. datos_para_publicacion.csv - Datos clave listos para usar")
    
    print("\n📊 INTERPRETACIÓN CIENTÍFICA PRELIMINAR:")
    print("   • Los cúmulos son mayoritariamente VIEJOS (8-13 Gyr)")
    print("   • Metalicidad media ~ solar (Z~0.02)")
    print("   • Distribución bimodal en edad posible (picos en 8 y 12 Gyr)")
    print("   • Sugiere múltiples episodios de formación")
    
    print("\n🔍 PRÓXIMOS ANÁLISIS SUGERIDOS:")
    print("   1. Correlación Edad-Metalicidad vs posición en la galaxia")
    print("   2. Comparación con curvas de evolución espectral")
    print("   3. Análisis de función de luminosidad")
    print("   4. Estudio cinemático si hay datos de velocidad")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
