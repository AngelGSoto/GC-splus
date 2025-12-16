#!/usr/bin/env python3
# analisis_gc_completo_final_CORREGIDO.py
# Versión CORREGIDA del script de análisis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
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

def diagnosticar_metalicidad(df):
    """Diagnóstico para entender qué tipo de valores tenemos"""
    
    print("\n🔍 DIAGNÓSTICO DE METALICIDAD CRUDA")
    print("="*50)
    
    if 'best.stellar.metallicity' in df.columns:
        raw_metal = df['best.stellar.metallicity'].values
        
        print(f"Valores crudos de 'best.stellar.metallicity':")
        print(f"• Media: {raw_metal.mean():.6f}")
        print(f"• Mediana: {np.median(raw_metal):.6f}")
        print(f"• Mínimo: {raw_metal.min():.6f}")
        print(f"• Máximo: {raw_metal.max():.6f}")
        print(f"• Primeros 5 valores: {raw_metal[:5]}")
        
        # Determinar si son Z o [Fe/H]
        if raw_metal.mean() > 0.001 and raw_metal.mean() < 0.05:
            print("\n✅ LOS VALORES PARECEN SER Z (fracción de masa)")
            print(f"   Z_sun = 0.02 (estándar)")
            print(f"   Tu media Z = {raw_metal.mean():.4f} ≈ {raw_metal.mean()/0.02:.1f} × Z_sun")
            return 'Z'
        elif raw_metal.mean() > -3 and raw_metal.mean() < 1:
            print("\n✅ LOS VALORES PARECEN SER [Fe/H] (escala logarítmica)")
            return '[Fe/H]'
        else:
            print("\n⚠️  NO SE PUEDE DETERMINAR EL FORMATO")
            return 'unknown'
    
    return None

def convertir_unidades(df):
    """Convierte unidades CORRECTAMENTE"""
    
    print("\n🔄 CONVIRTIENDO UNIDADES...")
    
    # 1. Convertir edades de Myr a Gyr
    if 'best.sfh.age_main' in df.columns:
        df['Edad_Gyr'] = df['best.sfh.age_main'] / 1000.0
        print(f"✅ Edades convertidas: Myr → Gyr")
    
    # 2. DIAGNÓSTICO Y CONVERSIÓN DE METALICIDAD
    if 'best.stellar.metallicity' in df.columns:
        raw_metal = df['best.stellar.metallicity'].values
        
        # Primero, diagnóstico
        format_type = diagnosticar_metalicidad(df)
        
        if format_type == 'Z':
            # VALOR CORREGIDO: Z_solar = 0.02
            Z_solar = 0.02
            
            # Convertir Z a [Fe/H]
            # [Fe/H] = log10(Z / Z_solar)
            mask = raw_metal > 0
            FeH = np.full_like(raw_metal, np.nan, dtype=float)
            FeH[mask] = np.log10(raw_metal[mask] / Z_solar)
            
            df['Fe_H'] = FeH
            df['Z'] = raw_metal  # Guardar también Z original
            
            print(f"✅ Metalicidad convertida: Z → [Fe/H]")
            print(f"   Z_sun = {Z_solar}")
            print(f"   [Fe/H] medio = {np.nanmean(FeH):.3f}")
            print(f"   [Fe/H] mediano = {np.nanmedian(FeH):.3f}")
            
        elif format_type == '[Fe/H]':
            # Ya son [Fe/H], no convertir
            df['Fe_H'] = raw_metal
            print(f"✅ Valores ya son [Fe/H], no se convirtieron")
            
        else:
            # Usar valor por defecto
            Z_solar = 0.02
            mask = raw_metal > 0
            FeH = np.full_like(raw_metal, np.nan, dtype=float)
            FeH[mask] = np.log10(raw_metal[mask] / Z_solar)
            df['Fe_H'] = FeH
            print(f"⚠️  Convertido asumiendo Z con Z_sun = {Z_solar}")
    
    # 3. SFR - usar log(SFR) para análisis
    if 'best.sfh.sfr10Myrs' in df.columns:
        df['SFR_10Myr'] = df['best.sfh.sfr10Myrs']
        # Crear columna log(SFR)
        mask = df['SFR_10Myr'] > 0
        df['log_SFR_10Myr'] = np.nan
        df.loc[mask, 'log_SFR_10Myr'] = np.log10(df.loc[mask, 'SFR_10Myr'])
    
    if 'best.sfh.sfr100Myrs' in df.columns:
        df['SFR_100Myr'] = df['best.sfh.sfr100Myrs']
        mask = df['SFR_100Myr'] > 0
        df['log_SFR_100Myr'] = np.nan
        df.loc[mask, 'log_SFR_100Myr'] = np.log10(df.loc[mask, 'SFR_100Myr'])
    
    return df

def analisis_estadistico_corregido(df):
    """Análisis estadístico CORREGIDO"""
    
    print("\n" + "="*60)
    print("📈 ANÁLISIS ESTADÍSTICO DETALLADO (CORREGIDO)")
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
        bins = [5, 8, 9, 10, 11, 12, 13]
        hist, _ = np.histogram(edades, bins=bins)
        
        print(f"\n📊 DISTRIBUCIÓN POR RANGOS DE EDAD:")
        for i in range(len(bins)-1):
            n = hist[i]
            pct = (n / len(edades)) * 100
            print(f"  {bins[i]}-{bins[i+1]} Gyr: {n:3d} cúmulos ({pct:5.1f}%)")
    
    # 2. Análisis CORREGIDO de metalicidades
    if 'Fe_H' in df.columns:
        feh = df['Fe_H'].dropna()
        
        print(f"\n🔬 METALICIDADES ([Fe/H]) - CORREGIDAS")
        print("-"*40)
        print(f"• Total cúmulos con metalicidad: {len(feh)}")
        print(f"• Rango: {feh.min():.2f} - {feh.max():.2f} dex")
        print(f"• Mediana: {feh.median():.2f} dex")
        print(f"• Media: {feh.mean():.2f} ± {feh.std():.2f} dex")
        
        # Mostrar también Z si está disponible
        if 'Z' in df.columns:
            Z_vals = df['Z'].dropna()
            print(f"• Z correspondiente: {Z_vals.median():.4f} (mediana)")
        
        # Clasificación por metalicidad
        pobres = feh[feh < -1.0]
        intermedios = feh[(feh >= -1.0) & (feh < -0.5)]
        ricos = feh[feh >= -0.5]
        
        print(f"\n📊 CLASIFICACIÓN POR METALICIDAD:")
        print(f"  Pobres ([Fe/H] < -1.0): {len(pobres):3d} ({len(pobres)/len(feh)*100:5.1f}%)")
        print(f"  Intermedios (-1.0 ≤ [Fe/H] < -0.5): {len(intermedios):3d} ({len(intermedios)/len(feh)*100:5.1f}%)")
        print(f"  Ricos ([Fe/H] ≥ -0.5): {len(ricos):3d} ({len(ricos)/len(feh)*100:5.1f}%)")
        
        # Interpretación
        print(f"\n🔍 INTERPRETACIÓN:")
        median_feh = feh.median()
        if median_feh > 0.2:
            print(f"  ⚠️  ALERTA: [Fe/H] mediana = {median_feh:.2f} (SUPER-SOLAR)")
            print(f"     • Esto es EXTREMADAMENTE inusual para GCs viejos")
            print(f"     • Verificar conversión y valores originales")
        elif median_feh > 0:
            print(f"  ⚠️  [Fe/H] mediana = {median_feh:.2f} (ligeramente super-solar)")
            print(f"     • Inusual pero posible en GCs ricos")
        elif median_feh > -0.5:
            print(f"  ✅ [Fe/H] mediana = {median_feh:.2f} (GCs ricos)")
            print(f"     • Típico de GCs rojos en galaxias masivas")
        else:
            print(f"  ✅ [Fe/H] mediana = {median_feh:.2f} (GCs pobres a intermedios)")
    
    # 3. Análisis de SFR (usando log)
    if 'log_SFR_10Myr' in df.columns:
        log_sfr = df['log_SFR_10Myr'].dropna()
        
        print(f"\n🔬 TASA DE FORMACIÓN ESTELAR (SFR)")
        print("-"*40)
        print(f"• Total cúmulos con SFR: {len(log_sfr)}")
        
        # Convertir de log a lineal para estadísticas
        sfr_linear = 10**log_sfr
        print(f"• Rango lineal: {sfr_linear.min():.2e} - {sfr_linear.max():.2e} M☉/año")
        print(f"• Mediana lineal: {sfr_linear.median():.2e} M☉/año")
        print(f"• Media log10(SFR): {log_sfr.mean():.2f} ± {log_sfr.std():.2f} dex")

def generar_graficos_corregidos(df):
    """Genera gráficos con valores CORREGIDOS"""
    
    print("\n🎨 GENERANDO GRÁFICOS CORREGIDOS...")
    
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
    
    # 2. Histograma de metalicidades CORREGIDAS
    ax2 = plt.subplot(2, 3, 2)
    if 'Fe_H' in df.columns:
        feh = df['Fe_H'].dropna()
        ax2.hist(feh, bins=20, alpha=0.7, color='forestgreen', edgecolor='black')
        ax2.axvline(feh.median(), color='red', linestyle='--',
                   label=f'Mediana: {feh.median():.2f} dex')
        
        # Líneas de clasificación
        ax2.axvline(-1.0, color='orange', linestyle=':', alpha=0.7, label='Pobre/Intermedio')
        ax2.axvline(-0.5, color='purple', linestyle=':', alpha=0.7, label='Intermedio/Rico')
        ax2.axvline(0.0, color='gold', linestyle=':', alpha=0.7, label='Solar')
        
        ax2.set_xlabel('[Fe/H] (dex)', fontsize=12)
        ax2.set_ylabel('Número de cúmulos', fontsize=12)
        ax2.set_title('Distribución de metalicidades (CORREGIDA)', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # 3. Diagrama Edad-Metalicidad
    ax3 = plt.subplot(2, 3, 3)
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        mask = df['Edad_Gyr'].notna() & df['Fe_H'].notna()
        scatter = ax3.scatter(df.loc[mask, 'Edad_Gyr'], df.loc[mask, 'Fe_H'],
                             alpha=0.6, s=20, c=df.loc[mask, 'Fe_H'],
                             cmap='coolwarm', vmin=-2, vmax=0.5)
        ax3.set_xlabel('Edad (Gyr)', fontsize=12)
        ax3.set_ylabel('[Fe/H] (dex)', fontsize=12)
        ax3.set_title('Diagrama Edad-Metalicidad (CORREGIDO)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Añadir colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('[Fe/H] (dex)', fontsize=12)
        
        # Añadir línea de regresión si hay suficientes datos
        if mask.sum() > 10:
            x = df.loc[mask, 'Edad_Gyr']
            y = df.loc[mask, 'Fe_H']
            coeffs = np.polyfit(x, y, 1)
            x_fit = np.array([x.min(), x.max()])
            y_fit = coeffs[0] * x_fit + coeffs[1]
            ax3.plot(x_fit, y_fit, 'k--', alpha=0.8, 
                    label=f'Pendiente: {coeffs[0]:.3f} dex/Gyr')
            ax3.legend()
    
    # 4. Histograma de SFR (log)
    ax4 = plt.subplot(2, 3, 4)
    if 'log_SFR_10Myr' in df.columns:
        log_sfr = df['log_SFR_10Myr'].dropna()
        ax4.hist(log_sfr, bins=20, alpha=0.7, color='crimson', edgecolor='black')
        ax4.axvline(log_sfr.median(), color='blue', linestyle='--',
                   label=f'Mediana: {log_sfr.median():.2f}')
        ax4.set_xlabel('log(SFR) [M☉/año]', fontsize=12)
        ax4.set_ylabel('Número de cúmulos', fontsize=12)
        ax4.set_title('Distribución de SFR (log, 10 Myr)', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Diagrama Edad-SFR
    ax5 = plt.subplot(2, 3, 5)
    if 'Edad_Gyr' in df.columns and 'log_SFR_10Myr' in df.columns:
        mask = df['Edad_Gyr'].notna() & df['log_SFR_10Myr'].notna()
        scatter = ax5.scatter(df.loc[mask, 'Edad_Gyr'], 
                             df.loc[mask, 'log_SFR_10Myr'],
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
    if 'Fe_H' in df.columns and 'log_SFR_10Myr' in df.columns:
        mask = df['Fe_H'].notna() & df['log_SFR_10Myr'].notna()
        scatter = ax6.scatter(df.loc[mask, 'Fe_H'], 
                             df.loc[mask, 'log_SFR_10Myr'],
                             alpha=0.6, s=20, c=df.loc[mask, 'Edad_Gyr'] if 'Edad_Gyr' in df.columns else 'green',
                             cmap='plasma')
        ax6.set_xlabel('[Fe/H] (dex)', fontsize=12)
        ax6.set_ylabel('log(SFR) [M☉/año]', fontsize=12)
        ax6.set_title('Metalicidad vs SFR (10 Myr)', fontsize=14)
        ax6.grid(True, alpha=0.3)
        
        if 'Edad_Gyr' in df.columns:
            cbar = plt.colorbar(scatter, ax=ax6)
            cbar.set_label('Edad (Gyr)', fontsize=12)
    
    plt.suptitle('ANÁLISIS CORREGIDO: CÚMULOS GLOBULARES EN NGC 5128', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('analisis_gc_corregido.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfico corregido guardado: analisis_gc_corregido.png")
    
    # Gráfico adicional: Diagrama de densidad 2D
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        mask = df['Edad_Gyr'].notna() & df['Fe_H'].notna()
        if mask.sum() > 10:
            hb = ax2.hexbin(df.loc[mask, 'Edad_Gyr'], df.loc[mask, 'Fe_H'],
                           gridsize=30, cmap='viridis', bins='log',
                           extent=[5, 13, -2, 1])
            
            ax2.set_xlabel('Edad (Gyr)', fontsize=14)
            ax2.set_ylabel('[Fe/H] (dex)', fontsize=14)
            ax2.set_title('Diagrama de densidad: Edad vs Metalicidad (CORREGIDO)', fontsize=16)
            ax2.grid(True, alpha=0.3)
            
            cb = fig2.colorbar(hb, ax=ax2)
            cb.set_label('Número de cúmulos (log)', fontsize=12)
            
            # Añadir líneas de referencia
            ax2.axhline(y=0, color='gold', linestyle='--', alpha=0.7, label='Solar')
            ax2.axhline(y=-0.5, color='purple', linestyle=':', alpha=0.5, label='Rico/Intermedio')
            ax2.axhline(y=-1.0, color='orange', linestyle=':', alpha=0.5, label='Intermedio/Pobre')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('densidad_edad_metal_corregido.png', dpi=150, bbox_inches='tight')
            print("✅ Gráfico de densidad corregido: densidad_edad_metal_corregido.png")
    
    plt.close('all')

def main():
    """Función principal CORREGIDA"""
    
    print("="*80)
    print("🔬 ANÁLISIS CIENTÍFICO CORREGIDO - CÚMULOS GLOBULARES NGC 5128")
    print("="*80)
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Convertir unidades CORRECTAMENTE
    df = convertir_unidades(df)
    
    # 3. Análisis estadístico CORREGIDO
    analisis_estadistico_corregido(df)
    
    # 4. Generar gráficos CORREGIDOS
    generar_graficos_corregidos(df)
    
    print("\n" + "="*80)
    print("🎉 ¡ANÁLISIS CORREGIDO COMPLETADO!")
    print("="*80)
    
    # Guardar datos corregidos
    if 'Fe_H' in df.columns and 'Edad_Gyr' in df.columns:
        df_corregido = df[['Fe_H', 'Edad_Gyr', 'Z' if 'Z' in df.columns else '']].dropna()
        df_corregido.to_csv('resultados_corregidos.csv', index=False)
        print("✅ Datos corregidos guardados: resultados_corregidos.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
