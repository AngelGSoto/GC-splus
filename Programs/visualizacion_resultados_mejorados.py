#!/usr/bin/env python3
# visualizacion_resultados_mejorados.py
# Visualiza los resultados con la rejilla fina

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    print("="*70)
    print("📊 VISUALIZACIÓN DE RESULTADOS CON REJILLA FINA")
    print("="*70)
    
    # Cargar datos
    try:
        df = pd.read_csv('resultados_gc_completos.csv')
        print(f"✅ Datos cargados: {len(df)} cúmulos")
    except:
        print("❌ No se pudo cargar resultados_gc_completos.csv")
        return
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Histograma de edades con KDE
    ax1 = plt.subplot(2, 3, 1)
    if 'Edad_Gyr' in df.columns:
        sns.histplot(df['Edad_Gyr'].dropna(), bins=15, kde=True, ax=ax1)
        ax1.axvline(df['Edad_Gyr'].median(), color='red', linestyle='--', 
                   label=f'Mediana: {df["Edad_Gyr"].median():.1f} Gyr')
        ax1.set_xlabel('Edad (Gyr)')
        ax1.set_ylabel('Número de cúmulos')
        ax1.set_title('Distribución de Edades (con KDE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de metalicidades
    ax2 = plt.subplot(2, 3, 2)
    if 'Fe_H' in df.columns:
        sns.histplot(df['Fe_H'].dropna(), bins=15, kde=True, ax=ax2, color='green')
        ax2.axvline(df['Fe_H'].median(), color='red', linestyle='--',
                   label=f'Mediana: {df["Fe_H"].median():.2f} dex')
        ax2.axvline(-1.0, color='orange', linestyle=':', alpha=0.7, label='[Fe/H] = -1.0')
        ax2.axvline(-0.5, color='purple', linestyle=':', alpha=0.7, label='[Fe/H] = -0.5')
        ax2.set_xlabel('[Fe/H] (dex)')
        ax2.set_ylabel('Número de cúmulos')
        ax2.set_title('Distribución de Metalicidades')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Diagrama Edad-Metalicidad
    ax3 = plt.subplot(2, 3, 3)
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        scatter = ax3.scatter(df['Edad_Gyr'], df['Fe_H'], alpha=0.6, s=20,
                            c=df['Fe_H'], cmap='coolwarm')
        plt.colorbar(scatter, ax=ax3, label='[Fe/H] (dex)')
        ax3.set_xlabel('Edad (Gyr)')
        ax3.set_ylabel('[Fe/H] (dex)')
        ax3.set_title('Diagrama Edad-Metalicidad')
        ax3.grid(True, alpha=0.3)
    
    # 4. Distribución acumulativa de edades
    ax4 = plt.subplot(2, 3, 4)
    if 'Edad_Gyr' in df.columns:
        edades_sorted = np.sort(df['Edad_Gyr'].dropna())
        cumul = np.arange(1, len(edades_sorted)+1) / len(edades_sorted)
        
        ax4.plot(edades_sorted, cumul, 'b-', linewidth=2)
        ax4.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='50%')
        ax4.axhline(0.25, color='g', linestyle='--', alpha=0.5, label='25%')
        ax4.axhline(0.75, color='g', linestyle='--', alpha=0.5, label='75%')
        
        ax4.set_xlabel('Edad (Gyr)')
        ax4.set_ylabel('Fracción acumulativa')
        ax4.set_title('Distribución Acumulativa de Edades')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Boxplot por rango de edad
    ax5 = plt.subplot(2, 3, 5)
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        # Crear bins de edad
        bins = [5, 8, 10, 12, 13]
        labels = ['5-8', '8-10', '10-12', '12-13']
        df['Rango_Edad'] = pd.cut(df['Edad_Gyr'], bins=bins, labels=labels)
        
        # Filtrar solo los que tienen ambos valores
        df_plot = df.dropna(subset=['Edad_Gyr', 'Fe_H', 'Rango_Edad'])
        
        sns.boxplot(data=df_plot, x='Rango_Edad', y='Fe_H', ax=ax5)
        ax5.set_xlabel('Rango de Edad (Gyr)')
        ax5.set_ylabel('[Fe/H] (dex)')
        ax5.set_title('Metalicidad por Rango de Edad')
        ax5.grid(True, alpha=0.3)
    
    # 6. Distribución 2D (hexbin)
    ax6 = plt.subplot(2, 3, 6)
    if 'Edad_Gyr' in df.columns and 'Fe_H' in df.columns:
        hb = ax6.hexbin(df['Edad_Gyr'], df['Fe_H'], gridsize=30, cmap='viridis', bins='log')
        cb = fig.colorbar(hb, ax=ax6)
        cb.set_label('Número de cúmulos (log)')
        ax6.set_xlabel('Edad (Gyr)')
        ax6.set_ylabel('[Fe/H] (dex)')
        ax6.set_title('Diagrama de Densidad Edad-Metalicidad')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Cúmulos Globulares en NGC 5128 - Resultados con Rejilla Fina', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('resultados_rejilla_fina.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfico guardado: resultados_rejilla_fina.png")
    
    # Análisis estadístico adicional
    print("\n📈 ESTADÍSTICAS ADICIONALES:")
    print("-"*40)
    
    if 'Edad_Gyr' in df.columns:
        edades = df['Edad_Gyr'].dropna()
        print(f"Distribución de edades:")
        for i in range(len(bins)-1):
            mask = (edades >= bins[i]) & (edades < bins[i+1])
            count = mask.sum()
            percent = count / len(edades) * 100
            print(f"  {labels[i]} Gyr: {count:3d} cúmulos ({percent:5.1f}%)")
    
    print("\n" + "="*70)
    print("🎉 VISUALIZACIÓN COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    main()
