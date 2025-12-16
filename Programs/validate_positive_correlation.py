#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_age_metal_correlation.py
==================================

Script simplificado para validar correlación edad-metalicidad
y generar tablas para paper.

Autor: Luis A. Gutiérrez Soto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats
import os
import sys

# Configuración simple
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
})

def load_cigale_results(results_file='out/results.fits'):
    """Carga resultados de CIGALE."""
    
    print(f"Cargando: {results_file}")
    
    if not os.path.exists(results_file):
        # Buscar alternativas
        alternatives = [
            "results.fits",
            "../out/results.fits",
            "pcigale/out/results.fits"
        ]
        
        for alt in alternatives:
            if os.path.exists(alt):
                results_file = alt
                print(f"Encontrado: {results_file}")
                break
        else:
            print("ERROR: No se encontró results.fits")
            sys.exit(1)
    
    try:
        with fits.open(results_file) as hdul:
            # Usar la primera extensión con datos
            data = hdul[1].data
            df = pd.DataFrame(data)
            print(f"Datos cargados: {len(df)} objetos")
            return df
    except Exception as e:
        print(f"Error cargando FITS: {e}")
        sys.exit(1)

def extract_age_metal(df):
    """Extrae edad y metalicidad."""
    
    # Buscar columnas automáticamente
    age_col = None
    metal_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'age' in col_lower and 'main' not in col_lower:
            age_col = col
        elif 'metal' in col_lower or col_lower == 'z':
            metal_col = col
    
    if not age_col or not metal_col:
        print("Columnas encontradas:", df.columns.tolist())
        print("ERROR: No se encontraron columnas de edad o metalicidad")
        sys.exit(1)
    
    print(f"Edad: {age_col}")
    print(f"Metalicidad: {metal_col}")
    
    # Extraer datos
    age = df[age_col].values.astype(float)
    metal = df[metal_col].values.astype(float)
    
    # Filtrar valores inválidos
    mask = np.isfinite(age) & np.isfinite(metal)
    age = age[mask]
    metal = metal[mask]
    
    print(f"Datos válidos: {len(age)} objetos")
    
    # Convertir edad a Gyr si está en Myr
    if np.mean(age) > 1000:
        age = age / 1000.0
        print("Edad convertida de Myr a Gyr")
    
    # Convertir Z a [Fe/H] si es necesario
    if np.mean(metal) < 1.0 and np.mean(metal) > 0.0001:  # Probablemente Z
        metal = np.log10(metal / 0.02)  # Z_sun = 0.02
        print("Metalicidad convertida de Z a [Fe/H]")
    
    return age, metal

def calculate_statistics(age, metal):
    """Calcula estadísticas básicas."""
    
    stats_dict = {
        'n': len(age),
        'age_mean': np.mean(age),
        'age_std': np.std(age),
        'age_median': np.median(age),
        'age_min': np.min(age),
        'age_max': np.max(age),
        'metal_mean': np.mean(metal),
        'metal_std': np.std(metal),
        'metal_median': np.median(metal),
        'metal_min': np.min(metal),
        'metal_max': np.max(metal),
    }
    
    # Correlación si hay suficientes datos
    if len(age) > 10:
        try:
            # Pearson
            r, p = stats.pearsonr(age, metal)
            stats_dict['pearson_r'] = r
            stats_dict['pearson_p'] = p
            
            # Spearman
            rho, p_rho = stats.spearmanr(age, metal)
            stats_dict['spearman_rho'] = rho
            stats_dict['spearman_p'] = p_rho
            
            # Regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(age, metal)
            stats_dict['slope'] = slope
            stats_dict['slope_err'] = std_err
            stats_dict['intercept'] = intercept
            stats_dict['r_squared'] = r_value**2
            stats_dict['regression_p'] = p_value
            
        except Exception as e:
            print(f"Advertencia: No se pudo calcular correlación: {e}")
    
    return stats_dict

def plot_simple_scatter(age, metal, stats):
    """Crea un gráfico simple."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(age, metal, s=20, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
    
    # Línea de regresión si existe
    if 'slope' in stats:
        x_fit = np.array([age.min(), age.max()])
        y_fit = stats['intercept'] + stats['slope'] * x_fit
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Fit: [Fe/H] = {stats["slope"]:.3f} × Age + {stats["intercept"]:.3f}')
    
    ax.set_xlabel('Age (Gyr)', fontweight='bold')
    ax.set_ylabel('[Fe/H]', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Texto con estadísticas
    text = f'N = {stats["n"]}\n'
    text += f'Age: {stats["age_mean"]:.2f} ± {stats["age_std"]:.2f} Gyr\n'
    text += f'[Fe/H]: {stats["metal_mean"]:.2f} ± {stats["metal_std"]:.2f}'
    
    if 'pearson_r' in stats:
        text += f'\nr = {stats["pearson_r"]:.3f} (p = {stats["pearson_p"]:.2e})'
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('age_metal_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráfico guardado: age_metal_plot.pdf")

def generate_text_table(age, metal, stats, filename='age_metal_stats.txt'):
    """Genera tabla de texto simple."""
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("AGE-METALLICITY STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"SAMPLE SIZE: {stats['n']}\n\n")
        
        f.write("AGE (Gyr):\n")
        f.write(f"  Mean ± SD: {stats['age_mean']:.2f} ± {stats['age_std']:.2f}\n")
        f.write(f"  Median:    {stats['age_median']:.2f}\n")
        f.write(f"  Range:     {stats['age_min']:.1f} to {stats['age_max']:.1f}\n\n")
        
        f.write("METALLICITY ([Fe/H]):\n")
        f.write(f"  Mean ± SD: {stats['metal_mean']:.2f} ± {stats['metal_std']:.2f}\n")
        f.write(f"  Median:    {stats['metal_median']:.2f}\n")
        f.write(f"  Range:     {stats['metal_min']:.2f} to {stats['metal_max']:.2f}\n\n")
        
        if 'pearson_r' in stats:
            f.write("CORRELATION:\n")
            f.write(f"  Pearson r:  {stats['pearson_r']:.3f} (p = {stats['pearson_p']:.2e})\n")
            f.write(f"  Spearman ρ: {stats['spearman_rho']:.3f} (p = {stats['spearman_p']:.2e})\n\n")
            
            f.write("LINEAR REGRESSION:\n")
            f.write(f"  Slope:      {stats['slope']:.4f} ± {stats['slope_err']:.4f}\n")
            f.write(f"  Intercept:  {stats['intercept']:.3f}\n")
            f.write(f"  R²:         {stats['r_squared']:.3f}\n")
            f.write(f"  Equation:   [Fe/H] = {stats['slope']:.3f} × Age + {stats['intercept']:.3f}\n\n")
        
        # Interpretación
        f.write("="*60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*60 + "\n\n")
        
        if 'slope' in stats:
            slope = stats['slope']
            
            if slope > 0.03:
                f.write("STRONG POSITIVE CORRELATION\n")
                f.write("-"*40 + "\n")
                f.write("Older GCs are MORE metal-rich.\n")
                f.write("This is UNUSUAL for globular clusters.\n\n")
                f.write("Possible explanations:\n")
                f.write("1. Age-metallicity degeneracy in photometric fitting\n")
                f.write("2. Selection bias in the sample\n")
                f.write("3. Complex formation history of NGC 5128\n")
                
            elif slope > 0.01:
                f.write("WEAK POSITIVE CORRELATION\n")
                f.write("-"*40 + "\n")
                f.write("Slight trend: older GCs are slightly more metal-rich.\n")
                
            elif abs(slope) < 0.01:
                f.write("NO SIGNIFICANT CORRELATION\n")
                f.write("-"*40 + "\n")
                f.write("Age and metallicity are independent.\n")
                
            elif slope < -0.01:
                f.write("NEGATIVE CORRELATION\n")
                f.write("-"*40 + "\n")
                f.write("Classical relation: older GCs are more metal-poor.\n")
    
    print(f"Tabla de texto guardada: {filename}")

def generate_latex_table_simple(stats, filename='age_metal_table.tex'):
    """Genera tabla LaTeX simple."""
    
    # Función para formatear números
    def fmt(num, dec=3):
        return f"{num:.{dec}f}"
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Statistics of the age-metallicity relation for globular clusters in NGC 5128.}
\\label{tab:age_metal_stats}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Parameter} & \\textbf{Value} & \\textbf{Units} \\\\
\\midrule
\\multicolumn{3}{l}{\\textbf{Sample}} \\\\
\\quad N & {n} & -- \\\\
\\midrule
\\multicolumn{3}{l}{\\textbf{Age}} \\\\
\\quad Mean & ${age_mean} \\pm {age_std}$ & Gyr \\\\
\\quad Median & ${age_median}$ & Gyr \\\\
\\quad Range & ${age_min}--{age_max}$ & Gyr \\\\
\\midrule
\\multicolumn{3}{l}{\\textbf{Metallicity ([Fe/H])}} \\\\
\\quad Mean & ${metal_mean} \\pm {metal_std}$ & dex \\\\
\\quad Median & ${metal_median}$ & dex \\\\
\\quad Range & ${metal_min}--{metal_max}$ & dex \\\\
"""
    
    if 'pearson_r' in stats:
        latex += """\\midrule
\\multicolumn{3}{l}{\\textbf{Correlation}} \\\\
\\quad Pearson $r$ & ${pearson_r}$ & -- \\\\
\\quad $p$-value & ${pearson_p}$ & -- \\\\
\\midrule
\\multicolumn{3}{l}{\\textbf{Linear Regression}} \\\\
\\quad Slope & ${slope} \\pm {slope_err}$ & dex Gyr$^{-1}$ \\\\
\\quad Intercept & ${intercept}$ & dex \\\\
\\quad $R^2$ & ${r_squared}$ & -- \\\\
\\quad Equation & $[\\text{Fe/H}] = {slope_short} \\times \\text{Age} + {intercept_short}$ & -- \\\\
"""
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Reemplazar valores
    replacements = {
        'n': str(stats['n']),
        'age_mean': fmt(stats['age_mean'], 2),
        'age_std': fmt(stats['age_std'], 2),
        'age_median': fmt(stats['age_median'], 2),
        'age_min': fmt(stats['age_min'], 1),
        'age_max': fmt(stats['age_max'], 1),
        'metal_mean': fmt(stats['metal_mean'], 2),
        'metal_std': fmt(stats['metal_std'], 2),
        'metal_median': fmt(stats['metal_median'], 2),
        'metal_min': fmt(stats['metal_min'], 2),
        'metal_max': fmt(stats['metal_max'], 2),
    }
    
    if 'pearson_r' in stats:
        replacements.update({
            'pearson_r': fmt(stats['pearson_r'], 3),
            'pearson_p': f"{stats['pearson_p']:.2e}",
            'slope': fmt(stats['slope'], 4),
            'slope_err': fmt(stats['slope_err'], 4),
            'intercept': fmt(stats['intercept'], 3),
            'r_squared': fmt(stats['r_squared'], 3),
            'slope_short': fmt(stats['slope'], 3),
            'intercept_short': fmt(stats['intercept'], 3),
        })
    
    # Aplicar reemplazos
    for key, value in replacements.items():
        latex = latex.replace(f"{{{key}}}", value)
    
    with open(filename, 'w') as f:
        f.write(latex)
    
    print(f"Tabla LaTeX guardada: {filename}")

def main():
    """Función principal."""
    
    print("="*60)
    print("VALIDACIÓN DE CORRELACIÓN EDAD-METALICIDAD")
    print("="*60)
    
    # 1. Cargar datos
    df = load_cigale_results()
    
    # 2. Extraer datos
    age, metal = extract_age_metal(df)
    
    # 3. Calcular estadísticas
    stats = calculate_statistics(age, metal)
    
    # 4. Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    
    print(f"\nMuestra: N = {stats['n']}")
    print(f"\nEdad (Gyr):")
    print(f"  Media: {stats['age_mean']:.2f} ± {stats['age_std']:.2f}")
    print(f"  Mediana: {stats['age_median']:.2f}")
    print(f"  Rango: {stats['age_min']:.1f} - {stats['age_max']:.1f}")
    
    print(f"\nMetalicidad ([Fe/H]):")
    print(f"  Media: {stats['metal_mean']:.2f} ± {stats['metal_std']:.2f}")
    print(f"  Mediana: {stats['metal_median']:.2f}")
    print(f"  Rango: {stats['metal_min']:.2f} - {stats['metal_max']:.2f}")
    
    if 'pearson_r' in stats:
        print(f"\nCorrelación:")
        print(f"  Pearson r = {stats['pearson_r']:.3f} (p = {stats['pearson_p']:.2e})")
        print(f"  Spearman ρ = {stats['spearman_rho']:.3f} (p = {stats['spearman_p']:.2e})")
        
        print(f"\nRegresión lineal:")
        print(f"  Pendiente: {stats['slope']:.4f} ± {stats['slope_err']:.4f}")
        print(f"  Intercepto: {stats['intercept']:.3f}")
        print(f"  R²: {stats['r_squared']:.3f}")
        print(f"  Ecuación: [Fe/H] = {stats['slope']:.3f} × Age + {stats['intercept']:.3f}")
        
        # Interpretación
        slope = stats['slope']
        print(f"\nINTERPRETACIÓN:")
        if slope > 0.02:
            print("  ⚠️  CORRELACIÓN POSITIVA FUERTE")
            print("     GCs más viejos → más metálicos")
            print("     (Inusual para cúmulos globulares)")
        elif slope > 0:
            print("  ✓ Correlación positiva débil")
        elif slope < -0.01:
            print("  ✓ Correlación negativa (clásica)")
        else:
            print("  ✓ Sin correlación significativa")
    
    # 5. Crear gráfico
    print("\n" + "="*60)
    print("CREANDO GRÁFICO")
    print("="*60)
    plot_simple_scatter(age, metal, stats)
    
    # 6. Generar tablas
    print("\n" + "="*60)
    print("GENERANDO TABLAS")
    print("="*60)
    generate_text_table(age, metal, stats, 'age_metal_statistics.txt')
    generate_latex_table_simple(stats, 'age_metal_table.tex')
    
    print("\n" + "="*60)
    print("¡ANÁLISIS COMPLETADO!")
    print("="*60)
    print("\nArchivos generados:")
    print("  • age_metal_plot.pdf (gráfico)")
    print("  • age_metal_statistics.txt (estadísticas)")
    print("  • age_metal_table.tex (tabla LaTeX para paper)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
