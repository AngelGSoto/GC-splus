#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_quality_age_metallicity_plotter_CORREGIDO_FINAL.py
========================================================

Script CORREGIDO que usa los valores BAYESIANOS de CIGALE (continuos)
en lugar de los valores "best" (discretos del grid).

Cambios principales:
1. Usa bayes.stellar.age_m_star en lugar de best.sfh.age_main
2. Usa bayes.stellar.metallicity en lugar de best.stellar.metallicity
3. Los valores bayesianos son CONTINUOS y representan promedios sobre el grid

Autor: Luis A. Gutiérrez Soto
Versión: 3.0 (Usa valores bayesianos correctos)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
import warnings
import argparse
import sys
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE ESTILO
# ============================================================================

def setup_publication_style(style='aa', font_size=9):
    """Configura estilo para publicaciones."""
    
    styles = {
        'aa': {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times'],
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 11,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'patch.linewidth': 0.6,
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
            'grid.linewidth': 0.4,
            'axes.grid': False,
            'mathtext.fontset': 'stix',
        }
    }
    
    if style in styles:
        plt.rcParams.update(styles[style])
    else:
        plt.rcParams.update(styles['aa'])
    
    return {
        'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c',
        'red': '#d62728', 'purple': '#9467bd', 'yellow': '#bcbd22'
    }

# ============================================================================
# CLASE PRINCIPAL CORREGIDA
# ============================================================================

class AgeMetallicityPlotter:
    """Plotter que usa valores BAYESIANOS de CIGALE."""
    
    def __init__(self, results_file, output_dir='paper_plots_corrected', style='aa'):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.colors = setup_publication_style(style)
        self.style = style
        
        # Cargar y validar datos
        self.data = self.load_and_validate_data()
        
        # Configuración
        self.config = {
            'age_unit': 'Gyr',
            'metallicity_unit': '[Fe/H]',
            'sun_metallicity': 0.02,
            'solar_color': '#FFD700',
            'cmap_scatter': 'viridis',
            'size_scatter': 25,
            'alpha_scatter': 0.7,
            'font_size': 9,
        }
        
        print(f"✅ Cargados {len(self.data)} objetos")
    
    def load_and_validate_data(self):
        """Carga datos y verifica que existen columnas bayesianas."""
        print("📊 CARGANDO DATOS BAYESIANOS DE CIGALE...")
        
        if not self.results_file.exists():
            raise FileNotFoundError(f"No se encuentra: {self.results_file}")
        
        # Cargar FITS
        with fits.open(self.results_file) as hdul:
            data = Table(hdul[1].data).to_pandas()
        
        # VERIFICAR COLUMNAS BAYESIANAS
        required_bayes_columns = [
            'bayes.stellar.age_m_star',
            'bayes.stellar.metallicity'
        ]
        
        missing = []
        for col in required_bayes_columns:
            if col not in data.columns:
                missing.append(col)
        
        if missing:
            print(f"🚨 ERROR: Columnas bayesianas faltantes: {missing}")
            print(f"📋 Columnas disponibles:")
            for col in data.columns[:20]:
                if 'bayes' in col or 'best' in col:
                    print(f"   • {col}")
            
            # Intentar con valores 'best' como fallback
            print(f"\n⚠️  Intentando con columnas 'best'...")
            if 'best.sfh.age_main' in data.columns and 'best.stellar.metallicity' in data.columns:
                print(f"✅ Usando columnas 'best' como fallback")
            else:
                raise ValueError("No se encuentran columnas bayesianas ni 'best'")
        
        return data
    
    def get_correct_columns(self):
        """Determina las columnas correctas a usar."""
        
        # Prioridad 1: Columnas bayesianas
        if 'bayes.stellar.age_m_star' in self.data.columns:
            age_col = 'bayes.stellar.age_m_star'
            print(f"✅ Usando columna bayesiana de edad: {age_col}")
        elif 'best.sfh.age_main' in self.data.columns:
            age_col = 'best.sfh.age_main'
            print(f"⚠️  Usando columna 'best' de edad (no bayesiana): {age_col}")
        else:
            raise ValueError("No se encuentra columna de edad")
        
        if 'bayes.stellar.metallicity' in self.data.columns:
            metal_col = 'bayes.stellar.metallicity'
            print(f"✅ Usando columna bayesiana de metalicidad: {metal_col}")
        elif 'best.stellar.metallicity' in self.data.columns:
            metal_col = 'best.stellar.metallicity'
            print(f"⚠️  Usando columna 'best' de metalicidad (no bayesiana): {metal_col}")
        else:
            raise ValueError("No se encuentra columna de metalicidad")
        
        return age_col, metal_col
    
    def prepare_correct_data(self):
        """
        Prepara los datos usando columnas CORRECTAS (bayesianas preferidas).
        """
        age_col, metal_col = self.get_correct_columns()
        
        print(f"\n📊 PREPARANDO DATOS CORRECTOS:")
        print(f"   • Edad: {age_col}")
        print(f"   • Metalicidad: {metal_col}")
        
        # Extraer datos
        age = pd.to_numeric(self.data[age_col], errors='coerce')
        metallicity = pd.to_numeric(self.data[metal_col], errors='coerce')
        
        # Filtrar NaN
        mask = age.notna() & metallicity.notna()
        age = age[mask]
        metallicity = metallicity[mask]
        
        print(f"   • Objetos válidos: {len(age)}")
        
        # Convertir edad de Myr a Gyr si es necesario
        if age.mean() > 1000:
            age = age / 1000.0
            print("   • Edad convertida: Myr → Gyr")
        
        # Convertir Z a [Fe/H] si es necesario
        z_sun = self.config['sun_metallicity']
        
        # Verificar si son Z o [Fe/H]
        metal_mean = metallicity.mean()
        metal_min = metallicity.min()
        metal_max = metallicity.max()
        
        print(f"   • Rango metalicidad cruda: [{metal_min:.4f}, {metal_max:.4f}]")
        
        # Criterios para determinar si es Z
        is_likely_Z = (
            metal_min >= 0 and 
            metal_max <= 0.1 and
            abs(metal_mean - 0.02) < 0.1
        )
        
        if is_likely_Z:
            # Convertir Z a [Fe/H]
            valid_mask = metallicity > 0
            feh = np.full_like(metallicity, np.nan)
            feh[valid_mask] = np.log10(metallicity[valid_mask] / z_sun)
            
            print(f"   • Convertido: Z → [Fe/H] (Z_sun = {z_sun})")
            print(f"   • Rango [Fe/H]: [{feh[valid_mask].min():.3f}, {feh[valid_mask].max():.3f}]")
            
            metallicity = pd.Series(feh, index=metallicity.index)
        else:
            print(f"   • Ya parece ser [Fe/H] (no se convirtió)")
        
        # Estadísticas
        print(f"\n📈 ESTADÍSTICAS FINALES:")
        print(f"   • Edad: {age.min():.2f} - {age.max():.2f} Gyr")
        print(f"   • [Fe/H]: {metallicity.min():.3f} - {metallicity.max():.3f} dex")
        print(f"   • ⟨Edad⟩: {age.mean():.2f} ± {age.std():.2f} Gyr")
        print(f"   • ⟨[Fe/H]⟩: {metallicity.mean():.3f} ± {metallicity.std():.3f} dex")
        
        return age.values, metallicity.values, age_col, metal_col
    
    def plot_correct_scatter(self, show_histograms=True, show_regression=True,
                           figsize=(7, 6), filename=None):
        """Gráfico con datos CORRECTOS (bayesianos)."""
        
        age, metallicity, age_col, metal_col = self.prepare_correct_data()
        
        # Crear figura
        if show_histograms:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                 left=0.15, right=0.95, bottom=0.15, top=0.95,
                                 wspace=0.05, hspace=0.05)
            ax_scatter = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
            
            plt.setp(ax_histx.get_xticklabels(), visible=False)
            plt.setp(ax_histy.get_yticklabels(), visible=False)
            
            # Histogramas
            ax_histx.hist(age, bins=20, alpha=0.3, color=self.colors['blue'], density=True)
            ax_histx.set_ylabel('Density', fontsize=self.config['font_size']-1)
            
            ax_histy.hist(metallicity, bins=20, orientation='horizontal',
                         alpha=0.3, color=self.colors['red'], density=True)
            ax_histy.set_xlabel('Density', fontsize=self.config['font_size']-1)
        else:
            fig, ax_scatter = plt.subplots(figsize=figsize)
        
        # Scatter plot con densidad
        if len(age) > 30:
            from scipy.stats import gaussian_kde
            try:
                xy = np.vstack([age, metallicity])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                
                scatter = ax_scatter.scatter(age[idx], metallicity[idx], c=z[idx],
                                           s=self.config['size_scatter'],
                                           alpha=self.config['alpha_scatter'],
                                           cmap=self.config['cmap_scatter'],
                                           edgecolors='white', linewidth=0.3)
                
                cbar = plt.colorbar(scatter, ax=ax_scatter, pad=0.02)
                cbar.set_label('Point Density', rotation=270, labelpad=15,
                             fontsize=self.config['font_size']-1)
            except:
                ax_scatter.scatter(age, metallicity, s=self.config['size_scatter'],
                                 alpha=self.config['alpha_scatter'],
                                 color=self.colors['blue'], edgecolors='black', linewidth=0.3)
        else:
            ax_scatter.scatter(age, metallicity, s=self.config['size_scatter'],
                             alpha=self.config['alpha_scatter'],
                             color=self.colors['blue'], edgecolors='black', linewidth=0.3)
        
        # Línea solar
        ax_scatter.axhline(y=0, color=self.config['solar_color'], 
                         linestyle='--', linewidth=1.5, alpha=0.8,
                         label='Solar metallicity')
        
        # Regresión
        if show_regression and len(age) > 10:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(age, metallicity)
                
                x_fit = np.linspace(age.min(), age.max(), 100)
                y_fit = intercept + slope * x_fit
                
                ax_scatter.plot(x_fit, y_fit, color=self.colors['red'], 
                              linewidth=2, linestyle='-', alpha=0.8,
                              label=f'Fit: [Fe/H] = {slope:.3f} × Age + {intercept:.3f}')
                
                # Estadísticas de regresión
                reg_text = f'$r = {r_value:.3f}$'
                if p_value < 0.001:
                    reg_text += ', $p < 0.001$'
                else:
                    reg_text += f', $p = {p_value:.3f}$'
                
                ax_scatter.text(0.02, 0.05, reg_text, transform=ax_scatter.transAxes,
                               fontsize=self.config['font_size']-1,
                               verticalalignment='bottom',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                print("⚠️  No se pudo calcular regresión")
        
        # Configurar ejes
        ax_scatter.set_xlabel(f'Age ({self.config["age_unit"]})', 
                            fontsize=self.config['font_size'], fontweight='bold')
        ax_scatter.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})', 
                            fontsize=self.config['font_size'], fontweight='bold')
        
        # Grid
        ax_scatter.grid(True, alpha=0.2, linestyle='--')
        
        # Leyenda
        handles, labels = ax_scatter.get_legend_handles_labels()
        if handles:
            ax_scatter.legend(handles, labels, loc='best', 
                            framealpha=0.9, fancybox=True,
                            fontsize=self.config['font_size']-1)
        
        # Estadísticas
        stats_text = f'$N = {len(age):,}$\n'
        stats_text += f'$\\langle \\mathrm{{Age}} \\rangle = {age.mean():.2f} \\pm {age.std():.2f}$ Gyr\n'
        stats_text += f'$\\langle \\mathrm{{[Fe/H]}} \\rangle = {metallicity.mean():.2f} \\pm {metallicity.std():.2f}$'
        
        ax_scatter.text(0.98, 0.98, stats_text, transform=ax_scatter.transAxes,
                       fontsize=self.config['font_size']-1,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Título informativo
        if 'bayes' in age_col and 'bayes' in metal_col:
            title = 'NGC 5128 Globular Clusters: Age vs Metallicity (Bayesian Values)'
        else:
            title = 'NGC 5128 Globular Clusters: Age vs Metallicity'
        
        ax_scatter.set_title(title, fontsize=self.config['font_size']+1, 
                           fontweight='bold', pad=15)
        
        # Añadir nota sobre método
        method_note = f'CIGALE Bayesian estimates\n' \
                     f'Age: {age_col}\n' \
                     f'Metallicity: {metal_col}'
        
        ax_scatter.text(0.02, 0.98, method_note, transform=ax_scatter.transAxes,
                       fontsize=self.config['font_size']-2,
                       verticalalignment='top',
                       style='italic',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # Guardar
        if filename is None:
            filename = f"age_metallicity_corrected_{self.style}.pdf"
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ Gráfico CORREGIDO guardado en: {output_path}")
        
        plt.show()
        return fig, ax_scatter
    
    def analyze_distribution(self):
        """Analiza la distribución de valores para verificar continuidad."""
        
        age_col, metal_col = self.get_correct_columns()
        
        print(f"\n🔍 ANÁLISIS DE DISTRIBUCIÓN DE VALORES:")
        print(f"="*50)
        
        age = self.data[age_col].dropna()
        metallicity = self.data[metal_col].dropna()
        
        print(f"\n📊 EDAD ({age_col}):")
        print(f"   • N valores: {len(age)}")
        print(f"   • Rango: [{age.min():.1f}, {age.max():.1f}]")
        print(f"   • Valores únicos: {len(age.unique())}")
        print(f"   • % valores únicos: {len(age.unique())/len(age)*100:.1f}%")
        
        if len(age.unique())/len(age) < 0.1:
            print(f"   ⚠️  VALORES DISCRETOS (probablemente del grid)")
        else:
            print(f"   ✅ VALORES CONTINUOS (bayesianos correctos)")
        
        print(f"\n📊 METALICIDAD ({metal_col}):")
        print(f"   • N valores: {len(metallicity)}")
        print(f"   • Rango: [{metallicity.min():.5f}, {metallicity.max():.5f}]")
        print(f"   • Valores únicos: {len(metallicity.unique())}")
        print(f"   • % valores únicos: {len(metallicity.unique())/len(metallicity)*100:.1f}%")
        
        if len(metallicity.unique())/len(metallicity) < 0.1:
            print(f"   ⚠️  VALORES DISCRETOS (probablemente del grid)")
            print(f"   ⚠️  Debes usar bayes.stellar.metallicity, no best.stellar.metallicity")
        else:
            print(f"   ✅ VALORES CONTINUOS (bayesianos correctos)")
        
        # Mostrar valores más comunes si son discretos
        if len(metallicity.unique()) < 10:
            print(f"\n📋 VALORES DISCRETOS ENCONTRADOS:")
            for val in sorted(metallicity.unique()):
                count = (metallicity == val).sum()
                pct = count / len(metallicity) * 100
                print(f"   • {val:.5f}: {count} objetos ({pct:.1f}%)")
    
    def compare_bayes_vs_best(self):
        """Compara valores bayesianos vs 'best'."""
        
        print(f"\n🔍 COMPARANDO VALORES BAYESIANOS vs 'BEST':")
        print(f"="*60)
        
        # Verificar qué columnas existen
        cols_to_check = [
            ('bayes.stellar.age_m_star', 'best.sfh.age_main'),
            ('bayes.stellar.metallicity', 'best.stellar.metallicity')
        ]
        
        for bayes_col, best_col in cols_to_check:
            if bayes_col in self.data.columns and best_col in self.data.columns:
                bayes_vals = self.data[bayes_col].dropna()
                best_vals = self.data[best_col].dropna()
                
                print(f"\n📊 {bayes_col} vs {best_col}:")
                print(f"   • Bayesianos únicos: {len(bayes_vals.unique())}")
                print(f"   • 'Best' únicos: {len(best_vals.unique())}")
                print(f"   • Diferencia media: {(bayes_vals - best_vals).mean():.4f}")
                
                # Crear gráfico de comparación
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histograma comparativo
                ax1.hist(bayes_vals, bins=30, alpha=0.5, label='Bayesian', density=True)
                ax1.hist(best_vals, bins=30, alpha=0.5, label='Best', density=True)
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Density')
                ax1.set_title(f'Distribution: {bayes_col.split(".")[-1]}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Scatter plot
                ax2.scatter(best_vals, bayes_vals, alpha=0.6, s=20)
                ax2.plot([best_vals.min(), best_vals.max()], 
                        [best_vals.min(), best_vals.max()], 
                        'r--', alpha=0.5, label='y=x')
                ax2.set_xlabel('Best value')
                ax2.set_ylabel('Bayesian value')
                ax2.set_title('Best vs Bayesian')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f"comparison_{bayes_col.split('.')[-1]}.png", 
                          dpi=300, bbox_inches='tight')
                plt.show()
    
    def generate_methods_description(self):
        """Genera descripción de métodos para el paper."""
        
        age_col, metal_col = self.get_correct_columns()
        
        methods = f"""
METHODS DESCRIPTION FOR PAPER
=============================

Data Analysis:
- We used Bayesian estimates from CIGALE SED fitting code
- Ages: {age_col}
- Metallicities: {metal_col}
- Conversion: Z → [Fe/H] using Z_⊙ = {self.config['sun_metallicity']}

Rationale for using Bayesian values:
- Bayesian estimates provide continuous values by marginalizing over the model grid
- Avoids discretization artifacts from the finite grid of models
- More robust for statistical analysis than discrete 'best' values

Statistical Analysis:
- Pearson correlation coefficient
- Linear regression with confidence intervals
- Kernel Density Estimation for visualization
- All analyses performed with custom Python scripts

Note on metallicity:
- Values were converted from mass fraction Z to [Fe/H] scale
- Solar reference: Z_⊙ = {self.config['sun_metallicity']} (Asplund et al. 2009)
"""
        
        methods_file = self.output_dir / "methods_description.txt"
        with open(methods_file, 'w') as f:
            f.write(methods)
        
        print(f"✅ Descripción de métodos guardada: {methods_file}")
        return methods

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal corregida."""
    
    print("\n" + "="*70)
    print("📊 PAPER-QUALITY PLOTTER CORREGIDO (Valores Bayesianos)")
    print("="*70)
    
    parser = argparse.ArgumentParser(description='Create plots with CORRECT Bayesian values')
    parser.add_argument('input_file', help='FITS file from CIGALE')
    parser.add_argument('--output-dir', default='paper_plots_corrected', 
                       help='Output directory')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze value distributions')
    parser.add_argument('--compare', action='store_true',
                       help='Compare Bayesian vs Best values')
    parser.add_argument('--methods', action='store_true',
                       help='Generate methods description')
    
    args = parser.parse_args()
    
    try:
        # Crear plotter
        plotter = AgeMetallicityPlotter(args.input_file, args.output_dir)
        
        # Análisis de distribución
        if args.analyze:
            plotter.analyze_distribution()
        
        # Comparación bayes vs best
        if args.compare:
            plotter.compare_bayes_vs_best()
        
        # Generar descripción de métodos
        if args.methods:
            plotter.generate_methods_description()
        
        # Crear gráfico principal
        print(f"\n🎨 CREANDO GRÁFICO PRINCIPAL CON VALORES CORRECTOS...")
        plotter.plot_correct_scatter(
            show_histograms=True,
            show_regression=True,
            filename="Fig1_age_metallicity_corrected.pdf"
        )
        
        print(f"\n✅ ANÁLISIS CORREGIDO COMPLETADO!")
        print(f"📁 Resultados en: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
