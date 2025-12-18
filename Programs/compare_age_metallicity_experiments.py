#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_experiments_direct_SIMPLE.py
=====================================

Versión simplificada con solo mediana en histogramas.

Autor: Luis A. Gutiérrez Soto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse
import sys
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE ESTILO PARA PAPER
# ============================================================================

def setup_style():
    """Configura estilo para paper."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.titlesize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'patch.linewidth': 1.0,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'grid.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'mathtext.fontset': 'stix',
    })

# ============================================================================
# CLASE PARA COMPARACIÓN DIRECTA - VERSIÓN SIMPLIFICADA
# ============================================================================

class SimpleExperimentComparator:
    """Comparación directa entre experimentos para paper."""
    
    def __init__(self):
        setup_style()
        
        # Paleta de colores atractivos para paper
        self.colors = {
            'scatter_blue': '#1E88E5',
            'scatter_orange': '#FF9800',
            'identity_line': '#4CAF50',
            'histogram': '#E53935',
            'text_box': '#3F51B5',
        }
        
        # Configuración para paper
        self.config = {
            'marker_size': 80,
            'marker_alpha': 0.7,
            'line_width': 2.5,
            'hist_alpha': 0.7,
            'edge_width': 0.8,
        }
    
    def load_data(self, results_file):
        """Carga y prepara datos de un experimento."""
        print(f"📊 Cargando: {results_file}")
        
        results_file = Path(results_file)
        if not results_file.exists():
            print(f"❌ Archivo no encontrado: {results_file}")
            return None
        
        # Cargar FITS
        with fits.open(results_file) as hdul:
            data = Table(hdul[1].data).to_pandas()
        
        # Usar columnas bayesianas
        age_col = 'bayes.stellar.age_m_star'
        metal_col = 'bayes.stellar.metallicity'
        
        age = pd.to_numeric(data[age_col], errors='coerce')
        metallicity = pd.to_numeric(data[metal_col], errors='coerce')
        
        # Filtrar NaN
        mask = age.notna() & metallicity.notna()
        age = age[mask]
        metallicity = metallicity[mask]
        
        print(f"   • Objetos válidos: {len(age)}")
        
        # Convertir edad de Myr a Gyr
        if age.mean() > 1000:
            age = age / 1000.0
        
        # Convertir Z a [Fe/H]
        z_sun = 0.02
        if metallicity.min() >= 0 and metallicity.max() <= 0.1:
            valid_mask = metallicity > 0
            feh = np.full_like(metallicity, np.nan)
            feh[valid_mask] = np.log10(metallicity[valid_mask] / z_sun)
            metallicity = pd.Series(feh, index=metallicity.index)
        
        return {
            'age': age.values,
            'metallicity': metallicity.values,
            'n_objects': len(age)
        }
    
    def create_metallicity_comparison(self, splus_data, splus_decam_data, output_dir):
        """Crea gráfico de comparación de metalicidad para paper."""
        
        fig, (ax_main, ax_hist) = plt.subplots(1, 2, figsize=(15, 6.5),
                                               gridspec_kw={'width_ratios': [3, 1]})
        
        # Extraer datos
        metal_splus = splus_data['metallicity']
        metal_decam = splus_decam_data['metallicity']
        
        # Verificar que tienen el mismo número de objetos
        if len(metal_splus) != len(metal_decam):
            print("⚠️  Diferente número de objetos. Recortando al mínimo común.")
            min_len = min(len(metal_splus), len(metal_decam))
            metal_splus = metal_splus[:min_len]
            metal_decam = metal_decam[:min_len]
        
        # ====================================================================
        # GRÁFICO PRINCIPAL: Metalicidad S-PLUS vs S-PLUS+DECam
        # ====================================================================
        
        # Scatter plot
        scatter = ax_main.scatter(metal_splus, metal_decam,
                                 s=self.config['marker_size'],
                                 alpha=self.config['marker_alpha'],
                                 color=self.colors['scatter_blue'],
                                 edgecolors='white',
                                 linewidth=self.config['edge_width'],
                                 zorder=5)
        
        # Línea de identidad
        min_val = min(metal_splus.min(), metal_decam.min())
        max_val = max(metal_splus.max(), metal_decam.max())
        margin = 0.1 * (max_val - min_val)
        
        identity_line = np.linspace(min_val - margin, max_val + margin, 100)
        ax_main.plot(identity_line, identity_line,
                    color=self.colors['identity_line'],
                    linestyle='--',
                    linewidth=self.config['line_width'],
                    alpha=0.8,
                    label='1:1 line',
                    zorder=3)
        
        # Configurar ejes
        ax_main.set_xlabel('Metallicity [Fe/H] (S-PLUS only)', 
                           fontsize=16, fontweight='bold', labelpad=12)
        ax_main.set_ylabel('Metallicity [Fe/H] (S-PLUS + DECam)', 
                           fontsize=16, fontweight='bold', labelpad=12)
        
        # Límites
        ax_main.set_xlim(min_val - margin, max_val + margin)
        ax_main.set_ylim(min_val - margin, max_val + margin)
        
        # Grid
        ax_main.grid(True, alpha=0.15, linestyle='--', zorder=1)
        ax_main.legend(loc='lower right', fontsize=14, framealpha=0.9)
        ax_main.set_title('Metallicity Comparison: S-PLUS vs S-PLUS+DECam',
                         fontsize=18, fontweight='bold', pad=15)
        
        # ====================================================================
        # ESTADÍSTICAS DE COMPARACIÓN
        # ====================================================================
        
        # Calcular diferencias
        diff = metal_decam - metal_splus
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        median_diff = np.median(diff)
        
        # Correlación
        r_value, p_value = stats.pearsonr(metal_splus, metal_decam)
        
        # Texto de estadísticas
        stats_text = f'$N = {len(metal_splus):,}$\n'
        stats_text += f'$\\langle \\Delta[Fe/H] \\rangle = {mean_diff:.3f}$\n'
        stats_text += f'$\\sigma_{{\\Delta[Fe/H]}} = {std_diff:.3f}$\n'
        stats_text += f'$\\mathrm{{median}}(\\Delta[Fe/H]) = {median_diff:.3f}$\n'
        stats_text += f'$r = {r_value:.3f}$'
        
        # Añadir estadísticas
        ax_main.text(0.02, 0.98, stats_text,
                    transform=ax_main.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             alpha=0.95,
                             edgecolor=self.colors['text_box'],
                             linewidth=1.5),
                    zorder=10)
        
        # ====================================================================
        # HISTOGRAMA DE DIFERENCIAS (SOLO MEDIANA)
        # ====================================================================
        
        # Histograma
        n_bins = 25
        n, bins, patches = ax_hist.hist(diff, bins=n_bins,
                                       color=self.colors['histogram'],
                                       alpha=self.config['hist_alpha'],
                                       edgecolor='darkred',
                                       linewidth=self.config['edge_width'],
                                       density=True,
                                       orientation='horizontal')
        
        # Línea vertical en cero
        ax_hist.axvline(x=0, color='black', linestyle='-', 
                       linewidth=1, alpha=0.3)
        
        # Configurar histograma
        ax_hist.set_xlabel('Density', fontsize=14, fontweight='bold')
        ax_hist.set_ylabel('Δ[Fe/H] (S-PLUS+DECam - S-PLUS)', 
                          fontsize=14, fontweight='bold', labelpad=12)
        ax_hist.set_title('Distribution of Differences',
                         fontsize=16, fontweight='bold', pad=15)
        
        # SOLO MEDIANA
        hist_stats = f'Median = {median_diff:.3f}'
        
        ax_hist.text(0.95, 0.95, hist_stats,
                    transform=ax_hist.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             alpha=0.9,
                             edgecolor='gray',
                             linewidth=1))
        
        ax_hist.grid(True, alpha=0.15, linestyle='--')
        
        # Ajustar límites
        hist_max = np.max(n) * 1.1
        ax_hist.set_xlim(0, hist_max)
        
        # ====================================================================
        # MEJORAS ESTÉTICAS
        # ====================================================================
        ax_main.tick_params(axis='both', which='major', labelsize=16)
        ax_hist.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        
        # ====================================================================
        # GUARDAR
        # ====================================================================
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_dir / "metallicity_comparison_simple.pdf"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ Gráfico de metalicidad guardado: {output_path}")
        
        png_path = output_dir / "metallicity_comparison_simple.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Versión PNG: {png_path}")
        
        #plt.show()
        
        return fig, (ax_main, ax_hist), diff
    
    def create_age_comparison(self, splus_data, splus_decam_data, output_dir):
        """Crea gráfico de comparación de edad para paper."""
        
        fig, (ax_main, ax_hist) = plt.subplots(1, 2, figsize=(15, 6.5),
                                               gridspec_kw={'width_ratios': [3, 1]})
        
        # Extraer datos
        age_splus = splus_data['age']
        age_decam = splus_decam_data['age']
        
        # Verificar que tienen el mismo número de objetos
        if len(age_splus) != len(age_decam):
            print("⚠️  Diferente número de objetos. Recortando al mínimo común.")
            min_len = min(len(age_splus), len(age_decam))
            age_splus = age_splus[:min_len]
            age_decam = age_decam[:min_len]
        
        # ====================================================================
        # GRÁFICO PRINCIPAL: Edad S-PLUS vs S-PLUS+DECam
        # ====================================================================
        
        # Scatter plot
        scatter = ax_main.scatter(age_splus, age_decam,
                                 s=self.config['marker_size'],
                                 alpha=self.config['marker_alpha'],
                                 color=self.colors['scatter_orange'],
                                 edgecolors='white',
                                 linewidth=self.config['edge_width'],
                                 zorder=5)
        
        # Línea de identidad
        min_val = min(age_splus.min(), age_decam.min())
        max_val = max(age_splus.max(), age_decam.max())
        margin = 0.1 * (max_val - min_val)
        
        identity_line = np.linspace(min_val - margin, max_val + margin, 100)
        ax_main.plot(identity_line, identity_line,
                    color=self.colors['identity_line'],
                    linestyle='--',
                    linewidth=self.config['line_width'],
                    alpha=0.8,
                    label='1:1 line',
                    zorder=3)
        
        # Configurar ejes
        ax_main.set_xlabel('Age (Gyr) (S-PLUS only)', 
                          fontsize=16, fontweight='bold', labelpad=12)
        ax_main.set_ylabel('Age (Gyr) (S-PLUS + DECam)', 
                          fontsize=16, fontweight='bold', labelpad=12)
        
        # Límites
        ax_main.set_xlim(min_val - margin, max_val + margin)
        ax_main.set_ylim(min_val - margin, max_val + margin)
        
        # Grid
        ax_main.grid(True, alpha=0.15, linestyle='--', zorder=1)
        ax_main.legend(loc='lower right', fontsize=14, framealpha=0.9)
        ax_main.set_title('Age Comparison: S-PLUS vs S-PLUS+DECam',
                         fontsize=18, fontweight='bold', pad=15)
        
        # ====================================================================
        # ESTADÍSTICAS DE COMPARACIÓN
        # ====================================================================
        
        # Calcular diferencias
        diff = age_decam - age_splus
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        median_diff = np.median(diff)
        
        # Correlación
        r_value, p_value = stats.pearsonr(age_splus, age_decam)
        
        # Texto de estadísticas
        stats_text = f'$N = {len(age_splus):,}$\n'
        stats_text += f'$\\langle \\Delta Age \\rangle = {mean_diff:.3f}$ Gyr\n'
        stats_text += f'$\\sigma_{{\\Delta Age}} = {std_diff:.3f}$ Gyr\n'
        stats_text += f'$\\mathrm{{median}}(\\Delta Age) = {median_diff:.3f}$ Gyr\n'
        stats_text += f'$r = {r_value:.3f}$'
        
        # Añadir estadísticas
        ax_main.text(0.02, 0.98, stats_text,
                    transform=ax_main.transAxes,
                    fontsize=14,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             alpha=0.95,
                             edgecolor=self.colors['text_box'],
                             linewidth=1.5),
                    zorder=10)
        
        # ====================================================================
        # HISTOGRAMA DE DIFERENCIAS (SOLO MEDIANA)
        # ====================================================================
        
        # Histograma
        n_bins = 25
        n, bins, patches = ax_hist.hist(diff, bins=n_bins,
                                       color=self.colors['histogram'],
                                       alpha=self.config['hist_alpha'],
                                       edgecolor='darkred',
                                       linewidth=self.config['edge_width'],
                                       density=True,
                                       orientation='horizontal')
        
        # Línea vertical en cero
        ax_hist.axvline(x=0, color='black', linestyle='-', 
                       linewidth=1, alpha=0.3)
        
        # Configurar histograma
        ax_hist.set_xlabel('Density', fontsize=12, fontweight='bold')
        ax_hist.set_ylabel('ΔAge (Gyr) (S-PLUS+DECam - S-PLUS)', 
                          fontsize=14, fontweight='bold', labelpad=12)
        ax_hist.set_title('Distribution of Differences',
                         fontsize=16, fontweight='bold', pad=15)
        
        # SOLO MEDIANA
        hist_stats = f'Median = {median_diff:.3f} Gyr'
        
        ax_hist.text(0.95, 0.95, hist_stats,
                    transform=ax_hist.transAxes,
                    fontsize=13,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             alpha=0.9,
                             edgecolor='gray',
                             linewidth=1))
        
        ax_hist.grid(True, alpha=0.15, linestyle='--')
        
        # Ajustar límites
        hist_max = np.max(n) * 1.1
        ax_hist.set_xlim(0, hist_max)
        
        # ====================================================================
        # MEJORAS ESTÉTICAS
        # ====================================================================
        ax_main.tick_params(axis='both', which='major', labelsize=16)
        ax_hist.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        
        # ====================================================================
        # GUARDAR
        # ====================================================================
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_dir / "age_comparison_simple.pdf"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ Gráfico de edad guardado: {output_path}")
        
        png_path = output_dir / "age_comparison_simple.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Versión PNG: {png_path}")
        
        #plt.show()
        
        return fig, (ax_main, ax_hist), diff

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal."""
    
    print("\n" + "="*70)
    print("📊 COMPARACIÓN PARA PAPER (SOLO MEDIANA)")
    print("="*70)
    
    parser = argparse.ArgumentParser(description='Paper-quality comparison with median only')
    
    parser.add_argument('--splus-only', required=True,
                       help='Path to results.fits from S-PLUS only experiment')
    parser.add_argument('--splus-decam', required=True,
                       help='Path to results.fits from S-PLUS+DECam experiment')
    parser.add_argument('--output-dir', default='paper_comparison_simple',
                       help='Output directory for paper plots')
    
    args = parser.parse_args()
    
    try:
        # Crear comparador
        comparator = SimpleExperimentComparator()
        
        # Cargar datos
        print(f"\n📂 CARGANDO DATOS...")
        splus_data = comparator.load_data(args.splus_only)
        splus_decam_data = comparator.load_data(args.splus_decam)
        
        if splus_data is None or splus_decam_data is None:
            print("❌ Error al cargar datos.")
            return 1
        
        # Crear directorio de salida
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Comparaciones
        print("\n🎨 CREANDO COMPARACIÓN DE METALICIDAD...")
        fig_metal, axes_metal, metal_diffs = comparator.create_metallicity_comparison(
            splus_data, splus_decam_data, output_dir
        )
        
        print("\n🎨 CREANDO COMPARACIÓN DE EDAD...")
        fig_age, axes_age, age_diffs = comparator.create_age_comparison(
            splus_data, splus_decam_data, output_dir
        )
        
        print(f"\n✅ ANÁLISIS COMPLETADO!")
        print(f"📁 Resultados en: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
