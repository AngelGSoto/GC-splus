#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_quality_age_metallicity_plotter_FINAL_SIMPLE_WITH_INVERTED.py
====================================================================

Versión final simplificada con texto unificado en una sola caja.
INCLUYE plot adicional con Age vs Metallicity (ejes invertidos).

Autor: Luis A. Gutiérrez Soto
Versión: 10.0 (Con plot invertido adicional)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
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

def setup_style():
    """Configura estilo para paper."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'patch.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'mathtext.fontset': 'stix',
    })

# ============================================================================
# CLASE PRINCIPAL SIMPLIFICADA CON PLOT ADICIONAL
# ============================================================================

class AgeMetallicityPlotter:
    """Plotter para paper con texto simplificado y plot invertido."""
    
    def __init__(self, results_file, experiment='narrow', output_dir='paper_plots_final'):
        self.results_file = Path(results_file)
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        setup_style()
        
        # Cargar datos
        self.data = self.load_data()
        
        # Configuración
        self.config = {
            'age_unit': 'Gyr',
            'metallicity_unit': '[Fe/H]',
            'sun_metallicity': 0.02,
            'solar_color': '#FF8C00',
            'cmap_scatter': 'viridis',
            'cmap_scatter_inverted': 'plasma',  # Diferente colormap para plot invertido
            'size_scatter': 35,
            'alpha_scatter': 0.8,
        }
        
        print(f"✅ Cargados {len(self.data)} objetos")
        print(f"📋 Experimento: {self.get_experiment_display_name()}")
    
    def get_experiment_display_name(self):
        """Obtiene el nombre para mostrar del experimento."""
        experiments = {
            'narrow': 'S-PLUS (narrow)',
            'narrow+broad': 'S-PLUS (narrow) + DECam',
            'splus_only': 'S-PLUS (narrow)',
            'splus_decam': 'S-PLUS (narrow) + DECam',
            'narrow_only': 'S-PLUS (narrow)',
            'full': 'S-PLUS (narrow) + DECam'
        }
        return experiments.get(self.experiment, f"Experiment: {self.experiment}")
    
    def get_experiment_filename_part(self):
        """Obtiene la parte del nombre de archivo para el experimento."""
        filename_parts = {
            'narrow': 'narrow_only',
            'narrow+broad': 'narrow_decam',
            'splus_only': 'splus_only',
            'splus_decam': 'splus_decam',
            'narrow_only': 'narrow_only',
            'full': 'narrow_decam'
        }
        return filename_parts.get(self.experiment, self.experiment.replace('+', '_'))
    
    def load_data(self):
        """Carga datos del archivo FITS."""
        print("📊 CARGANDO DATOS...")
        
        with fits.open(self.results_file) as hdul:
            data = Table(hdul[1].data).to_pandas()
        
        return data
    
    def prepare_data(self):
        """Prepara los datos para plotting."""
        print("\n📊 PREPARANDO DATOS...")
        
        # Usar columnas bayesianas
        age_col = 'bayes.stellar.age_m_star'
        metal_col = 'bayes.stellar.metallicity'
        
        age = pd.to_numeric(self.data[age_col], errors='coerce')
        metallicity = pd.to_numeric(self.data[metal_col], errors='coerce')
        
        # Filtrar NaN
        mask = age.notna() & metallicity.notna()
        age = age[mask]
        metallicity = metallicity[mask]
        
        print(f"   • Objetos válidos: {len(age)}")
        
        # Convertir edad de Myr a Gyr
        if age.mean() > 1000:
            age = age / 1000.0
        
        # Convertir Z a [Fe/H]
        z_sun = self.config['sun_metallicity']
        if metallicity.min() >= 0 and metallicity.max() <= 0.1:
            valid_mask = metallicity > 0
            feh = np.full_like(metallicity, np.nan)
            feh[valid_mask] = np.log10(metallicity[valid_mask] / z_sun)
            metallicity = pd.Series(feh, index=metallicity.index)
        
        return age.values, metallicity.values
    
    def create_final_plot(self):
        """Crea el gráfico final simplificado (Metallicity vs Age)."""
        
        age, metallicity = self.prepare_data()
        
        # ====================================================================
        # CONFIGURACIÓN DE LA FIGURA
        # ====================================================================
        fig = plt.figure(figsize=(8.5, 6))
        
        # Grid de 2 filas y 3 columnas
        gs = fig.add_gridspec(2, 3, 
                             width_ratios=[4, 0.15, 1],
                             height_ratios=[1, 4],
                             left=0.08, right=0.95,
                             bottom=0.12, top=0.90,
                             wspace=0.1, hspace=0.05)
        
        # ====================================================================
        # PANEL PRINCIPAL (scatter plot)
        # ====================================================================
        ax_main = fig.add_subplot(gs[1, 0])
        
        # Scatter plot con densidad
        from scipy.stats import gaussian_kde
        xy = np.vstack([age, metallicity])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        
        scatter = ax_main.scatter(age[idx], metallicity[idx], 
                                c=z[idx],
                                s=self.config['size_scatter'],
                                alpha=self.config['alpha_scatter'],
                                cmap=self.config['cmap_scatter'],
                                edgecolors='white',
                                linewidth=0.5,
                                zorder=5)
        
        # Línea solar
        ax_main.axhline(y=0, 
                       color=self.config['solar_color'],
                       linestyle='--', 
                       linewidth=2.0,
                       alpha=0.8,
                       zorder=3)
        
        # ====================================================================
        # BARRA DE COLOR - POSICIÓN MANUAL
        # ====================================================================
        main_pos = ax_main.get_position()

        # Definir posición manual de la barra de color
        bar_left = main_pos.x1 + 0.005
        bar_bottom = main_pos.y0
        bar_width = 0.02
        bar_height = main_pos.height

        # Crear eje para barra de color
        cax = fig.add_axes([bar_left, bar_bottom, bar_width, bar_height])
        cbar = plt.colorbar(scatter, cax=cax, orientation='vertical')
        cbar.set_label('Point Density', fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        
        # ====================================================================
        # HISTOGRAMA DE EDAD
        # ====================================================================
        ax_hist_age = fig.add_subplot(gs[0, 0], sharex=ax_main)
        plt.setp(ax_hist_age.get_xticklabels(), visible=False)
        
        ax_hist_age.hist(age, bins=20, 
                        color='skyblue', 
                        alpha=0.7,
                        edgecolor='navy',
                        linewidth=0.8,
                        density=True)
        
        ax_hist_age.set_ylabel('Density', fontsize=14)
        ax_hist_age.grid(True, alpha=0.2, linestyle='--')
        
        # ====================================================================
        # HISTOGRAMA DE METALICIDAD
        # ====================================================================
        ax_hist_metal = fig.add_subplot(gs[1, 2], sharey=ax_main)
        plt.setp(ax_hist_metal.get_yticklabels(), visible=False)
        
        ax_hist_metal.hist(metallicity, bins=20,
                          orientation='horizontal',
                          color='lightcoral',
                          alpha=0.7,
                          edgecolor='darkred',
                          linewidth=0.8,
                          density=True)
        
        ax_hist_metal.set_xlabel('Density', fontsize=14)
        ax_hist_metal.grid(True, alpha=0.2, linestyle='--')
        
        # ====================================================================
        # CELDA VACÍA
        # ====================================================================
        ax_unused = fig.add_subplot(gs[0, 2])
        ax_unused.axis('off')
        
        # ====================================================================
        # TEXTO SIMPLIFICADO EN UNA SOLA CAJA
        # ====================================================================
        experiment_name = self.get_experiment_display_name()
        
        # Texto simple: experimento y estadísticas
        text_lines = [
            f'{experiment_name}',
            f'',
            f'$N = {len(age):,}$',
            f'$\\langle Age \\rangle = {age.mean():.2f} \\pm {age.std():.2f}$ Gyr',
            f'$\\langle [Fe/H] \\rangle = {metallicity.mean():.3f} \\pm {metallicity.std():.3f}$'
        ]
        
        combined_text = '\n'.join(text_lines)
        
        # Posición en coordenadas de ejes (abajo derecha)
        # Usamos transform=ax_main.transAxes para coordenadas relativas (0-1)
        ax_main.text(0.98, 0.05, combined_text,
                    transform=ax_main.transAxes,
                    fontsize=11,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             alpha=0.95,
                             edgecolor='black',
                             linewidth=1),
                    zorder=10)
        
        # Texto "Solar" al lado de la línea
        # Usamos coordenadas de datos para la posición vertical
        solar_x = age.max() - 0.03 * (age.max() - age.min())
        ax_main.text(solar_x, 0.02, 'Solar',
                    color=self.config['solar_color'],
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.2',
                             facecolor='white',
                             alpha=0.9,
                             edgecolor=self.config['solar_color'],
                             linewidth=0.5),
                    zorder=10)
        
        # ====================================================================
        # CONFIGURACIÓN FINAL
        # ====================================================================
        # Ejes
        ax_main.set_xlabel(f'Age ({self.config["age_unit"]})', 
                          fontsize=14, fontweight='bold', labelpad=8)
        ax_main.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})', 
                          fontsize=14, fontweight='bold', labelpad=8)
        
        # Tamaño de los números de los ejes
        ax_main.tick_params(axis='both', labelsize=12)
        ax_hist_age.tick_params(axis='y', labelsize=12)
        ax_hist_metal.tick_params(axis='x', labelsize=12)
    
        # Grid
        ax_main.grid(True, alpha=0.2, linestyle='--', zorder=1)
        
        # Ajustar límites
        age_margin = 0.05 * (age.max() - age.min())
        metal_margin = 0.05 * (metallicity.max() - metallicity.min())
        
        ax_main.set_xlim(age.min() - age_margin, age.max() + age_margin)
        ax_main.set_ylim(metallicity.min() - metal_margin, 
                        metallicity.max() + metal_margin)
        
        # Título
        plt.suptitle('Metallicity vs Age', fontsize=16, fontweight='bold', y=0.98)
        
        # ====================================================================
        # GUARDAR
        # ====================================================================
        experiment_part = self.get_experiment_filename_part()
        filename = f"age_metallicity_{experiment_part}.pdf"
        output_path = self.output_dir / filename
        
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"\n✅ GRÁFICO PRINCIPAL GUARDADO EN: {output_path}")
        
        # Versión PNG para revisión
        png_path = self.output_dir / f"age_metallicity_{experiment_part}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Versión PNG: {png_path}")
        
        plt.close(fig)
        
        return fig, ax_main
    
    def create_inverted_plot(self):
        """Crea el gráfico adicional con Age vs Metallicity (ejes invertidos)."""
        
        age, metallicity = self.prepare_data()
        
        # ====================================================================
        # CONFIGURACIÓN DE LA FIGURA
        # ====================================================================
        fig = plt.figure(figsize=(9, 7))
        
        # Grid más simple para el plot invertido
        gs = fig.add_gridspec(2, 3, 
                             width_ratios=[4, 0.15, 1],
                             height_ratios=[1, 4],
                             left=0.1, right=0.95,
                             bottom=0.1, top=0.92,
                             wspace=0.1, hspace=0.05)
        
        # ====================================================================
        # PANEL PRINCIPAL (scatter plot - EJES INVERTIDOS)
        # ====================================================================
        ax_main = fig.add_subplot(gs[1, 0])
        
        # Scatter plot con densidad (ejes invertidos: x=metallicity, y=age)
        from scipy.stats import gaussian_kde
        xy = np.vstack([metallicity, age])  # Note: invertido para el plot invertido
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        
        scatter = ax_main.scatter(metallicity[idx], age[idx], 
                                c=z[idx],
                                s=self.config['size_scatter'],
                                alpha=self.config['alpha_scatter'],
                                cmap=self.config['cmap_scatter_inverted'],
                                edgecolors='white',
                                linewidth=0.5,
                                zorder=5)
        
        # Línea solar (vertical ahora)
        ax_main.axvline(x=0, 
                       color=self.config['solar_color'],
                       linestyle='--', 
                       linewidth=2.0,
                       alpha=0.8,
                       zorder=3)
        
        # ====================================================================
        # BARRA DE COLOR - POSICIÓN MANUAL
        # ====================================================================
        main_pos = ax_main.get_position()

        # Definir posición manual de la barra de color
        bar_left = main_pos.x1 + 0.005
        bar_bottom = main_pos.y0
        bar_width = 0.02
        bar_height = main_pos.height

        # Crear eje para barra de color
        cax = fig.add_axes([bar_left, bar_bottom, bar_width, bar_height])
        cbar = plt.colorbar(scatter, cax=cax, orientation='vertical')
        cbar.set_label('Point Density', fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
        
        # ====================================================================
        # HISTOGRAMA DE METALICIDAD (ahora horizontal superior)
        # ====================================================================
        ax_hist_metal = fig.add_subplot(gs[0, 0], sharex=ax_main)
        plt.setp(ax_hist_metal.get_xticklabels(), visible=False)
        
        ax_hist_metal.hist(metallicity, bins=20, 
                          color='lightcoral', 
                          alpha=0.7,
                          edgecolor='darkred',
                          linewidth=0.8,
                          density=True)
        
        ax_hist_metal.set_ylabel('Density', fontsize=14)
        ax_hist_metal.grid(True, alpha=0.2, linestyle='--')
        
        # ====================================================================
        # HISTOGRAMA DE EDAD (ahora vertical derecho)
        # ====================================================================
        ax_hist_age = fig.add_subplot(gs[1, 2], sharey=ax_main)
        plt.setp(ax_hist_age.get_yticklabels(), visible=False)
        
        ax_hist_age.hist(age, bins=20,
                        orientation='horizontal',
                        color='skyblue',
                        alpha=0.7,
                        edgecolor='navy',
                        linewidth=0.8,
                        density=True)
        
        ax_hist_age.set_xlabel('Density', fontsize=14)
        ax_hist_age.grid(True, alpha=0.2, linestyle='--')
        
        # ====================================================================
        # CELDA VACÍA
        # ====================================================================
        ax_unused = fig.add_subplot(gs[0, 2])
        ax_unused.axis('off')
        
        # ====================================================================
        # TEXTO SIMPLIFICADO EN UNA SOLA CAJA
        # ====================================================================
        experiment_name = self.get_experiment_display_name()
        
        # Texto simple: experimento y estadísticas
        text_lines = [
            f'{experiment_name}',
            f'',
            f'$N = {len(age):,}$',
            f'$\\langle Age \\rangle = {age.mean():.2f} \\pm {age.std():.2f}$ Gyr',
            f'$\\langle [Fe/H] \\rangle = {metallicity.mean():.3f} \\pm {metallicity.std():.3f}$'
        ]
        
        combined_text = '\n'.join(text_lines)
        
        # Posición en coordenadas de ejes (superior izquierda)
        ax_main.text(0.02, 0.98, combined_text,
                    transform=ax_main.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             alpha=0.95,
                             edgecolor='black',
                             linewidth=1),
                    zorder=10)
        
        # Texto "Solar" al lado de la línea (ahora vertical)
        solar_y = age.max() - 0.03 * (age.max() - age.min())
        ax_main.text(0.02, solar_y, 'Solar',
                    color=self.config['solar_color'],
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.2',
                             facecolor='white',
                             alpha=0.9,
                             edgecolor=self.config['solar_color'],
                             linewidth=0.5),
                    zorder=10)
        
        # ====================================================================
        # CONFIGURACIÓN FINAL
        # ====================================================================
        # Ejes (INVERTIDOS)
        ax_main.set_xlabel(f'Metallicity ({self.config["metallicity_unit"]})', 
                          fontsize=14, fontweight='bold', labelpad=8)
        ax_main.set_ylabel(f'Age ({self.config["age_unit"]})', 
                          fontsize=14, fontweight='bold', labelpad=8)
        
        # Tamaño de los números de los ejes
        ax_main.tick_params(axis='both', labelsize=12)
        ax_hist_metal.tick_params(axis='y', labelsize=12)
        ax_hist_age.tick_params(axis='x', labelsize=12)
    
        # Grid
        ax_main.grid(True, alpha=0.2, linestyle='--', zorder=1)
        
        # Ajustar límites
        age_margin = 0.05 * (age.max() - age.min())
        metal_margin = 0.05 * (metallicity.max() - metallicity.min())
        
        ax_main.set_xlim(metallicity.min() - metal_margin, metallicity.max() + metal_margin)
        ax_main.set_ylim(age.min() - age_margin, age.max() + age_margin)
        
        # Título
        plt.suptitle('Age vs Metallicity', fontsize=16, fontweight='bold', y=0.98)
        
        # ====================================================================
        # GUARDAR
        # ====================================================================
        experiment_part = self.get_experiment_filename_part()
        filename = f"age_metallicity_inverted_{experiment_part}.pdf"
        output_path = self.output_dir / filename
        
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ GRÁFICO INVERTIDO GUARDADO EN: {output_path}")
        
        # Versión PNG para revisión
        png_path = self.output_dir / f"age_metallicity_inverted_{experiment_part}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Versión PNG: {png_path}")
        
        plt.close(fig)
        
        return fig, ax_main
    
    def create_both_plots(self):
        """Crea ambos gráficos (principal e invertido)."""
        print("\n🎨 CREANDO AMBOS GRÁFICOS...")
        print("=" * 50)
        print("1. Creando gráfico principal (Metallicity vs Age)...")
        self.create_final_plot()
        print("\n2. Creando gráfico invertido (Age vs Metallicity)...")
        self.create_inverted_plot()

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal."""
    
    print("\n" + "="*70)
    print("📊 AGE-METALLICITY PLOTTER - VERSIÓN COMPLETA (2 PLOTS)")
    print("="*70)
    
    parser = argparse.ArgumentParser(
        description='Create age-metallicity plots with experiment info (two versions)'
    )
    parser.add_argument('input_file', help='FITS file from CIGALE')
    parser.add_argument('--experiment', default='narrow',
                       choices=['narrow', 'narrow+broad', 'splus_only', 'splus_decam'],
                       help='Type of experiment/filter combination')
    parser.add_argument('--output-dir', default='paper_plots_final', 
                       help='Output directory')
    parser.add_argument('--plot-type', default='both',
                       choices=['standard', 'inverted', 'both'],
                       help='Type of plot to generate: standard (Metallicity vs Age), inverted (Age vs Metallicity), or both')
    
    args = parser.parse_args()
    
    try:
        plotter = AgeMetallicityPlotter(args.input_file, args.experiment, args.output_dir)
        
        if args.plot_type == 'standard':
            print("\n🎨 CREANDO GRÁFICO ESTÁNDAR (Metallicity vs Age)...")
            plotter.create_final_plot()
        elif args.plot_type == 'inverted':
            print("\n🎨 CREANDO GRÁFICO INVERTIDO (Age vs Metallicity)...")
            plotter.create_inverted_plot()
        else:  # both
            plotter.create_both_plots()
        
        print(f"\n" + "="*70)
        print("✅ PROCESO COMPLETADO!")
        print("="*70)
        print(f"📁 Resultados en: {args.output_dir}")
        
        # Mostrar nombres de archivo generados
        experiment_part = plotter.get_experiment_filename_part()
        print(f"\n📄 ARCHIVOS GENERADOS:")
        
        if args.plot_type in ['standard', 'both']:
            print(f"   • age_metallicity_{experiment_part}.pdf")
            print(f"   • age_metallicity_{experiment_part}.png")
        
        if args.plot_type in ['inverted', 'both']:
            print(f"   • age_metallicity_inverted_{experiment_part}.pdf")
            print(f"   • age_metallicity_inverted_{experiment_part}.png")
        
        print(f"\n📊 RESUMEN ESTADÍSTICO:")
        age, metallicity = plotter.prepare_data()
        print(f"   • Número de objetos: {len(age):,}")
        print(f"   • Edad promedio: {age.mean():.2f} ± {age.std():.2f} Gyr")
        print(f"   • Rango de edades: {age.min():.2f} - {age.max():.2f} Gyr")
        print(f"   • Metalicidad promedio: {metallicity.mean():.3f} ± {metallicity.std():.3f} [Fe/H]")
        print(f"   • Rango de metalicidades: {metallicity.min():.3f} - {metallicity.max():.3f} [Fe/H]")
        
        print(f"\n💡 SUGERENCIAS PARA EL PAPER:")
        print(f"   • El plot estándar (Metallicity vs Age) es común en estudios de poblaciones estelares.")
        print(f"   • El plot invertido (Age vs Metallicity) es útil para ver la relación edad-metalicidad.")
        print(f"   • Ambos plots muestran la misma información pero con perspectivas diferentes.")
        print(f"   • Para A&A, recomiendo incluir ambos como figuras suplementarias o elegir uno.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
