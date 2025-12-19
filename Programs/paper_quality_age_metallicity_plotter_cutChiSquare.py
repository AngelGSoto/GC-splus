#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_quality_age_metallicity_plotter_FINAL_SIMPLE_WITH_CHI2_FILTER.py
=======================================================================

Versión final simplificada con filtrado por calidad de ajuste (χ² reducido).

Autor: Luis A. Gutiérrez Soto
Versión: 10.0 (Con filtrado por calidad)
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
# CLASE PRINCIPAL CON FILTRADO POR CALIDAD
# ============================================================================

class AgeMetallicityPlotter:
    """Plotter para paper con texto simplificado y filtrado por calidad."""
    
    def __init__(self, results_file, experiment='narrow', output_dir='paper_plots_final',
                 chi2_threshold=2.0, use_reduced_chi2=True):
        self.results_file = Path(results_file)
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Parámetros de filtrado por calidad
        self.chi2_threshold = chi2_threshold
        self.use_reduced_chi2 = use_reduced_chi2
        
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
            'size_scatter': 35,
            'alpha_scatter': 0.8,
            'good_color': '#2E8B57',  # Verde para buenos ajustes
            'bad_color': '#DC143C',   # Rojo para malos ajustes
        }
        
        print(f"✅ Cargados {len(self.data)} objetos inicialmente")
        print(f"📋 Experimento: {self.get_experiment_display_name()}")
        print(f"📊 Criterio de calidad: {'χ² reducido' if use_reduced_chi2 else 'χ²'} < {chi2_threshold}")
    
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
        """Prepara los datos para plotting con filtrado por calidad."""
        print("\n📊 PREPARANDO DATOS CON FILTRADO POR CALIDAD...")
        
        # Usar columnas bayesianas
        age_col = 'bayes.stellar.age_m_star'
        metal_col = 'bayes.stellar.metallicity'
        
        # Determinar columna de calidad
        if self.use_reduced_chi2:
            quality_col = 'best.reduced_chi_square'
            print(f"   • Usando χ² reducido ({quality_col})")
        else:
            quality_col = 'best.chi_square'
            print(f"   • Usando χ² ({quality_col})")
        
        # Verificar que existen las columnas necesarias
        missing_cols = []
        for col in [age_col, metal_col, quality_col]:
            if col not in self.data.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"❌ ERROR: Columnas faltantes: {missing_cols}")
            print(f"   Columnas disponibles: {list(self.data.columns[:20])}...")
            if 'best.reduced_chi_square' in self.data.columns:
                print(f"   Se encontró 'best.reduced_chi_square'")
            if 'best.chi_square' in self.data.columns:
                print(f"   Se encontró 'best.chi_square'")
            sys.exit(1)
        
        # Convertir a numérico
        age = pd.to_numeric(self.data[age_col], errors='coerce')
        metallicity = pd.to_numeric(self.data[metal_col], errors='coerce')
        quality = pd.to_numeric(self.data[quality_col], errors='coerce')
        
        # Aplicar filtro de calidad
        quality_mask = quality < self.chi2_threshold
        nan_mask = age.notna() & metallicity.notna() & quality.notna()
        
        # Máscara combinada
        mask = nan_mask & quality_mask
        
        # Datos antes y después del filtrado
        total_objects = len(age)
        good_objects = mask.sum()
        bad_objects = total_objects - good_objects
        
        print(f"   • Objetos totales: {total_objects}")
        print(f"   • Objetos con NaN: {nan_mask.sum() - mask.sum()}")
        print(f"   • Objetos con {quality_col} >= {self.chi2_threshold}: {bad_objects}")
        print(f"   • Objetos BUENOS ({quality_col} < {self.chi2_threshold}): {good_objects} ({good_objects/total_objects*100:.1f}%)")
        
        if good_objects == 0:
            print("❌ ERROR: No hay objetos que cumplan el criterio de calidad")
            sys.exit(1)
        
        # Aplicar máscara
        age_good = age[mask]
        metallicity_good = metallicity[mask]
        quality_good = quality[mask]
        
        # Datos rechazados (para análisis)
        age_bad = age[~mask]
        metallicity_bad = metallicity[~mask]
        quality_bad = quality[~mask]
        
        # Convertir edad de Myr a Gyr si es necesario
        if age_good.mean() > 1000:
            age_good = age_good / 1000.0
            if len(age_bad) > 0:
                age_bad = age_bad / 1000.0
        
        # Convertir Z a [Fe/H]
        z_sun = self.config['sun_metallicity']
        
        def convert_to_feh(z_values):
            if len(z_values) > 0 and z_values.min() >= 0 and z_values.max() <= 0.1:
                valid_mask = z_values > 0
                feh = np.full_like(z_values, np.nan)
                feh[valid_mask] = np.log10(z_values[valid_mask] / z_sun)
                return pd.Series(feh, index=z_values.index)
            return z_values
        
        metallicity_good = convert_to_feh(metallicity_good)
        metallicity_bad = convert_to_feh(metallicity_bad)
        
        # Estadísticas de calidad
        quality_stats = {
            'good_mean': quality_good.mean(),
            'good_std': quality_good.std(),
            'bad_mean': quality_bad.mean() if len(quality_bad) > 0 else np.nan,
            'bad_std': quality_bad.std() if len(quality_bad) > 0 else np.nan,
        }
        
        print(f"   • {quality_col} (buenos): {quality_stats['good_mean']:.2f} ± {quality_stats['good_std']:.2f}")
        if not np.isnan(quality_stats['bad_mean']):
            print(f"   • {quality_col} (rechazados): {quality_stats['bad_mean']:.2f} ± {quality_stats['bad_std']:.2f}")
        
        return {
            'age_good': age_good.values,
            'metallicity_good': metallicity_good.values,
            'quality_good': quality_good.values,
            'age_bad': age_bad.values,
            'metallicity_bad': metallicity_bad.values,
            'quality_bad': quality_bad.values,
            'quality_col': quality_col,
            'total_objects': total_objects,
            'good_objects': good_objects,
        }
    
    def create_final_plot(self, show_rejected=False):
        """Crea el gráfico final con filtrado por calidad."""
        
        data_dict = self.prepare_data()
        
        age = data_dict['age_good']
        metallicity = data_dict['metallicity_good']
        age_bad = data_dict['age_bad']
        metallicity_bad = data_dict['metallicity_bad']
        
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
        
        # Scatter plot de objetos BUENOS con densidad
        from scipy.stats import gaussian_kde
        if len(age) > 1:
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
                                    zorder=5,
                                    label='Good fit')
        else:
            scatter = ax_main.scatter(age, metallicity, 
                                    c='blue',
                                    s=self.config['size_scatter'],
                                    alpha=self.config['alpha_scatter'],
                                    edgecolors='white',
                                    linewidth=0.5,
                                    zorder=5,
                                    label='Good fit')
        
        # Opcional: mostrar objetos rechazados
        if show_rejected and len(age_bad) > 0:
            ax_main.scatter(age_bad, metallicity_bad,
                          c=self.config['bad_color'],
                          s=self.config['size_scatter'] * 0.5,
                          alpha=0.3,
                          edgecolors='none',
                          marker='x',
                          zorder=1,
                          label=f'Rejected ({data_dict["quality_col"]} ≥ {self.chi2_threshold})')
        
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
        if len(age) > 1:  # Solo si hay más de 1 punto para la densidad
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
        # HISTOGRAMA DE EDAD (solo buenos)
        # ====================================================================
        ax_hist_age = fig.add_subplot(gs[0, 0], sharex=ax_main)
        plt.setp(ax_hist_age.get_xticklabels(), visible=False)
        
        if len(age) > 0:
            ax_hist_age.hist(age, bins=20, 
                            color='skyblue', 
                            alpha=0.7,
                            edgecolor='navy',
                            linewidth=0.8,
                            density=True)
        
        ax_hist_age.set_ylabel('Density', fontsize=14)
        ax_hist_age.grid(True, alpha=0.2, linestyle='--')
        
        # ====================================================================
        # HISTOGRAMA DE METALICIDAD (solo buenos)
        # ====================================================================
        ax_hist_metal = fig.add_subplot(gs[1, 2], sharey=ax_main)
        plt.setp(ax_hist_metal.get_yticklabels(), visible=False)
        
        if len(metallicity) > 0:
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
        # TEXTO SIMPLIFICADO EN UNA SOLA CAJA (CON INFORMACIÓN DE CALIDAD)
        # ====================================================================
        experiment_name = self.get_experiment_display_name()
        quality_col_display = 'χ²ᵣ' if self.use_reduced_chi2 else 'χ²'
        
        # Texto con información de calidad
        text_lines = [
            f'{experiment_name}',
            f'',
            f'$N = {data_dict["good_objects"]:,}$ (de {data_dict["total_objects"]:,})',
            f'${quality_col_display} < {self.chi2_threshold}$',
            f'$\\langle Age \\rangle = {age.mean():.2f} \\pm {age.std():.2f}$ Gyr',
            f'$\\langle [Fe/H] \\rangle = {metallicity.mean():.3f} \\pm {metallicity.std():.3f}$'
        ]
        
        combined_text = '\n'.join(text_lines)
        
        # Posición en coordenadas de ejes
        ax_main.text(0.98, 0.05, combined_text,
                    transform=ax_main.transAxes,
                    fontsize=11,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             alpha=0.95,
                             edgecolor=self.config['good_color'],
                             linewidth=1.5),
                    zorder=10)
        
        # Texto "Solar" al lado de la línea
        if len(age) > 0:
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
        
        # Leyenda (si se muestran los rechazados)
        if show_rejected and len(age_bad) > 0:
            ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
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
        if len(age) > 0:
            age_margin = 0.05 * (age.max() - age.min())
            metal_margin = 0.05 * (metallicity.max() - metallicity.min())
            
            ax_main.set_xlim(age.min() - age_margin, age.max() + age_margin)
            ax_main.set_ylim(metallicity.min() - metal_margin, 
                            metallicity.max() + metal_margin)
        
        # ====================================================================
        # GUARDAR
        # ====================================================================
        experiment_part = self.get_experiment_filename_part()
        quality_str = f"chi2_{self.chi2_threshold}".replace('.', 'p')
        filename = f"age_metallicity_{experiment_part}_{quality_str}.pdf"
        output_path = self.output_dir / filename
        
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"\n✅ GRÁFICO GUARDADO EN: {output_path}")
        
        # Versión PNG para revisión
        png_path = self.output_dir / f"age_metallicity_{experiment_part}_{quality_str}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"✅ Versión PNG: {png_path}")
        
        # Guardar también los datos filtrados
        self.save_filtered_data(data_dict, output_path.parent)
        
        #plt.show()
        
        return fig, ax_main, data_dict
    
    def save_filtered_data(self, data_dict, output_dir):
        """Guarda los datos filtrados en un archivo CSV."""
        
        # Crear DataFrame con objetos buenos
        good_mask = np.zeros(len(self.data), dtype=bool)
        
        # Encontrar los índices de los objetos buenos
        age_col = 'bayes.stellar.age_m_star'
        quality_col = 'best.reduced_chi_square' if self.use_reduced_chi2 else 'best.chi_square'
        
        # Calcular máscara
        age = pd.to_numeric(self.data[age_col], errors='coerce')
        metallicity = pd.to_numeric(self.data['bayes.stellar.metallicity'], errors='coerce')
        quality = pd.to_numeric(self.data[quality_col], errors='coerce')
        
        nan_mask = age.notna() & metallicity.notna() & quality.notna()
        quality_mask = quality < self.chi2_threshold
        good_mask = nan_mask & quality_mask
        
        # DataFrame con objetos buenos
        good_data = self.data[good_mask].copy()
        
        # Añadir columnas calculadas
        if 'bayes.stellar.age_m_star' in good_data.columns:
            age_gyr = pd.to_numeric(good_data['bayes.stellar.age_m_star'], errors='coerce')
            if age_gyr.mean() > 1000:
                good_data['age_Gyr'] = age_gyr / 1000.0
        
        # Convertir Z a [Fe/H]
        z_sun = self.config['sun_metallicity']
        if 'bayes.stellar.metallicity' in good_data.columns:
            z = pd.to_numeric(good_data['bayes.stellar.metallicity'], errors='coerce')
            feh = np.log10(z / z_sun)
            good_data['Fe_H'] = feh
        
        # Guardar
        output_csv = output_dir / f"filtered_good_objects_chi2_{self.chi2_threshold}.csv"
        good_data.to_csv(output_csv, index=False)
        print(f"✅ Datos filtrados guardados en: {output_csv}")
        print(f"   • {len(good_data)} objetos con ajuste de buena calidad")
        
        # Estadísticas adicionales
        if 'best.reduced_chi_square' in good_data.columns:
            chi2_stats = good_data['best.reduced_chi_square'].describe()
            print(f"   • Estadísticas de χ² reducido (buenos objetos):")
            print(f"     - Media: {chi2_stats['mean']:.3f}")
            print(f"     - Mediana: {chi2_stats['50%']:.3f}")
            print(f"     - Min: {chi2_stats['min']:.3f}")
            print(f"     - Max: {chi2_stats['max']:.3f}")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal con argumentos para filtrado por calidad."""
    
    print("\n" + "="*70)
    print("📊 AGE-METALLICITY PLOTTER - CON FILTRADO POR CALIDAD (χ²)")
    print("="*70)
    
    parser = argparse.ArgumentParser(
        description='Create age-metallicity plots with quality filtering'
    )
    
    parser.add_argument('input_file', help='FITS file from CIGALE')
    parser.add_argument('--experiment', default='narrow',
                       choices=['narrow', 'narrow+broad', 'splus_only', 'splus_decam'],
                       help='Type of experiment/filter combination')
    parser.add_argument('--output-dir', default='paper_plots_final_quality', 
                       help='Output directory')
    parser.add_argument('--chi2-threshold', type=float, default=2.0,
                       help='Threshold for χ² or reduced χ² (default: 2.0)')
    parser.add_argument('--use-chi2', action='store_true',
                       help='Use χ² instead of reduced χ² (default: use reduced χ²)')
    parser.add_argument('--show-rejected', action='store_true',
                       help='Show rejected objects in the plot (transparent)')
    parser.add_argument('--no-filter', action='store_true',
                       help='Disable quality filtering (use all objects)')
    
    args = parser.parse_args()
    
    try:
        # Configurar parámetros de filtrado
        chi2_threshold = float('inf') if args.no_filter else args.chi2_threshold
        use_reduced_chi2 = not args.use_chi2
        
        print(f"\n📈 PARÁMETROS DE FILTRADO:")
        print(f"   • Filtrado por calidad: {'NO' if args.no_filter else 'SÍ'}")
        if not args.no_filter:
            print(f"   • Métrica: {'χ² reducido' if use_reduced_chi2 else 'χ²'}")
            print(f"   • Umbral: {chi2_threshold}")
            print(f"   • Mostrar rechazados: {'SÍ' if args.show_rejected else 'NO'}")
        
        plotter = AgeMetallicityPlotter(
            args.input_file, 
            args.experiment, 
            args.output_dir,
            chi2_threshold=chi2_threshold,
            use_reduced_chi2=use_reduced_chi2
        )
        
        print("\n🎨 CREANDO GRÁFICO CON FILTRADO POR CALIDAD...")
        fig, ax_main, data_dict = plotter.create_final_plot(
            show_rejected=args.show_rejected
        )
        
        print(f"\n✅ PROCESO COMPLETADO!")
        print(f"📁 Resultados en: {args.output_dir}")
        
        # Mostrar nombres de archivo generados
        experiment_part = plotter.get_experiment_filename_part()
        quality_str = "no_filter" if args.no_filter else f"chi2_{args.chi2_threshold}".replace('.', 'p')
        
        print(f"\n📄 Archivos generados:")
        print(f"   • age_metallicity_{experiment_part}_{quality_str}.pdf")
        print(f"   • age_metallicity_{experiment_part}_{quality_str}.png")
        print(f"   • filtered_good_objects_chi2_{args.chi2_threshold}.csv")
        
        print(f"\n📊 RESUMEN DE FILTRADO:")
        print(f"   • Objetos totales: {data_dict['total_objects']}")
        print(f"   • Objetos con buen ajuste: {data_dict['good_objects']}")
        print(f"   • Porcentaje retenido: {data_dict['good_objects']/data_dict['total_objects']*100:.1f}%")
        
        if data_dict['good_objects'] < data_dict['total_objects']:
            print(f"\n⚠️  RECOMENDACIÓN: Se eliminaron {data_dict['total_objects'] - data_dict['good_objects']} objetos.")
            print(f"   Los resultados presentados son más confiables al eliminar ajustes de baja calidad.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
