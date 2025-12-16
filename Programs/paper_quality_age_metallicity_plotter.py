#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_quality_age_metallicity_plotter.py
=========================================

Script profesional para crear gráficos de metalicidad vs edad de calidad
para publicaciones científicas.

Características:
- Estilo científico elegante (Nature/Science style)
- Múltiples opciones de visualización
- Personalización completa
- Exportación en alta resolución
- Compatible con resultados de CIGALE
- Colores y marcadores configurables
- Leyendas automáticas inteligentes
ff
Autor: Luis A. Gutiérrez Soto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings
import argparse
import os
import sys
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE ESTILO PROFESIONAL
# ============================================================================

def setup_publication_style(style='nature', font_size=9):
    """
    Configura el estilo de matplotlib para publicaciones científicas.
    
    Parameters
    ----------
    style : str
        Estilo preferido: 'nature', 'science', 'aas', 'classic'
    font_size : int
        Tamaño de fuente base
    """
    
    # Diccionario de estilos
    styles = {
        'nature': {
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': font_size,
            'axes.titlesize': font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1,
            'figure.titlesize': font_size + 1,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'patch.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'grid.linewidth': 0.6,
            'axes.grid': False,
            'mathtext.fontset': 'stix',
        },
        'science': {
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': font_size,
            'axes.titlesize': font_size + 1,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1,
            'figure.titlesize': font_size + 2,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'patch.linewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'grid.linewidth': 0.8,
            'axes.grid': False,
        }
    }
    
    # Aplicar estilo seleccionado
    if style in styles:
        plt.rcParams.update(styles[style])
    else:
        plt.rcParams.update(styles['nature'])  # Default
    
    # Colores personalizados para publicaciones
    colors = {
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'yellow': '#bcbd22',
        'cyan': '#17becf'
    }
    
    return colors

# ============================================================================
# CLASE PRINCIPAL PARA GRÁFICOS
# ============================================================================

class AgeMetallicityPlotter:
    """
    Clase para crear gráficos de calidad profesional de metalicidad vs edad.
    """
    
    def __init__(self, results_file, output_dir='plots', style='nature'):
        """
        Inicializa el plotter.
        
        Parameters
        ----------
        results_file : str
            Ruta al archivo de resultados (FITS o CSV)
        output_dir : str
            Directorio de salida para los gráficos
        style : str
            Estilo de visualización
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar estilo
        self.colors = setup_publication_style(style)
        self.style = style
        
        # Cargar datos
        self.data = self.load_data()
        
        # Configuración por defecto
        self.config = {
            'age_unit': 'Gyr',  # o 'Myr'
            'metallicity_unit': '[Fe/H]',  # o 'Z'
            'sun_metallicity': 0.02,  # Z_sol
            'solar_color': 'gold',
            'cmap_scatter': 'viridis',
            'cmap_density': 'plasma',
            'size_scatter': 20,
            'alpha_scatter': 0.7,
            'grid_alpha': 0.2,
            'hist_alpha': 0.3,
        }
        
        print(f"✅ Cargados {len(self.data)} objetos")
    
    def load_data(self):
        """
        Carga los datos desde diferentes formatos.
        """
        if not self.results_file.exists():
            raise FileNotFoundError(f"No se encuentra el archivo: {self.results_file}")
        
        # Determinar formato por extensión
        suffix = self.results_file.suffix.lower()
        
        try:
            if suffix == '.fits':
                return self.load_fits()
            elif suffix in ['.csv', '.txt', '.dat']:
                return self.load_text()
            else:
                raise ValueError(f"Formato no soportado: {suffix}")
        except Exception as e:
            print(f"Error cargando datos: {e}")
            sys.exit(1)
    
    def load_fits(self):
        """
        Carga datos desde archivo FITS (resultados de CIGALE).
        """
        with fits.open(self.results_file) as hdul:
            # Intentar diferentes extensiones
            for ext in [1, 2, 0]:
                try:
                    data = Table(hdul[ext].data).to_pandas()
                    # Buscar columnas relevantes
                    age_cols = [c for c in data.columns if 'age' in c.lower()]
                    metal_cols = [c for c in data.columns if 'metal' in c.lower() or 'z' == c.lower()]
                    
                    if age_cols and metal_cols:
                        print(f"✅ Columnas encontradas: {age_cols[0]}, {metal_cols[0]}")
                        return data
                except:
                    continue
        
        # Si no se encuentran, usar primera extensión y buscar manualmente
        data = Table(hdul[1].data).to_pandas()
        return data
    
    def load_text(self):
        """
        Carga datos desde archivo de texto.
        """
        # Intentar diferentes delimitadores
        for delimiter in [',', '\t', ' ', ';']:
            try:
                data = pd.read_csv(self.results_file, delimiter=delimiter)
                if len(data.columns) > 1:
                    print(f"✅ Archivo de texto cargado con delimitador: '{delimiter}'")
                    return data
            except:
                continue
        
        raise ValueError("No se pudo leer el archivo de texto")
    
    def prepare_data(self, age_column=None, metallicity_column=None):
        """
        Prepara los datos para graficar.
        """
        data = self.data.copy()
        
        # Encontrar columnas automáticamente si no se especifican
        if age_column is None:
            age_candidates = [c for c in data.columns if 'age' in c.lower()]
            age_column = age_candidates[0] if age_candidates else data.columns[0]
        
        if metallicity_column is None:
            metal_candidates = [c for c in data.columns 
                              if 'metal' in c.lower() 
                              or 'z' in c.lower()
                              or '[fe/h]' in c.lower()
                              or 'feh' in c.lower()]
            metallicity_column = metal_candidates[0] if metal_candidates else data.columns[1]
        
        print(f"📊 Usando columnas:")
        print(f"   Edad: {age_column}")
        print(f"   Metalicidad: {metallicity_column}")
        
        # Extraer datos
        age = pd.to_numeric(data[age_column], errors='coerce')
        metallicity = pd.to_numeric(data[metallicity_column], errors='coerce')
        
        # Filtrar valores no válidos
        mask = age.notna() & metallicity.notna()
        age = age[mask]
        metallicity = metallicity[mask]
        
        # Convertir edad a Gyr si está en Myr
        if age.mean() > 1000:  # Probablemente en Myr
            age = age / 1000
            print("⚠️  Convertida edad de Myr a Gyr")
        
        # Convertir Z a [Fe/H] si es necesario
        if metallicity.min() >= 0 and metallicity.max() <= 0.05:  # Probablemente Z
            metallicity = np.log10(metallicity / self.config['sun_metallicity'])
            print("⚠️  Convertida metalicidad de Z a [Fe/H]")
        
        return age.values, metallicity.values, age_column, metallicity_column
    
    # ========================================================================
    # MÉTODOS DE VISUALIZACIÓN PRINCIPALES
    # ========================================================================
    
    def plot_scatter(self, age_column=None, metallicity_column=None, 
                    color_by_density=True, show_solar=True,
                    show_histograms=True, show_regression=True,
                    figsize=(6, 5), filename=None):
        """
        Crea un gráfico de dispersión profesional.
        
        Parameters
        ----------
        color_by_density : bool
            Colorear puntos por densidad
        show_solar : bool
            Mostrar línea de metalicidad solar
        show_histograms : bool
            Mostrar histogramas marginales
        show_regression : bool
            Mostrar línea de regresión
        """
        # Preparar datos
        age, metallicity, age_col, metal_col = self.prepare_data(
            age_column, metallicity_column)
        
        # Crear figura
        if show_histograms:
            fig = plt.figure(figsize=figsize)
            # Definir grid para scatter + histogramas
            gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                 left=0.15, right=0.95, bottom=0.15, top=0.95,
                                 wspace=0.05, hspace=0.05)
            ax_scatter = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
            
            # Ocultar ticks de histogramas
            plt.setp(ax_histx.get_xticklabels(), visible=False)
            plt.setp(ax_histy.get_yticklabels(), visible=False)
            
            # Histograma de edad
            ax_histx.hist(age, bins=30, alpha=self.config['hist_alpha'], 
                         color=self.colors['blue'], density=True)
            ax_histx.set_ylabel('Density')
            
            # Histograma de metalicidad
            ax_histy.hist(metallicity, bins=30, orientation='horizontal',
                         alpha=self.config['hist_alpha'], color=self.colors['red'],
                         density=True)
            ax_histy.set_xlabel('Density')
        else:
            fig, ax_scatter = plt.subplots(figsize=figsize)
        
        # Colorear por densidad si se solicita
        if color_by_density and len(age) > 50:
            from scipy.stats import gaussian_kde
            try:
                # Calcular densidad 2D
                xy = np.vstack([age, metallicity])
                z = gaussian_kde(xy)(xy)
                
                # Ordenar por densidad para mejor visualización
                idx = z.argsort()
                age, metallicity, z = age[idx], metallicity[idx], z[idx]
                
                scatter = ax_scatter.scatter(age, metallicity, c=z, 
                                           s=self.config['size_scatter'],
                                           alpha=self.config['alpha_scatter'],
                                           cmap=self.config['cmap_scatter'],
                                           edgecolors='white', linewidth=0.3)
                
                # Añadir barra de color
                cbar = plt.colorbar(scatter, ax=ax_scatter, pad=0.02)
                cbar.set_label('Point Density', rotation=270, labelpad=15)
                
            except:
                # Fallback a scatter simple
                ax_scatter.scatter(age, metallicity, s=self.config['size_scatter'],
                                 alpha=self.config['alpha_scatter'],
                                 color=self.colors['blue'], edgecolors='black',
                                 linewidth=0.3)
        else:
            ax_scatter.scatter(age, metallicity, s=self.config['size_scatter'],
                             alpha=self.config['alpha_scatter'],
                             color=self.colors['blue'], edgecolors='black',
                             linewidth=0.3)
        
        # Línea de metalicidad solar
        if show_solar:
            ax_scatter.axhline(y=0, color=self.config['solar_color'], 
                             linestyle='--', linewidth=1.5, alpha=0.8,
                             label='Solar metallicity')
        
        # Regresión lineal
        if show_regression and len(age) > 10:
            try:
                # Calcular regresión
                slope, intercept, r_value, p_value, std_err = stats.linregress(age, metallicity)
                
                # Crear línea de regresión
                x_fit = np.linspace(age.min(), age.max(), 100)
                y_fit = intercept + slope * x_fit
                
                # Graficar
                ax_scatter.plot(x_fit, y_fit, color=self.colors['red'], 
                              linewidth=2, linestyle='-', alpha=0.8,
                              label=f'Fit: [Fe/H] = {slope:.3f} × Age + {intercept:.3f}\n$r$ = {r_value:.3f}')
                
                # Añadir intervalo de confianza
                if len(age) > 30:
                    # Calcular intervalo de confianza
                    y_err = std_err * np.sqrt(1/len(age) + (x_fit - np.mean(age))**2 / np.sum((age - np.mean(age))**2))
                    ax_scatter.fill_between(x_fit, y_fit - 1.96*y_err, y_fit + 1.96*y_err,
                                          alpha=0.2, color=self.colors['red'])
            except:
                print("⚠️  No se pudo calcular la regresión")
        
        # Configurar ejes
        ax_scatter.set_xlabel(f'Age ({self.config["age_unit"]})', fontweight='bold')
        ax_scatter.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})', fontweight='bold')
        
        # Grid
        ax_scatter.grid(True, alpha=self.config['grid_alpha'], linestyle='--')
        
        # Leyenda
        if show_solar or show_regression:
            ax_scatter.legend(loc='best', framealpha=0.8, fancybox=True)
        
        # Estadísticas en texto
        stats_text = f'N = {len(age):,}\n'
        stats_text += f'⟨Age⟩ = {age.mean():.2f} ± {age.std():.2f} Gyr\n'
        stats_text += f'⟨[Fe/H]⟩ = {metallicity.mean():.2f} ± {metallicity.std():.2f}'
        
        ax_scatter.text(0.02, 0.98, stats_text, transform=ax_scatter.transAxes,
                       fontsize=self.config.get('font_size', 9)-2,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Título
        if hasattr(self, 'title'):
            ax_scatter.set_title(self.title, fontweight='bold', pad=15)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar
        if filename is None:
            filename = f"age_metallicity_scatter_{self.style}.pdf"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {output_path}")
        
        plt.show()
        
        return fig, ax_scatter
    
    def plot_density_contour(self, age_column=None, metallicity_column=None,
                            levels=10, show_points=True, show_colorbar=True,
                            figsize=(7, 6), filename=None):
        """
        Crea un gráfico de contorno de densidad.
        """
        # Preparar datos
        age, metallicity, age_col, metal_col = self.prepare_data(
            age_column, metallicity_column)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calcular densidad 2D
        try:
            from scipy.stats import gaussian_kde
            
            # Crear grid para densidad
            xmin, xmax = age.min(), age.max()
            ymin, ymax = metallicity.min(), metallicity.max()
            
            # Añadir márgenes
            xmargin = (xmax - xmin) * 0.1
            ymargin = (ymax - ymin) * 0.1
            
            xx, yy = np.mgrid[xmin-xmargin:xmax+xmargin:100j, 
                             ymin-ymargin:ymax+ymargin:100j]
            
            # Calcular KDE
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([age, metallicity])
            kernel = gaussian_kde(values)
            z = np.reshape(kernel(positions).T, xx.shape)
            
            # Contornos de densidad
            contour = ax.contourf(xx, yy, z, levels=levels, cmap=self.config['cmap_density'], 
                                alpha=0.8)
            
            # Contornos de línea
            line_contours = ax.contour(xx, yy, z, levels=levels, colors='black', 
                                      linewidths=0.5, alpha=0.5)
            
            # Barra de color
            if show_colorbar:
                cbar = plt.colorbar(contour, ax=ax, pad=0.02)
                cbar.set_label('Density', rotation=270, labelpad=15)
            
        except Exception as e:
            print(f"⚠️  Error en KDE: {e}")
            # Fallback a hexbin
            hb = ax.hexbin(age, metallicity, gridsize=30, cmap=self.config['cmap_density'],
                          mincnt=1, alpha=0.8)
            if show_colorbar:
                plt.colorbar(hb, ax=ax, label='Count')
        
        # Puntos (opcional)
        if show_points:
            ax.scatter(age, metallicity, s=10, color='black', alpha=0.3, edgecolors='none')
        
        # Configurar ejes
        ax.set_xlabel(f'Age ({self.config["age_unit"]})', fontweight='bold')
        ax.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})', fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=self.config['grid_alpha'], linestyle=':')
        
        # Línea solar
        ax.axhline(y=0, color='gold', linestyle='--', linewidth=1.5, alpha=0.8,
                  label='Solar metallicity')
        
        # Estadísticas
        if len(age) > 0:
            # Calcular correlación
            corr_coef, p_value = stats.pearsonr(age, metallicity)
            
            stats_text = f'N = {len(age):,}\n'
            stats_text += f'ρ = {corr_coef:.3f}\n'
            if p_value < 0.001:
                stats_text += 'p < 0.001'
            else:
                stats_text += f'p = {p_value:.3f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=self.config.get('font_size', 9)-2,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Leyenda
        ax.legend(loc='best', framealpha=0.8)
        
        # Título
        if hasattr(self, 'title'):
            ax.set_title(self.title, fontweight='bold', pad=15)
        
        # Guardar
        if filename is None:
            filename = f"age_metallicity_density_{self.style}.pdf"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {output_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_hexbin(self, age_column=None, metallicity_column=None,
                   gridsize=30, show_colorbar=True, figsize=(6, 5), filename=None):
        """
        Gráfico hexbin para grandes conjuntos de datos.
        """
        # Preparar datos
        age, metallicity, age_col, metal_col = self.prepare_data(
            age_column, metallicity_column)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Hexbin plot
        hb = ax.hexbin(age, metallicity, gridsize=gridsize, cmap='viridis',
                      mincnt=1, edgecolors='none', alpha=0.9)
        
        # Barra de color
        if show_colorbar:
            cbar = plt.colorbar(hb, ax=ax, pad=0.02)
            cbar.set_label('Number of objects', rotation=270, labelpad=15)
        
        # Configurar ejes
        ax.set_xlabel(f'Age ({self.config["age_unit"]})', fontweight='bold')
        ax.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})', fontweight='bold')
        
        # Línea solar
        ax.axhline(y=0, color='gold', linestyle='--', linewidth=1.5, alpha=0.8,
                  label='Solar metallicity')
        
        # Grid
        ax.grid(True, alpha=self.config['grid_alpha'], linestyle='--')
        
        # Estadísticas
        stats_text = f'N = {len(age):,}\n'
        stats_text += f'⟨Age⟩ = {age.mean():.2f} Gyr\n'
        stats_text += f'⟨[Fe/H]⟩ = {metallicity.mean():.2f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=self.config.get('font_size', 9)-2,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Leyenda
        ax.legend(loc='best', framealpha=0.8)
        
        # Guardar
        if filename is None:
            filename = f"age_metallicity_hexbin_{self.style}.pdf"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_comparison(self, age_column=None, metallicity_column=None,
                       categories=None, figsize=(8, 6), filename=None):
        """
        Gráfico comparativo por categorías (si existen).
        """
        # Preparar datos
        age, metallicity, age_col, metal_col = self.prepare_data(
            age_column, metallicity_column)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Si hay categorías
        if categories is not None and len(categories) == len(age):
            unique_cats = np.unique(categories)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
            
            for cat, color in zip(unique_cats, colors):
                mask = categories == cat
                ax.scatter(age[mask], metallicity[mask], 
                         s=self.config['size_scatter'], 
                         alpha=self.config['alpha_scatter'],
                         color=color, label=str(cat),
                         edgecolors='black', linewidth=0.3)
        else:
            # Scatter simple
            ax.scatter(age, metallicity, s=self.config['size_scatter'],
                     alpha=self.config['alpha_scatter'],
                     color=self.colors['blue'], edgecolors='black',
                     linewidth=0.3)
        
        # Configuración
        ax.set_xlabel(f'Age ({self.config["age_unit"]})', fontweight='bold')
        ax.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})', fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=self.config['grid_alpha'], linestyle='--')
        
        # Línea solar
        ax.axhline(y=0, color='gold', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Leyenda si hay categorías
        if categories is not None:
            ax.legend(title='Categories', framealpha=0.8, fancybox=True)
        
        # Guardar
        if filename is None:
            filename = f"age_metallicity_comparison_{self.style}.pdf"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def create_multi_panel(self, age_column=None, metallicity_column=None,
                          figsize=(12, 10), filename=None):
        """
        Crea una figura multi-panel con diferentes visualizaciones.
        """
        # Preparar datos
        age, metallicity, age_col, metal_col = self.prepare_data(
            age_column, metallicity_column)
        
        fig = plt.figure(figsize=figsize)
        
        # Grid de subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Scatter plot
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(age, metallicity, c='blue', s=20, alpha=0.6)
        ax1.set_xlabel(f'Age ({self.config["age_unit"]})')
        ax1.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})')
        ax1.set_title('(a) Scatter Plot', loc='left', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gold', linestyle='--', alpha=0.7)
        
        # 2. Hexbin
        ax2 = fig.add_subplot(gs[0, 1])
        hb = ax2.hexbin(age, metallicity, gridsize=20, cmap='viridis', mincnt=1)
        ax2.set_xlabel(f'Age ({self.config["age_unit"]})')
        ax2.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})')
        ax2.set_title('(b) Density (Hexbin)', loc='left', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gold', linestyle='--', alpha=0.7)
        plt.colorbar(hb, ax=ax2, label='Count')
        
        # 3. Contour
        ax3 = fig.add_subplot(gs[1, 0])
        try:
            from scipy.stats import gaussian_kde
            xmin, xmax = age.min(), age.max()
            ymin, ymax = metallicity.min(), metallicity.max()
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([age, metallicity])
            kernel = gaussian_kde(values)
            z = np.reshape(kernel(positions).T, xx.shape)
            contour = ax3.contourf(xx, yy, z, cmap='plasma', alpha=0.8)
            ax3.set_xlabel(f'Age ({self.config["age_unit"]})')
            ax3.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})')
            ax3.set_title('(c) Density (Contour)', loc='left', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='gold', linestyle='--', alpha=0.7)
            plt.colorbar(contour, ax=ax3, label='Density')
        except:
            ax3.text(0.5, 0.5, 'KDE not available', ha='center', va='center')
        
        # 4. Histogram 2D
        ax4 = fig.add_subplot(gs[1, 1])
        hist = ax4.hist2d(age, metallicity, bins=30, cmap='hot', norm=LogNorm())
        ax4.set_xlabel(f'Age ({self.config["age_unit"]})')
        ax4.set_ylabel(f'Metallicity ({self.config["metallicity_unit"]})')
        ax4.set_title('(d) 2D Histogram', loc='left', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='gold', linestyle='--', alpha=0.7)
        plt.colorbar(hist[3], ax=ax4, label='Count')
        
        # Título general
        fig.suptitle('Age-Metallicity Relation: Multiple Visualizations', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Guardar
        if filename is None:
            filename = f"age_metallicity_multi_panel_{self.style}.pdf"
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✅ Figura multi-panel guardada en: {output_path}")
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def statistical_analysis(self, age_column=None, metallicity_column=None):
        """
        Realiza análisis estadístico de la relación edad-metalicidad.
        """
        # Preparar datos
        age, metallicity, age_col, metal_col = self.prepare_data(
            age_column, metallicity_column)
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS DE LA RELACIÓN EDAD-METALICIDAD")
        print("="*60)
        
        print(f"\n📊 Estadísticas descriptivas:")
        print(f"   Número de objetos: {len(age):,}")
        print(f"   Edad (Gyr):")
        print(f"     Mínimo: {age.min():.2f}")
        print(f"     Máximo: {age.max():.2f}")
        print(f"     Media: {age.mean():.2f} ± {age.std():.2f}")
        print(f"     Mediana: {np.median(age):.2f}")
        print(f"   Metalicidad ([Fe/H]):")
        print(f"     Mínimo: {metallicity.min():.2f}")
        print(f"     Máximo: {metallicity.max():.2f}")
        print(f"     Media: {metallicity.mean():.2f} ± {metallicity.std():.2f}")
        print(f"     Mediana: {np.median(metallicity):.2f}")
        
        print(f"\n📈 Análisis de correlación:")
        
        # Correlación de Pearson
        corr_pearson, p_pearson = stats.pearsonr(age, metallicity)
        print(f"   Pearson r = {corr_pearson:.4f}")
        print(f"   p-value = {p_pearson:.4e}")
        
        # Correlación de Spearman
        corr_spearman, p_spearman = stats.spearmanr(age, metallicity)
        print(f"   Spearman ρ = {corr_spearman:.4f}")
        print(f"   p-value = {p_spearman:.4e}")
        
        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = stats.linregress(age, metallicity)
        print(f"\n📐 Regresión lineal:")
        print(f"   Pendiente: {slope:.4f} ± {std_err:.4f}")
        print(f"   Intercepto: {intercept:.4f}")
        print(f"   R² = {r_value**2:.4f}")
        print(f"   Ecuación: [Fe/H] = {slope:.3f} × Age + {intercept:.3f}")
        
        # Guardar resultados en archivo
        results_file = self.output_dir / "statistical_analysis.txt"
        with open(results_file, 'w') as f:
            f.write("Statistical Analysis of Age-Metallicity Relation\n")
            f.write("="*50 + "\n\n")
            f.write(f"Sample size: {len(age)}\n")
            f.write(f"\nAge statistics (Gyr):\n")
            f.write(f"  Min: {age.min():.2f}\n")
            f.write(f"  Max: {age.max():.2f}\n")
            f.write(f"  Mean: {age.mean():.2f} ± {age.std():.2f}\n")
            f.write(f"  Median: {np.median(age):.2f}\n")
            f.write(f"\nMetallicity statistics ([Fe/H]):\n")
            f.write(f"  Min: {metallicity.min():.2f}\n")
            f.write(f"  Max: {metallicity.max():.2f}\n")
            f.write(f"  Mean: {metallicity.mean():.2f} ± {metallicity.std():.2f}\n")
            f.write(f"  Median: {np.median(metallicity):.2f}\n")
            f.write(f"\nCorrelation analysis:\n")
            f.write(f"  Pearson r = {corr_pearson:.4f} (p = {p_pearson:.4e})\n")
            f.write(f"  Spearman ρ = {corr_spearman:.4f} (p = {p_spearman:.4e})\n")
            f.write(f"\nLinear regression:\n")
            f.write(f"  Slope: {slope:.4f} ± {std_err:.4f}\n")
            f.write(f"  Intercept: {intercept:.4f}\n")
            f.write(f"  R² = {r_value**2:.4f}\n")
            f.write(f"  Equation: [Fe/H] = {slope:.3f} × Age + {intercept:.3f}\n")
        
        print(f"\n📝 Resultados guardados en: {results_file}")
        
        return {
            'age_stats': {
                'min': age.min(), 'max': age.max(),
                'mean': age.mean(), 'std': age.std(),
                'median': np.median(age)
            },
            'metal_stats': {
                'min': metallicity.min(), 'max': metallicity.max(),
                'mean': metallicity.mean(), 'std': metallicity.std(),
                'median': np.median(metallicity)
            },
            'correlations': {
                'pearson': {'r': corr_pearson, 'p': p_pearson},
                'spearman': {'rho': corr_spearman, 'p': p_spearman}
            },
            'regression': {
                'slope': slope, 'intercept': intercept,
                'r_squared': r_value**2, 'std_err': std_err
            }
        }

# ============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create professional age-metallicity plots for publications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.fits --style nature --output age_plot.pdf
  %(prog)s results.csv --plot-type density --colormap plasma
  %(prog)s data.txt --multi-panel --statistics
        """
    )
    
    parser.add_argument('input_file', help='Input file (FITS, CSV, TXT)')
    parser.add_argument('--age-col', help='Age column name (default: auto-detect)')
    parser.add_argument('--metal-col', help='Metallicity column name (default: auto-detect)')
    parser.add_argument('--plot-type', default='scatter',
                       choices=['scatter', 'density', 'hexbin', 'comparison', 'multi'],
                       help='Type of plot to create')
    parser.add_argument('--style', default='nature',
                       choices=['nature', 'science', 'aas', 'classic'],
                       help='Plot style')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--output-dir', default='plots', help='Output directory')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--colormap', default='viridis', help='Colormap for density plots')
    parser.add_argument('--multi-panel', action='store_true', 
                       help='Create multi-panel figure')
    parser.add_argument('--statistics', action='store_true',
                       help='Perform statistical analysis')
    parser.add_argument('--no-solar', action='store_true',
                       help='Do not show solar metallicity line')
    parser.add_argument('--no-regression', action='store_true',
                       help='Do not show regression line')
    parser.add_argument('--font-size', type=int, default=9,
                       help='Base font size')
    
    return parser.parse_args()

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Main function."""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("📊 PAPER-QUALITY AGE-METALLICITY PLOTTER")
    print("="*70)
    
    # Crear plotter
    try:
        plotter = AgeMetallicityPlotter(
            args.input_file,
            output_dir=args.output_dir,
            style=args.style
        )
        
        # Configurar título si se proporciona
        if args.title:
            plotter.title = args.title
        
        # Configurar colormap
        plotter.config['cmap_scatter'] = args.colormap
        plotter.config['cmap_density'] = args.colormap
        
        # Configurar estilo de fuente
        setup_publication_style(args.style, args.font_size)
        
        # Realizar análisis estadístico si se solicita
        if args.statistics:
            stats = plotter.statistical_analysis(args.age_col, args.metal_col)
        
        # Crear gráfico según tipo solicitado
        if args.multi_panel:
            # Figura multi-panel
            plotter.create_multi_panel(
                age_column=args.age_col,
                metallicity_column=args.metal_col,
                filename=args.output
            )
        
        elif args.plot_type == 'scatter':
            # Scatter plot
            plotter.plot_scatter(
                age_column=args.age_col,
                metallicity_column=args.metal_col,
                show_solar=not args.no_solar,
                show_regression=not args.no_regression,
                filename=args.output
            )
        
        elif args.plot_type == 'density':
            # Density contour plot
            plotter.plot_density_contour(
                age_column=args.age_col,
                metallicity_column=args.metal_col,
                filename=args.output
            )
        
        elif args.plot_type == 'hexbin':
            # Hexbin plot
            plotter.plot_hexbin(
                age_column=args.age_col,
                metallicity_column=args.metal_col,
                filename=args.output
            )
        
        elif args.plot_type == 'comparison':
            # Comparison plot (si hay categorías)
            plotter.plot_comparison(
                age_column=args.age_col,
                metallicity_column=args.metal_col,
                filename=args.output
            )
        
        print("\n✅ Proceso completado exitosamente!")
        print(f"📁 Gráficos guardados en: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# ============================================================================
# EJEMPLO DE USO RÁPIDO
# ============================================================================

def quick_example():
    """
    Ejemplo rápido de uso sin argumentos de línea de comandos.
    """
    # Ruta a tus resultados
    results_file = "out/results.fits"  # Cambia esto por tu archivo
    
    # Crear plotter
    plotter = AgeMetallicityPlotter(results_file, style='nature')
    
    # 1. Análisis estadístico
    print("\n📈 Realizando análisis estadístico...")
    stats = plotter.statistical_analysis()
    
    # 2. Gráfico scatter profesional (recomendado para paper)
    print("\n🎨 Creando gráfico scatter profesional...")
    plotter.plot_scatter(
        show_solar=True,
        show_regression=True,
        show_histograms=True,
        color_by_density=True,
        filename="age_metallicity_paper_quality.pdf"
    )
    
    # 3. Gráfico de densidad
    print("\n🎨 Creando gráfico de densidad...")
    plotter.plot_density_contour(
        filename="age_metallicity_density.pdf"
    )
    
    # 4. Figura multi-panel (para suplemento)
    print("\n🎨 Creando figura multi-panel...")
    plotter.create_multi_panel(
        filename="age_metallicity_multi_panel.pdf"
    )

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Para uso interactivo en Jupyter/script
    if len(sys.argv) > 1:
        # Modo línea de comandos
        sys.exit(main())
    else:
        # Modo interactivo - ejecutar ejemplo
        print("Modo interactivo. Ejecutando ejemplo...")
        print("Para uso por línea de comandos: python script.py --help")
        quick_example()
