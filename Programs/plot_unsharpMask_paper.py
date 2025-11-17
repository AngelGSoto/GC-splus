#!/usr/bin/env python3
"""
unsharp_mask_apj_figure_ra_dec.py
Figuras profesionales para ApJ - Versión final para paper
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import seaborn as sns
import logging
import os
from pathlib import Path

# Configuración de estilo profesional para ApJ
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
})

class APJUnsharpMaskPaper:
    def __init__(self):
        self.field_name = "CenA11"
        self.filter_name = "F660"
        self.output_dir = "APJ_Figures_Paper"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parámetros de procesamiento
        self.median_box = 25
        self.gaussian_sigma = 5
        
        # Configuración para resaltar el halo
        self.halo_contrast = 2.5
        
        # Configuración de estilo
        self.setup_paper_style()
        
    def setup_paper_style(self):
        """Configura el estilo para paper científico"""
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.2)
        
        # Colores profesionales
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#7f8c8d', 
            'accent': '#c0392b',
            'background': '#f8f9fa'
        }
        
        # Crear colormaps consistentes
        self.create_consistent_colormaps()
        
    def create_consistent_colormaps(self):
        """Crea colormaps consistentes para todas las imágenes"""
        # Misma escala de grises para original y modelo
        self.gray_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'paper_gray', 
            ['black', 'white']
        )
        
        # Escala divergente para residual
        colors_divergent = ['#202020', '#606060', 'white', '#606060', '#202020']
        self.divergent_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'paper_divergent', 
            colors_divergent
        )
        
    def load_data(self):
        """Carga los datos y el WCS para coordenadas celestes"""
        image_path = f"{self.field_name}/{self.field_name}_{self.filter_name}.fits.fz"
        
        if not os.path.exists(image_path):
            image_path_alt = f"{self.field_name}/{self.field_name}_{self.filter_name}.fits"
            if os.path.exists(image_path_alt):
                image_path = image_path_alt
                print(f"✅ Usando archivo: {image_path}")
            else:
                raise FileNotFoundError(f"No se encontró {image_path} ni {image_path_alt}")
            
        with fits.open(image_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data.astype(float)
                    header = hdu.header
                    break
            else:
                raise ValueError("No data found in FITS file")
        
        # Crear objeto WCS
        try:
            wcs = WCS(header)
            print("✅ WCS cargado correctamente")
        except Exception as e:
            print(f"⚠️  Error cargando WCS: {e}")
            wcs = None
                
        return data, header, wcs
    
    def apply_unsharp_mask(self, data):
        """Aplica el unsharp mask con parámetros optimizados"""
        # Filtro de mediana para eliminar fuentes puntuales
        median_filtered = median_filter(data, size=self.median_box)
        
        # Suavizado gaussiano para el fondo galáctico
        galaxy_background = gaussian_filter(median_filtered, sigma=self.gaussian_sigma)
        
        # Imagen residual (original - fondo)
        residual = data - galaxy_background
        
        return residual, galaxy_background, median_filtered
    
    def create_paper_figure(self, data, galaxy_background, residual, wcs):
        """Crea la figura principal para el paper"""
        # Tamaño optimizado para paper
        fig = plt.figure(figsize=(15, 10))
        
        # Layout mejorado: 2 filas, 3 columnas con más espacio
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.4,
                          height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        # Calcular escalas consistentes
        vmin_orig, vmax_orig = np.percentile(data, [5, 95])
        vmax_res = np.percentile(np.abs(residual), 99)
        vmin_res = -vmax_res
        
        # Panel A: Imagen original
        ax1 = fig.add_subplot(gs[0, 0], projection=wcs) if wcs else fig.add_subplot(gs[0, 0])
        self.plot_image_paper(ax1, data, wcs, "(a) Original Image", 
                             self.gray_cmap, vmin_orig, vmax_orig)
        
        # Panel B: Modelo de fondo
        ax2 = fig.add_subplot(gs[0, 1], projection=wcs) if wcs else fig.add_subplot(gs[0, 1])
        self.plot_image_paper(ax2, galaxy_background, wcs, "(b) Galaxy Background Model",
                             self.gray_cmap, vmin_orig, vmax_orig)
        
        # Panel C: Residual
        ax3 = fig.add_subplot(gs[0, 2], projection=wcs) if wcs else fig.add_subplot(gs[0, 2])
        self.plot_image_paper(ax3, residual, wcs, "(c) Residual Image",
                             self.divergent_cmap, vmin_res, vmax_res)
        
        # Panel D: Zoom región central - Original
        ax4 = fig.add_subplot(gs[1, 0], projection=wcs) if wcs else fig.add_subplot(gs[1, 0])
        self.plot_zoom_paper(ax4, data, wcs, "(d) Central Region - Original",
                            self.gray_cmap, None, None)  # Auto-scale para zoom
        
        # Panel E: Zoom región central - Residual  
        ax5 = fig.add_subplot(gs[1, 1], projection=wcs) if wcs else fig.add_subplot(gs[1, 1])
        self.plot_zoom_paper(ax5, residual, wcs, "(e) Central Region - Residual",
                            self.divergent_cmap, None, None)  # Auto-scale para zoom
        
        # Panel F: Información del procesamiento
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_processing_info(ax6, data, residual)
        
        # Título principal
        fig.suptitle(f'Galaxy Background Subtraction: {self.field_name} {self.filter_name}\n'
                    f'Unsharp Mask Technique',
                    fontsize=18, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_image_paper(self, ax, data, wcs, title, cmap, vmin, vmax):
        """Plot uniforme para imágenes del paper"""
        if wcs:
            im = ax.imshow(data, cmap=cmap, origin='lower', 
                          vmin=vmin, vmax=vmax, aspect='equal')
            
            # Configurar ejes en coordenadas celestes
            ax.coords['ra'].set_axislabel('Right Ascension (J2000)', fontsize=12)
            ax.coords['dec'].set_axislabel('Declination (J2000)', fontsize=12)
            ax.coords['ra'].set_ticklabel(size=10)
            ax.coords['dec'].set_ticklabel(size=10)
            
            # Formato para coordenadas
            ax.coords['ra'].set_major_formatter('hh:mm:ss')
            ax.coords['dec'].set_major_formatter('dd:mm:ss')
            
            # Configurar ticks
            ax.coords['ra'].set_ticks_position('bl')
            ax.coords['dec'].set_ticks_position('bl')
            ax.coords['ra'].set_ticklabel_position('b')
            ax.coords['dec'].set_ticklabel_position('l')
            
        else:
            im = ax.imshow(data, cmap=cmap, origin='lower',
                          vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_xlabel('X [pixels]', fontsize=12, labelpad=8)
            ax.set_ylabel('Y [pixels]', fontsize=12, labelpad=8)
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        ax.set_title(title, fontweight='bold', pad=12, fontsize=14)
        
        # Barra de color
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
        
        # Barra de escala
        self.add_scale_bar(ax, wcs, data.shape if wcs else None)
        
        ax.grid(False)
    
    def plot_zoom_paper(self, ax, data, wcs, title, cmap, vmin, vmax):
        """Zoom de región central para paper"""
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        zoom_size = 400
        
        y_slice = slice(center_y-zoom_size//2, center_y+zoom_size//2)
        x_slice = slice(center_x-zoom_size//2, center_x+zoom_size//2)
        
        zoom_data = data[y_slice, x_slice]
        
        # Calcular escalas automáticas para zoom
        if vmin is None or vmax is None:
            if cmap == self.divergent_cmap:
                vmax_zoom = np.percentile(np.abs(zoom_data), 99)
                vmin_zoom = -vmax_zoom
            else:
                vmin_zoom, vmax_zoom = np.percentile(zoom_data, [10, 90])
        else:
            vmin_zoom, vmax_zoom = vmin, vmax
        
        if wcs:
            zoom_wcs = wcs[y_slice, x_slice]
            im = ax.imshow(zoom_data, cmap=cmap, origin='lower',
                          vmin=vmin_zoom, vmax=vmax_zoom, aspect='equal')
            ax.coords['ra'].set_axislabel('Right Ascension', fontsize=11)
            ax.coords['dec'].set_axislabel('Declination', fontsize=11)
            ax.coords['ra'].set_ticklabel(size=9)
            ax.coords['dec'].set_ticklabel(size=9)
            
            ax.coords['ra'].set_ticks_position('bl')
            ax.coords['dec'].set_ticks_position('bl')
            ax.coords['ra'].set_ticklabel_position('b')
            ax.coords['dec'].set_ticklabel_position('l')
        else:
            im = ax.imshow(zoom_data, cmap=cmap, origin='lower',
                          vmin=vmin_zoom, vmax=vmax_zoom, aspect='equal')
            ax.set_xlabel('X [pixels]', fontsize=11, labelpad=6)
            ax.set_ylabel('Y [pixels]', fontsize=11, labelpad=6)
            ax.tick_params(axis='both', which='major', labelsize=9)
        
        ax.set_title(title, fontweight='bold', pad=10, fontsize=13)
        
        # Barra de color para zoom
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        
        # Barra de escala para zoom
        self.add_zoom_scale_bar(ax, wcs, zoom_data.shape if wcs else None)
        
        ax.grid(False)
    
    def plot_processing_info(self, ax, original, residual):
        """Información del procesamiento en lugar de histograma"""
        # Estadísticas
        orig_stats = sigma_clipped_stats(original, sigma=3.0)
        res_stats = sigma_clipped_stats(residual, sigma=3.0)
        
        # Texto informativo
        info_text = (
            "Processing Parameters:\n"
            f"• Median filter: {self.median_box}×{self.median_box} px\n"
            f"• Gaussian σ: {self.gaussian_sigma} px\n"
            f"• Field: {self.field_name} {self.filter_name}\n\n"
            "Statistics:\n"
            "Original Image:\n"
            f"  Mean: {orig_stats[0]:.1f}\n"
            f"  Median: {orig_stats[1]:.1f}\n"
            f"  Std: {orig_stats[2]:.1f}\n\n"
            "Residual Image:\n"
            f"  Mean: {res_stats[0]:.3f}\n"
            f"  Median: {res_stats[1]:.3f}\n"
            f"  Std: {res_stats[2]:.3f}\n\n"
            "Method:\n"
            "Data − Gaussian(Median(Data))"
        )
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', linespacing=1.4, fontfamily='monospace')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor('#f8f9fa')
        
        # Añadir título al panel
        ax.set_title("(f) Processing Information", fontweight='bold', pad=10, fontsize=13)
    
    def get_pixel_scale(self, wcs):
        """Obtiene la escala de píxeles en arcmin"""
        try:
            if hasattr(wcs, 'proj_plane_pixel_scales'):
                scales = wcs.proj_plane_pixel_scales()
                if hasattr(scales[0], 'to'):
                    scale_arcmin = scales[0].to(u.arcmin).value
                else:
                    scale_arcmin = scales[0] * 60
                return scale_arcmin
            else:
                try:
                    if hasattr(wcs.wcs, 'cdelt') and wcs.wcs.cdelt[0] is not None:
                        scale_deg = abs(wcs.wcs.cdelt[0])
                    else:
                        scale_deg = abs(wcs.wcs.cd[0, 0])
                    return scale_deg * 60
                except:
                    return 0.55 / 60
        except:
            return 0.55 / 60
    
    def add_scale_bar(self, ax, wcs, shape):
        """Añade barra de escala de 4.6 arcmin"""
        if wcs and shape:
            try:
                scale_arcmin_per_pixel = self.get_pixel_scale(wcs)
                pixels_for_4_6_arcmin = 4.6 / scale_arcmin_per_pixel
                
                x_start = shape[1] * 0.08
                y_pos = shape[0] * 0.08
                
                ax.plot([x_start, x_start + pixels_for_4_6_arcmin], [y_pos, y_pos], 
                       'k-', linewidth=4, alpha=0.8, transform=ax.get_transform('pixel'))
                ax.plot([x_start, x_start + pixels_for_4_6_arcmin], [y_pos, y_pos], 
                       'w-', linewidth=2, alpha=0.8, transform=ax.get_transform('pixel'))
                
                ax.text(x_start + pixels_for_4_6_arcmin/2, y_pos + shape[0]*0.03, 
                       '4.6 arcmin', ha='center', va='bottom', color='black', 
                       fontweight='bold', fontsize=10, transform=ax.get_transform('pixel'),
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.9))
            except:
                self.add_generic_scale_bar(ax, shape)
        else:
            self.add_generic_scale_bar(ax, shape)
    
    def add_zoom_scale_bar(self, ax, wcs, shape):
        """Barra de escala para zoom (1 arcmin)"""
        if wcs and shape:
            try:
                scale_arcmin_per_pixel = self.get_pixel_scale(wcs)
                pixels_for_1_arcmin = 1.0 / scale_arcmin_per_pixel
                
                x_start = shape[1] * 0.15
                y_pos = shape[0] * 0.10
                
                ax.plot([x_start, x_start + pixels_for_1_arcmin], [y_pos, y_pos], 
                       'k-', linewidth=3, alpha=0.8, transform=ax.get_transform('pixel'))
                ax.plot([x_start, x_start + pixels_for_1_arcmin], [y_pos, y_pos], 
                       'w-', linewidth=1.5, alpha=0.8, transform=ax.get_transform('pixel'))
                
                ax.text(x_start + pixels_for_1_arcmin/2, y_pos + shape[0]*0.04, 
                       '1 arcmin', ha='center', va='bottom', color='black',
                       fontweight='bold', fontsize=9, transform=ax.get_transform('pixel'),
                       bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.9))
            except:
                self.add_generic_zoom_scale_bar(ax, shape)
        else:
            self.add_generic_zoom_scale_bar(ax, shape)
    
    def add_generic_scale_bar(self, ax, shape):
        """Barra de escala genérica"""
        scale_pixels = 500
        x_start = shape[1] * 0.08
        y_pos = shape[0] * 0.08
        
        ax.plot([x_start, x_start + scale_pixels], [y_pos, y_pos], 
               'k-', linewidth=4, alpha=0.8)
        ax.plot([x_start, x_start + scale_pixels], [y_pos, y_pos], 
               'w-', linewidth=2, alpha=0.8)
        
        ax.text(x_start + scale_pixels/2, y_pos + shape[0]*0.03, 
               '4.6 arcmin', ha='center', va='bottom', color='black', fontweight='bold',
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.9))
    
    def add_generic_zoom_scale_bar(self, ax, shape):
        """Barra de escala genérica para zoom"""
        scale_pixels = 100
        x_start = shape[1] * 0.15
        y_pos = shape[0] * 0.10
        
        ax.plot([x_start, x_start + scale_pixels], [y_pos, y_pos], 
               'k-', linewidth=3, alpha=0.8)
        ax.plot([x_start, x_start + scale_pixels], [y_pos, y_pos], 
               'w-', linewidth=1.5, alpha=0.8)
        
        ax.text(x_start + scale_pixels/2, y_pos + shape[0]*0.04, 
               '1 arcmin', ha='center', va='bottom', color='black', fontweight='bold',
               fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.9))
    
    def save_figure(self, fig, filename, format='pdf'):
        """Guarda la figura en formatos de alta calidad"""
        path = Path(self.output_dir) / f"{filename}.{format}"
        
        for fmt in (['pdf', 'png'] if format == 'pdf' else [format]):
            save_path = Path(self.output_dir) / f"{filename}.{fmt}"
            fig.savefig(save_path, dpi=400, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            if fmt == 'pdf':
                print(f"✅ PDF para paper guardado: {save_path}")
            else:
                print(f"✅ {fmt.upper()} guardado: {save_path}")
    
    def run(self):
        """Ejecuta la generación de figuras para paper"""
        print(f"📄 Generando figura para PAPER: {self.field_name}")
        
        try:
            # Cargar datos
            print("📁 Cargando datos...")
            data, header, wcs = self.load_data()
            
            # Aplicar unsharp mask
            print("🔍 Aplicando unsharp mask...")
            residual, galaxy_background, _ = self.apply_unsharp_mask(data)
            
            # Crear figura para paper
            print("📊 Creando figura para paper...")
            fig_paper = self.create_paper_figure(data, galaxy_background, residual, wcs)
            self.save_figure(fig_paper, f"{self.field_name}_{self.filter_name}_paper_figure")
            plt.close(fig_paper)
            
            print("🎉 ¡Figura para paper generada exitosamente!")
            print(f"📁 Guardada en: {self.output_dir}/")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Función principal"""
    figure_generator = APJUnsharpMaskPaper()
    figure_generator.run()
    
    print("\n✅ CARACTERÍSTICAS PARA PAPER:")
    print("   • Layout optimizado (2×3)")
    print("   • Escalas de color consistentes") 
    print("   • Etiquetas de ejes claras y visibles")
    print("   • Barras de escala apropiadas")
    print("   • Información de procesamiento incluida")
    print("   • Calidad de publicación (400 DPI)")
    print("   • Tamaño optimizado para paper")

if __name__ == "__main__":
    main()
