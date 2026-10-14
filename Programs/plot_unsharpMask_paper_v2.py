#!/usr/bin/env python3
"""
---
Figuras profesionales para ApJ – Versión final con:
- Panel de contexto: DSS original (extent completo), aspect='equal'.
- Cruz en posición real de CenA11.
- Separación amplia entre paneles y colorbar para evitar solapamientos.
- Fuentes grandes, legibles para publicación.
- Memoria optimizada (float32, rasterizado, Agg, submuestreo DSS).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import median_filter, gaussian_filter, zoom
import matplotlib.colors as colors
import os
from pathlib import Path
import traceback
import gc

try:
    from astroquery.skyview import SkyView
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    print("⚠️  astroquery no instalado. El panel de contexto no se descargará automáticamente.")

# Estilo profesional con fuentes grandes
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.2,
})

class APJUnsharpMaskPaper:
    def __init__(self):
        self.field_name = "CenA11"
        self.filter_name = "F660"
        self.output_dir = "APJ_Figures_Paper"
        os.makedirs(self.output_dir, exist_ok=True)
        self.median_box = 25
        self.gaussian_sigma = 5
        self.halo_contrast = 2.5
        self.context_survey = "DSS2 Red"
        self.ngc5128_coords = SkyCoord("13h25m27.6s", "-43d01m09s", frame='icrs')
        self.local_context_file = Path(self.output_dir) / "CenA_DSS.fits"
        self.max_context_pixels = 900
        self.setup_paper_style()

    def setup_paper_style(self):
        self.colors = {'primary': '#2c3e50', 'secondary': '#7f8c8d',
                       'accent': '#c0392b', 'background': '#f8f9fa'}
        self.create_consistent_colormaps()

    def create_consistent_colormaps(self):
        self.gray_cmap = plt.cm.gray_r
        self.divergent_cmap = colors.LinearSegmentedColormap.from_list(
            'paper_divergent', ['#a6cee3', '#ffffff', '#1a1a1a'])

    def load_data(self):
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
                    data = hdu.data.astype(np.float32)
                    header = hdu.header
                    break
            else:
                raise ValueError("No data found in FITS file")
        try:
            wcs = WCS(header)
            print("✅ WCS cargado correctamente")
        except Exception as e:
            print(f"⚠️  Error cargando WCS: {e}")
            wcs = None
        return data, header, wcs

    def apply_unsharp_mask(self, data):
        median_filtered = median_filter(data, size=self.median_box).astype(np.float32)
        galaxy_background = gaussian_filter(median_filtered, sigma=self.gaussian_sigma).astype(np.float32)
        residual = data - galaxy_background
        return residual, galaxy_background, median_filtered

    # ============ FIGURA ORIGINAL (6 paneles) ============
    def create_paper_figure(self, data, galaxy_background, residual, wcs):
        fig = plt.figure(figsize=(14, 9))
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.53, wspace=0.58,
                          height_ratios=[1, 1], width_ratios=[1, 1, 1])

        vmin_orig = max(0, np.percentile(data, 1))
        vmax_orig = np.percentile(data, 99)
        vmax_res = np.percentile(np.abs(residual), 99)
        vmin_res = -vmax_res

        ax1 = fig.add_subplot(gs[0, 0], projection=wcs) if wcs else fig.add_subplot(gs[0, 0])
        self.plot_image_paper(ax1, data, wcs, "(a) Original", self.gray_cmap,
                              vmin_orig, vmax_orig, cbar_label='ADU')
        ax2 = fig.add_subplot(gs[0, 1], projection=wcs) if wcs else fig.add_subplot(gs[0, 1])
        self.plot_image_paper(ax2, galaxy_background, wcs, "(b) Smoothed Galaxy Model",
                              self.gray_cmap, vmin_orig, vmax_orig, cbar_label='ADU')
        ax3 = fig.add_subplot(gs[0, 2], projection=wcs) if wcs else fig.add_subplot(gs[0, 2])
        self.plot_image_paper(ax3, residual, wcs, "(c) Residual",
                              self.divergent_cmap, vmin_res, vmax_res,
                              cbar_label='Res. (ADU)')

        ax4 = self._make_zoom_axes(fig, gs[1, 0], data, wcs)
        self.plot_zoom_paper(ax4, data, wcs, "(d) Original Zoom",
                             self.gray_cmap, vmin_orig, vmax_orig, cbar_label='ADU')
        ax5 = self._make_zoom_axes(fig, gs[1, 1], data, wcs)
        self.plot_zoom_paper(ax5, residual, wcs, "(e) Residual Zoom",
                             self.divergent_cmap, vmin_res, vmax_res,
                             cbar_label='Res. (ADU)')
        self.plot_context_panel(gs[1, 2], fig, wcs, data, residual)

        fig.suptitle(f'{self.field_name} J0660 – Unsharp Mask Subtraction',
                     fontsize=18, fontweight='bold', y=0.98)
        return fig

    # ============ NUEVA FIGURA SOLO FILA SUPERIOR (3 paneles) ============
    def create_top_only_figure(self, data, galaxy_background, residual, wcs):
        """Genera una figura compacta con solo los paneles superiores."""
        fig = plt.figure(figsize=(14, 5))
        gs = plt.GridSpec(1, 3, figure=fig, hspace=0.35, wspace=0.55)

        vmin_orig = max(0, np.percentile(data, 1))
        vmax_orig = np.percentile(data, 99)
        vmax_res = np.percentile(np.abs(residual), 99)
        vmin_res = -vmax_res

        ax1 = fig.add_subplot(gs[0, 0], projection=wcs) if wcs else fig.add_subplot(gs[0, 0])
        self.plot_image_paper(ax1, data, wcs, "(a) Original", self.gray_cmap,
                              vmin_orig, vmax_orig, cbar_label='ADU')
        ax2 = fig.add_subplot(gs[0, 1], projection=wcs) if wcs else fig.add_subplot(gs[0, 1])
        self.plot_image_paper(ax2, galaxy_background, wcs, "(b) Smoothed Galaxy Model",
                              self.gray_cmap, vmin_orig, vmax_orig, cbar_label='ADU')
        ax3 = fig.add_subplot(gs[0, 2], projection=wcs) if wcs else fig.add_subplot(gs[0, 2])
        self.plot_image_paper(ax3, residual, wcs, "(c) Residual",
                              self.divergent_cmap, vmin_res, vmax_res,
                              cbar_label='Res. (ADU)')

        fig.suptitle(f'{self.field_name} J0660 – Unsharp Mask Subtraction (Top Panels)',
                     fontsize=18, fontweight='bold', y=0.98)
        return fig

    # El resto de los métodos se mantienen exactamente igual
    def _make_zoom_axes(self, fig, subplot_spec, data, wcs):
        if wcs is None:
            return fig.add_subplot(subplot_spec)
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        zoom_size = 400
        y_slice = slice(center_y-zoom_size//2, center_y+zoom_size//2)
        x_slice = slice(center_x-zoom_size//2, center_x+zoom_size//2)
        zoom_wcs = wcs[y_slice, x_slice]
        return fig.add_subplot(subplot_spec, projection=zoom_wcs)

    def _resample_dss(self, data, wcs):
        ny, nx = data.shape
        if max(ny, nx) <= self.max_context_pixels:
            return data, wcs
        factor = self.max_context_pixels / max(ny, nx)
        print(f"🌌 Submuestreando imagen DSS {nx}x{ny} a factor {factor:.2f}")
        new_data = zoom(data, factor, order=1).astype(np.float32)
        new_wcs = wcs.deepcopy()
        if hasattr(new_wcs.wcs, 'cd'):
            new_wcs.wcs.cd *= 1.0 / factor
        elif hasattr(new_wcs.wcs, 'pc'):
            new_wcs.wcs.pc *= 1.0 / factor
            new_wcs.wcs.cdelt *= 1.0 / factor
        new_wcs.wcs.crpix = np.array(new_wcs.wcs.crpix) * factor
        return new_data, new_wcs

    def _load_context_image(self):
        data, wcs = None, None
        if self.local_context_file.exists():
            print(f"🌌 Usando imagen de contexto local: {self.local_context_file}")
            try:
                with fits.open(self.local_context_file) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None:
                            data = hdu.data.astype(np.float32)
                            wcs = WCS(hdu.header)
                            break
            except Exception as e:
                print(f"⚠️  Error leyendo archivo local: {e}")
        if data is None and ASTROQUERY_AVAILABLE:
            try:
                print("🌌 Descargando imagen de contexto (DSS) desde SkyView...")
                hdu_list = SkyView.get_images(position=self.ngc5128_coords,
                                              survey=[self.context_survey],
                                              radius=45*u.arcmin)
                if hdu_list:
                    for hdu in hdu_list[0]:
                        if hdu.data is not None:
                            data = hdu.data.astype(np.float32)
                            wcs = WCS(hdu.header)
                            try:
                                hdu_list[0].writeto(self.local_context_file, overwrite=True)
                                print(f"💾 Imagen guardada localmente como {self.local_context_file}")
                            except:
                                pass
                            break
            except Exception as e:
                print(f"⚠️  Fallo descarga SkyView: {e}")
        if data is None or wcs is None:
            return None, None, None

        data, wcs = self._resample_dss(data, wcs)

        ny, nx = data.shape
        corners = np.array([[0, 0], [0, ny], [nx, 0], [nx, ny]], dtype=np.float64)
        world = wcs.wcs_pix2world(corners, 0)
        ra_vals = world[:, 0]
        dec_vals = world[:, 1]
        ra_min, ra_max = ra_vals.min(), ra_vals.max()
        dec_min, dec_max = dec_vals.min(), dec_vals.max()

        ra_left = wcs.wcs_pix2world([[0, ny//2]], 0)[0][0]
        ra_right = wcs.wcs_pix2world([[nx-1, ny//2]], 0)[0][0]
        if ra_left > ra_right:
            extent = [ra_max, ra_min, dec_min, dec_max]
        else:
            extent = [ra_min, ra_max, dec_min, dec_max]

        return data, wcs, extent

    def plot_context_panel(self, subplot_spec, fig, wcs, data, residual):
        if not wcs:
            ax = fig.add_subplot(subplot_spec)
            self.plot_processing_info(ax, data, residual)
            return

        context_data, context_wcs, extent = self._load_context_image()
        if context_data is None:
            ax = fig.add_subplot(subplot_spec)
            self.plot_processing_info(ax, data, residual)
            return

        try:
            ax = fig.add_subplot(subplot_spec)
            vmin, vmax = np.percentile(context_data, [1, 99])
            ax.imshow(context_data, cmap='gray_r', origin='lower',
                      extent=extent, vmin=vmin, vmax=vmax, aspect='equal',
                      rasterized=True)

            footprint = wcs.calc_footprint()
            center_ra = np.mean(footprint[:, 0])
            center_dec = np.mean(footprint[:, 1])

            ax.plot(center_ra, center_dec, marker='+', color='yellow',
                    markersize=15, markeredgewidth=2, zorder=10)
            ax.text(center_ra, center_dec, f" {self.field_name}",
                    color='yellow', fontsize=12, fontweight='bold',
                    ha='left', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel('RA (J2000)', fontsize=13)
            ax.set_ylabel('Dec (J2000)', fontsize=13)
            ax.tick_params(labelsize=11)
            ax.set_title("(f) NGC 5128 field", fontweight='bold', pad=10, fontsize=14)

            scale_deg = 5.0 / 60.0
            x0 = extent[0] + 0.05 * (extent[1] - extent[0])
            y0 = extent[2] + 0.1 * (extent[3] - extent[2])
            ax.plot([x0, x0 + scale_deg], [y0, y0], color='cyan', lw=3)
            ax.plot([x0, x0 + scale_deg], [y0, y0], color='black', lw=1)
            ax.text(x0 + scale_deg/2, y0 + 0.07*(extent[3]-extent[2]),
                    "5'", color='cyan', fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
            ax.grid(False)

        except Exception as e:
            print(f"❌ Error en panel contexto: {e}")
            traceback.print_exc()
            ax = fig.add_subplot(subplot_spec)
            self.plot_processing_info(ax, data, residual)

    def plot_image_paper(self, ax, data, wcs, title, cmap, vmin, vmax, cbar_label=''):
        if wcs:
            im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                           aspect='equal', rasterized=True,
                           transform=ax.get_transform('pixel'))
            ax.coords['ra'].set_axislabel('RA (J2000)', fontsize=13)
            ax.coords['dec'].set_axislabel('Dec (J2000)', fontsize=13)
            ax.coords['ra'].set_ticklabel(size=11)
            ax.coords['dec'].set_ticklabel(size=11)
            ax.coords['ra'].set_major_formatter('hh:mm:ss')
            ax.coords['dec'].set_major_formatter('dd:mm:ss')
            ax.coords['ra'].set_ticks_position('bl')
            ax.coords['dec'].set_ticks_position('bl')
            ax.coords['ra'].set_ticklabel_position('b')
            ax.coords['dec'].set_ticklabel_position('l')
        else:
            im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                           aspect='equal', rasterized=True)
            ax.set_xlabel('X [pixels]', fontsize=13)
            ax.set_ylabel('Y [pixels]', fontsize=13)
            ax.tick_params(labelsize=11)
        ax.set_title(title, fontweight='bold', pad=10, fontsize=14)
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.10, shrink=0.75)
        cbar.ax.tick_params(labelsize=11)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=12, labelpad=2)
        self.add_scale_bar(ax, wcs, data.shape)
        ax.grid(False)

    def plot_zoom_paper(self, ax, data, wcs, title, cmap, vmin, vmax, cbar_label=''):
        center_y, center_x = data.shape[0]//2, data.shape[1]//2
        zoom_size = 400
        y_slice = slice(center_y-zoom_size//2, center_y+zoom_size//2)
        x_slice = slice(center_x-zoom_size//2, center_x+zoom_size//2)
        zoom_data = data[y_slice, x_slice]

        if vmin is None or vmax is None:
            if cmap == self.divergent_cmap:
                vmax_zoom = np.percentile(np.abs(zoom_data), 99)
                vmin_zoom = -vmax_zoom
            else:
                vmin_zoom, vmax_zoom = np.percentile(zoom_data, [10, 90])
                vmin_zoom = max(0, vmin_zoom)
        else:
            vmin_zoom, vmax_zoom = vmin, vmax

        if wcs:
            im = ax.imshow(zoom_data, cmap=cmap, origin='lower',
                           vmin=vmin_zoom, vmax=vmax_zoom,
                           aspect='equal', rasterized=True,
                           transform=ax.get_transform('pixel'))
            ax.coords['ra'].set_axislabel('RA', fontsize=12)
            ax.coords['dec'].set_axislabel('Dec', fontsize=12)
            ax.coords['ra'].set_ticklabel(size=10)
            ax.coords['dec'].set_ticklabel(size=10)
            ax.coords['ra'].set_ticks_position('bl')
            ax.coords['dec'].set_ticks_position('bl')
            ax.coords['ra'].set_ticklabel_position('b')
            ax.coords['dec'].set_ticklabel_position('l')
        else:
            im = ax.imshow(zoom_data, cmap=cmap, origin='lower',
                           vmin=vmin_zoom, vmax=vmax_zoom,
                           aspect='equal', rasterized=True)
            ax.set_xlabel('X [pixels]', fontsize=12)
            ax.set_ylabel('Y [pixels]', fontsize=12)
            ax.tick_params(labelsize=10)

        ax.set_title(title, fontweight='bold', pad=10, fontsize=13)
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.10, shrink=0.75)
        cbar.ax.tick_params(labelsize=10)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=10, labelpad=1)
        self.add_zoom_scale_bar(ax, wcs, zoom_data.shape)
        ax.grid(False)

    def plot_processing_info(self, ax, original, residual):
        orig_stats = sigma_clipped_stats(original, sigma=3.0)
        res_stats = sigma_clipped_stats(residual, sigma=3.0)
        info_text = (
            "Parameters:\n"
            f"• Median: {self.median_box}×{self.median_box} px\n"
            f"• Gauss σ: {self.gaussian_sigma} px\n"
            f"• Field: {self.field_name} {self.filter_name}\n\n"
            "Stats (original):\n"
            f"  Mean: {orig_stats[0]:.1f}  Median: {orig_stats[1]:.1f}  Std: {orig_stats[2]:.1f}\n\n"
            "Stats (residual):\n"
            f"  Mean: {res_stats[0]:.3f}  Median: {res_stats[1]:.3f}  Std: {res_stats[2]:.3f}"
        )
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', linespacing=1.4, fontfamily='monospace')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#f8f9fa')

    def get_pixel_scale(self, wcs):
        try:
            if hasattr(wcs, 'proj_plane_pixel_scales'):
                scales = wcs.proj_plane_pixel_scales()
                return (scales[0].to(u.arcmin).value if hasattr(scales[0],'to')
                        else scales[0]*60)
            elif hasattr(wcs.wcs, 'cdelt') and wcs.wcs.cdelt[0] is not None:
                return abs(wcs.wcs.cdelt[0])*60
            else:
                return abs(wcs.wcs.cd[0,0])*60
        except:
            return 0.55/60

    def add_scale_bar(self, ax, wcs, shape):
        if wcs and shape:
            try:
                scale_arcmin_per_pixel = self.get_pixel_scale(wcs)
                pixels = 4.6 / scale_arcmin_per_pixel
                x_start = shape[1]*0.08
                y_pos = shape[0]*0.08
                ax.plot([x_start, x_start+pixels], [y_pos, y_pos],
                        'k-', lw=3, alpha=0.8, transform=ax.get_transform('pixel'))
                ax.plot([x_start, x_start+pixels], [y_pos, y_pos],
                        'w-', lw=1.5, alpha=0.8, transform=ax.get_transform('pixel'))
                ax.text(x_start+pixels/2, y_pos+shape[0]*0.05, '4.6′',
                        ha='center', va='bottom', color='black', fontweight='bold',
                        fontsize=11, transform=ax.get_transform('pixel'),
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.9))
            except:
                self.add_generic_scale_bar(ax, shape)
        else:
            self.add_generic_scale_bar(ax, shape)

    def add_zoom_scale_bar(self, ax, wcs, shape):
        if wcs and shape:
            try:
                scale_arcmin_per_pixel = self.get_pixel_scale(wcs)
                pixels = 1.0 / scale_arcmin_per_pixel
                x_start = shape[1]*0.15
                y_pos = shape[0]*0.10
                ax.plot([x_start, x_start+pixels], [y_pos, y_pos],
                        'k-', lw=2.5, alpha=0.8, transform=ax.get_transform('pixel'))
                ax.plot([x_start, x_start+pixels], [y_pos, y_pos],
                        'w-', lw=1, alpha=0.8, transform=ax.get_transform('pixel'))
                ax.text(x_start+pixels/2, y_pos+shape[0]*0.04, '1′',
                        ha='center', va='bottom', color='black', fontweight='bold',
                        fontsize=10, transform=ax.get_transform('pixel'),
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.9))
            except:
                self.add_generic_zoom_scale_bar(ax, shape)
        else:
            self.add_generic_zoom_scale_bar(ax, shape)

    def add_generic_scale_bar(self, ax, shape):
        scale_pixels = 500
        x_start = shape[1]*0.08
        y_pos = shape[0]*0.08
        ax.plot([x_start, x_start+scale_pixels], [y_pos, y_pos], 'k-', lw=3, alpha=0.8)
        ax.plot([x_start, x_start+scale_pixels], [y_pos, y_pos], 'w-', lw=1.5, alpha=0.8)
        ax.text(x_start+scale_pixels/2, y_pos+shape[0]*0.05, '4.6′',
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.9))

    def add_generic_zoom_scale_bar(self, ax, shape):
        scale_pixels = 100
        x_start = shape[1]*0.15
        y_pos = shape[0]*0.10
        ax.plot([x_start, x_start+scale_pixels], [y_pos, y_pos], 'k-', lw=2.5, alpha=0.8)
        ax.plot([x_start, x_start+scale_pixels], [y_pos, y_pos], 'w-', lw=1, alpha=0.8)
        ax.text(x_start+scale_pixels/2, y_pos+shape[0]*0.04, '1′',
                ha='center', va='bottom', color='black', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.9))

    def save_figure(self, fig, filename, format='pdf'):
        for fmt in (['pdf', 'png'] if format == 'pdf' else [format]):
            save_path = Path(self.output_dir) / f"{filename}.{fmt}"
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"{fmt.upper()} guardado: {save_path}")

    def run(self, top_only=False):
        print(f" Generando figura para PAPER: {self.field_name}")
        try:
            print(" Cargando datos (float32)...")
            data, header, wcs = self.load_data()
            print(" Aplicando unsharp mask...")
            residual, galaxy_background, _ = self.apply_unsharp_mask(data)

            if top_only:
                print(" Creando figura SOLO FILA SUPERIOR...")
                fig_paper = self.create_top_only_figure(data, galaxy_background, residual, wcs)
                filename = f"{self.field_name}_{self.filter_name}_paper_figure_top"
            else:
                print(" Creando figura completa original...")
                fig_paper = self.create_paper_figure(data, galaxy_background, residual, wcs)
                filename = f"{self.field_name}_{self.filter_name}_paper_figure"

            self.save_figure(fig_paper, filename)
            plt.close(fig_paper)
            del data, residual, galaxy_background, fig_paper
            gc.collect()
            print(" ¡Figura generada exitosamente!")
            print(f" Guardada en: {self.output_dir}/")
        except Exception as e:
            print(f" Error: {e}")
            traceback.print_exc()

def main():
    figure_generator = APJUnsharpMaskPaper()
    # Genera la figura completa original (6 paneles)
    figure_generator.run(top_only=False)
    # Genera la figura compacta con solo la fila superior (3 paneles)
    figure_generator.run(top_only=True)
    print("\n    Figuras generadas:")
    print("   • CenA11_F660_paper_figure (original completa)")
    print("   • CenA11_F660_paper_figure_top (solo paneles superiores)")

if __name__ == "__main__":
    main()
