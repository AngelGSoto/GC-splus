#!/usr/bin/env python3
"""
gc_two_panel_figure.py
Figura de dos paneles para el paper: SNR + Densidad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.colors as colors
from matplotlib import rcParams
from scipy.stats import gaussian_kde
import os

def setup_publication_style():
    """Configura estilo profesional para figura de dos paneles"""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 18
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 400
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.linewidth'] = 1.0

class GCTwoPanelFigure:
    def __init__(self):
        setup_publication_style()
        self.fig = None
        self.axs = None
        
    def create_two_panel_figure(self, df, snr_column='SNR_F660_3', 
                              output_path="Fig1_two_panel_gc_distribution.pdf"):
        """Crea figura de dos paneles: SNR + Densidad"""
        
        # Crear figura con 2 subplots lado a lado
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.axs = (ax1, ax2)
        
        # =======================================================================
        # PANEL A: DISTRIBUCIÓN CON SNR (CALIDAD FOTOMÉTRICA)
        # =======================================================================
        
        snr_values = df[snr_column]
        snr_min, snr_max = 0, 20
        
        # Scatter plot para SNR
        sc1 = ax1.scatter(
            df['galactic_l'], 
            df['galactic_b'],
            c=df[snr_column],
            s=25,
            alpha=0.8,
            cmap='plasma',
            vmin=snr_min,
            vmax=snr_max,
            edgecolors='black',
            linewidths=0.08,
            rasterized=True
        )
        
        ax1.set_xlabel('Galactic Longitude [deg]', fontsize=14, fontweight='bold', labelpad=8)
        ax1.set_ylabel('Galactic Latitude [deg]', fontsize=14, fontweight='bold', labelpad=8)
        ax1.tick_params(axis='both', which='major', labelsize=11)
        ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.4)
        
        # Barra de color para SNR
        cbar1 = self.fig.colorbar(sc1, ax=ax1, shrink=0.8, pad=0.03, aspect=25, extend='max')
        cbar1.set_label('J0660-band Signal-to-Noise Ratio', rotation=270, 
                       labelpad=15, fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # Marcas de referencia en barra de SNR
        cbar1.ax.axhline(y=3, color='red', linestyle='--', alpha=0.8, linewidth=0.8)
        cbar1.ax.axhline(y=5, color='orange', linestyle='--', alpha=0.8, linewidth=0.8)
        cbar1.ax.axhline(y=10, color='green', linestyle='--', alpha=0.8, linewidth=0.8)
        
        ax1.set_title('(a) Photometric Quality (SNR)', fontsize=16, fontweight='bold', pad=12)
        
        # =======================================================================
        # PANEL B: DENSIDAD ESPACIAL (CONCENTRACIÓN FÍSICA)
        # =======================================================================
        
        # Calcular densidad con Gaussian KDE
        x, y = df['galactic_l'], df['galactic_b']
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        
        # Normalizar densidad para mejor visualización
        z_normalized = (z - z.min()) / (z.max() - z.min())
        
        # Scatter plot para densidad
        sc2 = ax2.scatter(
            df['galactic_l'], 
            df['galactic_b'],
            c=z_normalized,
            s=25,
            alpha=0.8,
            cmap='plasma',  # Alternativa: 'hot', 'plasma', 'inferno'
            edgecolors='black',
            linewidths=0.05,
            rasterized=True
        )
        
        ax2.set_xlabel('Galactic Longitude [deg]', fontsize=14, fontweight='bold', labelpad=8)
        ax2.set_ylabel('Galactic Latitude [deg]', fontsize=14, fontweight='bold', labelpad=8)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.4)
        
        # Barra de color para densidad
        cbar2 = self.fig.colorbar(sc2, ax=ax2, shrink=0.8, pad=0.03, aspect=25)
        cbar2.set_label('Normalized Spatial Density', rotation=270, 
                       labelpad=15, fontsize=12, fontweight='bold')
        cbar2.ax.tick_params(labelsize=10)
        
        ax2.set_title('(b) Spatial Distribution', fontsize=16, fontweight='bold', pad=12)
        
        # =======================================================================
        # ESTADÍSTICAS Y ANOTACIONES COMPARTIDAS
        # =======================================================================
        
        stats_text = f'N = {len(df):,} GCs\nMedian SNR = {snr_values.median():.1f}'
        
        # Añadir estadísticas a ambos paneles
        for ax in [ax1, ax2]:
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes,
                   fontsize=10,
                   fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor="white", 
                            alpha=0.95,
                            edgecolor='black',
                            linewidth=0.8))
        
        # =======================================================================
        # SINCRONIZAR EJES Y LÍMITES
        # =======================================================================
        
        # Usar los mismos límites en ambos paneles
        l_min, l_max = df['galactic_l'].min(), df['galactic_l'].max()
        b_min, b_max = df['galactic_b'].min(), df['galactic_b'].max()
        
        l_margin = (l_max - l_min) * 0.02
        b_margin = (b_max - b_min) * 0.02
        
        for ax in [ax1, ax2]:
            ax.set_xlim(l_min - l_margin, l_max + l_margin)
            ax.set_ylim(b_min - b_margin, b_max + b_margin)
            ax.set_aspect('equal')  # Misma escala en ambos ejes
        
        plt.tight_layout(pad=2.0)
        
        # =======================================================================
        # GUARDAR FIGURA
        # =======================================================================
        
        base_path = output_path.replace('.pdf', '')
        
        # PDF (calidad máxima)
        plt.savefig(f"{base_path}.pdf", dpi=400, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        # PNG (alta resolución)
        plt.savefig(f"{base_path}.png", dpi=400, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print(f"💾 FIGURA DE DOS PANELES GUARDADA:")
        print(f"   {base_path}.pdf")
        print(f"   {base_path}.png")
        
        plt.close()
        
        return self.fig

    def load_and_clean_data(self, catalog_path, snr_column='SNR_F660_3'):
        """Carga y limpia los datos (misma función que antes)"""
        try:
            df = pd.read_csv(catalog_path)
            initial_count = len(df)
            
            print("=" * 60)
            print("📊 CARGA Y LIMPIEZA DE DATOS")
            print("=" * 60)
            print(f"Fuentes iniciales: {initial_count:,}")
            
            # Eliminar duplicados
            coord_duplicates = df.duplicated(subset=['RAJ2000', 'DEJ2000']).sum()
            df = df[~df.duplicated(subset=['RAJ2000', 'DEJ2000'])].copy()
            print(f"Duplicados eliminados: {coord_duplicates:,}")
            
            # Convertir a coordenadas galácticas
            coords = SkyCoord(ra=df['RAJ2000'].values*u.deg, 
                            dec=df['DEJ2000'].values*u.deg, 
                            frame='icrs')
            galactic_coords = coords.galactic
            
            df['galactic_l'] = galactic_coords.l.deg
            df['galactic_b'] = galactic_coords.b.deg
            
            # Filtrar por SNR válido
            valid_mask = (
                (df[snr_column] > 0) & 
                (df[snr_column] < 100) & 
                np.isfinite(df[snr_column]) &
                np.isfinite(df['galactic_l']) & 
                np.isfinite(df['galactic_b'])
            )
            
            filtered_df = df[valid_mask].copy()
            final_count = len(filtered_df)
            
            print(f"Fuentes finales: {final_count:,}")
            print(f"Tasa de retención: {(final_count/initial_count*100):.1f}%")
            print(f"Mediana SNR: {filtered_df[snr_column].median():.2f}")
            
            return filtered_df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def main():
    """Función principal"""
    CATALOG_PATH = "Results_Corrected/all_fields_photometry_COMPLETE.csv"
    SNR_COLUMN = 'SNR_F660_3'
    
    os.makedirs("paper_figures", exist_ok=True)
    
    plotter = GCTwoPanelFigure()
    
    print("📁 Cargando datos...")
    df_cleaned = plotter.load_and_clean_data(CATALOG_PATH, SNR_COLUMN)
    
    if df_cleaned is None:
        return
    
    print("\n🎨 Creando figura de dos paneles...")
    plotter.create_two_panel_figure(
        df_cleaned, 
        SNR_COLUMN,
        output_path="paper_figures/Fig1_two_panel_gc_distribution.pdf"
    )
    
    print("\n" + "="*60)
    print("✅ FIGURA DE DOS PANELES CREADA EXITOSAMENTE")
    print("="*60)
    print("LA FIGURA MUESTRA:")
    print("• Panel (a): Calidad fotométrica (SNR en banda J0660)")
    print("• Panel (b): Densidad espacial de cúmulos globulares")
    print("• Ambos comparten la misma escala espacial")
    print("• Permite correlacionar calidad con distribución física")

if __name__ == "__main__":
    main()
