#!/usr/bin/env python3
# individual_gc_plots_CORREGIDO.py - Figuras individuales para paper de cúmulos globulares
# VERSIÓN CORREGIDA con proyección astropy exacta y mejoras para ApJ
# AHORA: Figura espacial incluye todos los GCs con coordenadas (χ² ≤ 2 y χ² > 2)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import gaussian_kde, linregress, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import os
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import angular_separation

# ============================================================================
# CONFIGURACIÓN AVANZADA PARA FIGURAS DE PAPER (ApJ style)
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'serif',
    'mathtext.it': 'serif:italic',
    'mathtext.bf': 'serif:bold',
})

# Paletas de colores profesionales
metallicity_cmap = plt.cm.RdYlBu_r  # Rojo (metálico) a Azul (pobre en metales)
age_cmap = plt.cm.plasma  # Paleta para edad
color_cmap = plt.cm.viridis  # Paleta para colores fotométricos

# Centro de NGC 5128 (J2000)
RA_CENTER, DEC_CENTER = 201.365063, -43.019112  # grados
DISTANCE_MPC = 3.8  # Mpc
DISTANCE_KPC = DISTANCE_MPC * 1000  # kpc
DISTANCE_PC = DISTANCE_MPC * 1e6  # pc

# ============================================================================
# FUNCIONES DE CARGA DE DATOS (MEJORADAS)
# ============================================================================

def load_complete_gc_data(filter_chi2=True):
    """
    Carga datos completos de cúmulos globulares incluyendo fotometría y coordenadas.
    
    Parámetros:
        filter_chi2 (bool): Si True, filtra por best.reduced_chi_square ≤ 2.0.
                            Si False, incluye todos los objetos.
    """
    print("📊 Cargando datos de cúmulos globulares...")
    
    # Cargar datos fotométricos del FITS
    fits_path = 'out/results.fits'
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"No se encontró el archivo: {fits_path}")
    
    print(f"📂 Leyendo archivo FITS: {fits_path}")
    table = Table.read(fits_path, hdu=1)
    df_fits = table.to_pandas()
    
    original_count = len(df_fits)
    
    # Filtrar por chi2 reducido si se solicita
    if filter_chi2:
        chi2_threshold = 2.0
        df_fits = df_fits[df_fits['best.reduced_chi_square'] <= chi2_threshold]
        print(f"  ✓ Filtrado por χ² ≤ {chi2_threshold}: {len(df_fits)}/{original_count} objetos")
    else:
        print(f"  • Sin filtrar por χ²: {len(df_fits)} objetos")
    
    # Calcular [Fe/H] (metalicidad)
    Z_solar = 0.02
    df_fits['FeH'] = np.log10(df_fits['bayes.stellar.metallicity'] / Z_solar)

    # Calcular [Fe/H] error
    metallicity = df_fits['bayes.stellar.metallicity'].values
    metallicity_err = df_fits['bayes.stellar.metallicity_err'].values

    FeH_err = np.full_like(metallicity, np.nan, dtype=float)
    valid_err_mask = (metallicity > 0) & (metallicity_err > 0)
    if valid_err_mask.sum() > 0:
        relative_error = metallicity_err[valid_err_mask] / metallicity[valid_err_mask]
        FeH_err[valid_err_mask] = relative_error / np.log(10)

    # Rellenar NaN con la mediana de los errores válidos
    median_err = np.nanmedian(FeH_err) if np.any(valid_err_mask) else 0.1
    FeH_err = np.where(np.isnan(FeH_err), median_err, FeH_err)

    df_fits['FeH_err'] = FeH_err
    
    # Convertir edad a Gyr
    df_fits['Age'] = df_fits['bayes.stellar.age_m_star'] / 1000.0  # Myr -> Gyr
    
    # Filtrar outliers en metalicidad (suave) - solo si hay suficientes datos
    if len(df_fits) > 10:
        feh_Q1 = df_fits['FeH'].quantile(0.05)
        feh_Q3 = df_fits['FeH'].quantile(0.95)
        feh_IQR = feh_Q3 - feh_Q1
        feh_mask = (df_fits['FeH'] >= feh_Q1 - 2.5*feh_IQR) & (df_fits['FeH'] <= feh_Q3 + 2.5*feh_IQR)
        df_fits = df_fits[feh_mask]
        print(f"  ✓ Filtrado outliers [Fe/H]: {len(df_fits)} objetos")
    
    # Cargar coordenadas del catálogo
    try:
        catalog_path = '../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv'
        if not os.path.exists(catalog_path):
            print(f"⚠️  No se encontró el catálogo: {catalog_path}")
            print("  Usando solo datos del FITS...")
            df_fits['RA'] = np.nan
            df_fits['DEC'] = np.nan
            df_fits['Rgc'] = np.nan
            return df_fits
        
        catalog = pd.read_csv(catalog_path)
        
        # Preparar IDs para merge
        df_fits['id_str'] = df_fits['id'].astype(str).str.strip()
        catalog['T17ID_str'] = catalog['T17ID'].astype(str).str.strip()
        
        # Columnas a cargar
        catalog_columns = ['RAJ2000', 'DEJ2000', 'Rgc', 'T17ID_str',
                           'gmag_corr', 'imag_corr', 'zmag_corr', 'rmag_corr',
                           'gmag', 'imag', 'zmag', 'e_gmag_corr', 'e_imag_corr', 'e_zmag_corr' ]
        
        # Filtrar columnas existentes
        available_columns = [col for col in catalog_columns if col in catalog.columns]
        print(f"  Columnas disponibles en catálogo: {available_columns}")
        
        # Merge con inner join para solo objetos en ambos
        df_complete = pd.merge(
            df_fits,
            catalog[available_columns],
            left_on='id_str',
            right_on='T17ID_str',
            how='inner'
        )
        
        # Renombrar columnas de coordenadas
        rename_dict = {'RAJ2000': 'RA', 'DEJ2000': 'DEC'}
        df_complete = df_complete.rename(columns=rename_dict)
        
        print(f"✓ Merge exitoso: {len(df_complete)} objetos con datos completos")
        
        # Verificar tipos de datos
        print(f"  Tipos de coordenadas:")
        print(f"    RA: {df_complete['RA'].dtype}, DEC: {df_complete['DEC'].dtype}")
        
    except Exception as e:
        print(f"⚠️  Error cargando coordenadas: {e}")
        import traceback
        traceback.print_exc()
        print("  Usando solo datos del FITS...")
        df_complete = df_fits
        df_complete['RA'] = np.nan
        df_complete['DEC'] = np.nan
        df_complete['Rgc'] = np.nan
    
    return df_complete

def calculate_spatial_parameters_corrected(df):
    """
    Calcula parámetros espaciales CORREGIDOS usando astropy para proyección exacta.
    
    Returns:
        radii_kpc, x_kpc, y_kpc, valid_mask
    """
    if 'RA' not in df.columns or 'DEC' not in df.columns:
        print("  ⚠️  No hay columnas RA/DEC en el DataFrame")
        return None, None, None, None
    
    # Máscara de valores válidos
    valid_mask = df['RA'].notna() & df['DEC'].notna()
    
    if not valid_mask.any():
        print("  ⚠️  No hay coordenadas válidas")
        return None, None, None, None
    
    n_valid = valid_mask.sum()
    print(f"  ✓ {n_valid}/{len(df)} objetos con coordenadas válidas")
    
    # Coordenadas del centro de NGC 5128
    center_coord = SkyCoord(
        ra=RA_CENTER * u.deg,
        dec=DEC_CENTER * u.deg,
        distance=DISTANCE_MPC * u.Mpc
    )
    
    # Coordenadas de los clusters
    cluster_coords = SkyCoord(
        ra=df.loc[valid_mask, 'RA'].values * u.deg,
        dec=df.loc[valid_mask, 'DEC'].values * u.deg,
        distance=DISTANCE_MPC * u.Mpc
    )
    
    # 1. Calcular separación angular (método robusto)
    separations_angular = center_coord.separation(cluster_coords)
    
    # 2. Convertir a distancia proyectada en kpc
    radii_kpc = (separations_angular.radian * DISTANCE_KPC)
    
    # 3. Calcular coordenadas cartesianas proyectadas
    delta_ra = (cluster_coords.ra - center_coord.ra).to(u.rad).value
    delta_dec = (cluster_coords.dec - center_coord.dec).to(u.rad).value
    cos_dec_center = np.cos(center_coord.dec.radian)
    
    x_angle = delta_ra * cos_dec_center  # radianes
    y_angle = delta_dec  # radianes
    
    x_kpc = x_angle * DISTANCE_KPC
    y_kpc = y_angle * DISTANCE_KPC
    
    # 4. Verificación de consistencia
    radii_from_xy = np.sqrt(x_kpc**2 + y_kpc**2)
    diff = np.abs(radii_kpc - radii_from_xy)
    rel_diff = diff / radii_kpc * 100  # Porcentaje
    
    print(f"  ✓ Proyección calculada:")
    print(f"    • Radio máximo: {radii_kpc.max():.1f} kpc ({separations_angular.max().to(u.deg):.2f})")
    print(f"    • Radio mínimo: {radii_kpc.min():.1f} kpc")
    print(f"    • Error máximo de proyección: {rel_diff.max():.3f}%")
    print(f"    • Error medio de proyección: {rel_diff.mean():.3f}%")
    
    if rel_diff.max() > 0.1:
        print(f"  ⚠️  Advertencia: Algunos errores de proyección > 0.1%")
        print(f"    Esto puede deberse a clusters muy alejados (>5° del centro)")
    
    return radii_kpc, x_kpc, y_kpc, valid_mask

# ============================================================================
# FIGURA 1: DISTRIBUCIÓN ESPACIAL 2D (TAMAÑO POR EDAD) - CON TODOS LOS GCs
# ============================================================================

def create_spatial_distribution_2d(df, radii_kpc, x_kpc, y_kpc, valid_mask, output_dir='figures'):
    """
    Crea figura de distribución espacial 2D con color por metalicidad
    y tamaño continuo por edad. Incluye todos los objetos con coordenadas,
    distinguiendo calidad del ajuste: círculos (χ² ≤ 2.0) y cuadrados (χ² > 2.0).
    VERSIÓN CON TAMAÑOS DE FUENTE AUMENTADOS.
    """
    print("\n🎨 Creando figura de distribución espacial 2D (incluye todos los GCs)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear figura con tamaño apropiado para ApJ
    fig = plt.figure(figsize=(10, 8))
    
    # Layout con barra de color integrada
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, width_ratios=[15, 1], wspace=0.05, hspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    # Datos para el plot (solo con coordenadas válidas)
    feh = df.loc[valid_mask, 'FeH'].values
    x = x_kpc[valid_mask]
    y = y_kpc[valid_mask]
    
    # Máscaras de calidad usando el nombre correcto de la columna
    chi2_col = 'best.reduced_chi_square'
    good_qual = df.loc[valid_mask, chi2_col] <= 2.0
    bad_qual = ~good_qual
    
    n_good = np.sum(good_qual)
    n_bad = np.sum(bad_qual)
    print(f"   • {n_good} GCs con χ² ≤ 2.0 (círculos)")
    print(f"   • {n_bad} GCs con χ² > 2.0 (cuadrados)")
    
    # Normalizador para metalicidad (excluir outliers extremos)
    feh_min = np.percentile(feh, 1)
    feh_max = np.percentile(feh, 99)
    norm_feh = Normalize(vmin=feh_min, vmax=feh_max)
    
    # Crear ScalarMappable para la barra de color
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=metallicity_cmap, norm=norm_feh)
    sm.set_array([])
    
    # ===== PUNTOS DE MALA CALIDAD (cuadrados grises) =====
    if n_bad > 0:
        ax.scatter(x[bad_qual], y[bad_qual],
                   color='#BBBBBB',          # Gris medio
                   marker='s',                # Cuadrado
                   s=50,                      # Tamaño fijo, más pequeños
                   alpha=0.45,
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=1,
                   label=rf'$\chi^2_r > 2.0$ (N={n_bad})')
    
    # ===== PUNTOS DE BUENA CALIDAD =====
    # Verificar si tenemos datos de edad
    has_age_data = 'Age' in df.columns
    
    if n_good > 0:
        age = df.loc[valid_mask, 'Age'].values[good_qual]
        feh_good = feh[good_qual]
        x_good = x[good_qual]
        y_good = y[good_qual]
        
        age_valid = np.isfinite(age) & (age > 0) & (age <= 15)
        
        if has_age_data and age_valid.sum() > 10:
            age_for_plot = age.copy()
            age_for_plot[~age_valid] = np.median(age[age_valid])
            
            age_min = np.min(age_for_plot[age_valid])
            age_max = np.max(age_for_plot[age_valid])
            
            base_size = 10
            max_size = 70
            log_age = np.log10(age_for_plot)
            log_min = np.log10(age_min)
            log_max = np.log10(age_max)
            
            if log_max > log_min:
                age_norm = (log_age - log_min) / (log_max - log_min)
            else:
                age_norm = np.ones_like(log_age) * 0.5
            
            sizes = base_size  + (max_size - base_size) * (age_norm ** 2)
            
            # Puntos con edad válida
            ax.scatter(x_good[age_valid], y_good[age_valid],
                       c=feh_good[age_valid], cmap=metallicity_cmap, norm=norm_feh,
                       s=sizes[age_valid], alpha=0.85,
                       edgecolors='k', linewidth=0.5, zorder=5)
            
            # Puntos con edad no válida (tamaño fijo pequeño)
            if (~age_valid).sum() > 0:
                ax.scatter(x_good[~age_valid], y_good[~age_valid],
                           c=feh_good[~age_valid], cmap=metallicity_cmap, norm=norm_feh,
                           s=20, alpha=0.3, edgecolors='k', linewidth=0.2, zorder=4)
        else:
            # Sin edades válidas, tamaño fijo
            ax.scatter(x_good, y_good,
                       c=feh_good, cmap=metallicity_cmap, norm=norm_feh,
                       s=40, alpha=0.85, edgecolors='k', linewidth=0.5, zorder=5)
            has_age_data = False
    
    # ===== BARRA DE COLOR =====
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('[Fe/H] (dex)', fontsize=18, labelpad=12)
    cbar.ax.tick_params(labelsize=16)
    
    # ===== LEYENDA DE EDAD (para buenos)         =====
    if has_age_data and n_good > 0 and age_valid.sum() > 10:
        ages_rep = np.array([3.0, 11.0])
        log_ages_rep = np.log10(ages_rep)
        age_norm_rep = (log_ages_rep - log_min) / (log_max - log_min)
        sizes_rep = base_size + (max_size - base_size) * (age_norm_rep ** 2)
        diam_rep = 2 * np.sqrt(sizes_rep / np.pi)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k',
                   markersize=diam_rep[0], label=f'Age ~{ages_rep[0]:.0f} Gyr', linewidth=0),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='k',
                   markersize=diam_rep[1], label=f'Age ~{ages_rep[1]:.0f} Gyr', linewidth=0)
           ]
        age_legend = ax.legend(handles=legend_elements, loc='upper right',
                               fontsize=12, title='Larger symbols = older clusters',
                               title_fontsize=12, framealpha=0.9)
        ax.add_artist(age_legend)
    
    
    # ===== ELIPSES CONCÉNTRICAS =====
    max_data_radius = np.max(radii_kpc)
    possible_radii = [10, 30, 60, 100, 150, 200, 250]
    ellipse_radii = [r for r in possible_radii if r <= max_data_radius * 1.1]
    if len(ellipse_radii) < 3:
        for p in [25, 50, 75]:
            r = np.percentile(radii_kpc, p)
            if r > 5:
                ellipse_radii.append(int(np.round(r, -1)))
    ellipse_radii = sorted(set(ellipse_radii))
    
    for i, r in enumerate(ellipse_radii):
        ellipse = Ellipse((0, 0), width=2*r, height=2*r,
                          edgecolor='gray', facecolor='none',
                          linestyle='--' if i % 2 == 0 else '-.',
                          linewidth=1.0, alpha=0.6, zorder=1)
        ax.add_patch(ellipse)
        
        angle = i * (2*np.pi / len(ellipse_radii))
        label_x = r * np.cos(angle) * 1.05
        label_y = r * np.sin(angle) * 1.05
        
        if -np.pi/4 <= angle <= np.pi/4:
            ha, va = 'left', 'center'
        elif np.pi/4 < angle <= 3*np.pi/4:
            ha, va = 'center', 'bottom'
        elif 3*np.pi/4 < angle <= 5*np.pi/4:
            ha, va = 'right', 'center'
        else:
            ha, va = 'center', 'top'
        
        ax.text(label_x, label_y, f'{r:.0f} kpc',
                fontsize=12, ha=ha, va=va, color='gray', alpha=0.8,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')],
                zorder=10)
    
    # Marcar centro
    ax.scatter(0, 0, color='gold', s=300, marker='*',
               edgecolors='darkorange', linewidth=1.5, zorder=20)
    
    # Círculo central
    center_circle = Circle((0, 0), 5, color='gold', alpha=0.15, zorder=15)
    ax.add_artist(center_circle)
    
    # Líneas de ejes
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=2)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=2)
    
    # Configuración de ejes
    ax.set_xlabel(r'$\Delta$X (kpc)', fontsize=22)
    ax.set_ylabel(r'$\Delta$Y (kpc)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    max_abs = max(np.abs(x).max(), np.abs(y).max()) * 1.5
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_aspect('equal')
    
    stats_text = f'Total GCs with coordinates: {len(feh)}\n'
    stats_text += f'● Good quality (χ² ≤ 2): {n_good}\n'
    stats_text += f'■ Poor quality (χ² > 2): {n_bad}\n'
    stats_text += f'R$_{{\\mathrm{{max}}}}$ = {max_data_radius:.0f} kpc'
    
    # ax.text(0.9, 0.02, stats_text, transform=ax.transAxes,
    #         fontsize=14, verticalalignment='bottom', linespacing=1.5,
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
    #                  edgecolor='gray', linewidth=0.5), zorder=10)

    # ===== LEYENDA DE CALIDAD UNIFICADA =====
    qual_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='k', markersize=10,
               label=rf'Good quality ($\chi^2_r \leq 2.0$): {n_good}'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#BBBBBB',
               markeredgecolor='k', markersize=10, 
               label=rf'Poor quality ($\chi^2_r > 2.0$): {n_bad}'),
        Line2D([0], [0], marker='None', color='w', label=f'Total GCs: {len(feh)}'),
        Line2D([0], [0], marker='None', color='w', label=f'R$_{{\\mathrm{{max}}}}$ = {max_data_radius:.0f} kpc')
        ]
    qual_legend = ax.legend(handles=qual_elements, loc='lower left',
                        fontsize=12, framealpha=0.9, 
                        title_fontsize=12)
    # Importante: si ya tienes otras leyendas (por ejemplo, de edad), debes añadir esta manualmente.
    # Si no hay otras leyendas, puedes usar directamente ax.legend(...) sin add_artist.
    ax.add_artist(qual_legend)
    
    # Flechas de orientación
    arrow_length = max_abs * 0.1
    if max_abs > 50:
        arrow_x, arrow_y = max_abs * 0.85, -max_abs * 0.85
        ax.arrow(arrow_x, arrow_y, arrow_length, 0,
                 head_width=arrow_length*0.3, head_length=arrow_length*0.2,
                 fc='k', ec='k', alpha=0.7, zorder=10)
        ax.text(arrow_x + arrow_length*1.3, arrow_y, 'E',
                fontsize=14, ha='center', va='center', alpha=0.7, zorder=10)
        ax.arrow(arrow_x, arrow_y, 0, arrow_length,
                 head_width=arrow_length*0.3, head_length=arrow_length*0.2,
                 fc='k', ec='k', alpha=0.7, zorder=10)
        ax.text(arrow_x, arrow_y + arrow_length*1.3, 'N',
                fontsize=14, ha='center', va='center', alpha=0.7, zorder=10)
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    # Guardar
    output_path = f'{output_dir}/Figure_Spatial_Distribution_2D.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura espacial 2D guardada: {output_path}")
    
    return fig

def create_radial_gradient_scatter(df, radii_kpc, valid_mask, df_all=None, radii_kpc_all=None, valid_mask_all=None, output_dir='figures'):
    """
    Figura 2: Gradiente radial de metalicidad (scatter plot).
    Si se proporcionan df_all, radii_kpc_all y valid_mask_all, se añaden los puntos de mala calidad (χ² > 2) en gris como fondo.
    El ajuste lineal se calcula solo con la muestra de buena calidad (df).
    """
    print("\n🎨 Creando figura del gradiente radial (scatter)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ===== Puntos de mala calidad (fondo) si se proporcionan =====
    if df_all is not None and radii_kpc_all is not None and valid_mask_all is not None:
        feh_bad = df_all.loc[valid_mask_all, 'FeH'].values
        radii_bad = radii_kpc_all[valid_mask_all]
        chi2_bad = df_all.loc[valid_mask_all, 'best.reduced_chi_square'].values
        
        # Filtrar solo los de mala calidad (χ² > 2)
        bad_mask = chi2_bad > 2.0
        if bad_mask.sum() > 0:
            log_radii_bad = np.log10(radii_bad[bad_mask])
            ax.scatter(log_radii_bad, feh_bad[bad_mask],
                       color='#BBBBBB', marker='s', s=30, alpha=0.3,
                       edgecolors='black', linewidth=0.2, zorder=1,
                       label=rf'$\chi^2_r > 2.0$ (N={bad_mask.sum()})')
    
    # ===== Puntos de buena calidad =====
    feh_valid = df.loc[valid_mask, 'FeH'].values
    radii_valid = radii_kpc[valid_mask]
    
    if len(radii_valid) < 50:
        print(f"⚠️  Solo {len(radii_valid)} puntos válidos, insuficiente para gradiente")
        plt.close(fig)
        return None
    
    log_radii = np.log10(radii_valid)
    
    ax.scatter(log_radii, feh_valid, 
               s=45, alpha=0.7, color='steelblue',
               edgecolors='navy', linewidth=0.3, zorder=3,
               label=rf'$\chi^2_r \leq 2.0$ (N={len(feh_valid)})')
    
    # Ajuste lineal (solo buena calidad)
    mask_finite = np.isfinite(log_radii) & np.isfinite(feh_valid)
    
    if mask_finite.sum() > 10:
        slope, intercept, r_value, p_value, std_err = linregress(
            log_radii[mask_finite], feh_valid[mask_finite]
        )
        
        x_fit = np.linspace(log_radii.min(), log_radii.max(), 100)
        y_fit = intercept + slope * x_fit
        
        # Error en la predicción
        n = mask_finite.sum()
        x_mean = log_radii[mask_finite].mean()
        Sxx = np.sum((log_radii[mask_finite] - x_mean)**2)
        s_resid = std_err * np.sqrt(Sxx)          # error estándar residual
        y_err = s_resid * np.sqrt(1/n + (x_fit - x_mean)**2 / Sxx)
        
        # Línea de ajuste
        ax.plot(x_fit, y_fit, 'k-.', linewidth=2.5, alpha=0.9, 
                label=f'Linear fit (good quality)', zorder=4)
        
        # Banda de incertidumbre (1σ)
        ax.fill_between(x_fit, y_fit - y_err, y_fit + y_err,
                        color='gray', alpha=0.2, zorder=2)
        
        # Calcular también correlación de Spearman
        rho, p_rho = spearmanr(log_radii[mask_finite], feh_valid[mask_finite])
        
        # Estadísticas para mostrar
        stats_text = f'Slope = {slope:.3f} $\\pm$ {std_err:.3f} dex/dex\n'
        stats_text += f'Pearson $r$ = {r_value:.3f}\n'
        stats_text += f'Spearman $\\rho$ = {rho:.3f}\n'
        
        if p_value < 0.001:
            stats_text += f'$p$ < 0.001'
        else:
            stats_text += f'$p$ = {p_value:.3f}'
        
        ax.text(0.3, 0.05, stats_text,
                transform=ax.transAxes, fontsize=15,
                verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                         edgecolor='gray', pad=0.3))
        
        print(f"  Resultados del ajuste:")
        print(f"    • Pendiente: {slope:.3f} ± {std_err:.3f} dex/dex")
        print(f"    • Pearson r = {r_value:.3f}, p = {p_value:.4f}")
        print(f"    • Spearman ρ = {rho:.3f}, p = {p_rho:.4f}")
        
        gradient_result = {
            'slope': float(slope),
            'slope_err': float(std_err),
            'intercept': float(intercept),
            'r_pearson': float(r_value),
            'p_pearson': float(p_value),
            'rho_spearman': float(rho),
            'p_spearman': float(p_rho),
            'n_points': int(mask_finite.sum())
        }
    else:
        print("⚠️  No hay suficientes puntos finitos para ajuste lineal")
        gradient_result = None
    
    # Ajustar límites del eje Y con margen extra
    if df_all is not None:
        # Usar todos los puntos para los límites
        feh_all_valid = df_all.loc[valid_mask_all, 'FeH'].values
        ymin, ymax = np.nanmin(feh_all_valid), np.nanmax(feh_all_valid)
    else:
        ymin, ymax = np.nanmin(feh_valid), np.nanmax(feh_valid)
    
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.52 * yrange, ymax + 0.45 * yrange)
    
    # Configurar ejes
    ax.set_xlabel(r'$\log_{10}(R / \mathrm{kpc})$', fontsize=22)
    ax.set_ylabel('[Fe/H] (dex)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Leyenda
    ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    # Guardar figura
    output_path = f'{output_dir}/Radial_Gradient_Scatter.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura 2 (gradiente radial scatter) guardada: {output_path}")
    
    return gradient_result

# ============================================================================
# FIGURA ADICIONAL: GRADIENTE RADIAL SEPARADO POR EDAD (con ajustes lineales)
# ============================================================================

def create_radial_gradient_by_age(df_all, radii_kpc_all, valid_mask_all, output_dir='figures'):
    """
    Crea figura de gradiente radial separando por edad (≤8 Gyr y >8 Gyr) solo para objetos con χ² ≤ 2.
    Incluye puntos de mala calidad (χ² > 2) en gris como referencia.
    Ajustes lineales para cada grupo de edad (buena calidad).
    """
    print("\n🎨 Creando figura de gradiente radial separado por edad...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    valid = valid_mask_all
    feh_all = df_all.loc[valid, 'FeH'].values
    radii_all = radii_kpc_all[valid]
    age_all = df_all.loc[valid, 'Age'].values
    chi2_all = df_all.loc[valid, 'best.reduced_chi_square'].values
    
    if len(radii_all) < 50:
        print(f"⚠️  Solo {len(radii_all)} puntos válidos, no se crea figura.")
        return None
    
    log_radii = np.log10(radii_all)
    
    good_qual = chi2_all <= 2.0
    bad_qual = ~good_qual
    age_threshold = 8.0
    good_young = good_qual & (age_all <= age_threshold)
    good_old   = good_qual & (age_all > age_threshold)
    
    n_bad = bad_qual.sum()
    n_young = good_young.sum()
    n_old = good_old.sum()
    
    print(f"   • Mala calidad (χ²>2): {n_bad}")
    print(f"   • Buena calidad, edad ≤{age_threshold} Gyr: {n_young}")
    print(f"   • Buena calidad, edad >{age_threshold} Gyr: {n_old}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Puntos de mala calidad
    if n_bad > 0:
        ax.scatter(log_radii[bad_qual], feh_all[bad_qual],
                   color='#BBBBBB', marker='s', s=40, alpha=0.4,
                   edgecolors='black', linewidth=0.3, zorder=1,
                   label=rf'$\chi^2_r > 2.0$ (N={n_bad})')
    
    # Jóvenes (≤8 Gyr)
    if n_young > 0:
        ax.scatter(log_radii[good_young], feh_all[good_young],
                   color='darkorange', marker='o', s=60, alpha=0.8,
                   edgecolors='darkred', linewidth=0.5, zorder=3,
                   label=rf'$\chi^2_r \leq 2.0$, Age $\leq$ {age_threshold} Gyr (N={n_young})')
    
    # Viejos (>8 Gyr)
    if n_old > 0:
        ax.scatter(log_radii[good_old], feh_all[good_old],
                   color='steelblue', marker='o', s=60, alpha=0.8,
                   edgecolors='navy', linewidth=0.5, zorder=3,
                   label=rf'$\chi^2_r \leq 2.0$, Age $>$ {age_threshold} Gyr (N={n_old})')
    
    # Ajustes lineales para cada grupo (solo buena calidad)
    x_fit = np.linspace(log_radii.min(), log_radii.max(), 100)
    
    if n_young > 5:
        slope_y, intercept_y, r_y, p_y, err_y = linregress(log_radii[good_young], feh_all[good_young])
        y_fit_young = intercept_y + slope_y * x_fit
        ax.plot(x_fit, y_fit_young, color='darkorange', linestyle='--', linewidth=2.5, alpha=0.9,
                label=rf'Young fit: slope = {slope_y:.3f} $\pm$ {err_y:.3f}')
        print(f"   • Jóvenes: pendiente = {slope_y:.3f} ± {err_y:.3f}, p = {p_y:.4f}")
    
    if n_old > 5:
        slope_o, intercept_o, r_o, p_o, err_o = linregress(log_radii[good_old], feh_all[good_old])
        y_fit_old = intercept_o + slope_o * x_fit
        ax.plot(x_fit, y_fit_old, color='steelblue', linestyle='--', linewidth=2.5, alpha=0.9,
                label=rf'Old fit: slope = {slope_o:.3f} $\pm$ {err_o:.3f}')
        print(f"   • Viejos: pendiente = {slope_o:.3f} ± {err_o:.3f}, p = {p_o:.4f}")
    
    # Configurar ejes
    x_min, x_max = log_radii.min() - 0.05, log_radii.max() + 0.05
    y_min, y_max = feh_all.min() - 0.8, feh_all.max() + 0.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel(r'$\log_{10}(R / \mathrm{kpc})$', fontsize=22)
    ax.set_ylabel('[Fe/H] (dex)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=14, loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    output_path = f'{output_dir}/Radial_Gradient_by_Age.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura guardada: {output_path}")
    return fig

# ============================================================================
# FIGURA 3: GRADIENTE RADIAL BINNEADO
# ============================================================================

def create_radial_gradient_binned(df, radii_kpc, valid_mask, output_dir='figures'):
    """
    Figura 3: Gradiente radial binneado.
    Versión para paper de ApJ.
    (Usa solo objetos con χ² ≤ 2, según df)
    """
    print("\n🎨 Creando figura del gradiente radial binneado...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Datos válidos
    feh_valid = df.loc[valid_mask, 'FeH'].values
    radii_valid = radii_kpc[valid_mask]
    
    if len(radii_valid) < 100:
        print(f"⚠️  Solo {len(radii_valid)} puntos, considerando menos bins")
        n_bins = 5
    else:
        n_bins = 8
    
    log_radii = np.log10(radii_valid)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear bins con igual número de objetos (percentiles)
    try:
        bin_edges = np.percentile(log_radii, np.linspace(0, 100, n_bins + 1))
    except:
        # Si hay valores idénticos, usar bins equiespaciados
        bin_edges = np.linspace(log_radii.min(), log_radii.max(), n_bins + 1)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Estadísticas por bin
    feh_means = np.zeros(n_bins)
    feh_medians = np.zeros(n_bins)
    feh_stds = np.zeros(n_bins)
    feh_sems = np.zeros(n_bins)  # Standard Error of the Mean
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (log_radii >= bin_edges[i]) & (log_radii < bin_edges[i+1])
        if mask.sum() > 0:
            feh_bin = feh_valid[mask]
            feh_means[i] = np.mean(feh_bin)
            feh_medians[i] = np.median(feh_bin)
            feh_stds[i] = np.std(feh_bin)
            feh_sems[i] = feh_stds[i] / np.sqrt(mask.sum())
            counts[i] = mask.sum()
        else:
            feh_means[i] = np.nan
            feh_medians[i] = np.nan
            feh_stds[i] = np.nan
            feh_sems[i] = np.nan
            counts[i] = 0
    
    # Filtrar bins vacíos
    valid_bins = ~np.isnan(feh_means)
    
    if valid_bins.sum() < 3:
        print("⚠️  Menos de 3 bins válidos, no se crea figura")
        return
    
    # Plot datos binneados
    ax.errorbar(bin_centers[valid_bins], feh_means[valid_bins], 
                yerr=feh_sems[valid_bins],
                fmt='s', color='crimson', ecolor='crimson', elinewidth=2,
                capsize=4, capthick=2, markersize=8, markeredgewidth=1.5,
                markeredgecolor='darkred', zorder=5,
                label='Binned mean ± SEM')
    
    # También mostrar la mediana
    ax.scatter(bin_centers[valid_bins], feh_medians[valid_bins],
               marker='D', color='darkblue', s=60, alpha=0.7,
               edgecolors='navy', linewidth=1, zorder=4,
               label='Binned median')
    
    # Añadir número de objetos en cada bin
    for i in range(n_bins):
        if counts[i] > 0:
            ax.annotate(f'N={int(counts[i])}',
                       xy=(bin_centers[i], feh_means[i]),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='normal',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.7, edgecolor='gray'))
    
    # Ajuste lineal a los datos originales (para comparación)
    mask_finite = np.isfinite(log_radii) & np.isfinite(feh_valid)
    if mask_finite.sum() > 10:
        slope, intercept, r_value, p_value, _ = linregress(
            log_radii[mask_finite], feh_valid[mask_finite]
        )
        
        x_fit = np.linspace(log_radii.min(), log_radii.max(), 100)
        y_fit = intercept + slope * x_fit
        
        ax.plot(x_fit, y_fit, 'k-.', linewidth=2.0, alpha=0.7,
                label=f'Linear fit to all data\n(slope = {slope:.3f})')
    
    # Configurar ejes
    ax.set_xlabel(r'$\log_{10}(R / \mathrm{kpc})$', fontsize=22)
    ax.set_ylabel('Mean [Fe/H] (dex)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Información estadística
    stats_text = f'Number of bins = {n_bins}\n'
    if 'slope' in locals():
        stats_text += f'Linear slope = {slope:.3f} dex/dex\n'
        stats_text += f'$r$ = {r_value:.3f}, $p$ = {p_value:.3e}' if p_value < 0.001 else f'$p$ = {p_value:.3f}'
    
    ax.text(0.05, 0.2, stats_text,
            transform=ax.transAxes, fontsize=15,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='gray', pad=0.3))
    
    # Leyenda
    ax.legend(
     fontsize=15,
     loc='center left',
     bbox_to_anchor=(0.03, 0.35),
     framealpha=0.9
    )
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/Radial_Gradient_Binned.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura 3 (gradiente radial binneado) guardada: {output_path}")

# ============================================================================
# FIGURA 4: METALICIDAD VS COLOR (g-i)
# ============================================================================

def create_metallicity_vs_gi(df, output_dir='figures'):
    """
    Crea figura de metalicidad vs color (g-i) usando magnitudes corregidas.
    (Usa solo objetos con χ² ≤ 2, según df)
    """
    print("\n🎨 Creando figura de metalicidad vs (g-i)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar columnas
    required_cols = ['gmag_corr', 'imag_corr', 'FeH', 'Age']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Columnas faltantes: {missing_cols}")
        return None
    
    # Calcular color (g-i)
    df = df.copy()
    df['g_i'] = df['gmag_corr'] - df['imag_corr']
    
    # Filtrar valores extremos (suave)
    g_i_Q1 = df['g_i'].quantile(0.05)
    g_i_Q3 = df['g_i'].quantile(0.95)
    g_i_IQR = g_i_Q3 - g_i_Q1
    g_i_mask = (df['g_i'] >= g_i_Q1 - 2*g_i_IQR) & (df['g_i'] <= g_i_Q3 + 2*g_i_IQR)
    
    df_filtered = df[g_i_mask].copy()
    
    if len(df_filtered) < 50:
        print(f"⚠️  Solo {len(df_filtered)} puntos después de filtrar")
        return None
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    feh = df_filtered['FeH'].values
    age = df_filtered['Age'].values
    g_i = df_filtered['g_i'].values
    
    # Scatter con color por edad
    scatter = ax.scatter(g_i, feh, c=age, cmap=age_cmap,
                        s=45, alpha=0.85, edgecolors='k', linewidth=0.4,
                        vmin=5, vmax=12)  # Rango típico para GCs viejos
    
    # Barra de color para edad
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age (Gyr)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Ajuste lineal
    mask_finite = np.isfinite(g_i) & np.isfinite(feh)
    if mask_finite.sum() > 10:
        slope, intercept, r_value, p_value, std_err = linregress(
            g_i[mask_finite], feh[mask_finite]
        )
        
        x_fit = np.linspace(g_i.min(), g_i.max(), 100)
        y_fit = intercept + slope * x_fit
        
        ax.plot(x_fit, y_fit, 'k-', linewidth=2.0, alpha=0.8,
                label=f'Linear fit: r = {r_value:.3f}')
        
        # Calcular también ajuste para poblaciones viejas (>8 Gyr)
        old_mask = mask_finite & (age >= 8)
        if old_mask.sum() > 10:
            slope_old, intercept_old, r_old, p_old, _ = linregress(
                g_i[old_mask], feh[old_mask]
            )
            y_fit_old = intercept_old + slope_old * x_fit
            ax.plot(x_fit, y_fit_old, 'k--', linewidth=1.5, alpha=0.6,
                    label=f'Old GCs (>8 Gyr): r = {r_old:.3f}')
    
    # Configuración
    ax.set_xlabel(r'$(g - i)_0$', fontsize=16)
    ax.set_ylabel(r'[Fe/H] (dex)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Estadísticas
    stats_text = f'N = {len(feh)}\n'
    stats_text += f'⟨g-i⟩ = {g_i.mean():.3f} ± {g_i.std():.3f}\n'
    stats_text += f'⟨[Fe/H]⟩ = {feh.mean():.3f} ± {feh.std():.3f}'
    
    if 'r_value' in locals():
        stats_text += f'\nCorrelation: r = {r_value:.3f}\n'
        if p_value < 0.001:
            stats_text += f'p < 0.001'
        else:
            stats_text += f'p = {p_value:.3f}'
        
        # Mostrar ecuación de la recta
        eq_text = f'[Fe/H] = {slope:.3f}·(g-i) + {intercept:.3f}'
        ax.text(0.05, 0.95, eq_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                         edgecolor='gray'))
    
    ax.text(0.05, 0.05, stats_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                     edgecolor='gray', linewidth=0.5))
    
    # Leyenda
    if 'r_value' in locals():
        ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/Figure_Metallicity_vs_g-i.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura 4 (metalicidad vs g-i) guardada: {output_path}")
    
    return fig

# ============================================================================
# FIGURA 5: METALICIDAD VS COLOR (g-z) -- with marginal histograms
# ============================================================================

def create_metallicity_vs_gz_with_marginals(df, output_dir='figures'):
    """
    Figura de metalicidad vs color (g-z) con histogramas marginales por edad
    y barras de error en [Fe/H] y en (g-z), coloreadas por grupo de edad.
    """
    print("\n🎨 Creando figura de metalicidad vs (g-z) con errores...")
    os.makedirs(output_dir, exist_ok=True)

    required_cols = ['gmag_corr', 'zmag_corr', 'FeH', 'FeH_err', 'Age',
                     'e_gmag_corr', 'e_zmag_corr']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Columnas faltantes: {missing}")
        return None

    df = df.copy()
    df['g_z'] = df['gmag_corr'] - df['zmag_corr']

    # Error en g-z: propagación
    valid_err = df['e_gmag_corr'].notna() & df['e_zmag_corr'].notna()
    df['g_z_err'] = np.nan
    if valid_err.any():
        df.loc[valid_err, 'g_z_err'] = np.sqrt(
            df.loc[valid_err, 'e_gmag_corr']**2 + df.loc[valid_err, 'e_zmag_corr']**2
        )
    median_gz_err = df['g_z_err'].median()
    df['g_z_err'] = df['g_z_err'].fillna(median_gz_err if not np.isnan(median_gz_err) else 0.02)

    # Filtrado suave de valores extremos en g_z
    q1 = df['g_z'].quantile(0.05)
    q3 = df['g_z'].quantile(0.95)
    iqr = q3 - q1
    mask = (df['g_z'] >= q1 - 2*iqr) & (df['g_z'] <= q3 + 2*iqr)
    df_filt = df[mask].copy()

    if len(df_filt) < 50:
        print(f"⚠️  Solo {len(df_filt)} puntos tras filtrar")
        return None

    # Grupos de edad
    df_filt['Age group'] = np.where(df_filt['Age'] <= 8, 'Young (≤ 8 Gyr)', 'Old (> 8 Gyr)')
    young_mask = df_filt['Age'] <= 8
    old_mask   = df_filt['Age'] > 8

    # JointGrid
    g = sns.JointGrid(data=df_filt, x='g_z', y='FeH')
    colors = {'Young (≤ 8 Gyr)': 'darkorange', 'Old (> 8 Gyr)': 'steelblue'}

    for grp_name, mask in [('Young (≤ 8 Gyr)', young_mask),
                           ('Old (> 8 Gyr)', old_mask)]:
        if mask.sum() == 0:
            continue
        x = df_filt.loc[mask, 'g_z']
        y = df_filt.loc[mask, 'FeH']
        xerr = df_filt.loc[mask, 'g_z_err']
        yerr = np.clip(df_filt.loc[mask, 'FeH_err'], 0, 0.4)   # recorte visual

        # Barras de error sutiles pero del color del grupo
        g.ax_joint.errorbar(x, y,
                            xerr=xerr, yerr=yerr,
                            fmt='none', ecolor=colors[grp_name],
                            alpha=0.22, capsize=0, elinewidth=0.6,
                            zorder=1)
        # Puntos encima
        g.ax_joint.scatter(x, y, s=60, alpha=0.85,
                           edgecolor='k', linewidth=0.3,
                           color=colors[grp_name], label=grp_name, zorder=3)

    # Marginales
    sns.histplot(data=df_filt, x='g_z', hue='Age group', ax=g.ax_marg_x,
                 element="step", kde=True, bins=20, alpha=0.6,
                 palette=colors)
    sns.histplot(data=df_filt, y='FeH', hue='Age group', ax=g.ax_marg_y,
                 element="step", kde=True, bins=20, alpha=0.6,
                 palette=colors)

    # Eliminar leyendas duplicadas de los marginales
    if g.ax_marg_x.get_legend() is not None:
        g.ax_marg_x.get_legend().remove()
    if g.ax_marg_y.get_legend() is not None:
        g.ax_marg_y.get_legend().remove()

    # Tamaños de fuente mejorados
    g.ax_joint.tick_params(labelsize=14)
    g.ax_marg_x.tick_params(labelsize=12)
    g.ax_marg_y.tick_params(labelsize=12)
    g.ax_joint.xaxis.label.set_size(18)
    g.ax_joint.yaxis.label.set_size(18)

    # Leyenda del panel central
    g.ax_joint.legend(fontsize=12, loc="lower right",  title=None)

    g.set_axis_labels(r'$(g - z)_0$', '[Fe/H] (dex)')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'Figure_Metallicity_vs_g-z_marginals.pdf')
    g.savefig(output_path, dpi=600)
    plt.close()
    print(f"✓ Figura guardada: {output_path}")
    return g


# ============================================================================
# FIGURA 5: METALICIDAD VS r-MAG -- marginals
# ============================================================================

def create_metallicity_vs_r_with_marginals(df, output_dir='figures'):
    """
    Crea figura de metalicidad vs magnitud r con histogramas marginales por edad.
    Solo objetos con χ² ≤ 2.
    """
    print("\n🎨 Creando figura de metalicidad vs r (con marginales)...")
    os.makedirs(output_dir, exist_ok=True)

    required_cols = ['rmag_corr', 'FeH', 'Age']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️  Columnas faltantes: {missing}")
        return None

    df = df.copy()
    df['r_mag'] = df['rmag_corr']

    # Filtrar valores extremos (suave)
    q1 = df['r_mag'].quantile(0.05)
    q3 = df['r_mag'].quantile(0.95)
    iqr = q3 - q1
    mask = (df['r_mag'] >= q1 - 2*iqr) & (df['r_mag'] <= q3 + 2*iqr)
    df_filt = df[mask].copy()

    if len(df_filt) < 50:
        print(f"Solo {len(df_filt)} puntos tras filtrar")
        return None

    # Crear etiquetas de edad
    df_filt['Age group'] = np.where(df_filt['Age'] <= 8, 'Young (≤ 8 Gyr)', 'Old (> 8 Gyr)')

    # Joint plot con histogramas marginales
    g = sns.JointGrid(data=df_filt, x='r_mag', y='FeH', hue='Age group',
                      palette={'Young (≤ 8 Gyr)': 'darkorange', 'Old (> 8 Gyr)': 'steelblue'})
    g.plot_joint(sns.scatterplot, s=60, alpha=0.8, edgecolor='k', linewidth=0.3)
    g.plot_marginals(sns.histplot, element="step", kde=True, bins=20, alpha=0.6)

    g.set_axis_labels(r'$r_0$ (mag)', '[Fe/H] (dex)')
    g.ax_joint.legend(fontsize=12, title=None)
    g.ax_joint.get_legend().remove()

    # Invertir eje x (magnitudes más brillantes a la izquierda)
    g.ax_joint.invert_xaxis()

    # Estadísticas en el gráfico
    # stats = f'N = {len(df_filt)}\n⟨r₀⟩ = {df_filt["r_mag"].mean():.2f} ± {df_filt["r_mag"].std():.2f}\n⟨[Fe/H]⟩ = {df_filt["FeH"].mean():.3f} ± {df_filt["FeH"].std():.3f}'
    # g.ax_joint.text(0.05, 0.95, stats, transform=g.ax_joint.transAxes,
    #                 fontsize=11, verticalalignment='top',
    #                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # g.ax_joint.margins(x=0.05, y=0)  # margen solo en x
    # ylim = g.ax_joint.get_ylim()
    # extra_bottom = 0.05 * (ylim[1] - ylim[0])  # 5% extra abajo
    # g.ax_joint.set_ylim(ylim[0] - extra_bottom, ylim[1] * 3.1)  # pequeño margen arriba también (2%)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'Figure_Metallicity_vs_r_marginals.pdf')
    g.savefig(output_path, dpi=600)
    plt.close()
    print(f"✓ Figura guardada: {output_path}")
    return g

# ============================================================================
# FIGURA 6: DISTRIBUCIÓN DE METALICIDAD (MDF)
# ============================================================================

def create_metallicity_distribution(df, output_dir='figures'):
    """
    Crea figura de distribución de metalicidad (MDF).
    (Usa solo objetos con χ² ≤ 2, según df)
    """
    print("\n🎨 Creando figura de distribución de metalicidad...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    feh = df['FeH'].values
    
    # Filtrar valores finitos
    feh_finite = feh[np.isfinite(feh)]
    
    if len(feh_finite) < 50:
        print(f"⚠️  Solo {len(feh_finite)} valores finitos")
        return None
    
    # Crear figura con dos paneles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ========== PANEL A: Histograma con KDE ==========
    
    # Determinar número óptimo de bins (regla de Freedman-Diaconis)
    iqr = np.percentile(feh_finite, 75) - np.percentile(feh_finite, 25)
    bin_width = 2 * iqr / (len(feh_finite) ** (1/3))
    n_bins = int((feh_finite.max() - feh_finite.min()) / bin_width)
    n_bins = min(max(n_bins, 20), 50)  # Limitar entre 20 y 50
    
    # Histograma
    ax1.hist(feh_finite, bins=n_bins, density=True,
             alpha=0.5, color='steelblue',
             edgecolor='navy', linewidth=0.8,
             label=f'N = {len(feh_finite)} GCs')
    
    # KDE
    try:
        kde = gaussian_kde(feh_finite, bw_method=0.15)
        x_kde = np.linspace(feh_finite.min() - 0.3, feh_finite.max() + 0.3, 500)
        y_kde = kde(x_kde)
        
        ax1.plot(x_kde, y_kde, 'darkblue', linewidth=2.0, alpha=0.9, label='KDE')
        
        # Líneas de estadísticas
        mean_feh = feh_finite.mean()
        median_feh = np.median(feh_finite)
        std_feh = feh_finite.std()
        
        ax1.axvline(mean_feh, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.8, label=f'Mean: {mean_feh:.3f}')
        ax1.axvline(median_feh, color='green', linestyle='-.', linewidth=1.5,
                    alpha=0.8, label=f'Median: {median_feh:.3f}')
        
        # Detectar bimodalidad (si existe)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(y_kde, height=0.1)
        
        if len(peaks) >= 2:
            peak_heights = properties['peak_heights']
            sorted_indices = np.argsort(peak_heights)[::-1]
            sorted_peaks = peaks[sorted_indices]
            
            for i, peak_idx in enumerate(sorted_peaks[:2]):
                peak_x = x_kde[peak_idx]
                peak_y = y_kde[peak_idx]
                ax1.plot(peak_x, peak_y, 'o', color='orange', markersize=8)
                ax1.annotate(f'Peak {i+1}\n[Fe/H] = {peak_x:.2f}',
                            xy=(peak_x, peak_y),
                            xytext=(10, 20*(i+1)), textcoords='offset points',
                            ha='left', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
                
    except:
        print("  ⚠️  gaussian_kde no disponible, omitiendo KDE")
        mean_feh = feh_finite.mean()
        median_feh = np.median(feh_finite)
        std_feh = feh_finite.std()
        
        ax1.axvline(mean_feh, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.8, label=f'Mean: {mean_feh:.3f}')
        ax1.axvline(median_feh, color='green', linestyle='-.', linewidth=1.5,
                    alpha=0.8, label=f'Median: {median_feh:.3f}')
    
    ax1.set_xlabel('[Fe/H] (dex)', fontsize=14)
    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # Estadísticas detalladas
    stats_text = f'Mean = {mean_feh:.3f} ± {std_feh:.3f}\n'
    stats_text += f'Median = {median_feh:.3f}\n'
    stats_text += f'Std Dev = {std_feh:.3f}\n'
    stats_text += f'Skewness = {pd.Series(feh_finite).skew():.3f}\n'
    stats_text += f'Kurtosis = {pd.Series(feh_finite).kurtosis():.3f}\n'
    stats_text += f'Range = [{feh_finite.min():.2f}, {feh_finite.max():.2f}]'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                     edgecolor='gray', linewidth=0.5))
    
    # ========== PANEL B: Boxplot y Violín ==========
    
    try:
        from scipy.stats import gaussian_kde as gkde
        density = gkde(feh_finite, bw_method=0.15)
        xs = np.linspace(feh_finite.min() - 0.3, feh_finite.max() + 0.3, 200)
        ax2.fill_betweenx(xs, 0, density(xs), alpha=0.4, color='steelblue', label='Density')
    except:
        print("  ⚠️  No se pudo crear el plot de densidad")
        ax2.hist(feh_finite, bins=30, orientation='horizontal', 
                alpha=0.4, color='steelblue', density=True, label='Density')
    
    # Boxplot horizontal
    bp = ax2.boxplot(feh_finite, vert=False, positions=[0], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = np.percentile(feh_finite, percentiles)
    
    for p_val, p_label in zip(percentile_values, percentiles):
        ax2.axvline(p_val, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
        ax2.text(p_val, 0.5, f'{p_label}%', ha='center', va='center',
                fontsize=9, color='gray', transform=ax2.get_xaxis_transform())
    
    ax2.set_xlabel('[Fe/H] (dex)', fontsize=14)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.2, linestyle=':', axis='x')
    
    # Título general
    plt.suptitle('Metallicity Distribution Function of NGC 5128 Globular Clusters', fontsize=16)
    plt.tight_layout()
    
    output_path = f'{output_dir}/Figure6_Metallicity_Distribution.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura 6 (distribución metalicidad) guardada: {output_path}")
    
    return fig

# ============================================================================
# FIGURA 7: EDAD VS METALICIDAD CON COLOR BAR (g-i)
# ============================================================================

def create_age_vs_metallicity(df, output_dir='figures'):
    """
    Crea figura de edad vs metalicidad con color bar por (g-i).
    (Usa solo objetos con χ² ≤ 2, según df)
    """
    print("\n🎨 Creando figura de edad vs metalicidad...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar columnas necesarias
    required_cols = ['FeH', 'Age', 'gmag_corr', 'imag_corr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Columnas faltantes: {missing_cols}")
        return None
    
    # Calcular (g-i)
    df = df.copy()
    df['g_i'] = df['gmag_corr'] - df['imag_corr']
    
    # Filtrar valores extremos
    masks = []
    
    # Metalicidad
    feh_Q1, feh_Q3 = df['FeH'].quantile(0.05), df['FeH'].quantile(0.95)
    feh_IQR = feh_Q3 - feh_Q1
    masks.append((df['FeH'] >= feh_Q1 - 2*feh_IQR) & (df['FeH'] <= feh_Q3 + 2*feh_IQR))
    
    # Edad
    age_Q1, age_Q3 = df['Age'].quantile(0.05), df['Age'].quantile(0.95)
    age_IQR = age_Q3 - age_Q1
    masks.append((df['Age'] >= age_Q1 - 2*age_IQR) & (df['Age'] <= age_Q3 + 2*age_IQR))
    
    # Color (g-i)
    g_i_Q1, g_i_Q3 = df['g_i'].quantile(0.05), df['g_i'].quantile(0.95)
    g_i_IQR = g_i_Q3 - g_i_Q1
    masks.append((df['g_i'] >= g_i_Q1 - 2*g_i_IQR) & (df['g_i'] <= g_i_Q3 + 2*g_i_IQR))
    
    # Combinar máscaras
    combined_mask = np.all(masks, axis=0)
    df_filtered = df[combined_mask].copy()
    
    if len(df_filtered) < 50:
        print(f"⚠️  Solo {len(df_filtered)} puntos después de filtrar")
        return None
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))
    
    feh = df_filtered['FeH'].values
    age = df_filtered['Age'].values
    g_i = df_filtered['g_i'].values
    
    # Scatter plot: Edad vs Metalicidad, coloreado por (g-i)
    scatter = ax.scatter(feh, age, c=g_i, cmap='Spectral_r',
                        s=50, alpha=0.85, edgecolors='k', linewidth=0.4,
                        vmin=g_i.min(), vmax=g_i.max())
    
    # Barra de color
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$(g - i)_0$', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Líneas de referencia
    ax.axhline(y=8, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
    
    # Configuración
    ax.set_xlabel(r'[Fe/H] (dex)', fontsize=16)
    ax.set_ylabel(r'Age (Gyr)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Estadísticas
    stats_text = f'N = {len(feh)}\n'
    stats_text += f'⟨Age⟩ = {age.mean():.1f} ± {age.std():.1f} Gyr\n'
    stats_text += f'⟨[Fe/H]⟩ = {feh.mean():.2f} ± {feh.std():.2f}\n'
    stats_text += f'⟨g-i⟩ = {g_i.mean():.3f} ± {g_i.std():.3f}'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                     edgecolor='gray', linewidth=0.5))
    
    # Añadir región de "old, metal-poor" y "intermediate-age, metal-rich"
    ax.text(-1.5, 11, 'Old, metal-poor', fontsize=10, alpha=0.7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(0.0, 6, 'Intermediate-age,\nmetal-rich', fontsize=10, alpha=0.7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/Figure7_Age_vs_Metallicity.pdf'
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figura 7 (edad vs metalicidad) guardada: {output_path}")
    
    return fig

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que genera todas las figuras individuales para el paper.
    """
    print("\n" + "="*70)
    print("NGC 5128 - GLOBULAR CLUSTER PLOTS FOR ApJ PAPER")
    print("VERSION CORREGIDA con proyección astropy exacta")
    print("="*70)
    
    # Crear directorio de figuras
    output_dir = "figures_paper"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Directorio de figuras: {output_dir}")
    
    # 1. Cargar datos con filtro (para la mayoría de figuras)
    print("\n[1] CARGANDO DATOS COMPLETOS (con filtro χ² ≤ 2)...")
    df = load_complete_gc_data(filter_chi2=True)
    print(f"✓ {len(df)} cúmulos globulares cargados (buena calidad)")
    
    # 2. Cargar datos sin filtrar (para figura espacial)
    print("\n[2] CARGANDO DATOS SIN FILTRAR (para incluir χ² > 2)...")
    df_all = load_complete_gc_data(filter_chi2=False)
    print(f"✓ {len(df_all)} cúmulos globulares cargados (todos)")
    
    # 3. Calcular parámetros espaciales para df_all (todos con coordenadas)
    print("\n[3] CALCULANDO PARÁMETROS ESPACIALES (para todos los GCs)...")
    spatial_results_all = calculate_spatial_parameters_corrected(df_all)
    
    figures_created = []
    gradient_results = None
    
    if spatial_results_all[0] is not None:
        radii_kpc_all, x_kpc_all, y_kpc_all, valid_mask_all = spatial_results_all
        print(f"✓ Parámetros espaciales calculados para {len(radii_kpc_all)} objetos")
        print(f"  Radio proyectado: {radii_kpc_all.min():.1f} - {radii_kpc_all.max():.0f} kpc")
        
        # Añadir radios al DataFrame (opcional)
        df_all['R_proj_kpc'] = np.nan
        df_all.loc[valid_mask_all, 'R_proj_kpc'] = radii_kpc_all
        
        # 4. Figura de distribución espacial 2D (usando df_all)
        print("\n[4] CREANDO FIGURA DE DISTRIBUCIÓN ESPACIAL 2D (con todos los GCs)...")
        fig1 = create_spatial_distribution_2d(df_all, radii_kpc_all, x_kpc_all, y_kpc_all, valid_mask_all, output_dir)
        if fig1:
            figures_created.append('Figura 1: Spatial Distribution 2D (all GCs)')
    else:
        print("⚠️  No se pudieron calcular parámetros espaciales para df_all")
        radii_kpc_all = None
        valid_mask_all = None
    
    # 5. Calcular parámetros espaciales para df (solo buena calidad)
    print("\n[5] CALCULANDO PARÁMETROS ESPACIALES (para GCs de buena calidad)...")
    spatial_results = calculate_spatial_parameters_corrected(df)
    
    if spatial_results[0] is not None:
        radii_kpc, x_kpc, y_kpc, valid_mask = spatial_results
        print(f"✓ Parámetros espaciales calculados para {len(radii_kpc)} objetos")
        print(f"  Radio proyectado: {radii_kpc.min():.1f} - {radii_kpc.max():.0f} kpc")
        
        # Añadir radios al DataFrame df
        df['R_proj_kpc'] = np.nan
        df.loc[valid_mask, 'R_proj_kpc'] = radii_kpc
    else:
        print("⚠️  No se pudieron calcular parámetros espaciales para df")
        radii_kpc = None
        valid_mask = None
    
    # ========================================================================
    # FIGURAS QUE DEPENDEN DE RADIOS (solo si se pudieron calcular)
    # ========================================================================
    if radii_kpc is not None and valid_mask is not None:
        # 6. Figura del gradiente radial (scatter) - original
        print("\n[6] CREANDO FIGURA DEL GRADIENTE RADIAL (SCATTER)...")
        gradient_results = create_radial_gradient_scatter(
    df, radii_kpc, valid_mask,
    df_all=df_all, radii_kpc_all=radii_kpc_all, valid_mask_all=valid_mask_all,
    output_dir=output_dir
           )
        if gradient_results:
            figures_created.append('Figura 2: Radial Gradient Scatter')
            print(f"  • Pendiente: {gradient_results['slope']:.3f} ± {gradient_results['slope_err']:.3f} dex/dex")
            print(f"  • Pearson r = {gradient_results['r_pearson']:.3f}, p = {gradient_results['p_pearson']:.4f}")
            if gradient_results['p_pearson'] < 0.05:
                print(f"  → ¡GRADIENTE SIGNIFICATIVO! (p < 0.05)")
        else:
            print("  ⚠️  No se pudo generar la figura del gradiente radial scatter.")
        
        # 7. Figura del gradiente radial binneado
        print("\n[7] CREANDO FIGURA DEL GRADIENTE RADIAL BINNEADO...")
        create_radial_gradient_binned(df, radii_kpc, valid_mask, output_dir)
        figures_created.append('Figura 3: Radial Gradient Binned')
    else:
        print("\n⚠️  No se pueden generar figuras de gradiente radial por falta de datos espaciales.")
        gradient_results = None
    
    # 8. Figura adicional: gradiente radial separado por edad (usando todos los GCs)
    if radii_kpc_all is not None and valid_mask_all is not None:
        print("\n[8] CREANDO FIGURA DE GRADIENTE RADIAL POR EDAD...")
        fig_age = create_radial_gradient_by_age(df_all, radii_kpc_all, valid_mask_all, output_dir)
        if fig_age:
            figures_created.append('Figura adicional: Radial Gradient by Age')
    else:
        print("\n⚠️  No se puede generar la figura de gradiente por edad (sin datos espaciales completos).")
    
    # ========================================================================
    # FIGURAS QUE NO DEPENDEN DE RADIOS
    # ========================================================================
    # 9. Figura de metalicidad vs (g-i)
    print("\n[9] CREANDO FIGURA DE METALICIDAD VS (g-i)...")
    fig4 = create_metallicity_vs_gi(df, output_dir)
    if fig4:
        figures_created.append('Figura 4: Metallicity vs g-i')
    
    # 10. Figura de metalicidad vs (g-z)
    print("\n[10] CREANDO FIGURA DE METALICIDAD VS (g-z)...")
    fig5 = create_metallicity_vs_gz_with_marginals(df, output_dir)
    if fig5:
        figures_created.append('Figura 5: Metallicity vs g-z')
    
    # 11. Figura de metalicidad vs r-mag
    print("\n[11] CREANDO FIGURA DE METALICIDAD VS r...")
    fig55 = create_metallicity_vs_r_with_marginals(df, output_dir)
    if fig55:
        figures_created.append('Figura 6: Metallicity vs r')
    
    # 12. Figura de distribución de metalicidad
    print("\n[12] CREANDO FIGURA DE DISTRIBUCIÓN DE METALICIDAD...")
    fig6 = create_metallicity_distribution(df, output_dir)
    if fig6:
        figures_created.append('Figura 7: Metallicity Distribution')
    
    # 13. Figura de edad vs metalicidad
    print("\n[13] CREANDO FIGURA DE EDAD VS METALICIDAD...")
    fig7 = create_age_vs_metallicity(df, output_dir)
    if fig7:
        figures_created.append('Figura 8: Age vs Metallicity')
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "="*70)
    print("✅ GENERACIÓN DE FIGURAS COMPLETADA")
    print("="*70)
    
    print(f"\n📁 FIGURAS GUARDADAS en '{output_dir}/':")
    for i, fig_name in enumerate(figures_created, 1):
        print(f"   {i}. {fig_name}")
    
    print(f"\n📊 ESTADÍSTICAS DEL DATASET (buena calidad):")
    print(f"   • Total GCs analizados: {len(df)}")
    print(f"   • [Fe/H] medio: {df['FeH'].mean():.3f} ± {df['FeH'].std():.3f}")
    
    if 'Age' in df.columns:
        valid_age = df['Age'].dropna()
        if len(valid_age) > 0:
            print(f"   • Edad media: {valid_age.mean():.1f} ± {valid_age.std():.1f} Gyr")
    
    if 'R_proj_kpc' in df.columns:
        valid_radius = df['R_proj_kpc'].dropna()
        if len(valid_radius) > 0:
            print(f"   • Radio máximo (buena calidad): {valid_radius.max():.0f} kpc")
            print(f"   • GCs con posición: {len(valid_radius)} ({len(valid_radius)/len(df)*100:.1f}%)")
    
    print(f"\n📊 ESTADÍSTICAS DEL DATASET COMPLETO (todos los GCs):")
    print(f"   • Total GCs con coordenadas: {len(df_all)}")
    n_good_all = (df_all['best.reduced_chi_square'] <= 2.0).sum()
    n_bad_all = (df_all['best.reduced_chi_square'] > 2.0).sum()
    print(f"   • Buena calidad: {n_good_all}")
    print(f"   • Mala calidad: {n_bad_all}")
    
    if gradient_results:
        print(f"\n📈 RESULTADOS DEL GRADIENTE RADIAL (buena calidad):")
        print(f"   • Pendiente: {gradient_results['slope']:.3f} ± {gradient_results['slope_err']:.3f} dex/dex")
        print(f"   • Significancia: p = {gradient_results['p_pearson']:.4f}")
        if gradient_results['p_pearson'] < 0.05:
            print(f"   • Interpretación: Gradiente negativo significativo")
            print(f"     (más metálico en el centro, más pobre en metales en el halo)")
    
    print("\n🎨 CARACTERÍSTICAS DE LAS FIGURAS:")
    print("   • Formato: PDF de alta resolución (600 DPI)")
    print("   • Estilo: Publicación científica (ApJ compliant)")
    print("   • Proyección espacial: Corregida con astropy")
    print("   • Estadísticas incluidas en cada figura")
    print("   • Paletas de colores profesionales")
    
    print("\n" + "="*70)
    print("📄 LISTO PARA INCLUIR EN EL PAPER DE ApJ")
    print("="*70)
    
    # Guardar estadísticas en un archivo de texto
    stats_file = f"{output_dir}/plot_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("NGC 5128 Globular Cluster Analysis - Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total GCs analyzed (good quality): {len(df)}\n")
        f.write(f"Mean [Fe/H] (good): {df['FeH'].mean():.3f} ± {df['FeH'].std():.3f}\n")
        
        if 'Age' in df.columns:
            valid_age = df['Age'].dropna()
            if len(valid_age) > 0:
                f.write(f"Mean Age (good): {valid_age.mean():.1f} ± {valid_age.std():.1f} Gyr\n")
        
        if gradient_results:
            f.write("\nRadial Metallicity Gradient (good quality):\n")
            f.write(f"  Slope: {gradient_results['slope']:.3f} ± {gradient_results['slope_err']:.3f} dex/dex\n")
            f.write(f"  Pearson r: {gradient_results['r_pearson']:.3f}\n")
            f.write(f"  p-value: {gradient_results['p_pearson']:.4f}\n")
            if 'rho_spearman' in gradient_results:
                f.write(f"  Spearman ρ: {gradient_results['rho_spearman']:.3f}\n")
            f.write(f"  N points: {gradient_results['n_points']}\n")
        
        f.write(f"\nFigures generated: {len(figures_created)}\n")
        for fig_name in figures_created:
            f.write(f"  - {fig_name}\n")
    
    print(f"\n📝 Estadísticas guardadas en: {stats_file}")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
