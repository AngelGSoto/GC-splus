#!/usr/bin/env python3
# comparison_cigale_literature_final_optimal.py
# Gráfico optimizado para ApJ - Con distinción de GCs jóvenes (borde dorado)
# MANTIENE TODOS LOS PARÁMETROS ESTÉTICOS ORIGINALES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN ELEGANTE PARA ApJ - CON TAMAÑOS AUMENTADOS (ORIGINAL)
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 12,
    'axes.labelsize': 20,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
})

# Colores elegantes (ORIGINAL)
COLORS = {
    'beasley': '#D55E00',
    'hernandez': '#0072B2',
    'band': '#CCCCCC',
    'one_to_one': '#000000',
    'reference': '#999999',
}

# Estilos para cada estudio - CON ZORDER ESPECIFICADO (ORIGINAL)
STYLES = {
    'beasley': {
        'marker': '^',
        'color': COLORS['beasley'],
        'label': 'Beasley et al. (2008) [M/H]',
        'edgecolor': 'black',
        'linewidth': 0.8,
        'alpha': 0.85,
        's': 110,
        'zorder_scatter': 4,
        'zorder_error': 3,
    },
    'hernandez': {
        'marker': 'o',
        'color': COLORS['hernandez'],
        'label': 'Hernández et al. (2018) [Fe/H]',
        'edgecolor': 'black',
        'linewidth': 0.8,
        'alpha': 0.85,
        's': 100,
        'zorder_scatter': 6,
        'zorder_error': 5,
    }
}

# Parámetros globales (ORIGINAL)
REDUCED_CHI2_THRESHOLD = 2.
Z_SOLAR = 0.02
YOUNG_THRESHOLD = 8.0  # Gyr - NUEVO

# ============================================================================
# FUNCIONES (ORIGINALES, CON PEQUEÑAS MODIFICACIONES)
# ============================================================================

def load_cigale_results_with_errors():
    """Carga resultados de CIGALE con errores (ORIGINAL)."""
    try:
        print("  • Cargando resultados de CIGALE...")
        
        table = Table.read('out/results.fits', hdu=1)
        df = table.to_pandas()
        
        # Mostrar columnas disponibles para debugging
        print("  • Columnas disponibles:")
        for col in ['id', 'bayes.stellar.metallicity', 'bayes.stellar.metallicity_err',
                   'best.chi_square', 'best.reduced_chi_square']:
            if col in df.columns:
                print(f"    → {col}")
        
        print(f"  ✓ Columna de error encontrada: bayes.stellar.metallicity_err")
        
        # Crear DataFrame
        cigale_df = pd.DataFrame()
        cigale_df['ID'] = df['id'].astype(str).str.strip()
        cigale_df['age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0
        
        # Calcular [Fe/H] = log10(metallicity/Z_solar)
        metallicity = df['bayes.stellar.metallicity'].copy()
        metallicity_err = df['bayes.stellar.metallicity_err'].copy()
        
        # Paso 1: Filtrar valores no positivos para evitar log(0) o log(negativo)
        valid_mask = (metallicity > 0) & (metallicity_err > 0)
        print(f"  • Objetos con metallicity > 0: {valid_mask.sum()}/{len(metallicity)}")
        
        # Inicializar arrays con NaN
        FeH_CIGALE = np.full_like(metallicity, np.nan, dtype=float)
        FeH_err_CIGALE = np.full_like(metallicity, np.nan, dtype=float)
        
        # Solo calcular para valores válidos
        if valid_mask.any():
            # [Fe/H] = log10(M/Z_solar)
            FeH_CIGALE[valid_mask] = np.log10(metallicity[valid_mask] / Z_SOLAR)
            
            # ERROR CORREGIDO:
            with np.errstate(invalid='ignore', divide='ignore'):
                relative_error = metallicity_err[valid_mask] / metallicity[valid_mask]
                FeH_err_CIGALE[valid_mask] = relative_error / np.log(10)
            
            print(f"\n  • Estadísticas de errores CORREGIDAS:")
            print(f"    - Error relativo en metallicity (dM/M):")
            print(f"      Min: {relative_error.min():.3f}")
            print(f"      Mediana: {np.median(relative_error):.3f}")
            print(f"      Max: {relative_error.max():.3f}")
            print(f"\n    - Error en [Fe/H] (dex):")
            print(f"      Min: {FeH_err_CIGALE[valid_mask].min():.3f}")
            print(f"      Mediana: {np.median(FeH_err_CIGALE[valid_mask]):.3f}")
            print(f"      Max: {FeH_err_CIGALE[valid_mask].max():.3f}")
            
            large_errors = (FeH_err_CIGALE[valid_mask] > 1.0).sum()
            if large_errors > 0:
                print(f"    ⚠️  {large_errors} objetos tienen error > 1.0 dex")
        
        cigale_df['FeH_CIGALE'] = FeH_CIGALE
        cigale_df['FeH_err_CIGALE'] = FeH_err_CIGALE
        
        # ⚠️ USAR CHI-CUADRADO REDUCIDO
        cigale_df['reduced_chi2'] = df['best.reduced_chi_square']
        cigale_df['chi2'] = df['best.chi_square']
        
        # Filtrar por chi-cuadrado reducido y valores válidos de [Fe/H]
        cigale_df = cigale_df[
            (cigale_df['reduced_chi2'] <= REDUCED_CHI2_THRESHOLD) &
            (cigale_df['FeH_CIGALE'].notna())
        ].copy()
        
        print(f"\n  • Después de filtros:")
        print(f"    - Objetos con χ²_reducido ≤ {REDUCED_CHI2_THRESHOLD}: {len(cigale_df)}")
        print(f"    - Rango de [Fe/H]: {cigale_df['FeH_CIGALE'].min():.2f} a {cigale_df['FeH_CIGALE'].max():.2f}")
        
        # Reemplazar NaN por la mediana (solo si hay datos)
        if not cigale_df.empty and cigale_df['FeH_err_CIGALE'].notna().any():
            median_err = cigale_df['FeH_err_CIGALE'].median()
            print(f"    - Mediana de error: {median_err:.3f} dex")
            
            if median_err > 0.5:
                print(f"    ⚠️  ¡ADVERTENCIA! Error mediano muy grande: {median_err:.3f} dex")
            
            cigale_df['FeH_err_CIGALE'] = cigale_df['FeH_err_CIGALE'].fillna(median_err)
        
        return cigale_df
        
    except Exception as e:
        print(f"  ❌ Error cargando CIGALE: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def simple_crossmatch(my_cat, lit_cat, radius=5.0):
    """Crossmatch simple con diagnóstico (ORIGINAL)."""
    try:
        coords1 = SkyCoord(ra=my_cat['RAJ2000'].values*u.degree, 
                          dec=my_cat['DEJ2000'].values*u.degree)
        coords2 = SkyCoord(ra=lit_cat['RA'].values*u.degree, 
                          dec=lit_cat['DEC'].values*u.degree)
        
        idx1, idx2, sep2d, _ = coords2.search_around_sky(coords1, radius*u.arcsec)
        
        if len(idx1) == 0:
            return pd.DataFrame()
        
        matches = []
        for i, j, sep in zip(idx1, idx2, sep2d.arcsec):
            match_data = {
                'idx_my': i,
                'idx_lit': j,
                'separation': sep,
                'T17ID': str(my_cat.iloc[i]['T17ID']),
            }
            
            for col in lit_cat.columns:
                match_data[f'lit_{col}'] = lit_cat.iloc[j][col]
                
            matches.append(match_data)
        
        df = pd.DataFrame(matches)
        
        if not df.empty:
            idx_closest = df.groupby('T17ID')['separation'].idxmin()
            df = df.loc[idx_closest].reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"  ❌ Error en crossmatch: {e}")
        return pd.DataFrame()

# ============================================================================
# FUNCIÓN PRINCIPAL - MODIFICADA PARA DISTINGUIR JÓVENES
# ============================================================================

def main():
    print("="*70)
    print("COMPARACIÓN CIGALE vs LITERATURA - CON DISTINCIÓN JÓVENES (borde dorado)")
    print(f"Umbral de χ² reducido: {REDUCED_CHI2_THRESHOLD}")
    print(f"GCs jóvenes: edad < {YOUNG_THRESHOLD} Gyr (borde dorado)")
    print("="*70)
    
    # 1. CARGAR DATOS (ORIGINAL)
    print("\n1. CARGANDO DATOS...")
    
    my_cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    print(f"    → {len(my_cat)} objetos")
    
    beasley = pd.read_csv('../../globular_clusters_ngc5128_Beasley.csv')
    beasley = beasley.rename(columns={
        'RAJ2000': 'RA', 'DEJ2000': 'DEC', 
        '[M/H]': 'FeH', 'e_[M/H]': 'FeH_err',
    })
    beasley['FeH'] = pd.to_numeric(beasley['FeH'], errors='coerce')
    beasley['FeH_err'] = pd.to_numeric(beasley['FeH_err'], errors='coerce')
    print(f"    → Beasley: {len(beasley)} objetos")
    
    hernandez = pd.read_csv('../../globular_clusters_ngc5128.csv')
    hernandez = hernandez.rename(columns={
        'RA (J2000)': 'RA', 'Dec (J2000)': 'DEC',
        '[Fe/H] (dex)': 'FeH', 'Error (dex)': 'FeH_err',
    })
    hernandez['FeH'] = pd.to_numeric(hernandez['FeH'], errors='coerce')
    hernandez['FeH_err'] = pd.to_numeric(hernandez['FeH_err'], errors='coerce')
    print(f"    → Hernández: {len(hernandez)} objetos")
    
    cigale_df = load_cigale_results_with_errors()
    
    if cigale_df.empty:
        print("  ⚠️  No hay datos después del filtro de χ² reducido")
        return
    
    # 2. CROSSMATCH (ORIGINAL + age)
    print("\n2. PROCESANDO COINCIDENCIAS...")
    
    def combine_data(matches_df, cigale_df, source_name):
        if matches_df.empty:
            return pd.DataFrame()
        
        combined = []
        for _, row in matches_df.iterrows():
            t17id = str(row['T17ID']).strip()
            
            cigale_match = cigale_df[cigale_df['ID'].str.strip() == t17id]
            if cigale_match.empty:
                cigale_match = cigale_df[cigale_df['ID'].str.contains(t17id, na=False)]
            
            if not cigale_match.empty:
                cigale_row = cigale_match.iloc[0]
                combined.append({
                    'T17ID': t17id,
                    'ID_CIGALE': cigale_row['ID'],
                    'FeH_CIGALE': cigale_row['FeH_CIGALE'],
                    'FeH_err_CIGALE': cigale_row['FeH_err_CIGALE'],
                    'age': cigale_row['age_Gyr'],                     # <-- AÑADIDO
                    'reduced_chi2': cigale_row['reduced_chi2'],
                    'chi2': cigale_row['chi2'],
                    'FeH_lit': row.get('lit_FeH', np.nan),
                    'FeH_err_lit': row.get('lit_FeH_err', np.nan),
                    'source': source_name,
                    'separation': row.get('separation', np.nan)
                })
        
        return pd.DataFrame(combined)
    
    df_beasley = combine_data(simple_crossmatch(my_cat, beasley, 3.0), cigale_df, 'Beasley')
    df_hernandez = combine_data(simple_crossmatch(my_cat, hernandez, 4.0), cigale_df, 'Hernandez')
    
    print(f"  • Coincidencias: Beasley={len(df_beasley)}, Hernández={len(df_hernandez)}")
    
    # Mostrar cuántos jóvenes hay en cada muestra
    if not df_beasley.empty and 'age' in df_beasley.columns:
        n_young_beasley = (df_beasley['age'] < YOUNG_THRESHOLD).sum()
        print(f"    → Beasley: {n_young_beasley} GCs jóvenes (<{YOUNG_THRESHOLD} Gyr)")
    if not df_hernandez.empty and 'age' in df_hernandez.columns:
        n_young_hernandez = (df_hernandez['age'] < YOUNG_THRESHOLD).sum()
        print(f"    → Hernández: {n_young_hernandez} GCs jóvenes (<{YOUNG_THRESHOLD} Gyr)")
    
    if df_beasley.empty and df_hernandez.empty:
        print("❌ No hay datos para graficar")
        return
    
    # 3. MÉTRICAS (sin ε_CIGALE en el texto)
    print("\n3. CALCULANDO MÉTRICAS...")
    
    def calculate_metrics(df, name):
        if df.empty:
            return None
        
        diffs = df['FeH_CIGALE'] - df['FeH_lit']
        abs_diffs = np.abs(diffs)
        
        return {
            'name': name,
            'N': len(df),
            'MAE': np.mean(abs_diffs),
            'bias': np.mean(diffs),
            'std': np.std(diffs),
            'frac_0.3': np.sum(abs_diffs <= 0.3) / len(df) * 100,
        }
    
    metrics_b = calculate_metrics(df_beasley, 'Beasley+2008')
    metrics_h = calculate_metrics(df_hernandez, 'Hernández+2018')
    
    for metrics in [metrics_b, metrics_h]:
        if metrics:
            print(f"\n  • {metrics['name']}:")
            print(f"    - N = {metrics['N']}")
            print(f"    - MAE = {metrics['MAE']:.3f} dex")
            print(f"    - Bias = {metrics['bias']:+.3f} dex")
            print(f"    - Std = {metrics['std']:.3f} dex")
            print(f"    - Fracción dentro de 0.3 dex = {metrics['frac_0.3']:.1f}%")
    
    # 4. CREAR GRÁFICO (MODIFICADO PARA DISTINGUIR JÓVENES)
    print("\n4. CREANDO GRÁFICO (con distinción jóvenes)...")
    
    fig = plt.figure(figsize=(9, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.08)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Determinar límites (EXACTAMENTE IGUAL QUE ORIGINAL)
    all_data = pd.concat([df_beasley, df_hernandez], ignore_index=True)
    if all_data.empty:
        xmin, xmax = -2.5, 0.5
    else:
        xmin = min(all_data['FeH_lit'].min(), all_data['FeH_CIGALE'].min()) - 1.
        xmax = max(all_data['FeH_lit'].max(), all_data['FeH_CIGALE'].max()) + 1.
    
    x_band = np.linspace(xmin, xmax, 100)
    
    # Panel superior - Capas de fondo (IGUAL QUE ORIGINAL)
    ax1.fill_between(x_band, x_band - 0.3, x_band + 0.3,
                     color=COLORS['band'], alpha=0.25,
                     label='±0.3 dex', zorder=1)
    
    ax1.plot([xmin, xmax], [xmin, xmax], 
             color=COLORS['one_to_one'], linestyle='--',
             linewidth=1.8, alpha=0.8, label='1:1', zorder=2)
    
    # Función para plot con errores - MODIFICADA para distinguir jóvenes
    def plot_data_with_young(ax, df, style):
        if df.empty or 'age' not in df.columns:
            return
        
        young_mask = df['age'] < YOUNG_THRESHOLD
        old_mask = ~young_mask
        
        # Limitar errores para visualización (IGUAL QUE ORIGINAL)
        yerr_limited = df['FeH_err_CIGALE'].clip(upper=1.0)
        xerr_limited = df['FeH_err_lit'].clip(upper=1.0)
        
        # Error bars para TODOS los puntos (IGUAL QUE ORIGINAL)
        ax.errorbar(df['FeH_lit'], df['FeH_CIGALE'],
                   xerr=xerr_limited,
                   yerr=yerr_limited,
                   fmt='none', 
                   ecolor=style['color'], 
                   alpha=0.35,
                   capsize=2.5,
                   elinewidth=0.8,
                   zorder=style['zorder_error'])
        
        # 1. GCs viejos (borde negro, IGUAL QUE ORIGINAL)
        if old_mask.any():
            ax.scatter(df.loc[old_mask, 'FeH_lit'], 
                      df.loc[old_mask, 'FeH_CIGALE'],
                      color=style['color'],
                      marker=style['marker'],
                      s=style['s'],
                      edgecolor=style['edgecolor'],
                      linewidth=style['linewidth'],
                      alpha=style['alpha'],
                      zorder=style['zorder_scatter'])
        
        # 2. GCs jóvenes (borde dorado, MISMOS parámetros excepto edgecolor y zorder+1)
        if young_mask.any():
            ax.scatter(df.loc[young_mask, 'FeH_lit'], 
                      df.loc[young_mask, 'FeH_CIGALE'],
                      color=style['color'],
                      marker=style['marker'],
                      s=style['s'],                     # MISMO tamaño
                      edgecolor='gold',                  # Borde dorado
                      linewidth=1.8,                      # Un poco más grueso para destacar
                      alpha=style['alpha'],               # MISMA opacidad
                      zorder=style['zorder_scatter'] + 1) # Encima de los viejos
    
    # Dibujar Beasley primero, luego Hernández
    plot_data_with_young(ax1, df_beasley, STYLES['beasley'])
    plot_data_with_young(ax1, df_hernandez, STYLES['hernandez'])
    
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(xmin, xmax)
    ax1.set_ylabel('[Fe/H]$_{\\rm This\\ Work}$ (dex)', fontsize=20, labelpad=10)
    ax1.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    
    # Leyenda personalizada (para reflejar los bordes dorados)
    from matplotlib.lines import Line2D
    legend_elements = [
        # Beasley
        Line2D([0], [0], marker='^', color='w', markerfacecolor=STYLES['beasley']['color'],
               markersize=12, label='Beasley+2008 (old)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=STYLES['beasley']['color'],
               markersize=12, label='Beasley+2008 (young)', markeredgecolor='gold', markeredgewidth=1.8),
        # Hernández
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLES['hernandez']['color'],
               markersize=10, label='Hernández+2018 (old)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLES['hernandez']['color'],
               markersize=10, label='Hernández+2018 (young)', markeredgecolor='gold', markeredgewidth=1.8),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=16, framealpha=0.95, markerscale=1.2)
    
    # Texto con métricas (SIN ε_CIGALE)
    text_str = f"χ² reducido ≤ {REDUCED_CHI2_THRESHOLD}\n"
    if metrics_b:
        text_str += f"\nBeasley+2008:\nN={metrics_b['N']}\nMAE={metrics_b['MAE']:.2f} dex"
    if metrics_h:
        text_str += f"\n\nHernández+2018:\nN={metrics_h['N']}\nMAE={metrics_h['MAE']:.2f} dex"
    
    if text_str:
        ax1.text(0.03, 0.97, text_str, transform=ax1.transAxes,
                fontsize=16.5, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.95, edgecolor='gray', linewidth=0.5),
                zorder=10)
    
    # Panel inferior - MODIFICADO para distinguir jóvenes
    # Líneas horizontales (IGUAL QUE ORIGINAL)
    ax2.axhline(y=0, color='black', linewidth=1.5, alpha=0.7, zorder=1)
    ax2.axhspan(-0.3, 0.3, color=COLORS['band'], alpha=0.2, zorder=2)
    ax2.axhspan(-0.15, 0.15, color=COLORS['reference'], alpha=0.15, zorder=2)
    
    # Función para residuos con distinción
    def plot_residuals_with_young(ax, df, style):
        if df.empty or 'age' not in df.columns:
            return
        
        df['diff'] = df['FeH_CIGALE'] - df['FeH_lit']
        young_mask = df['age'] < YOUNG_THRESHOLD
        old_mask = ~young_mask
        
        # Viejos (borde negro, IGUAL QUE ORIGINAL)
        if old_mask.any():
            ax.scatter(df.loc[old_mask, 'FeH_lit'], 
                      df.loc[old_mask, 'diff'],
                      color=style['color'],
                      marker=style['marker'],
                      s=style['s'] * 0.8,
                      edgecolor=style['edgecolor'],
                      linewidth=style['linewidth'],
                      alpha=style['alpha'],
                      zorder=style['zorder_scatter'])
        
        # Jóvenes (borde dorado)
        if young_mask.any():
            ax.scatter(df.loc[young_mask, 'FeH_lit'], 
                      df.loc[young_mask, 'diff'],
                      color=style['color'],
                      marker=style['marker'],
                      s=style['s'] * 0.8,               # MISMO tamaño reducido
                      edgecolor='gold',
                      linewidth=1.8,
                      alpha=style['alpha'],
                      zorder=style['zorder_scatter'] + 1)
    
    plot_residuals_with_young(ax2, df_beasley, STYLES['beasley'])
    plot_residuals_with_young(ax2, df_hernandez, STYLES['hernandez'])
    
    ax2.set_xlabel('[Fe/H]$_{\\rm Literature}$ (dex)', fontsize=20, labelpad=8)
    ax2.set_ylabel('Δ[Fe/H] (dex)', fontsize=20, labelpad=8)
    ax2.set_ylim(-1.0, 1.0)  # IGUAL QUE ORIGINAL
    ax2.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    
    ax2.text(0.03, 0.85, 'Δ[Fe/H] = This Work − Literature',
            transform=ax2.transAxes, fontsize=13,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.8, edgecolor='none'),
            zorder=10)
    
    # Ajustes finales (IGUAL QUE ORIGINAL)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.tick_params(axis='both', labelsize=18)
    ax1.tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.07, hspace=0.08)
    
    # Guardar
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/Comparison_metalicity_luis_literature_v2.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=600, pad_inches=0.05)
    plt.savefig('figures/Comparison_metalicity_luis_literature_v2.png', bbox_inches='tight', dpi=300, pad_inches=0.05)
    
    print(f"\n✅ Gráfico guardado en: {output_path}")
    print(f"   • GCs jóvenes (<{YOUNG_THRESHOLD} Gyr) destacados con borde dorado")
    print(f"   • Eliminada línea ε_CIGALE del cuadro de métricas")
    print(f"   • Todos los demás parámetros (límites, tamaños, opacidades) IGUALES al original")
    
    # Guardar datos
    if not df_hernandez.empty:
        df_hernandez.to_csv('figures/hernandez_detailed_young.csv', index=False)
    if not df_beasley.empty:
        df_beasley.to_csv('figures/beasley_detailed_young.csv', index=False)
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO CON ÉXITO ✓")
    print("="*70)

if __name__ == "__main__":
    main()
