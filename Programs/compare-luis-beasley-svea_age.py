#!/usr/bin/env python3
# comparison_age_cigale_woodley_exact.py
# Réplica exacta del método de metalicidad pero para edades

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
import os
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN IDÉNTICA AL SCRIPT DE METALICIDAD
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

COLORS = {
    'woodley': '#2E86AB',
    'band': '#CCCCCC',
    'one_to_one': '#000000',
}

REDUCED_CHI2_THRESHOLD = 2.0

# ============================================================================
# FUNCIONES - IDÉNTICAS AL SCRIPT DE METALICIDAD
# ============================================================================

def load_cigale_results_with_ages():
    """Carga resultados de CIGALE - MÉTODO IDÉNTICO AL DE METALICIDAD."""
    print("  • Cargando resultados de CIGALE...")
    
    try:
        table = Table.read('out/results.fits', hdu=1)
        df = table.to_pandas()
        
        # Crear DataFrame exactamente como en metalicidad
        cigale_df = pd.DataFrame()
        cigale_df['ID'] = df['id'].astype(str).str.strip()
        
        # Edad en Gyr (CIGALE da en Myr)
        cigale_df['age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0
        cigale_df['age_err_Gyr'] = df['bayes.stellar.age_m_star_err'] / 1000.0
        
        # Chi-cuadrado reducido
        cigale_df['reduced_chi2'] = df['best.reduced_chi_square']
        
        # Filtrar por chi-cuadrado reducido (EXACTO como en metalicidad)
        cigale_df = cigale_df[
            (cigale_df['reduced_chi2'] <= REDUCED_CHI2_THRESHOLD) &
            (cigale_df['age_Gyr'].notna())
        ].copy()
        
        print(f"  • Objetos con χ² ≤ {REDUCED_CHI2_THRESHOLD}: {len(cigale_df)}")
        
        return cigale_df
        
    except Exception as e:
        print(f"  ❌ Error cargando CIGALE: {e}")
        return pd.DataFrame()

def load_woodley_data_fixed():
    """Carga Woodley - Asegurando formato correcto."""
    print("  • Cargando datos de Woodley...")
    
    try:
        # Cargar tablas
        woodley_coords = pd.read_csv('../../TAP_1_J_Apj_703_1335_table2_Woodley2010_age.csv')
        woodley_ages = pd.read_csv('../../TAP_1_J_Apj_703_1335_table5_Woodley2010_age.csv')
        
        # Para cada cúmulo en woodley_coords, tomar la primera entrada
        woodley_coords = woodley_coords.drop_duplicates(subset='GC', keep='first')
        
        # Las coordenadas en Woodley ya están en grados decimales
        # Según el sample que mostraste: RA=201.171667, DEC=-43.029333
        woodley_coords['RA_deg'] = woodley_coords['RAJ2000']
        woodley_coords['DEC_deg'] = woodley_coords['DEJ2000']
        
        # Combinar
        woodley_combined = pd.merge(
            woodley_coords[['GC', 'RA_deg', 'DEC_deg']],
            woodley_ages[['GC', 'Age', 'E_Age', 'e_Age']],
            on='GC',
            how='inner'
        )
        
        # Calcular error simétrico
        woodley_combined['age_err_Gyr'] = woodley_combined[['E_Age', 'e_Age']].max(axis=1)
        woodley_combined = woodley_combined.rename(columns={'Age': 'age_Gyr'})
        
        print(f"  • Woodley combinado: {len(woodley_combined)} objetos")
        
        return woodley_combined
        
    except Exception as e:
        print(f"  ❌ Error cargando Woodley: {e}")
        return pd.DataFrame()

def simple_crossmatch(my_cat, lit_cat, radius=2.0):
    """Crossmatch IDÉNTICO al de metalicidad."""
    try:
        coords1 = SkyCoord(ra=my_cat['RAJ2000'].values*u.degree, 
                          dec=my_cat['DEJ2000'].values*u.degree)
        coords2 = SkyCoord(ra=lit_cat['RA_deg'].values*u.degree, 
                          dec=lit_cat['DEC_deg'].values*u.degree)
        
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

def combine_data_age(matches_df, cigale_df):
    """Combina datos IDÉNTICO al método de metalicidad."""
    if matches_df.empty:
        return pd.DataFrame()
    
    combined = []
    for _, row in matches_df.iterrows():
        t17id = str(row['T17ID']).strip()
        
        # Buscar en CIGALE (EXACTO como en metalicidad)
        cigale_match = cigale_df[cigale_df['ID'].str.strip() == t17id]
        if cigale_match.empty:
            cigale_match = cigale_df[cigale_df['ID'].str.contains(t17id, na=False)]
        
        if not cigale_match.empty:
            cigale_row = cigale_match.iloc[0]
            combined.append({
                'T17ID': t17id,
                'age_CIGALE': cigale_row['age_Gyr'],
                'age_err_CIGALE': cigale_row['age_err_Gyr'],
                'reduced_chi2': cigale_row['reduced_chi2'],
                'age_Woodley': row.get('lit_age_Gyr', np.nan),
                'age_err_Woodley': row.get('lit_age_err_Gyr', np.nan),
                'separation': row.get('separation', np.nan),
                'GC_Woodley': row.get('lit_GC', np.nan),
            })
    
    return pd.DataFrame(combined)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("="*70)
    print("COMPARACIÓN DE EDADES - MÉTODO IDÉNTICO A METALICIDAD")
    print(f"Umbral de χ² reducido: {REDUCED_CHI2_THRESHOLD}")
    print("="*70)
    
    # 1. CARGAR DATOS
    print("\n1. CARGANDO DATOS...")
    
    my_cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    print(f"    → Mi catálogo: {len(my_cat)} objetos")
    
    cigale_df = load_cigale_results_with_ages()
    if cigale_df.empty:
        return
    
    woodley_df = load_woodley_data_fixed()
    if woodley_df.empty:
        return
    
    # 2. CROSSMATCH
    print("\n2. PROCESANDO COINCIDENCIAS...")
    
    matches = simple_crossmatch(my_cat, woodley_df, radius=2.0)
    print(f"    → Coincidencias encontradas: {len(matches)}")
    
    if matches.empty:
        return
    
    # 3. COMBINAR CON CIGALE
    df_combined = combine_data_age(matches, cigale_df)
    print(f"    → Objetos combinados con CIGALE: {len(df_combined)}")
    
    if df_combined.empty:
        return
    
    # 4. CALCULAR MÉTRICAS
    print("\n3. CALCULANDO MÉTRICAS...")
    
    diffs = df_combined['age_CIGALE'] - df_combined['age_Woodley']
    abs_diffs = np.abs(diffs)
    
    metrics = {
        'N': len(df_combined),
        'MAE': np.mean(abs_diffs),
        'bias': np.mean(diffs),
        'std': np.std(diffs),
        'correlation': df_combined['age_CIGALE'].corr(df_combined['age_Woodley']),
        'within_1Gyr': (abs_diffs <= 1.0).sum() / len(df_combined) * 100,
        'within_2Gyr': (abs_diffs <= 2.0).sum() / len(df_combined) * 100,
    }
    
    print(f"\n  • MÉTRICAS (N={metrics['N']}):")
    print(f"    - MAE = {metrics['MAE']:.2f} Gyr")
    print(f"    - Bias = {metrics['bias']:+.2f} Gyr")
    print(f"    - Std = {metrics['std']:.2f} Gyr")
    print(f"    - Correlación = {metrics['correlation']:.3f}")
    print(f"    - Dentro de 1 Gyr = {metrics['within_1Gyr']:.1f}%")
    print(f"    - Dentro de 2 Gyr = {metrics['within_2Gyr']:.1f}%")
    
    # 5. CREAR GRÁFICO
    print("\n4. CREANDO GRÁFICO...")
    
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    # Determinar límites
    xmin = min(df_combined['age_Woodley'].min(), df_combined['age_CIGALE'].min()) - 5
    xmax = max(df_combined['age_Woodley'].max(), df_combined['age_CIGALE'].max()) + 5
    
    # Panel superior
    x_band = np.linspace(xmin, xmax, 100)
    ax1.fill_between(x_band, x_band - 1, x_band + 1,
                     color=COLORS['band'], alpha=0.25,
                     label=r'$\pm$1 Gyr', zorder=1)
    
    ax1.plot([xmin, xmax], [xmin, xmax], 
             color=COLORS['one_to_one'], linestyle='--',
             linewidth=1.8, alpha=0.8, label='1:1', zorder=2)
    
    # Error bars
    ax1.errorbar(df_combined['age_Woodley'], df_combined['age_CIGALE'],
                xerr=df_combined['age_err_Woodley'].clip(upper=5.0),
                yerr=df_combined['age_err_CIGALE'].clip(upper=5.0),
                fmt='none', ecolor=COLORS['woodley'], 
                alpha=0.35, capsize=2.5, elinewidth=0.8, zorder=3)
    
    # Scatter
    ax1.scatter(df_combined['age_Woodley'], df_combined['age_CIGALE'],
               color=COLORS['woodley'], s=100,
               edgecolor='black', linewidth=0.8, alpha=0.85,
               label='Woodley et al. (2010)', zorder=4)
    
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(xmin, xmax)
    ax1.set_ylabel(r'Age$_{\rm This\ Work}$ (Gyr)', fontsize=20, labelpad=10)
    ax1.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', fontsize=15, framealpha=0.95)
    
    # Texto con métricas
    text_str = f"χ² reducido ≤ {REDUCED_CHI2_THRESHOLD}\n"
    text_str += f"N = {metrics['N']}\n"
    text_str += f"MAE = {metrics['MAE']:.2f} Gyr\n"
    text_str += f"Bias = {metrics['bias']:+.2f} Gyr\n"
    text_str += f"r = {metrics['correlation']:.3f}"
    
    ax1.text(0.03, 0.97, text_str, transform=ax1.transAxes,
            fontsize=15, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.95, edgecolor='gray', linewidth=0.5))
    
    # Panel inferior
    ax2.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
    ax2.axhspan(-1, 1, color=COLORS['band'], alpha=0.2)
    
    ax2.scatter(df_combined['age_Woodley'], diffs,
               color=COLORS['woodley'], s=80,
               edgecolor='black', linewidth=0.8, alpha=0.85)
    
    ax2.set_xlabel(r'Age$_{\rm Woodley\ et\ al.\ (2010)}$ (Gyr)', fontsize=20, labelpad=8)
    ax2.set_ylabel(r'$\Delta$Age (Gyr)', fontsize=20, labelpad=8)
    ax2.set_ylim(-8, 8)
    ax2.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    
    ax2.text(0.03, 0.85, r'$\Delta$Age = This Work $-$ Woodley',
            transform=ax2.transAxes, fontsize=13,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.8, edgecolor='none'))
    
    # Ajustes finales
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.08, hspace=0.08)
    
    # Guardar
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/Comparison_age_woodley_exact_method.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=600, pad_inches=0.05)
    print(f"\n    ✓ Gráfico guardado en: {output_path}")
    
    # 6. GUARDAR DATOS
    print("\n5. GUARDANDO DATOS...")
    df_combined['delta_age'] = diffs
    df_combined['abs_delta_age'] = abs_diffs
    
    output_data_path = 'figures/age_comparison_woodley_exact.csv'
    df_combined.to_csv(output_data_path, index=False)
    print(f"    ✓ Datos guardados en: {output_data_path}")
    
    # 7. MOSTRAR MAYORES DIFERENCIAS
    print(f"\n6. TOP 5 DIFERENCIAS MÁS GRANDES:")
    top_diffs = df_combined.nlargest(5, 'abs_delta_age')
    for idx, row in top_diffs.iterrows():
        print(f"    - {row['T17ID']}: {row['age_CIGALE']:.1f} vs {row['age_Woodley']:.1f} Gyr "
              f"(Δ={row['delta_age']:+.1f} Gyr, χ²={row['reduced_chi2']:.2f})")
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO CON ÉXITO")
    print("="*70)

if __name__ == "__main__":
    main()
