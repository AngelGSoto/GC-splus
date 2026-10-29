#!/usr/bin/env python3
# comparison_cigale_literature_final_v3.py
# Comparacion CIGALE vs literatura (Beasley+2008, Hernandez+2018), con GCs jovenes (borde dorado).
# v3: las metricas y los puntos usan solo los GCs con metalicidad en la literatura
#     (Beasley: 94 cruzados, 81 con [M/H]); recuadro con "N = 81 (94 matched)";
#     chi2_nu en ingles; ejes algo mas grandes; grillas opcionales (SHOW_GRID).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACION
# ============================================================================
FS_AXES = 23          # nombres de los ejes (antes 20)
FS_TICKS = 20         # valores de los ejes (antes 18)
SHOW_GRID = False     # A&A: sin grillas (poner True para mostrarlas)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 12,
    'axes.labelsize': FS_AXES,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': FS_TICKS,
    'ytick.labelsize': FS_TICKS,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
})

COLORS = {
    'beasley': '#D55E00',
    'hernandez': '#0072B2',
    'band': '#CCCCCC',
    'one_to_one': '#000000',
    'reference': '#999999',
}

STYLES = {
    'beasley': {
        'marker': '^', 'color': COLORS['beasley'], 'label': 'Beasley et al. (2008) [M/H]',
        'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85, 's': 110,
        'zorder_scatter': 4, 'zorder_error': 3,
    },
    'hernandez': {
        'marker': 'o', 'color': COLORS['hernandez'], 'label': 'Hernández et al. (2018) [Fe/H]',
        'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.85, 's': 100,
        'zorder_scatter': 6, 'zorder_error': 5,
    }
}

REDUCED_CHI2_THRESHOLD = 2.
Z_SOLAR = 0.02
YOUNG_THRESHOLD = 8.0  # Gyr


# ============================================================================
# FUNCIONES
# ============================================================================

def load_cigale_results_with_errors():
    """Carga resultados de CIGALE con errores."""
    try:
        print("  • Cargando resultados de CIGALE...")
        df = Table.read('out/results.fits', hdu=1).to_pandas()

        cigale_df = pd.DataFrame()
        cigale_df['ID'] = df['id'].astype(str).str.strip()
        cigale_df['age_Gyr'] = df['bayes.stellar.age_m_star'] / 1000.0

        metallicity = df['bayes.stellar.metallicity'].copy()
        metallicity_err = df['bayes.stellar.metallicity_err'].copy()
        valid_mask = (metallicity > 0) & (metallicity_err > 0)

        FeH_CIGALE = np.full_like(metallicity, np.nan, dtype=float)
        FeH_err_CIGALE = np.full_like(metallicity, np.nan, dtype=float)
        if valid_mask.any():
            FeH_CIGALE[valid_mask] = np.log10(metallicity[valid_mask] / Z_SOLAR)
            with np.errstate(invalid='ignore', divide='ignore'):
                FeH_err_CIGALE[valid_mask] = (metallicity_err[valid_mask] / metallicity[valid_mask]) / np.log(10)

        cigale_df['FeH_CIGALE'] = FeH_CIGALE
        cigale_df['FeH_err_CIGALE'] = FeH_err_CIGALE
        cigale_df['reduced_chi2'] = df['best.reduced_chi_square']
        cigale_df['chi2'] = df['best.chi_square']

        cigale_df = cigale_df[(cigale_df['reduced_chi2'] <= REDUCED_CHI2_THRESHOLD) &
                              (cigale_df['FeH_CIGALE'].notna())].copy()
        print(f"    - Objetos con chi2_nu <= {REDUCED_CHI2_THRESHOLD}: {len(cigale_df)}")

        if not cigale_df.empty and cigale_df['FeH_err_CIGALE'].notna().any():
            median_err = cigale_df['FeH_err_CIGALE'].median()
            print(f"    - Mediana del error en [Fe/H]: {median_err:.3f} dex")
            cigale_df['FeH_err_CIGALE'] = cigale_df['FeH_err_CIGALE'].fillna(median_err)
        return cigale_df

    except Exception as e:
        print(f"  ❌ Error cargando CIGALE: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def simple_crossmatch(my_cat, lit_cat, radius=5.0):
    """Crossmatch simple."""
    try:
        coords1 = SkyCoord(ra=my_cat['RAJ2000'].values * u.degree, dec=my_cat['DEJ2000'].values * u.degree)
        coords2 = SkyCoord(ra=lit_cat['RA'].values * u.degree, dec=lit_cat['DEC'].values * u.degree)
        idx1, idx2, sep2d, _ = coords2.search_around_sky(coords1, radius * u.arcsec)
        if len(idx1) == 0:
            return pd.DataFrame()
        matches = []
        for i, j, sep in zip(idx1, idx2, sep2d.arcsec):
            md = {'idx_my': i, 'idx_lit': j, 'separation': sep, 'T17ID': str(my_cat.iloc[i]['T17ID'])}
            for col in lit_cat.columns:
                md[f'lit_{col}'] = lit_cat.iloc[j][col]
            matches.append(md)
        df = pd.DataFrame(matches)
        if not df.empty:
            df = df.loc[df.groupby('T17ID')['separation'].idxmin()].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  ❌ Error en crossmatch: {e}")
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("COMPARACION CIGALE vs LITERATURA (v3: N con metalicidad valida)")
    print("=" * 70)

    # 1. DATOS
    my_cat = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    beasley = pd.read_csv('../../globular_clusters_ngc5128_Beasley.csv').rename(columns={
        'RAJ2000': 'RA', 'DEJ2000': 'DEC', '[M/H]': 'FeH', 'e_[M/H]': 'FeH_err'})
    beasley['FeH'] = pd.to_numeric(beasley['FeH'], errors='coerce')
    beasley['FeH_err'] = pd.to_numeric(beasley['FeH_err'], errors='coerce')
    hernandez = pd.read_csv('../../globular_clusters_ngc5128.csv').rename(columns={
        'RA (J2000)': 'RA', 'Dec (J2000)': 'DEC', '[Fe/H] (dex)': 'FeH', 'Error (dex)': 'FeH_err'})
    hernandez['FeH'] = pd.to_numeric(hernandez['FeH'], errors='coerce')
    hernandez['FeH_err'] = pd.to_numeric(hernandez['FeH_err'], errors='coerce')

    cigale_df = load_cigale_results_with_errors()
    if cigale_df.empty:
        print("  ⚠️  No hay datos despues del filtro de chi2_nu")
        return

    # 2. CRUCE
    def combine_data(matches_df, cigale_df, source_name):
        if matches_df.empty:
            return pd.DataFrame()
        combined = []
        for _, row in matches_df.iterrows():
            t17id = str(row['T17ID']).strip()
            cm = cigale_df[cigale_df['ID'].str.strip() == t17id]
            if cm.empty:
                cm = cigale_df[cigale_df['ID'].str.contains(t17id, na=False)]
            if not cm.empty:
                cr = cm.iloc[0]
                combined.append({
                    'T17ID': t17id, 'ID_CIGALE': cr['ID'],
                    'FeH_CIGALE': cr['FeH_CIGALE'], 'FeH_err_CIGALE': cr['FeH_err_CIGALE'],
                    'age': cr['age_Gyr'], 'reduced_chi2': cr['reduced_chi2'], 'chi2': cr['chi2'],
                    'FeH_lit': row.get('lit_FeH', np.nan), 'FeH_err_lit': row.get('lit_FeH_err', np.nan),
                    'source': source_name, 'separation': row.get('separation', np.nan)})
        return pd.DataFrame(combined)

    df_beasley_all = combine_data(simple_crossmatch(my_cat, beasley, 3.0), cigale_df, 'Beasley')
    df_hernandez_all = combine_data(simple_crossmatch(my_cat, hernandez, 4.0), cigale_df, 'Hernandez')

    # solo los GCs con metalicidad en la literatura (Beasley tiene celdas vacias en [M/H])
    df_beasley = df_beasley_all.dropna(subset=['FeH_lit']).copy() if not df_beasley_all.empty else df_beasley_all
    df_hernandez = df_hernandez_all.dropna(subset=['FeH_lit']).copy() if not df_hernandez_all.empty else df_hernandez_all
    nmatch_b, nmatch_h = len(df_beasley_all), len(df_hernandez_all)
    print(f"  • Beasley: {nmatch_b} cruzados, {len(df_beasley)} con [M/H]")
    print(f"  • Hernandez: {nmatch_h} cruzados, {len(df_hernandez)} con [Fe/H]")
    for d, n in [(df_beasley, 'Beasley'), (df_hernandez, 'Hernandez')]:
        if not d.empty:
            print(f"    → {n}: {(d['age'] < YOUNG_THRESHOLD).sum()} GCs jovenes (<{YOUNG_THRESHOLD} Gyr)")

    if df_beasley.empty and df_hernandez.empty:
        print("❌ No hay datos para graficar")
        return

    # 3. METRICAS
    def calculate_metrics(df, name):
        if df.empty:
            return None
        diffs = df['FeH_CIGALE'] - df['FeH_lit']
        return {'name': name, 'N': len(df), 'MAE': np.mean(np.abs(diffs)), 'bias': np.mean(diffs),
                'std': np.std(diffs), 'frac_0.3': np.mean(np.abs(diffs) <= 0.3) * 100}

    metrics_b = calculate_metrics(df_beasley, 'Beasley+2008')
    metrics_h = calculate_metrics(df_hernandez, 'Hernández+2018')
    for m in [metrics_b, metrics_h]:
        if m:
            print(f"\n  • {m['name']}: N = {m['N']}, MAE = {m['MAE']:.3f}, bias = {m['bias']:+.3f}, "
                  f"std = {m['std']:.3f} dex, dentro de 0.3 dex = {m['frac_0.3']:.1f}%")

    # 4. FIGURA
    fig = plt.figure(figsize=(9, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.08)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    all_data = pd.concat([df_beasley, df_hernandez], ignore_index=True)
    xmin = min(all_data['FeH_lit'].min(), all_data['FeH_CIGALE'].min()) - 1.
    xmax = max(all_data['FeH_lit'].max(), all_data['FeH_CIGALE'].max()) + 1.
    x_band = np.linspace(xmin, xmax, 100)

    ax1.fill_between(x_band, x_band - 0.3, x_band + 0.3, color=COLORS['band'], alpha=0.25,
                     label='±0.3 dex', zorder=1)
    ax1.plot([xmin, xmax], [xmin, xmax], color=COLORS['one_to_one'], linestyle='--',
             linewidth=1.8, alpha=0.8, label='1:1', zorder=2)

    def plot_data_with_young(ax, df, style):
        if df.empty:
            return
        young = df['age'] < YOUNG_THRESHOLD
        ax.errorbar(df['FeH_lit'], df['FeH_CIGALE'],
                    xerr=df['FeH_err_lit'].clip(upper=1.0), yerr=df['FeH_err_CIGALE'].clip(upper=1.0),
                    fmt='none', ecolor=style['color'], alpha=0.35, capsize=2.5, elinewidth=0.8,
                    zorder=style['zorder_error'])
        if (~young).any():
            ax.scatter(df.loc[~young, 'FeH_lit'], df.loc[~young, 'FeH_CIGALE'], color=style['color'],
                       marker=style['marker'], s=style['s'], edgecolor=style['edgecolor'],
                       linewidth=style['linewidth'], alpha=style['alpha'], zorder=style['zorder_scatter'])
        if young.any():
            ax.scatter(df.loc[young, 'FeH_lit'], df.loc[young, 'FeH_CIGALE'], color=style['color'],
                       marker=style['marker'], s=style['s'], edgecolor='gold', linewidth=1.8,
                       alpha=style['alpha'], zorder=style['zorder_scatter'] + 1)

    plot_data_with_young(ax1, df_beasley, STYLES['beasley'])
    plot_data_with_young(ax1, df_hernandez, STYLES['hernandez'])

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(xmin, xmax)
    ax1.set_ylabel('[Fe/H]$_{\\rm This\\ Work}$ (dex)', fontsize=FS_AXES, labelpad=10)
    if SHOW_GRID:
        ax1.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor=STYLES['beasley']['color'],
               markersize=12, label='Beasley+2008 (old)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=STYLES['beasley']['color'],
               markersize=12, label='Beasley+2008 (young)', markeredgecolor='gold', markeredgewidth=1.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLES['hernandez']['color'],
               markersize=10, label='Hernández+2018 (old)', markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLES['hernandez']['color'],
               markersize=10, label='Hernández+2018 (young)', markeredgecolor='gold', markeredgewidth=1.8),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=16, framealpha=0.95, markerscale=1.2)

    # recuadro: N con metalicidad valida (y cruzados entre parentesis, solo si difieren)
    def ntxt(m, nmatch):
        return f"N={m['N']}" + (f" ({nmatch} matched)" if nmatch != m['N'] else "")
    text_str = f"$\\chi^2_\\nu \\leq$ {REDUCED_CHI2_THRESHOLD:g}\n"
    if metrics_b:
        text_str += f"\nBeasley+2008:\n{ntxt(metrics_b, nmatch_b)}\nMAE={metrics_b['MAE']:.2f} dex"
    if metrics_h:
        text_str += f"\n\nHernández+2018:\n{ntxt(metrics_h, nmatch_h)}\nMAE={metrics_h['MAE']:.2f} dex"
    ax1.text(0.03, 0.97, text_str, transform=ax1.transAxes, fontsize=16.5, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.5),
             zorder=10)

    # panel inferior
    ax2.axhline(y=0, color='black', linewidth=1.5, alpha=0.7, zorder=1)
    ax2.axhspan(-0.3, 0.3, color=COLORS['band'], alpha=0.2, zorder=2)
    ax2.axhspan(-0.15, 0.15, color=COLORS['reference'], alpha=0.15, zorder=2)

    def plot_residuals_with_young(ax, df, style):
        if df.empty:
            return
        diff = df['FeH_CIGALE'] - df['FeH_lit']
        young = df['age'] < YOUNG_THRESHOLD
        if (~young).any():
            ax.scatter(df.loc[~young, 'FeH_lit'], diff[~young], color=style['color'],
                       marker=style['marker'], s=style['s'] * 0.8, edgecolor=style['edgecolor'],
                       linewidth=style['linewidth'], alpha=style['alpha'], zorder=style['zorder_scatter'])
        if young.any():
            ax.scatter(df.loc[young, 'FeH_lit'], diff[young], color=style['color'],
                       marker=style['marker'], s=style['s'] * 0.8, edgecolor='gold', linewidth=1.8,
                       alpha=style['alpha'], zorder=style['zorder_scatter'] + 1)

    plot_residuals_with_young(ax2, df_beasley, STYLES['beasley'])
    plot_residuals_with_young(ax2, df_hernandez, STYLES['hernandez'])

    ax2.set_xlabel('[Fe/H]$_{\\rm Literature}$ (dex)', fontsize=FS_AXES, labelpad=8)
    ax2.set_ylabel('Δ[Fe/H] (dex)', fontsize=FS_AXES, labelpad=8)
    ax2.set_ylim(-1.0, 1.0)
    if SHOW_GRID:
        ax2.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
    ax2.text(0.03, 0.85, 'Δ[Fe/H] = This Work − Literature', transform=ax2.transAxes, fontsize=13,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                                                edgecolor='none'), zorder=10)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis='both', labelsize=FS_TICKS)
    ax2.tick_params(axis='both', labelsize=FS_TICKS)
    plt.tight_layout()
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.08, hspace=0.08)

    os.makedirs('figures', exist_ok=True)
    out = 'figures/Comparison_metalicity_luis_literature_v3'
    plt.savefig(out + '.pdf', bbox_inches='tight', dpi=600, pad_inches=0.05)
    plt.savefig(out + '.png', bbox_inches='tight', dpi=300, pad_inches=0.05)
    print(f"\n✅ Grafico guardado en: {out}.pdf")

    df_hernandez.to_csv('figures/hernandez_detailed_young.csv', index=False)
    df_beasley.to_csv('figures/beasley_detailed_young.csv', index=False)


if __name__ == "__main__":
    main()
