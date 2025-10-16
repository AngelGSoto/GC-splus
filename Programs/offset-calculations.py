import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import os
from pathlib import Path

# Configuración
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

def calculate_homogenization_stats(splus_mag, taylor_mag, filter_name):
    """
    Calcula estadísticas de homogenización entre magnitudes SPLUS y Taylor
    """
    # Filtrar valores válidos
    mask = (~np.isnan(splus_mag)) & (~np.isnan(taylor_mag)) & (splus_mag < 90) & (taylor_mag < 90)
    splus_valid = splus_mag[mask]
    taylor_valid = taylor_mag[mask]
    
    if len(splus_valid) < 3:
        return None
    
    # Calcular diferencias
    differences = taylor_valid - splus_valid
    
    # Estadísticas básicas
    stats_dict = {
        'n_stars': len(splus_valid),
        'mean_diff': np.mean(differences),
        'median_diff': np.median(differences),
        'std_diff': np.std(differences),
        'mad_diff': stats.median_abs_deviation(differences),
        'min_diff': np.min(differences),
        'max_diff': np.max(differences)
    }
    
    # Regresión lineal
    if len(splus_valid) > 5:
        slope, intercept, r_value, p_value, std_err = stats.linregress(splus_valid, taylor_valid)
        stats_dict.update({
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        })
    else:
        stats_dict.update({
            'slope': np.nan,
            'intercept': np.nan,
            'r_value': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_err': np.nan
        })
    
    return stats_dict, splus_valid, taylor_valid, differences

def create_comparison_plot(splus_mag, taylor_mag, differences, stats_dict, field_name, filter_name, aperture, output_dir):
    """
    Crea gráficos de comparación entre magnitudes SPLUS y Taylor
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Homogenización: {field_name} - {filter_name} - Apertura {apertura}"', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Dispersión SPLUS vs Taylor
    axes[0, 0].scatter(splus_mag, taylor_mag, alpha=0.6, s=30)
    min_val = min(np.min(splus_mag), np.min(taylor_mag))
    max_val = max(np.max(splus_mag), np.max(taylor_mag))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='y=x')
    
    if not np.isnan(stats_dict.get('slope', np.nan)):
        x_range = np.linspace(min_val, max_val, 100)
        y_pred = stats_dict['slope'] * x_range + stats_dict['intercept']
        axes[0, 0].plot(x_range, y_pred, 'g-', linewidth=2, 
                       label=f'y = {stats_dict["slope"]:.3f}x + {stats_dict["intercept"]:.3f}')
    
    axes[0, 0].set_xlabel('Magnitud SPLUS')
    axes[0, 0].set_ylabel('Magnitud Taylor')
    axes[0, 0].set_title(f'Correlación (r = {stats_dict.get("r_value", 0):.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma de diferencias
    axes[0, 1].hist(differences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(stats_dict['median_diff'], color='red', linestyle='--', linewidth=2, 
                      label=f'Mediana: {stats_dict["median_diff"]:.3f}')
    axes[0, 1].axvline(stats_dict['mean_diff'], color='orange', linestyle='--', linewidth=2,
                      label=f'Media: {stats_dict["mean_diff"]:.3f}')
    axes[0, 1].set_xlabel('Diferencia (Taylor - SPLUS)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Diferencias')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Diferencias vs Magnitud SPLUS
    axes[1, 0].scatter(splus_mag, differences, alpha=0.6, s=30)
    axes[1, 0].axhline(0, color='red', linestyle='-', alpha=0.8)
    axes[1, 0].axhline(stats_dict['median_diff'], color='blue', linestyle='--', 
                      label=f'Δ mediana: {stats_dict["median_diff"]:.3f}')
    axes[1, 0].set_xlabel('Magnitud SPLUS')
    axes[1, 0].set_ylabel('Diferencia (Taylor - SPLUS)')
    axes[1, 0].set_title('Diferencias vs Magnitud')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Boxplot de diferencias
    axes[1, 1].boxplot(differences, vert=True)
    axes[1, 1].set_ylabel('Diferencia (Taylor - SPLUS)')
    axes[1, 1].set_title('Boxplot de Diferencias')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Texto con estadísticas
    stats_text = (
        f'Estadísticas:\n'
        f'N estrellas: {stats_dict["n_stars"]}\n'
        f'Δ Media: {stats_dict["mean_diff"]:.3f} ± {stats_dict["std_diff"]:.3f}\n'
        f'Δ Mediana: {stats_dict["median_diff"]:.3f}\n'
        f'MAD: {stats_dict["mad_diff"]:.3f}\n'
        f'R²: {stats_dict.get("r_squared", 0):.3f}\n'
        f'Min: {stats_dict["min_diff"]:.3f}, Max: {stats_dict["max_diff"]:.3f}'
    )
    
    axes[1, 1].text(0.95, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontfamily='monospace')
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_filename = f"{output_dir}/homogenization_{field_name}_{filter_name}_aper{apertura}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def process_field_file(file_path, output_dir):
    """
    Procesa un archivo de campo individual y calcula offsets para todos los filtros y aperturas
    """
    field_name = os.path.basename(file_path).split('_')[0]
    print(f"Procesando campo: {field_name}")
    
    # Leer datos
    df = pd.read_csv(file_path)
    
    # Mapeo de filtros SPLUS a Taylor
    filter_mapping = {
        'F378': 'u',
        'F395': 'u', 
        'F410': 'u',
        'F430': 'g',
        'F515': 'g',
        'F660': 'r',
        'F861': 'i'  # Podríamos probar también con 'z'
    }
    
    results = []
    plots_created = []
    
    # Procesar cada filtro y apertura
    for splus_filter, taylor_filter in filter_mapping.items():
        taylor_col = f'taylor_{taylor_filter}mag'
        
        for aperture in [2, 3]:
            splus_col = f'MAG_{splus_filter}_{aperture}'
            
            if splus_col in df.columns and taylor_col in df.columns:
                stats_dict, splus_valid, taylor_valid, differences = calculate_homogenization_stats(
                    df[splus_col].values, df[taylor_col].values, splus_filter
                )
                
                if stats_dict is not None:
                    # Guardar resultados
                    result_row = {
                        'field': field_name,
                        'splus_filter': splus_filter,
                        'taylor_filter': taylor_filter,
                        'aperture': aperture,
                        **stats_dict
                    }
                    results.append(result_row)
                    
                    # Crear gráfico
                    plot_path = create_comparison_plot(
                        splus_valid, taylor_valid, differences, stats_dict,
                        field_name, splus_filter, aperture, output_dir
                    )
                    plots_created.append(plot_path)
                    
                    print(f"  {splus_filter} (aper {aperture}\"): "
                          f"Δ_mediana = {stats_dict['median_diff']:.3f}, "
                          f"n = {stats_dict['n_stars']}")
    
    return results, plots_created

def create_summary_plots(all_results, output_dir):
    """
    Crea gráficos resumen de todos los campos
    """
    df_results = pd.DataFrame(all_results)
    
    # Gráfico 1: Offsets por filtro (todos los campos)
    plt.figure(figsize=(14, 8))
    
    filters = sorted(df_results['splus_filter'].unique())
    for i, filter_name in enumerate(filters):
        plt.subplot(2, 4, i+1)
        
        filter_data = df_results[df_results['splus_filter'] == filter_name]
        
        # Usar mediana como offset principal
        offsets = filter_data['median_diff'].values
        fields = filter_data['field'].values
        
        plt.bar(range(len(offsets)), offsets, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        plt.axhline(y=np.median(offsets), color='blue', linestyle='--', 
                   label=f'Mediana global: {np.median(offsets):.3f}')
        
        plt.xlabel('Campos')
        plt.ylabel('Offset (Taylor - SPLUS)')
        plt.title(f'{filter_name}\n(n={len(offsets)} campos)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_offsets_by_filter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Boxplot de offsets por filtro
    plt.figure(figsize=(12, 6))
    
    plot_data = []
    labels = []
    for filter_name in filters:
        filter_data = df_results[df_results['splus_filter'] == filter_name]
        if len(filter_data) > 0:
            plot_data.append(filter_data['median_diff'].values)
            labels.append(filter_name)
    
    plt.boxplot(plot_data, labels=labels)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    plt.ylabel('Offset (Taylor - SPLUS)')
    plt.title('Distribución de Offsets por Filtro')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_boxplot_offsets.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3: Heatmap de correlaciones
    plt.figure(figsize=(10, 8))
    
    # Pivot table para correlaciones
    pivot_r = df_results.pivot_table(values='r_value', 
                                   index='field', 
                                   columns='splus_filter', 
                                   aggfunc='mean')
    
    sns.heatmap(pivot_r, annot=True, cmap='RdYlBu', center=0, 
                vmin=-1, vmax=1, fmt='.3f')
    plt.title('Correlación (r) por Campo y Filtro')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 4: Número de estrellas por campo
    plt.figure(figsize=(12, 6))
    
    n_stars_data = []
    for filter_name in filters:
        filter_data = df_results[df_results['splus_filter'] == filter_name]
        n_stars_by_field = filter_data.groupby('field')['n_stars'].mean()
        n_stars_data.append(n_stars_by_field.values)
    
    plt.boxplot(n_stars_data, labels=filters)
    plt.ylabel('Número de Estrellas')
    plt.title('Número de Estrellas de Referencia por Filtro')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_n_stars.png", dpi=300, bbox_inches='tight')
    plt.close()

def calculate_robust_offsets(all_results):
    """
    Calcula offsets robustos usando múltiples métodos
    """
    df = pd.DataFrame(all_results)
    
    # Calcular offsets por filtro usando diferentes métodos
    offsets_summary = []
    
    for filter_name in df['splus_filter'].unique():
        filter_data = df[df['splus_filter'] == filter_name]
        
        # Diferentes métodos de agregación
        median_offset = filter_data['median_diff'].median()
        weighted_offset = np.average(filter_data['median_diff'].values, 
                                   weights=filter_data['n_stars'].values)
        mad_weighted_offset = np.average(filter_data['median_diff'].values,
                                       weights=1/filter_data['mad_diff'].values)
        
        # Estadísticas
        n_fields = len(filter_data)
        n_stars_total = filter_data['n_stars'].sum()
        std_offset = filter_data['median_diff'].std()
        mad_global = stats.median_abs_deviation(filter_data['median_diff'].values)
        
        offsets_summary.append({
            'filter': filter_name,
            'offset_median': median_offset,
            'offset_weighted_n': weighted_offset,
            'offset_weighted_mad': mad_weighted_offset,
            'recommended_offset': median_offset,  # Usar mediana como valor recomendado
            'n_fields': n_fields,
            'n_stars_total': n_stars_total,
            'std_across_fields': std_offset,
            'mad_across_fields': mad_global,
            'min_offset': filter_data['median_diff'].min(),
            'max_offset': filter_data['median_diff'].max()
        })
    
    return pd.DataFrame(offsets_summary)

def main():
    """
    Función principal para calcular offsets de homogenización
    """
    # Configurar directorios
    input_pattern = "*_reference_stars_photometry_v17.csv"
    output_dir = "plot_homogenization"
    
    # Crear directorio de salida
    Path(output_dir).mkdir(exist_ok=True)
    
    # Encontrar archivos
    field_files = glob.glob(input_pattern)
    print(f"Encontrados {len(field_files)} archivos de campo")
    
    # Procesar todos los campos
    all_results = []
    all_plots = []
    
    for file_path in field_files:
        try:
            results, plots = process_field_file(file_path, output_dir)
            all_results.extend(results)
            all_plots.extend(plots)
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
    
    # Crear DataFrames con resultados
    df_results = pd.DataFrame(all_results)
    
    # Calcular offsets robustos
    df_offsets = calculate_robust_offsets(all_results)
    
    # Crear gráficos resumen
    create_summary_plots(all_results, output_dir)
    
    # Guardar resultados
    df_results.to_csv(f"{output_dir}/homogenization_detailed_results.csv", index=False)
    df_offsets.to_csv(f"{output_dir}/homogenization_recommended_offsets.csv", index=False)
    
    # Imprimir resumen
    print("\n" + "="*60)
    print("RESUMEN DE OFFSETS RECOMENDADOS")
    print("="*60)
    
    for _, row in df_offsets.iterrows():
        print(f"{row['filter']}: {row['recommended_offset']:.3f} "
              f"(±{row['mad_across_fields']:.3f}, "
              f"n={row['n_fields']} campos, "
              f"{row['n_stars_total']} estrellas)")
    
    print(f"\nResultados guardados en: {output_dir}/")
    print(f"- homogenization_detailed_results.csv")
    print(f"- homogenization_recommended_offsets.csv")
    print(f"- Gráficos individuales y resumen")
    
    # Recomendaciones para campos con pocas estrellas
    low_n_fields = df_results[df_results['n_stars'] < 10]['field'].unique()
    if len(low_n_fields) > 0:
        print(f"\n⚠️  Campos con pocas estrellas (<10): {list(low_n_fields)}")
        print("   Considerar usar offsets de campos vecinos o el promedio global")
    
    return df_results, df_offsets

if __name__ == "__main__":
    df_results, df_offsets = main()
