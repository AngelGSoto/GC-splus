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
    fig.suptitle(f'Homogenización: {field_name} - {filter_name} - Aperture {aperture}"', fontsize=16, fontweight='bold')
    
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
    plot_filename = f"{output_dir}/homogenization_{field_name}_{filter_name}_aper{aperture}.png"
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
    
    # Procesar cada filtro y apertura (SOLO 3 arcsec como mencionaste)
    for splus_filter, taylor_filter in filter_mapping.items():
        taylor_col = f'taylor_{taylor_filter}mag'
        
        # Solo procesar apertura de 3 arcsec
        aperture = 3
        splus_col = f'MAG_{splus_filter}_{aperture}'
        
        if splus_col in df.columns and taylor_col in df.columns:
            result = calculate_homogenization_stats(
                df[splus_col].values, df[taylor_col].values, splus_filter
            )
            
            if result is not None:
                stats_dict, splus_valid, taylor_valid, differences = result
                
                # Guardar resultados
                result_row = {
                    'field': field_name,
                    'splus_filter': splus_filter,
                    'taylor_filter': taylor_filter,
                    'aperture': aperture,
                    **stats_dict
                }
                results.append(result_row)
                
                # Crear gráfico solo si hay suficientes estrellas
                if stats_dict['n_stars'] >= 10:  # Solo crear gráficos para campos con al menos 10 estrellas
                    plot_path = create_comparison_plot(
                        splus_valid, taylor_valid, differences, stats_dict,
                        field_name, splus_filter, aperture, output_dir
                    )
                    plots_created.append(plot_path)
                
                print(f"  {splus_filter} (aper {aperture}\"): "
                      f"Δ_mediana = {stats_dict['median_diff']:.3f}, "
                      f"n = {stats_dict['n_stars']}")
            else:
                print(f"  {splus_filter}: No hay suficientes datos válidos")
        else:
            print(f"  {splus_filter}: Columnas no encontradas ({splus_col}, {taylor_col})")
    
    return results, plots_created

def create_summary_plots(all_results, output_dir):
    """
    Crea gráficos resumen de todos los campos
    """
    if not all_results:
        print("No hay resultados para crear gráficos resumen")
        return
        
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
        
        # Color diferente para campos con pocas estrellas
        colors = ['red' if n < 30 else 'skyblue' for n in filter_data['n_stars'].values]
        
        bars = plt.bar(range(len(offsets)), offsets, alpha=0.7, color=colors)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        if len(offsets) > 0:
            # Calcular mediana solo con campos buenos (≥30 estrellas)
            good_offsets = filter_data[filter_data['n_stars'] >= 30]['median_diff'].values
            if len(good_offsets) > 0:
                global_median = np.median(good_offsets)
                plt.axhline(y=global_median, color='blue', linestyle='--', 
                           label=f'Mediana global: {global_median:.3f}')
        
        plt.xlabel('Campos')
        plt.ylabel('Offset (Taylor - SPLUS)')
        plt.title(f'{filter_name}\n(n={len(offsets)} campos)')
        plt.xticks(range(len(fields)), fields, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir leyenda para colores
        if i == 0:
            plt.text(0.02, 0.98, 'Rojo: <30 estrellas\nAzul: ≥30 estrellas', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_offsets_by_filter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Boxplot de offsets por filtro (solo campos buenos)
    plt.figure(figsize=(12, 6))
    
    plot_data = []
    labels = []
    for filter_name in filters:
        filter_data = df_results[(df_results['splus_filter'] == filter_name) & 
                                (df_results['n_stars'] >= 30)]
        if len(filter_data) > 0:
            plot_data.append(filter_data['median_diff'].values)
            labels.append(filter_name)
    
    if plot_data:
        plt.boxplot(plot_data, tick_labels=labels)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        plt.ylabel('Offset (Taylor - SPLUS)')
        plt.title('Distribución de Offsets por Filtro (solo campos con ≥30 estrellas)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_boxplot_offsets.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Gráfico 3: Número de estrellas por campo
    plt.figure(figsize=(12, 6))
    
    n_stars_data = []
    field_labels = []
    for field in df_results['field'].unique():
        field_data = df_results[df_results['field'] == field]
        if len(field_data) > 0:
            n_stars_avg = field_data['n_stars'].mean()
            n_stars_data.append(n_stars_avg)
            field_labels.append(field)
    
    if n_stars_data:
        colors = ['red' if n < 30 else 'skyblue' for n in n_stars_data]
        bars = plt.bar(range(len(n_stars_data)), n_stars_data, color=colors, alpha=0.7)
        plt.axhline(y=30, color='red', linestyle='--', label='Umbral (30 estrellas)')
        plt.xlabel('Campos')
        plt.ylabel('Número Promedio de Estrellas')
        plt.title('Número de Estrellas de Referencia por Campo')
        plt.xticks(range(len(field_labels)), field_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir valores encima de las barras
        for i, v in enumerate(n_stars_data):
            plt.text(i, v + 5, str(int(v)), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_n_stars_by_field.png", dpi=300, bbox_inches='tight')
        plt.close()

def calculate_robust_offsets(all_results, min_stars_threshold=30):
    """
    Calcula offsets robustos usando múltiples métodos, considerando solo campos con buena estadística
    """
    if not all_results:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_results)
    
    # Separar campos buenos y problemáticos
    good_fields_mask = df['n_stars'] >= min_stars_threshold
    good_df = df[good_fields_mask]
    problematic_df = df[~good_fields_mask]
    
    print(f"\n📊 CLASIFICACIÓN DE CAMPOS (umbral: {min_stars_threshold} estrellas):")
    print(f"   - Campos buenos: {len(good_df['field'].unique())}")
    print(f"   - Campos problemáticos: {len(problematic_df['field'].unique())}")
    
    # Calcular offsets por filtro usando diferentes métodos
    offsets_summary = []
    
    for filter_name in df['splus_filter'].unique():
        filter_data = df[df['splus_filter'] == filter_name]
        good_filter_data = good_df[good_df['splus_filter'] == filter_name]
        
        # Si no hay campos buenos para este filtro, usar todos los campos
        if len(good_filter_data) == 0:
            good_filter_data = filter_data
            print(f"   ⚠️  Filtro {filter_name}: No hay campos buenos, usando todos los campos")
        
        # Diferentes métodos de agregación (solo con campos buenos)
        median_offset = good_filter_data['median_diff'].median()
        weighted_offset = np.average(good_filter_data['median_diff'].values, 
                                   weights=good_filter_data['n_stars'].values)
        
        # Estadísticas
        n_good_fields = len(good_filter_data)
        n_total_fields = len(filter_data)
        n_stars_total = good_filter_data['n_stars'].sum()
        std_offset = good_filter_data['median_diff'].std()
        mad_global = stats.median_abs_deviation(good_filter_data['median_diff'].values)
        
        offsets_summary.append({
            'filter': filter_name,
            'offset_median': median_offset,
            'offset_weighted': weighted_offset,
            'recommended_offset': median_offset,  # Usar mediana como valor recomendado
            'n_good_fields': n_good_fields,
            'n_total_fields': n_total_fields,
            'n_stars_total': n_stars_total,
            'std_across_fields': std_offset,
            'mad_across_fields': mad_global,
            'min_offset': good_filter_data['median_diff'].min(),
            'max_offset': good_filter_data['median_diff'].max()
        })
    
    return pd.DataFrame(offsets_summary), good_df, problematic_df

def create_final_offset_recommendations(df_results, df_offsets, problematic_df, output_dir):
    """
    Crea recomendaciones finales de offsets para cada campo, usando campos vecinos para los problemáticos
    """
    if df_results.empty or df_offsets.empty:
        return pd.DataFrame()
    
    # Crear tabla de offsets globales
    global_offsets = df_offsets.set_index('filter')['recommended_offset'].to_dict()
    
    # Obtener todos los campos únicos
    all_fields = sorted(df_results['field'].unique())
    all_filters = df_results['splus_filter'].unique()
    
    final_recommendations = []
    
    print(f"\n🎯 RECOMENDACIONES FINALES POR CAMPO:")
    
    for field in all_fields:
        field_data = df_results[df_results['field'] == field]
        field_n_stars = field_data['n_stars'].mean()
        
        # Determinar si el campo es problemático
        is_problematic = field in problematic_df['field'].unique()
        
        for filter_name in all_filters:
            filter_data = field_data[field_data['splus_filter'] == filter_name]
            
            if len(filter_data) == 0:
                continue
                
            if not is_problematic:
                # Campo bueno: usar su propio offset
                recommended_offset = filter_data['median_diff'].iloc[0]
                recommendation_source = 'propio'
                recommendation_notes = f'Campo bueno ({int(field_n_stars)} estrellas)'
            else:
                # Campo problemático: usar offset de campos vecinos o global
                field_num = int(field.replace('CenA', ''))
                
                # Buscar campos vecinos
                neighbor_nums = [field_num-1, field_num+1]
                neighbor_fields = [f"CenA{num:02d}" for num in neighbor_nums if f"CenA{num:02d}" in all_fields]
                
                neighbor_offsets = []
                for neighbor in neighbor_fields:
                    neighbor_data = df_results[(df_results['field'] == neighbor) & 
                                             (df_results['splus_filter'] == filter_name)]
                    if len(neighbor_data) > 0 and neighbor_data['n_stars'].iloc[0] >= 30:
                        neighbor_offsets.append(neighbor_data['median_diff'].iloc[0])
                
                if len(neighbor_offsets) >= 1:
                    # Usar promedio de vecinos
                    recommended_offset = np.mean(neighbor_offsets)
                    recommendation_source = 'vecinos'
                    recommendation_notes = f'Promedio de {len(neighbor_offsets)} vecino(s)'
                else:
                    # Usar offset global
                    recommended_offset = global_offsets.get(filter_name, 0.0)
                    recommendation_source = 'global'
                    recommendation_notes = 'Sin vecinos disponibles'
            
            final_recommendations.append({
                'field': field,
                'filter': filter_name,
                'n_stars': field_data['n_stars'].mean(),
                'original_offset': filter_data['median_diff'].iloc[0],
                'recommended_offset': recommended_offset,
                'recommendation_source': recommendation_source,
                'recommendation_notes': recommendation_notes,
                'is_problematic': is_problematic
            })
        
        # Imprimir resumen por campo
        field_recs = [r for r in final_recommendations if r['field'] == field]
        sources = [r['recommendation_source'] for r in field_recs]
        unique_sources = set(sources)
        
        status = "⚠️ PROBLEMÁTICO" if is_problematic else "✅ BUENO"
        print(f"   {field}: {status} ({int(field_n_stars)} estrellas) - Fuentes: {', '.join(unique_sources)}")
    
    # Crear DataFrame final
    df_final = pd.DataFrame(final_recommendations)
    
    # Guardar recomendaciones
    df_final.to_csv(f"{output_dir}/final_offset_recommendations.csv", index=False)
    
    # Crear tabla pivote para uso fácil
    pivot_final = df_final.pivot_table(
        values='recommended_offset', 
        index='field', 
        columns='filter', 
        aggfunc='first'
    )
    pivot_final.to_csv(f"{output_dir}/final_offset_table.csv")
    
    print(f"\n💾 Archivos finales guardados:")
    print(f"   - final_offset_recommendations.csv (detallado)")
    print(f"   - final_offset_table.csv (tabla para uso)")
    
    return df_final

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
            if results:  # Solo añadir si hay resultados
                all_results.extend(results)
                all_plots.extend(plots)
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("No se generaron resultados. Verifica los archivos de entrada.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Crear DataFrames con resultados
    df_results = pd.DataFrame(all_results)
    
    # Calcular offsets robustos (usando mínimo 30 estrellas por campo)
    df_offsets, good_df, problematic_df = calculate_robust_offsets(all_results, min_stars_threshold=30)
    
    # Crear gráficos resumen
    create_summary_plots(all_results, output_dir)
    
    # Crear recomendaciones finales
    df_final = create_final_offset_recommendations(df_results, df_offsets, problematic_df, output_dir)
    
    # Guardar resultados
    df_results.to_csv(f"{output_dir}/homogenization_detailed_results.csv", index=False)
    if not df_offsets.empty:
        df_offsets.to_csv(f"{output_dir}/homogenization_recommended_offsets.csv", index=False)
    
    # Imprimir resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL DE HOMOGENIZACIÓN")
    print("="*60)
    
    if not df_offsets.empty:
        print("\n📊 OFFSETS GLOBALES RECOMENDADOS (basados en campos buenos):")
        for _, row in df_offsets.iterrows():
            print(f"   {row['filter']}: {row['recommended_offset']:.3f} "
                  f"(±{row['mad_across_fields']:.3f}) - "
                  f"{row['n_good_fields']}/{row['n_total_fields']} campos, "
                  f"{row['n_stars_total']} estrellas")
    
    problematic_fields = problematic_df['field'].unique()
    if len(problematic_fields) > 0:
        print(f"\n⚠️  CAMPOS PROBLEMÁTICOS (<30 estrellas): {list(problematic_fields)}")
        print("   Para estos campos se usan offsets de campos vecinos o el offset global")
    
    print(f"\n💾 TODOS LOS RESULTADOS GUARDADOS EN: {output_dir}/")
    print(f"   - homogenization_detailed_results.csv")
    print(f"   - homogenization_recommended_offsets.csv") 
    print(f"   - final_offset_recommendations.csv")
    print(f"   - final_offset_table.csv")
    print(f"   - Gráficos individuales y resumen")
    
    return df_results, df_offsets, df_final

if __name__ == "__main__":
    df_results, df_offsets, df_final = main()
