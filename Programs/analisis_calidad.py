import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def aanda_validation_suite():
    """
    Suite de validación específica para requisitos de A&A
    """
    # Cargar los datos desde los archivos guardados
    try:
        df_results = pd.read_csv('plot_homogenization/homogenization_detailed_results.csv')
        df_final = pd.read_csv('plot_homogenization/final_offset_recommendations.csv')
    except FileNotFoundError as e:
        print(f"Error: No se encontraron los archivos de resultados. Asegúrate de haber ejecutado el script principal primero. {e}")
        return

    print("="*70)
    print("VALIDACIÓN PARA PUBLICACIÓN EN A&A")
    print("="*70)
    
    # 1. Test de normalidad de residuos
    print("\n1. TEST DE NORMALIDAD DE RESIDUOS:")
    for filt in df_results['splus_filter'].unique():
        filt_data = df_results[df_results['splus_filter'] == filt]
        stat, p_value = stats.normaltest(filt_data['median_diff'])
        print(f"   {filt}: p-value = {p_value:.4f} {'✅' if p_value > 0.05 else '⚠️'}")
    
    # 2. Análisis de correlación espacial
    print("\n2. CORRELACIÓN ESPACIAL DE OFFSETS:")
    # Por ahora, no tenemos coordenadas de campos, así que omitimos este análisis.
    # En una versión futura podrías agregar coordenadas RA/DEC de los campos.
    print("   (Omitido por falta de coordenadas de campos)")
    
    # 3. Validación con submuestreo
    print("\n3. VALIDACIÓN POR SUBMUESTREO:")
    validation_results = []
    for filt in df_results['splus_filter'].unique():
        filt_data = df_results[df_results['splus_filter'] == filt]
        
        # Bootstrap de campos
        n_bootstraps = 1000
        bootstrap_offsets = []
        
        for _ in range(n_bootstraps):
            sample = filt_data.sample(frac=0.7, replace=True)  # 70% de campos
            if len(sample) > 0:
                bootstrap_offsets.append(sample['median_diff'].median())
        
        bootstrap_mean = np.mean(bootstrap_offsets)
        bootstrap_std = np.std(bootstrap_offsets)
        
        validation_results.append({
            'filter': filt,
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'n_bootstraps': n_bootstraps
        })
        
        print(f"   {filt}: {bootstrap_mean:.3f} ± {bootstrap_std:.3f}")
    
    # 4. Análisis de sensibilidad a umbral de estrellas
    print("\n4. SENSIBILIDAD AL UMBRAL DE ESTRELLAS:")
    thresholds = [20, 30, 40, 50]
    sensitivity_analysis = []
    
    for threshold in thresholds:
        good_data = df_results[df_results['n_stars'] >= threshold]
        if len(good_data) > 0:
            offsets_by_threshold = good_data.groupby('splus_filter')['median_diff'].median()
            sensitivity_analysis.append({
                'threshold': threshold,
                'offsets': offsets_by_threshold.to_dict(),
                'n_fields': len(good_data['field'].unique())
            })
    
    # 5. Crear figuras de validación para el paper
    create_validation_figures(df_results, validation_results, sensitivity_analysis, df_final)
    
    return validation_results

def create_validation_figures(df_results, validation_results, sensitivity_analysis, df_final):
    """
    Crear figuras específicas para el paper
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Figura 1: Estabilidad bootstrap
    axes[0, 0].bar([r['filter'] for r in validation_results], 
                   [r['bootstrap_mean'] for r in validation_results],
                   yerr=[r['bootstrap_std'] for r in validation_results],
                   capsize=5, alpha=0.7)
    axes[0, 0].set_ylabel('Offset (mag)')
    axes[0, 0].set_title('Estabilidad Bootstrap de Offsets')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Figura 2: Sensibilidad al umbral
    for filt in df_results['splus_filter'].unique():
        thresholds = [s['threshold'] for s in sensitivity_analysis]
        offsets = [s['offsets'].get(filt, np.nan) for s in sensitivity_analysis]
        axes[0, 1].plot(thresholds, offsets, 'o-', label=filt, markersize=8)
    axes[0, 1].set_xlabel('Umbral de estrellas')
    axes[0, 1].set_ylabel('Offset (mag)')
    axes[0, 1].set_title('Sensibilidad al Umbral de Estrellas')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Figura 3: Distribución de offsets por filtro
    for i, filt in enumerate(df_results['splus_filter'].unique()):
        filt_data = df_results[df_results['splus_filter'] == filt]
        axes[1, 0].boxplot(filt_data['median_diff'], positions=[i], widths=0.6)
    axes[1, 0].set_xticks(range(len(df_results['splus_filter'].unique())))
    axes[1, 0].set_xticklabels(df_results['splus_filter'].unique())
    axes[1, 0].set_ylabel('Offset (mag)')
    axes[1, 0].set_title('Distribución de Offsets por Filtro')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Figura 4: Mapa de campos problemáticos
    problematic_fields = df_final[df_final['is_problematic']]['field'].unique()
    all_fields = sorted(df_results['field'].unique())
    
    # Simular distribución espacial (podrías usar coordenadas reales)
    field_avg_offsets = df_results.groupby('field')['median_diff'].mean()
    
    colors = ['red' if field in problematic_fields else 'blue' for field in all_fields]
    
    for i, field in enumerate(all_fields):
        avg_offset = field_avg_offsets.get(field, 0)
        axes[1, 1].scatter(i, avg_offset, c=colors[i], s=100, alpha=0.7)
        axes[1, 1].text(i, avg_offset + 0.1, field, ha='center', fontsize=8)
    
    axes[1, 1].axhline(0, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Campo (orden arbitrario)')
    axes[1, 1].set_ylabel('Offset promedio')
    axes[1, 1].set_title('Mapa de campos problemáticos\n(Rojo: <30 estrellas)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plot_homogenization/aanda_validation_figures.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_aanda_tables():
    """
    Genera tablas en formato apropiado para A&A
    """
    # Cargar datos
    df_offsets = pd.read_csv('plot_homogenization/homogenization_recommended_offsets.csv')
    df_final = pd.read_csv('plot_homogenization/final_offset_recommendations.csv')
    
    print("\n" + "="*70)
    print("TABLAS PARA PUBLICACIÓN EN A&A")
    print("="*70)
    
    # Tabla 1: Offsets globales recomendados
    print("\nTabla 1: Offsets de homogenización globales recomendados")
    print("-" * 80)
    print("Filtro  |  Offset (mag) |  Incertidumbre (MAD) |  N° Campos |  N° Estrellas")
    print("-" * 80)
    for _, row in df_offsets.iterrows():
        print(f"{row['filter']:6} | {row['recommended_offset']:13.3f} | {row['mad_across_fields']:21.3f} | "
              f"{row['n_good_fields']:11} | {row['n_stars_total']:12}")
    
    # Tabla 2: Resumen por campo
    print("\n\nTabla 2: Resumen de homogenización por campo")
    print("-" * 100)
    print("Campo  |  N° Estrellas |  Estado      |  Fuente offset  |  Offset promedio")
    print("-" * 100)
    
    field_summary = df_final.groupby('field').agg({
        'n_stars': 'first',
        'is_problematic': 'first',
        'recommendation_source': 'first',
        'recommended_offset': 'mean'
    }).round(3)
    
    for field, data in field_summary.iterrows():
        status = "Problemático" if data['is_problematic'] else "Bueno"
        print(f"{field:6} | {data['n_stars']:13} | {status:12} | {data['recommendation_source']:15} | {data['recommended_offset']:16.3f}")
    
    # Guardar tablas en archivos LaTeX
    with open('plot_homogenization/aanda_table_offsets.tex', 'w') as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\caption{Offsets de homogenización recomendados entre fotometría S-PLUS y el catálogo de Taylor et al. (2017).}\n")
        f.write("\\label{tab:offsets}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Filtro & Offset (mag) & $\\sigma_{\\mathrm{MAD}}$ (mag) & N\\textsuperscript{o} Campos \\\\\n")
        f.write("\\hline\n")
        for _, row in df_offsets.iterrows():
            f.write(f"{row['filter']} & {row['recommended_offset']:.3f} & {row['mad_across_fields']:.3f} & {row['n_good_fields']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n💾 Tablas LaTeX guardadas en:")
    print(f"   - plot_homogenization/aanda_table_offsets.tex")

if __name__ == "__main__":
    # Ejecutar validación
    validation_results = aanda_validation_suite()
    
    # Generar tablas para el paper
    generate_aanda_tables()
    
    print(f"\n🎯 VALIDACIÓN COMPLETADA")
    print("Revisa los archivos generados en 'plot_homogenization/'")
    print("- aanda_validation_figures.png")
    print("- aanda_table_offsets.tex")
