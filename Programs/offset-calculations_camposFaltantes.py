import pandas as pd
import numpy as np

def generate_missing_offsets():
    """
    Genera offsets para campos faltantes usando campos vecinos
    """
    # Cargar offsets existentes
    df_offsets = pd.read_csv("plot_homogenization/final_offset_recommendations.csv")
    df_global = pd.read_csv("plot_homogenization/homogenization_recommended_offsets.csv")
    
    # Campos faltantes
    missing_fields = ['CenA06', 'CenA07']
    
    # Estrategia: usar campos vecinos
    neighbor_strategy = {
        'CenA06': ['CenA05', 'CenA07', 'CenA10'],  # Vecinos aproximados
        'CenA07': ['CenA06', 'CenA08', 'CenA11']   # Vecinos aproximados
    }
    
    new_offsets = []
    
    for field in missing_fields:
        neighbors = neighbor_strategy[field]
        available_neighbors = [n for n in neighbors if n in df_offsets['field'].unique()]
        
        print(f"Generando offsets para {field} usando vecinos: {available_neighbors}")
        
        if available_neighbors:
            # Calcular promedio de vecinos disponibles
            for filter_name in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
                neighbor_offsets = []
                for neighbor in available_neighbors:
                    neighbor_data = df_offsets[
                        (df_offsets['field'] == neighbor) & 
                        (df_offsets['splus_filter'] == filter_name)
                    ]
                    if len(neighbor_data) > 0:
                        neighbor_offsets.append(neighbor_data['recommended_offset'].iloc[0])
                
                if neighbor_offsets:
                    new_offset = np.mean(neighbor_offsets)
                    # Usar la misma incertidumbre que el promedio global
                    global_uncertainty = df_global[df_global['filter'] == filter_name]['mad_across_fields'].iloc[0]
                    
                    new_offsets.append({
                        'field': field,
                        'splus_filter': filter_name,
                        'taylor_filter': df_offsets[df_offsets['splus_filter'] == filter_name]['taylor_filter'].iloc[0],
                        'aperture': 3,
                        'n_stars': 0,  # Indicar que son estimados
                        'mean_diff': new_offset,
                        'median_diff': new_offset,
                        'std_diff': global_uncertainty,
                        'mad_diff': global_uncertainty,
                        'min_diff': new_offset,
                        'max_diff': new_offset,
                        'slope': np.nan,
                        'intercept': np.nan,
                        'r_value': np.nan,
                        'r_squared': np.nan,
                        'p_value': np.nan,
                        'std_err': np.nan,
                        'recommended_offset': new_offset,
                        'recommendation_source': 'vecinos_estimado',
                        'recommendation_notes': f'Estimado de {len(available_neighbors)} vecinos',
                        'is_problematic': True
                    })
        
        else:
            # Si no hay vecinos, usar offsets globales
            print(f"  ⚠️  No hay vecinos disponibles para {field}, usando offsets globales")
            for filter_name in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
                global_data = df_global[df_global['filter'] == filter_name]
                if len(global_data) > 0:
                    new_offset = global_data['recommended_offset'].iloc[0]
                    global_uncertainty = global_data['mad_across_fields'].iloc[0]
                    
                    new_offsets.append({
                        'field': field,
                        'splus_filter': filter_name,
                        'taylor_filter': df_offsets[df_offsets['splus_filter'] == filter_name]['taylor_filter'].iloc[0],
                        'aperture': 3,
                        'n_stars': 0,
                        'mean_diff': new_offset,
                        'median_diff': new_offset,
                        'std_diff': global_uncertainty,
                        'mad_diff': global_uncertainty,
                        'min_diff': new_offset,
                        'max_diff': new_offset,
                        'slope': np.nan,
                        'intercept': np.nan,
                        'r_value': np.nan,
                        'r_squared': np.nan,
                        'p_value': np.nan,
                        'std_err': np.nan,
                        'recommended_offset': new_offset,
                        'recommendation_source': 'global_estimado',
                        'recommendation_notes': 'Sin datos, usando promedio global',
                        'is_problematic': True
                    })
    
    if new_offsets:
        # Añadir nuevos offsets al DataFrame
        df_new_offsets = pd.DataFrame(new_offsets)
        df_updated = pd.concat([df_offsets, df_new_offsets], ignore_index=True)
        
        # Guardar versión actualizada
        df_updated.to_csv("plot_homogenization/final_offset_recommendations_complete.csv", index=False)
        print(f"✅ Offsets generados para campos faltantes")
        print(f"💾 Archivo guardado: final_offset_recommendations_complete.csv")
        
        return df_updated
    else:
        print("❌ No se pudieron generar offsets para campos faltantes")
        return df_offsets

# Ejecutar la corrección
df_complete = generate_missing_offsets()
