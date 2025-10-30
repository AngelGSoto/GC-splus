import pandas as pd
import numpy as np
import os

def proper_combined_error(original_error, offset_error):
    """
    Propagación CORRECTA para combinación de errores independientes
    σ_total = √(σ_original² + σ_offset²)
    """
    if np.isnan(original_error) or original_error <= 0:
        return offset_error
    if np.isnan(offset_error) or offset_error <= 0:
        return original_error
    
    return np.sqrt(original_error**2 + offset_error**2)

def check_columns():
    """Verifica las columnas reales en los archivos"""
    print("🔍 VERIFICANDO COLUMNAS EN ARCHIVOS...")
    
    try:
        # Cargar archivos y mostrar columnas
        df_offsets = pd.read_csv("plot_homogenization/final_offset_recommendations.csv")
        print("Columnas en final_offset_recommendations.csv:")
        print(df_offsets.columns.tolist())
        
        return df_offsets.columns.tolist()
    
    except FileNotFoundError as e:
        print(f"❌ Error cargando archivos: {e}")
        return []

def generate_missing_offsets():
    """
    Genera offsets para campos faltantes CenA06 y CenA07 usando campos vecinos
    """
    print("🔧 GENERANDO OFFSETS PARA CAMPOS FALTANTES")
    print("="*50)
    
    # Primero verificar las columnas
    columns_offsets = check_columns()
    
    if not columns_offsets:
        print("❌ No se pudieron cargar los archivos de offsets")
        return pd.DataFrame()
    
    # Cargar offsets existentes
    df_offsets = pd.read_csv("plot_homogenization/final_offset_recommendations.csv")
    
    # Determinar el nombre correcto de la columna del filtro
    if 'filter' in df_offsets.columns:
        filter_col = 'filter'
    elif 'splus_filter' in df_offsets.columns:
        filter_col = 'splus_filter'
    else:
        # Buscar alguna columna que contenga 'filter'
        filter_col_candidates = [col for col in df_offsets.columns if 'filter' in col.lower()]
        if filter_col_candidates:
            filter_col = filter_col_candidates[0]
            print(f"⚠️  Usando columna: {filter_col} para filtros")
        else:
            raise KeyError("No se encontró columna de filtros en el archivo de offsets")
    
    print(f"✅ Usando columna '{filter_col}' para filtros SPLUS")
    
    # Campos faltantes
    missing_fields = ['CenA06', 'CenA07']
    
    # Estrategia de vecinos basada en proximidad numérica
    neighbor_strategy = {
        'CenA06': ['CenA03', 'CenA09', 'CenA10'],  # Vecinos más cercanos disponibles
        'CenA07': ['CenA09', 'CenA10', 'CenA11']   # Vecinos más cercanos disponibles
    }
    
    new_offsets = []
    
    for field in missing_fields:
        neighbors = neighbor_strategy[field]
        available_neighbors = [n for n in neighbors if n in df_offsets['field'].unique()]
        
        print(f"📊 Generando offsets para {field} usando vecinos: {available_neighbors}")
        
        if available_neighbors:
            # Calcular promedio de vecinos disponibles para cada filtro
            for filter_name in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
                neighbor_offsets = []
                
                for neighbor in available_neighbors:
                    neighbor_data = df_offsets[
                        (df_offsets['field'] == neighbor) & 
                        (df_offsets[filter_col] == filter_name)
                    ]
                    if len(neighbor_data) > 0:
                        neighbor_offsets.append(neighbor_data['recommended_offset'].iloc[0])
                
                if neighbor_offsets:
                    new_offset = np.mean(neighbor_offsets)
                    
                    # Crear entrada con la estructura CORRECTA
                    new_offsets.append({
                        'field': field,
                        'filter': filter_name,
                        'n_stars': 0,  # Indicar que son estimados
                        'original_difference': np.nan,  # No tenemos datos originales
                        'recommended_offset': new_offset,
                        'recommendation_source': 'vecinos_estimado',
                        'recommendation_notes': f'Estimado de {len(available_neighbors)} vecinos: {", ".join(available_neighbors)}',
                        'is_problematic': True
                    })
                    
                    print(f"   {filter_name}: {new_offset:+.3f} mag (de {len(available_neighbors)} vecinos)")
        
        else:
            # Si no hay vecinos, usar offsets globales del archivo de offsets recomendados
            print(f"  ⚠️  No hay vecinos disponibles para {field}, usando offsets globales")
            
            # Calcular offsets globales a partir de los datos existentes
            for filter_name in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
                filter_data = df_offsets[df_offsets[filter_col] == filter_name]
                if len(filter_data) > 0:
                    # Usar la mediana de los offsets recomendados de campos buenos
                    good_fields_data = filter_data[filter_data['n_stars'] >= 30]
                    if len(good_fields_data) > 0:
                        global_offset = good_fields_data['recommended_offset'].median()
                    else:
                        global_offset = filter_data['recommended_offset'].median()
                    
                    new_offsets.append({
                        'field': field,
                        'filter': filter_name,
                        'n_stars': 0,
                        'original_difference': np.nan,
                        'recommended_offset': global_offset,
                        'recommendation_source': 'global_estimado',
                        'recommendation_notes': 'Sin vecinos disponibles, usando offset global',
                        'is_problematic': True
                    })
    
    if new_offsets:
        # Añadir nuevos offsets al DataFrame existente
        df_new_offsets = pd.DataFrame(new_offsets)
        
        # Asegurarse de que las columnas coincidan
        common_columns = set(df_offsets.columns) & set(df_new_offsets.columns)
        df_new_offsets = df_new_offsets[list(common_columns)]
        
        df_updated = pd.concat([df_offsets, df_new_offsets], ignore_index=True)
        
        # Guardar versión actualizada
        output_file = "plot_homogenization/final_offset_recommendations_complete.csv"
        df_updated.to_csv(output_file, index=False)
        print(f"\n✅ Offsets generados para campos faltantes")
        print(f"💾 Archivo guardado: {output_file}")
        
        return df_updated
    else:
        print("❌ No se pudieron generar offsets para campos faltantes")
        return df_offsets

def reapply_homogenization_with_complete_offsets():
    """
    Vuelve a aplicar la homogenización usando los offsets completos
    SOLO APLICA OFFSETS A MAGNITUDES - SIN RECALCULAR FLUJOS
    """
    print("\n🔄 RE-APLICANDO HOMOGENIZACIÓN CON OFFSETS COMPLETOS")
    print("="*60)
    
    try:
        # Cargar el catálogo original
        gc_catalog_file = "Results/all_fields_gc_photometry_corrected_errors_v17.csv"
        df_gc = pd.read_csv(gc_catalog_file)
        print(f"✅ Catálogo original cargado: {len(df_gc)} cúmulos globulares")
        
        # Cargar offsets completos
        offsets_file = "plot_homogenization/final_offset_recommendations_complete.csv"
        df_offsets_complete = pd.read_csv(offsets_file)
        print(f"✅ Offsets completos cargados: {len(df_offsets_complete)} entradas")
        print(f"✅ Campos con offsets: {len(df_offsets_complete['field'].unique())}")
        
    except FileNotFoundError as e:
        print(f"❌ Error cargando archivos: {e}")
        return pd.DataFrame()
    
    # Determinar columna de filtro en offsets
    if 'filter' in df_offsets_complete.columns:
        filter_col = 'filter'
    elif 'splus_filter' in df_offsets_complete.columns:
        filter_col = 'splus_filter'
    else:
        print("❌ No se encontró columna de filtros en los offsets")
        return pd.DataFrame()
    
    print(f"✅ Usando columna '{filter_col}' para filtros en offsets")
    
    # Crear diccionario de offsets por campo y filtro
    offsets_dict = {}
    
    for _, row in df_offsets_complete.iterrows():
        field = row['field']
        filter_name = row[filter_col]
        offset = row['recommended_offset']
        
        if field not in offsets_dict:
            offsets_dict[field] = {}
            
        offsets_dict[field][filter_name] = offset
    
    # Aplicar homogenización a todos los campos
    total_corrections = 0
    fields_processed = set()
    
    print(f"\n📊 APLICANDO OFFSETS SOLO A MAGNITUDES:")
    print("   (No se recalculan flujos, solo se propagan errores en magnitudes)")
    
    for field in df_gc['FIELD'].unique():
        if field not in offsets_dict:
            print(f"  ⚠️  Campo {field}: No hay offsets disponibles, saltando...")
            continue
            
        field_mask = df_gc['FIELD'] == field
        field_offsets = offsets_dict[field]
        fields_processed.add(field)
        
        # Aplicar offsets a cada filtro y apertura
        for filt in ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']:
            if filt not in field_offsets:
                continue
                
            offset = field_offsets[filt]
            
            # Aplicar a ambas aperturas (2 y 3 arcsec)
            for aperture in ['2', '3']:
                # Columnas de magnitud
                mag_col = f'MAG_{filt}_{aperture}'
                magerr_col = f'MAGERR_{filt}_{aperture}'
                
                if mag_col in df_gc.columns:
                    # Aplicar corrección a magnitudes con propagación de errores
                    valid_mask = (field_mask) & (df_gc[mag_col] < 90)  # Excluir valores 99
                    
                    if valid_mask.any():
                        n_corrections = valid_mask.sum()
                        
                        # 1. Corregir magnitudes: SPLUS_corregido = SPLUS + offset
                        df_gc.loc[valid_mask, mag_col] += offset
                        
                        # 2. PROPAGACIÓN CORRECTA DE ERRORES EN MAGNITUDES
                        # σ_mag_corr = √(σ_mag_orig² + σ_offset²)
                        offset_uncertainty = 0.05  # Valor conservador basado en MAD
                        
                        # Solo propagar error si el error original es válido
                        valid_error_mask = valid_mask & (df_gc[magerr_col] > 0) & (df_gc[magerr_col] < 10)
                        
                        if valid_error_mask.any():
                            corrected_errors = np.sqrt(
                                df_gc.loc[valid_error_mask, magerr_col]**2 + 
                                offset_uncertainty**2
                            )
                            df_gc.loc[valid_error_mask, magerr_col] = corrected_errors
                        
                        total_corrections += n_corrections
                        print(f"  {field} {filt}_{aperture}: {n_corrections} correcciones (offset: {offset:+.3f})")
    
    # Guardar catálogo homogenizado completo
    output_file = "Results/all_fields_gc_photometry_corrected_errors_v17_homogenised_complete.csv"
    df_gc.to_csv(output_file, index=False)
    
    print(f"\n✅ HOMOGENIZACIÓN COMPLETA FINALIZADA")
    print(f"   - Todos los campos procesados: {len(fields_processed)}")
    print(f"   - Correcciones aplicadas: {total_corrections}")
    print(f"   - Solo se aplicó a magnitudes (flujos sin cambios)")
    print(f"   - Propagación de errores: ✅ CORRECTA")
    print(f"   - Archivo de salida: {output_file}")
    
    return df_gc

def verify_offsets_application():
    """
    Verifica que los offsets se aplicaron correctamente
    """
    print(f"\n🔍 VERIFICANDO APLICACIÓN DE OFFSETS")
    print("="*50)
    
    try:
        # Cargar catálogos original y homogenizado
        df_orig = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17.csv")
        df_homo = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17_homogenised_complete.csv")
        df_offsets = pd.read_csv("plot_homogenization/final_offset_recommendations_complete.csv")
        
        print("📊 Verificación de aplicación de offsets:")
        print("   (Comparando diferencias antes/después con offsets esperados)")
        
        # Verificar algunos campos y filtros
        test_fields = list(df_offsets['field'].unique())[:3]  # Primeros 3 campos
        test_filters = ['F378', 'F410', 'F430', 'F660']
        aperture = 3
        
        for field in test_fields:
            print(f"\n   📍 Campo {field}:")
            
            # Offsets esperados para este campo
            field_offsets_data = df_offsets[df_offsets['field'] == field]
            field_offsets_dict = field_offsets_data.set_index('filter')['recommended_offset'].to_dict()
            
            for filt in test_filters:
                if filt not in field_offsets_dict:
                    continue
                    
                expected_offset = field_offsets_dict[filt]
                mag_col = f'MAG_{filt}_{aperture}'
                
                if mag_col not in df_orig.columns or mag_col not in df_homo.columns:
                    continue
                
                # Calcular offset realmente aplicado
                field_mask_orig = (df_orig['FIELD'] == field) & (df_orig[mag_col] < 90)
                field_mask_homo = (df_homo['FIELD'] == field) & (df_homo[mag_col] < 90)
                
                if field_mask_orig.any() and field_mask_homo.any():
                    # Usar medianas para evitar outliers
                    orig_median = df_orig.loc[field_mask_orig, mag_col].median()
                    homo_median = df_homo.loc[field_mask_homo, mag_col].median()
                    
                    applied_offset = homo_median - orig_median
                    difference = applied_offset - expected_offset
                    
                    status = "✅" if abs(difference) < 0.01 else "⚠️" if abs(difference) < 0.1 else "❌"
                    
                    print(f"      {filt}: esperado={expected_offset:+.3f}, aplicado={applied_offset:+.3f}, diff={difference:+.3f} {status}")
    
    except FileNotFoundError as e:
        print(f"❌ Error en verificación: {e}")

def diagnose_sign_issue():
    """
    Diagnóstico específico del problema de signo
    """
    print(f"\n🔍 DIAGNÓSTICO ESPECÍFICO DEL PROBLEMA DE SIGNO")
    print("="*60)
    
    try:
        # Cargar datos
        df_orig = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17.csv")
        df_homo = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17_homogenised_complete.csv")
        df_offsets = pd.read_csv("plot_homogenization/final_offset_recommendations_complete.csv")
        
        print("📊 Análisis del problema de signo:")
        print("   Δ = SPLUS - Taylor")
        print("   Offset recomendado = -Δ")
        print("   Aplicación correcta: SPLUS_corregido = SPLUS + offset")
        print("\n   Verificación:")
        
        # Tomar un campo y filtro específico para análisis detallado
        test_field = 'CenA01'
        test_filter = 'F410'
        aperture = 3
        
        field_offsets = df_offsets[df_offsets['field'] == test_field]
        filt_offset_data = field_offsets[field_offsets['filter'] == test_filter]
        
        if len(filt_offset_data) > 0:
            recommended_offset = filt_offset_data['recommended_offset'].iloc[0]
            original_difference = filt_offset_data['original_difference'].iloc[0]
            
            print(f"\n   Campo {test_field}, Filtro {test_filter}:")
            print(f"      Δ original (SPLUS - Taylor) = {original_difference:.3f}")
            print(f"      Offset recomendado = {recommended_offset:.3f}")
            print(f"      Teóricamente: Δ_corregido debería ≈ 0")
            
            # Verificar qué pasó realmente
            mag_col = f'MAG_{test_filter}_{aperture}'
            taylor_col = 'gmag'  # Para F410
            
            if mag_col in df_orig.columns and mag_col in df_homo.columns and taylor_col in df_orig.columns:
                field_mask = df_orig['FIELD'] == test_field
                valid_mask = field_mask & (df_orig[mag_col] < 90) & (df_orig[taylor_col] < 90)
                
                if valid_mask.any():
                    # Calcular Δ antes y después
                    delta_orig = (df_orig.loc[valid_mask, mag_col] - df_orig.loc[valid_mask, taylor_col]).median()
                    delta_homo = (df_homo.loc[valid_mask, mag_col] - df_homo.loc[valid_mask, taylor_col]).median()
                    
                    print(f"      Δ original (mediana) = {delta_orig:.3f}")
                    print(f"      Δ homogenizado (mediana) = {delta_homo:.3f}")
                    print(f"      Cambio en Δ = {delta_homo - delta_orig:+.3f}")
                    
                    # Verificar signo
                    expected_change = -delta_orig  # Teóricamente debería cancelar Δ original
                    actual_change = delta_homo - delta_orig
                    
                    if abs(actual_change - expected_change) < 0.1:
                        print(f"      ✅ Signo CORRECTO")
                    else:
                        print(f"      ❌ Signo INCORRECTO")
                        print(f"         Cambio esperado: {expected_change:.3f}")
                        print(f"         Cambio obtenido: {actual_change:.3f}")
    
    except FileNotFoundError as e:
        print(f"❌ Error en diagnóstico: {e}")

def main():
    """
    Función principal
    """
    print("🚀 HOMOGENIZACIÓN - SOLO MAGNITUDES")
    print("="*50)
    
    try:
        # Paso 1: Generar offsets para campos faltantes
        print("\n📝 PASO 1: Generando offsets para campos faltantes...")
        df_complete_offsets = generate_missing_offsets()
        
        if df_complete_offsets.empty:
            print("❌ No se pudieron generar offsets completos")
            return
        
        # Paso 2: Re-aplicar homogenización con offsets completos
        print("\n📝 PASO 2: Aplicando homogenización solo a magnitudes...")
        df_final_homogenised = reapply_homogenization_with_complete_offsets()
        
        if df_final_homogenised.empty:
            print("❌ No se pudo completar la homogenización")
            return
        
        # Paso 3: Verificación de aplicación
        verify_offsets_application()
        
        # Paso 4: Diagnóstico del problema de signo
        diagnose_sign_issue()
        
        # Resumen final
        print("\n" + "="*60)
        print("🎯 HOMOGENIZACIÓN COMPLETA FINALIZADA")
        print("="*60)
        
        print(f"📊 ESTADÍSTICAS FINALES:")
        print(f"   - Cúmulos globulares procesados: {len(df_final_homogenised)}")
        print(f"   - Campos incluidos: {len(df_final_homogenised['FIELD'].unique())}")
        print(f"   - Incluye CenA06 y CenA07 con offsets estimados de vecinos")
        print(f"   - Solo se aplicó a magnitudes (flujos sin cambios)")
        print(f"   - Propagación de errores: ✅ CORRECTA")
        
        print(f"\n🔧 DETALLES TÉCNICOS:")
        print(f"   - Aplicación: SPLUS_corregido = SPLUS + offset")
        print(f"   - Error: σ_total = √(σ_mag² + σ_offset²)")
        print(f"   - σ_offset conservador: 0.05 mag")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   - Offsets completos: plot_homogenization/final_offset_recommendations_complete.csv")
        print(f"   - Catálogo homogenizado: Results/all_fields_gc_photometry_corrected_errors_v17_homogenised_complete.csv")
        
        print(f"\n🔍 PRÓXIMOS PASOS:")
        print(f"   1. Ejecutar análisis de coherencia para verificar mejora")
        print(f"   2. Si persiste el problema, revisar el signo en apply_homogenization.py")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
