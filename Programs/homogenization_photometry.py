import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from astropy.stats import mad_std

def calculate_and_apply_proper_homogenization():
    """
    Calcula y aplica offsets de homogenización SIN PROPAGAR ERRORES DEL OFFSET
    """
    
    # Leer archivo original SIN homogenización
    df = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17.csv")
    
    print("🔍 CALCULANDO OFFSETS CON MAPEO FÍSICO MEJORADO")
    print("=" * 60)
    
    # MAPEO MEJORADO basado en la física real de los filtros
    filter_mapping = {
        'MAG_F378_3': 'umag',    # F378 (378 nm) → u (354 nm)
        'MAG_F395_3': 'umag',    # F395 (395 nm) → u (354 nm)  
        'MAG_F410_3': 'umag',    # F410 (410 nm) → u (354 nm)
        'MAG_F430_3': 'umag',    # F430 (430 nm) → u (354 nm)
        'MAG_F515_3': 'gmag',    # F515 (515 nm) → g (477 nm)
        'MAG_F660_3': 'rmag',    # F660 (660 nm) → r (622 nm)
        'MAG_F861_3': 'imag'     # F861 (861 nm) → i (763 nm)
    }
    
    offsets = {}
    uncertainties = {}
    
    print("\n📈 CALCULANDO OFFSETS:")
    print("=" * 50)
    
    # Calcular offsets como: offset = median(Taylor - SPLUS)
    for splus_col, taylor_col in filter_mapping.items():
        if splus_col not in df.columns or taylor_col not in df.columns:
            continue
            
        # Filtrar datos válidos
        mask = (
            df[splus_col].notna() & 
            df[taylor_col].notna() &
            (df[splus_col] < 90) & (df[taylor_col] < 90) &
            (df[splus_col] > 10) & (df[taylor_col] > 10)
        )
        
        valid_data = df.loc[mask]
        if len(valid_data) < 10:
            continue
        
        # Diferencia: Taylor - SPLUS
        differences = valid_data[taylor_col] - valid_data[splus_col]
        valid_diff = differences[np.isfinite(differences)]
        
        if len(valid_diff) == 0:
            continue
            
        # Estadísticas
        median_offset = np.median(valid_diff)
        mad_uncertainty = mad_std(valid_diff)
        
        filter_name = splus_col.replace('MAG_', '').replace('_3', '')
        offsets[filter_name] = median_offset
        uncertainties[filter_name] = mad_uncertainty
        
        print(f"  {filter_name}: Offset = {median_offset:+.3f} ± {mad_uncertainty:.3f} mag")
    
    # Aplicar offsets al archivo original
    print("\n🔄 APLICANDO OFFSETS AL CATÁLOGO ORIGINAL")
    print("=" * 50)
    
    for filtro, offset in offsets.items():
        mag_column = f"MAG_{filtro}_3"
        
        if mag_column in df.columns:
            # Aplicar offset: SPLUS_corregido = SPLUS_original + offset
            df[mag_column] = df[mag_column] + offset
            
            # ⚠️ NO PROPAGAMOS ERRORES DEL OFFSET - mantenemos errores originales
            print(f"  ✅ {mag_column}: +{offset:+.3f} mag")
            print(f"     ✓ Errores originales mantenidos (sin propagación)")
    
    # Guardar archivo homogenizado
    output_path = "Results/all_fields_gc_photometry_properly_homogenized_v5.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Archivo homogenizado guardado: {output_path}")
    
    # Guardar tabla de offsets
    offsets_df = pd.DataFrame({
        'filter': list(offsets.keys()),
        'offset_mag': list(offsets.values()),
        'uncertainty_mag': [uncertainties[f] for f in offsets.keys()]
    })
    offsets_df.to_csv("Results/proper_homogenization_offsets_v5.csv", index=False)
    print(f"📋 Tabla de offsets: Results/proper_homogenization_offsets_v5.csv")
    
    return output_path, offsets

def verify_error_preservation(original_path, homogenized_path):
    """
    Verifica que los errores se mantengan iguales
    """
    print("\n🔍 VERIFICANDO CONSERVACIÓN DE ERRORES")
    print("=" * 50)
    
    df_orig = pd.read_csv(original_path)
    df_homog = pd.read_csv(homogenized_path)
    
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    
    print("COMPARACIÓN DE ERRORES ANTES/DESPUÉS:")
    print("Filtro  | Error_original | Error_homogenizado | Diferencia")
    print("-" * 65)
    
    for filtro in splus_filters:
        err_col = f'MAGERR_{filtro}_3'
        
        if err_col in df_orig.columns and err_col in df_homog.columns:
            # Usar medianas para comparación robusta
            orig_error = df_orig[err_col].median()
            homog_error = df_homog[err_col].median()
            diff = homog_error - orig_error
            
            print(f"{filtro:6} | {orig_error:13.3f} | {homog_error:17.3f} | {diff:+.3f}")
            
            if abs(diff) < 0.001:
                print(f"        ✅ Errores idénticos para {filtro}")
            else:
                print(f"        ⚠️  Pequeña diferencia para {filtro}")

def create_clean_cigale_input_with_original_errors():
    """
    Crea archivo de entrada para CIGALE manteniendo errores originales
    """
    print("\n🎯 CREANDO ARCHIVO CIGALE CON ERRORES ORIGINALES")
    print("=" * 50)
    
    df = pd.read_csv("Results/all_fields_gc_photometry_properly_homogenized_v5.csv")
    
    # Columnas para CIGALE
    cigale_columns = ['id', 'redshift']
    
    # Bandas Taylor
    taylor_bands = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
    
    # Bandas SPLUS homogenizadas
    splus_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    
    # Añadir bandas Taylor
    for band in taylor_bands:
        if band in df.columns:
            cigale_columns.append(band)
            if f'e_{band}' in df.columns:
                cigale_columns.append(f'{band}_err')
            elif f'{band}_err' in df.columns:
                cigale_columns.append(f'{band}_err')
    
    # Añadir bandas SPLUS
    for band in splus_bands:
        mag_col = f'MAG_{band}_3'
        err_col = f'MAGERR_{band}_3'
        if mag_col in df.columns:
            cigale_columns.append(band)
            if err_col in df.columns:
                cigale_columns.append(f'{band}_err')
    
    print(f"📋 Columnas seleccionadas para CIGALE: {cigale_columns}")
    
    # Crear DataFrame para CIGALE
    cigale_data = {}
    
    # Identificador
    if 'T17ID' in df.columns:
        cigale_data['id'] = df['T17ID']
    elif 'recno' in df.columns:
        cigale_data['id'] = df['recno']
    else:
        cigale_data['id'] = range(1, len(df) + 1)
    
    # Redshift
    cigale_data['redshift'] = 0.00183
    
    # Copiar datos de magnitudes Taylor
    for band in taylor_bands:
        if band in df.columns:
            cigale_data[band] = df[band]
            if f'e_{band}' in df.columns:
                cigale_data[f'{band}_err'] = df[f'e_{band}']
            elif f'{band}_err' in df.columns:
                cigale_data[f'{band}_err'] = df[f'{band}_err']
    
    # Copiar datos de magnitudes SPLUS homogenizadas CON ERRORES ORIGINALES
    for band in splus_bands:
        mag_col = f'MAG_{band}_3'
        err_col = f'MAGERR_{band}_3'
        
        if mag_col in df.columns:
            cigale_data[band] = df[mag_col]  # Magnitudes homogenizadas
            if err_col in df.columns:
                cigale_data[f'{band}_err'] = df[err_col]  # Errores ORIGINALES
    
    # Crear DataFrame
    cigale_df = pd.DataFrame(cigale_data)
    
    # Reordenar columnas
    available_columns = [col for col in cigale_columns if col in cigale_df.columns]
    cigale_df = cigale_df[available_columns]
    
    # Filtrar fuentes con mínimo de bandas válidas
    min_bands_required = 5
    magnitude_cols = [col for col in cigale_df.columns if not col.endswith('_err') and col not in ['id', 'redshift']]
    valid_bands = cigale_df[magnitude_cols].notna().sum(axis=1)
    cigale_df = cigale_df[valid_bands >= min_bands_required]
    
    output_path = "Results/cigale_input_ngc5128_clean_v5.csv"
    cigale_df.to_csv(output_path, index=False)
    
    print(f"💾 Archivo CIGALE guardado: {output_path}")
    print(f"📊 Estructura: {len(cigale_df)} fuentes, {len(cigale_df.columns)} columnas")
    print(f"🎯 Fuentes conservadas: {len(cigale_df)} (de {len(df)} originales)")
    
    return output_path

def analyze_original_errors():
    """
    Analiza la distribución de errores en el catálogo original
    """
    print("\n🔍 ANALIZANDO DISTRIBUCIÓN DE ERRORES ORIGINALES")
    print("=" * 50)
    
    df = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17.csv")
    
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    
    print("ESTADÍSTICAS DE ERRORES ORIGINALES:")
    print("Filtro  | Percentil 5% | Mediana | Percentil 95% | % < 1.0 mag")
    print("-" * 70)
    
    for filtro in splus_filters:
        err_col = f'MAGERR_{filtro}_3'
        
        if err_col in df.columns:
            errors = df[err_col]
            valid_errors = errors[(errors > 0) & (errors < 100)]
            
            if len(valid_errors) > 0:
                p5 = np.percentile(valid_errors, 5)
                median = np.median(valid_errors)
                p95 = np.percentile(valid_errors, 95)
                low_errors = (valid_errors < 1.0).sum() / len(valid_errors) * 100
                
                print(f"{filtro:6} | {p5:11.3f} | {median:7.3f} | {p95:12.3f} | {low_errors:10.1f}%")

if __name__ == "__main__":
    print("🎯 HOMOGENIZACIÓN CON ERRORES ORIGINALES")
    print("=" * 70)
    print("MEJORA PRINCIPAL: No propagamos errores del offset")
    print("Justificación: Los offsets son correcciones sistemáticas,")
    print("no aumentan la incertidumbre de medición individual")
    print("=" * 70)
    
    # Analizar errores originales primero
    analyze_original_errors()
    
    # Calcular y aplicar homogenización SIN propagación de errores
    output_path, offsets = calculate_and_apply_proper_homogenization()
    
    # Verificar que los errores se mantengan
    verify_error_preservation(
        "Results/all_fields_gc_photometry_corrected_errors_v17.csv",
        output_path
    )
    
    # Crear archivo para CIGALE
    cigale_path = create_clean_cigale_input_with_original_errors()
    
    print("\n✅ PROCESO COMPLETADO - ERRORES ORIGINALES MANTENIDOS")
    print("=" * 70)
    print("📊 Resumen:")
    print(f"   - Archivo homogenizado: {output_path}")
    print(f"   - Archivo CIGALE: {cigale_path}")
    print(f"   - Offsets aplicados: {len(offsets)} filtros")
    print("   - Errores: MANTENIDOS ORIGINALES (sin propagación)")
    print("   - Los foto-espectros ahora tendrán errores realistas")
