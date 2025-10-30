import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from astropy.stats import mad_std

def revert_and_correct_homogenization():
    """
    Revierte la homogenización incorrecta y aplica la corrección adecuada
    """
    
    # Leer el archivo homogenizado (que tiene corrección duplicada)
    df_homogenized = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17_homogenized.csv")
    
    # Offsets que se aplicaron incorrectamente (del archivo de offsets)
    offsets_applied = {
        'F378': -1.285,
        'F395': -1.461, 
        'F410': 0.003,
        'F430': -0.048,
        'F515': -0.603,
        'F660': -0.454,
        'F861': -0.260
    }
    
    print("🔄 REVERTIENDO HOMOGENIZACIÓN INCORRECTA")
    print("=" * 50)
    
    # Revertir: restar los offsets que se aplicaron
    for filtro, offset in offsets_applied.items():
        mag_column = f"MAG_{filtro}_3"
        if mag_column in df_homogenized.columns:
            df_homogenized[mag_column] = df_homogenized[mag_column] - offset
            print(f"  ✅ {mag_column}: revertido offset {offset:+.3f}")
    
    # Guardar archivo revertido
    reverted_path = "Results/all_fields_gc_photometry_corrected_errors_v17_reverted.csv"
    df_homogenized.to_csv(reverted_path, index=False)
    print(f"💾 Archivo revertido: {reverted_path}")
    
    return reverted_path, df_homogenized

def calculate_proper_offsets(catalog_path):
    """
    Calcula los offsets correctos desde el archivo revertido
    """
    print("\n🔍 CALCULANDO OFFSETS CORRECTOS")
    print("=" * 50)
    
    df = pd.read_csv(catalog_path)
    
    filter_mapping = {
        'MAG_F378_3': 'umag',
        'MAG_F395_3': 'umag', 
        'MAG_F410_3': 'gmag',
        'MAG_F430_3': 'gmag',
        'MAG_F515_3': 'gmag',
        'MAG_F660_3': 'rmag',
        'MAG_F861_3': 'zmag'
    }
    
    proper_offsets = {}
    
    for splus_col, taylor_col in filter_mapping.items():
        if splus_col not in df.columns or taylor_col not in df.columns:
            continue
            
        # Filtrar datos válidos
        mask = (
            df[splus_col].notna() & 
            df[taylor_col].notna() &
            (df[splus_col] < 90) &
            (df[taylor_col] < 90) &
            (df[splus_col] > 10) &
            (df[taylor_col] > 10)
        )
        
        valid_data = df.loc[mask]
        if len(valid_data) < 10:
            continue
        
        x_data = valid_data[taylor_col]
        y_data = valid_data[splus_col]
        
        differences = y_data - x_data
        valid_diff = differences[np.isfinite(differences)]
        
        if len(valid_diff) == 0:
            continue
            
        median_diff = np.median(valid_diff)
        proper_offsets[splus_col.replace('MAG_', '').replace('_3', '')] = -median_diff  # INVERTIR el offset
    
    print("📊 OFFSETS CORRECTOS CALCULADOS:")
    for filtro, offset in proper_offsets.items():
        print(f"  {filtro}: {offset:+.3f} mag")
    
    return proper_offsets

def apply_proper_homogenization(input_path, offsets, output_suffix="_proper_homogenized"):
    """
    Aplica la homogenización correcta
    """
    print("\n🔄 APLICANDO HOMOGENIZACIÓN CORRECTA")
    print("=" * 50)
    
    df = pd.read_csv(input_path)
    
    for filtro, offset in offsets.items():
        mag_column = f"MAG_{filtro}_3"
        err_column = f"MAGERR_{filtro}_3"
        
        if mag_column in df.columns:
            # Aplicar offset CORRECTO
            df[mag_column] = df[mag_column] + offset
            print(f"  ✅ {mag_column}: aplicado offset {offset:+.3f} mag")
    
    output_path = input_path.replace('.csv', f'{output_suffix}.csv')
    df.to_csv(output_path, index=False)
    print(f"💾 Archivo correctamente homogenizado: {output_path}")
    
    return output_path, df

def verify_correction(original_path, corrected_path):
    """
    Verifica que la corrección sea correcta
    """
    print("\n🔍 VERIFICANDO CORRECCIÓN")
    print("=" * 50)
    
    df_orig = pd.read_csv(original_path)
    df_corr = pd.read_csv(corrected_path)
    
    filter_mapping = {
        'MAG_F378_3': 'umag',
        'MAG_F395_3': 'umag', 
        'MAG_F410_3': 'gmag',
        'MAG_F430_3': 'gmag',
        'MAG_F515_3': 'gmag',
        'MAG_F660_3': 'rmag',
        'MAG_F861_3': 'zmag'
    }
    
    print("COMPARACIÓN DE DIFERENCIAS:")
    print("Filtro      | Δ_original | Δ_corregido")
    print("-" * 40)
    
    for splus_col, taylor_col in filter_mapping.items():
        if splus_col not in df_orig.columns or taylor_col not in df_orig.columns:
            continue
            
        # Calcular diferencias en original
        mask_orig = (
            df_orig[splus_col].notna() & df_orig[taylor_col].notna() &
            (df_orig[splus_col] < 90) & (df_orig[taylor_col] < 90) &
            (df_orig[splus_col] > 10) & (df_orig[taylor_col] > 10)
        )
        
        mask_corr = (
            df_corr[splus_col].notna() & df_corr[taylor_col].notna() &
            (df_corr[splus_col] < 90) & (df_corr[taylor_col] < 90) &
            (df_corr[splus_col] > 10) & (df_corr[taylor_col] > 10)
        )
        
        if mask_orig.sum() > 10 and mask_corr.sum() > 10:
            diff_orig = np.median(df_orig.loc[mask_orig, splus_col] - df_orig.loc[mask_orig, taylor_col])
            diff_corr = np.median(df_corr.loc[mask_corr, splus_col] - df_corr.loc[mask_corr, taylor_col])
            
            print(f"{splus_col:11} | {diff_orig:+.3f}     | {diff_corr:+.3f}")

def main():
    """
    Proceso completo de corrección
    """
    print("🎯 CORRECCIÓN DE HOMOGENIZACIÓN DUPLICADA")
    print("=" * 60)
    
    # 1. Revertir homogenización incorrecta
    reverted_path, df_reverted = revert_and_correct_homogenization()
    
    # 2. Calcular offsets correctos
    proper_offsets = calculate_proper_offsets(reverted_path)
    
    # 3. Aplicar homogenización correcta
    final_path, df_final = apply_proper_homogenization(reverted_path, proper_offsets)
    
    # 4. Verificar
    verify_correction(reverted_path, final_path)
    
    print("\n✅ CORRECCIÓN COMPLETADA")
    print("=" * 60)
    print(f"📁 Archivo final listo para CIGALE: {final_path}")
    print(f"📊 Offsets aplicados: {len(proper_offsets)} filtros")
    
    # Guardar tabla de offsets correctos
    offsets_df = pd.DataFrame({
        'filter': list(proper_offsets.keys()),
        'proper_offset': list(proper_offsets.values())
    })
    offsets_df.to_csv("Results/proper_homogenization_offsets.csv", index=False)
    print(f"📋 Tabla de offsets correctos: Results/proper_homogenization_offsets.csv")

if __name__ == "__main__":
    main()
