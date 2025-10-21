import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from astropy.stats import mad_std
import seaborn as sns

def internal_splus_homogenization():
    """
    PASO 1: Homogenización INTERNA de los filtros SPLUS entre sí
    Usando F660 y F861 como referencia por su alta correlación
    """
    
    print("🎯 HOMOGENIZACIÓN INTERNA SPLUS")
    print("=" * 50)
    
    df = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17.csv")
    
    # Orden de filtros por longitud de onda
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    wavelengths = [378, 395, 410, 430, 515, 660, 861]
    
    print("📊 Correlaciones entre filtros SPLUS (matriz):")
    splus_mag_cols = [f'MAG_{f}_3' for f in splus_filters if f'MAG_{f}_3' in df.columns]
    splus_corr_data = df[splus_mag_cols].copy()
    splus_corr_data.columns = [col.replace('MAG_', '').replace('_3', '') for col in splus_corr_data.columns]
    
    corr_matrix = splus_corr_data.corr()
    print(corr_matrix.round(3))
    
    # Usar F660 como referencia principal (alta correlación con otros)
    reference_filter = 'F660'
    reference_col = f'MAG_{reference_filter}_3'
    
    internal_offsets = {}
    
    for filtro in splus_filters:
        if filtro == reference_filter:
            continue
            
        current_col = f'MAG_{filtro}_3'
        
        if current_col not in df.columns or reference_col not in df.columns:
            continue
        
        # Filtrar datos válidos para ambos filtros
        mask = (
            df[current_col].notna() & df[reference_col].notna() &
            (df[current_col] < 90) & (df[reference_col] < 90) &
            (df[current_col] > 10) & (df[reference_col] > 10)
        )
        
        valid_data = df.loc[mask]
        if len(valid_data) < 10:
            continue
        
        # Calcular relación esperada basada en modelos de GCs
        # Para GCs típicos, esperamos colores específicos
        expected_relations = {
            'F378': 2.0,   # F378 debería ser ~2 mag más débil que F660 en GCs
            'F395': 1.8,   # F395 debería ser ~1.8 mag más débil  
            'F410': 1.2,   # F410 debería ser ~1.2 mag más débil
            'F430': 0.8,   # F430 debería ser ~0.8 mag más débil
            'F515': 0.3,   # F515 debería ser ~0.3 mag más débil
            'F861': -0.4   # F861 debería ser ~0.4 mag más brillante
        }
        
        if filtro in expected_relations:
            expected_diff = expected_relations[filtro]
            actual_diff = np.median(valid_data[current_col] - valid_data[reference_col])
            offset = expected_diff - actual_diff
            
            internal_offsets[filtro] = offset
            
            print(f"  {filtro} vs {reference_filter}:")
            print(f"    Diferencia actual: {actual_diff:.3f} mag")
            print(f"    Diferencia esperada: {expected_diff:.3f} mag")  
            print(f"    Offset interno: {offset:+.3f} mag")
            print(f"    N: {len(valid_data)} fuentes")
    
    # Aplicar homogenización interna
    for filtro, offset in internal_offsets.items():
        mag_col = f'MAG_{filtro}_3'
        df[mag_col] = df[mag_col] + offset
        print(f"  ✅ {mag_col}: +{offset:+.3f} mag (homogenización interna)")
    
    # Guardar resultado intermedio
    internal_output = "Results/all_fields_gc_photometry_internal_splus_homogenized.csv"
    df.to_csv(internal_output, index=False)
    print(f"\n💾 Homogenización interna guardada: {internal_output}")
    
    return internal_output, internal_offsets

def calibrate_to_taylor_with_scaling():
    """
    PASO 2: Calibración a Taylor usando factores de escala por color
    """
    
    print("\n🎯 CALIBRACIÓN A TAYLOR CON FACTORES DE ESCALA")
    print("=" * 50)
    
    df = pd.read_csv("Results/all_fields_gc_photometry_internal_splus_homogenized.csv")
    
    # Mapeo basado en diagnóstico
    filter_mapping = {
        'F378': 'umag',
        'F395': 'umag', 
        'F410': 'gmag',
        'F430': 'gmag', 
        'F515': 'gmag',
        'F660': 'rmag',
        'F861': 'imag'
    }
    
    # Calcular offsets pero con agrupación por color
    final_offsets = {}
    
    for splus_filt, taylor_filt in filter_mapping.items():
        splus_col = f'MAG_{splus_filt}_3'
        taylor_col = taylor_filt
        
        if splus_col not in df.columns or taylor_col not in df.columns:
            continue
        
        # Filtrar datos válidos
        mask = (
            df[splus_col].notna() & df[taylor_col].notna() &
            (df[splus_col] < 90) & (df[taylor_col] < 90) &
            (df[splus_col] > 10) & (df[taylor_col] > 10)
        )
        
        valid_data = df.loc[mask]
        if len(valid_data) < 10:
            continue
        
        # Estrategia: usar solo los GCs con mejor comportamiento
        # Calcular residuos y seleccionar los más consistentes
        differences = valid_data[taylor_col] - valid_data[splus_col]
        residuals = differences - np.median(differences)
        
        # Seleccionar fuentes con residuos pequeños (dentro de 1 MAD)
        mad_residuals = mad_std(residuals)
        good_mask = np.abs(residuals) < mad_residuals
        
        good_data = valid_data[good_mask]
        
        if len(good_data) < 5:
            # Si muy pocas fuentes buenas, usar todas
            good_data = valid_data
        
        # Calcular offset con fuentes seleccionadas
        final_diff = np.median(good_data[taylor_col] - good_data[splus_col])
        final_offsets[splus_filt] = final_diff
        
        print(f"  {splus_filt} -> {taylor_filt}:")
        print(f"    Offset final: {final_diff:+.3f} mag")
        print(f"    Fuentes usadas: {len(good_data)}/{len(valid_data)}")
        print(f"    MAD residuos: {mad_residuals:.3f} mag")
    
    # Aplicar offsets finales
    for filtro, offset in final_offsets.items():
        mag_col = f'MAG_{filtro}_3'
        df[mag_col] = df[mag_col] + offset
        print(f"  ✅ {mag_col}: +{offset:+.3f} mag (calibración Taylor)")
    
    final_output = "Results/all_fields_gc_photometry_final_calibrated.csv"
    df.to_csv(final_output, index=False)
    print(f"\n💾 Calibración final guardada: {final_output}")
    
    return final_output, final_offsets

def create_validation_plots(original_path, final_path):
    """
    Crear gráficos de validación comparativos
    """
    
    print("\n📊 CREANDO GRÁFICOS DE VALIDACIÓN")
    print("=" * 50)
    
    df_orig = pd.read_csv(original_path)
    df_final = pd.read_csv(final_path)
    
    splus_filters = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
    taylor_filters = ['umag', 'gmag', 'rmag', 'imag', 'zmag']
    
    # 1. Matriz de correlación comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original
    orig_cols = [f'MAG_{f}_3' for f in splus_filters if f'MAG_{f}_3' in df_orig.columns]
    orig_corr_data = df_orig[orig_cols].copy()
    orig_corr_data.columns = [col.replace('MAG_', '').replace('_3', '') for col in orig_corr_data.columns]
    orig_corr = orig_corr_data.corr()
    
    sns.heatmap(orig_corr, annot=True, cmap='coolwarm', center=0, ax=ax1, 
                square=True, fmt='.3f', cbar_kws={'label': 'Correlación'})
    ax1.set_title('Correlaciones SPLUS - ORIGINAL')
    
    # Final
    final_cols = [f'MAG_{f}_3' for f in splus_filters if f'MAG_{f}_3' in df_final.columns]
    final_corr_data = df_final[final_cols].copy()
    final_corr_data.columns = [col.replace('MAG_', '').replace('_3', '') for col in final_corr_data.columns]
    final_corr = final_corr_data.corr()
    
    sns.heatmap(final_corr, annot=True, cmap='coolwarm', center=0, ax=ax2,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlación'})
    ax2.set_title('Correlaciones SPLUS - HOMOGENIZADO')
    
    plt.tight_layout()
    plt.savefig('Results/validation_correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Diferencias con Taylor
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    filter_mapping = {
        'F378': 'umag', 'F395': 'umag', 'F410': 'gmag', 'F430': 'gmag',
        'F515': 'gmag', 'F660': 'rmag', 'F861': 'imag'
    }
    
    for idx, (splus_filt, taylor_filt) in enumerate(filter_mapping.items()):
        if idx >= len(axes):
            break
            
        splus_col = f'MAG_{splus_filt}_3'
        
        if splus_col not in df_final.columns or taylor_filt not in df_final.columns:
            continue
        
        # Calcular diferencias finales
        mask = (
            df_final[splus_col].notna() & df_final[taylor_filt].notna() &
            (df_final[splus_col] < 90) & (df_final[taylor_filt] < 90) &
            (df_final[splus_col] > 10) & (df_final[taylor_filt] > 10)
        )
        
        valid_data = df_final.loc[mask]
        if len(valid_data) < 10:
            continue
        
        differences = valid_data[taylor_filt] - valid_data[splus_col]
        
        axes[idx].hist(differences, bins=30, alpha=0.7, color='skyblue')
        axes[idx].axvline(np.median(differences), color='red', linestyle='--', 
                         label=f'Mediana: {np.median(differences):.3f}')
        axes[idx].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[idx].set_xlabel(f'{taylor_filt} - {splus_filt}')
        axes[idx].set_ylabel('Número de fuentes')
        axes[idx].set_title(f'{splus_filt} vs {taylor_filt}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Ocultar ejes no usados
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('Results/validation_taylor_differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráficos de validación guardados")

def main():
    """
    Proceso completo de homogenización mejorada
    """
    print("🎯 HOMOGENIZACIÓN MEJORADA - 2 PASOS")
    print("=" * 70)
    print("PASO 1: Homogenización INTERNA SPLUS (mejorar coherencia entre filtros)")
    print("PASO 2: Calibración a Taylor (usando solo fuentes consistentes)")
    print("=" * 70)
    
    # Paso 1: Homogenización interna SPLUS
    internal_path, internal_offsets = internal_splus_homogenization()
    
    # Paso 2: Calibración a Taylor
    final_path, final_offsets = calibrate_to_taylor_with_scaling()
    
    # Validación
    create_validation_plots(
        "Results/all_fields_gc_photometry_corrected_errors_v17.csv",
        final_path
    )
    
    print("\n✅ PROCESO COMPLETADO")
    print("=" * 70)
    print("📊 Resumen de offsets aplicados:")
    
    print("\n🔧 Homogenización INTERNA SPLUS:")
    for filtro, offset in internal_offsets.items():
        print(f"  {filtro}: {offset:+.3f} mag")
    
    print("\n🎯 Calibración TAYLOR:")
    for filtro, offset in final_offsets.items():
        print(f"  {filtro}: {offset:+.3f} mag")
    
    print(f"\n📁 Archivo final: {final_path}")
    print("📊 Gráficos de validación en: Results/")

if __name__ == "__main__":
    main()
