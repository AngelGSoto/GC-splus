import pandas as pd
import numpy as np
import os

def abmag_to_mjy(mag_ab):
    """Convierte magnitud AB a flujo en mJy"""
    return 3631.0 * 10**(-mag_ab / 2.5)

def mjy_error(mag_ab, mag_err):
    """Propaga el error de magnitud a error en flujo"""
    flux = abmag_to_mjy(mag_ab)
    return flux * (10**(mag_err/2.5) - 1)

def preparar_cigale_custom_filters(input_file, output_file, n_objects=10):
    """
    Prepara datos para CIGALE usando los nombres exactos de tus archivos de filtros
    """
    
    # Leer tus datos
    df = pd.read_csv(input_file)
    print(f"📊 Datos cargados: {len(df)} objetos")
    
    # Seleccionar muestra
    sample_df = df.head(n_objects).copy()
    print(f"🔬 Usando {len(sample_df)} objetos para prueba")
    
    # Crear DataFrame para CIGALE
    cigale_data = pd.DataFrame()
    
    # Columnas básicas
    cigale_data['id'] = sample_df['T17ID'].fillna(sample_df['recno'])
    cigale_data['redshift'] = 0.0018  # NGC 5128
    
    # 🔹 MAPEO: columnas internas → nombres de archivos de filtros
    filter_mapping = {
        'F378': 'F0378',  # MAG_F378_3 → F0378 (nombre archivo)
        'F395': 'F0395',
        'F410': 'F0410', 
        'F430': 'F0430',
        'F515': 'F0515',
        'F660': 'F0660',
        'F861': 'F0861'
    }
    
    print("🔄 Conversión de magnitudes AB a flujos mJy...")
    
    for internal_name, file_name in filter_mapping.items():
        mag_col = f'MAG_{internal_name}_3'
        err_col = f'MAGERR_{internal_name}_3'
        
        if mag_col not in sample_df.columns:
            print(f"⚠️  Columna {mag_col} no encontrada")
            continue
            
        try:
            # Convertir magnitudes a flujos (mJy)
            magnitudes = sample_df[mag_col]
            errores_mag = sample_df[err_col]
            
            # Usar el nombre del archivo de filtro en el output
            cigale_data[file_name] = abmag_to_mjy(magnitudes)
            cigale_data[f'{file_name}_err'] = mjy_error(magnitudes, errores_mag)
            
            # Mostrar ejemplo de conversión
            if sample_df.index[0] == 0:
                mag_val = magnitudes.iloc[0]
                flux_val = cigale_data[file_name].iloc[0]
                print(f"   {internal_name} → {file_name}: {mag_val:.2f} mag → {flux_val:.2e} mJy")
                
        except Exception as e:
            print(f"❌ Error procesando {internal_name}: {e}")
    
    # Verificar que los archivos de filtros existan
    print("\n🔍 Verificando archivos de filtros...")
    missing_filters = []
    for file_name in filter_mapping.values():
        filter_file = f"{file_name}.dat"
        if os.path.exists(filter_file):
            print(f"   ✅ {filter_file} encontrado")
        else:
            print(f"   ❌ {filter_file} NO encontrado")
            missing_filters.append(filter_file)
    
    if missing_filters:
        print(f"⚠️  Faltan {len(missing_filters)} archivos de filtros")
    
    # Guardar archivo
    cigale_data.to_csv(output_file, sep=' ', index=False, float_format='%.6e')
    
    print(f"\n💾 Archivo CIGALE guardado: {output_file}")
    print("📋 Columnas en el archivo:")
    print(f"   {list(cigale_data.columns)}")
    
    return cigale_data, filter_mapping

if __name__ == "__main__":
    input_file = "../Results_Corrected/all_fields_photometry_COMPLETE_high_quality.csv"
    output_file = "gc_splus_cigale_custom.txt"
    
    datos_cigale, mapeo = preparar_cigale_custom_filters(input_file, output_file, n_objects=100)
    
    print(f"\n🎯 Para pcigale.ini usa:")
    bands_str = ", ".join([f"{name}, {name}_err" for name in mapeo.values()])
    print(f"bands = {bands_str}")
