# preparar_cigale_corregido.py
import pandas as pd
import numpy as np
import os

def abmag_to_mjy(mag_ab):
    """Convierte magnitud AB a flujo en mJy"""
    if mag_ab >= 90 or np.isnan(mag_ab):  # Manejar valores no válidos
        return 0.0
    return 3631.0 * 10**(-mag_ab / 2.5) * 1000

def mjy_error_exacta(mag_ab, mag_err):
    """
    Propagación EXACTA de errores usando derivadas
    σ_flux = |dflux/dmag| * σ_mag = flux * (ln(10)/2.5) * σ_mag
    """
    if mag_err <= 0 or mag_ab >= 90 or np.isnan(mag_ab) or np.isnan(mag_err):
        return 0.0
    
    try:
        flux = abmag_to_mjy(mag_ab)
        if flux <= 0:
            return 0.0
            
        # Fórmula exacta: σ_f = f * (ln(10)/2.5) * σ_m
        error_factor = (np.log(10) / 2.5) * mag_err
        flux_error = flux * error_factor
        
        # Validación de resultados
        if np.isnan(flux_error) or np.isinf(flux_error) or flux_error <= 0:
            return flux * 0.1
            
        return flux_error
        
    except Exception as e:
        print(f"⚠️  Error en propagación: mag={mag_ab}, err={mag_err}, error={e}")
        flux = abmag_to_mjy(mag_ab)
        return flux * 0.1

def preparar_cigale_custom_filters(input_file, output_file, n_objects=100):
    """
    Prepara datos para CIGALE usando el formato CORRECTO con # en el header
    """
    
    # Leer tus datos
    df = pd.read_csv(input_file)
    print(f"📊 Datos cargados: {len(df)} objetos")
    
    # Seleccionar muestra
    sample_df = df.head(n_objects).copy()
    print(f"🔬 Usando {len(sample_df)} objetos para prueba")
    
    # Crear DataFrame para CIGALE
    cigale_data = pd.DataFrame()
    
    # Columnas básicas - EXACTAMENTE como en el ejemplo
    cigale_data['id'] = sample_df['T17ID'].fillna(sample_df['recno']).astype(str)
    cigale_data['redshift'] = 0.001825  # NGC 5128
    
    # 🔹 MAPEO: columnas internas → nombres de archivos de filtros
    filter_mapping = {
        'F378': 'F0378',
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
            magnitudes = sample_df[mag_col].fillna(99.0)
            errores_mag = sample_df[err_col].fillna(0.0)
            
            # Usar el nombre del archivo de filtro en el output
            cigale_data[file_name] = [abmag_to_mjy(mag) for mag in magnitudes]
            cigale_data[f'{file_name}_err'] = [
                mjy_error_exacta(mag, err) for mag, err in zip(magnitudes, errores_mag)
            ]
            
        except Exception as e:
            print(f"❌ Error procesando {internal_name}: {e}")
    
    # 🔥 CORRECCIÓN: Guardar con formato adecuado para cada tipo de dato
    print(f"\n💾 Guardando archivo con formato CIGALE correcto (# en header)...")
    
    with open(output_file, 'w') as f:
        # Escribir header con #, separado por espacios
        header = "# " + " ".join(cigale_data.columns)
        f.write(header + "\n")
        
        # Escribir datos - formatear según el tipo de dato
        for idx, row in cigale_data.iterrows():
            formatted_values = []
            for col, val in row.items():
                if col == 'id':
                    # ID como string
                    formatted_values.append(str(val))
                elif col == 'redshift':
                    # Redshift con formato fijo
                    formatted_values.append(f"{val:.6f}")
                else:
                    # Flujos con formato científico
                    formatted_values.append(f"{val:.6e}")
            
            line = " ".join(formatted_values)
            f.write(line + "\n")
    
    print(f"✅ Archivo CIGALE guardado: {output_file}")
    
    # Mostrar preview del archivo generado
    print(f"\n📋 PREVIEW del archivo generado:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:  # Mostrar primeras 3 líneas
                print(f"   {line.strip()}")
            else:
                break
    
    # Mostrar estadísticas
    print(f"\n📊 Estadísticas de conversión:")
    total_fluxes = 0
    for file_name in filter_mapping.values():
        if file_name in cigale_data.columns:
            n_valid = (cigale_data[file_name] > 0).sum()
            total_fluxes += n_valid
            print(f"   {file_name}: {n_valid}/{len(cigale_data)} flujos válidos")
    
    print(f"   TOTAL: {total_fluxes} flujos convertidos")
    
    return cigale_data, filter_mapping

def verificar_formato_archivo(output_file):
    """Verifica que el archivo tenga el formato correcto para CIGALE"""
    
    print(f"\n{'='*60}")
    print("🔍 VERIFICANDO FORMATO DEL ARCHIVO CIGALE")
    print(f"{'='*60}")
    
    if not os.path.exists(output_file):
        print(f"❌ Archivo no encontrado: {output_file}")
        return False
    
    with open(output_file, 'r') as f:
        lineas = f.readlines()
    
    if len(lineas) < 2:
        print("❌ Archivo vacío o con muy pocas líneas")
        return False
    
    # Verificar header
    header = lineas[0].strip()
    if not header.startswith("#"):
        print("❌ ERROR CRÍTICO: Header no comienza con #")
        print(f"   Header actual: {header}")
        return False
    
    print("✅ Header comienza con # (correcto)")
    
    # Verificar estructura de datos
    datos_linea = lineas[1].strip().split()
    n_columnas_datos = len(datos_linea)
    n_columnas_header = len(header.split()) - 1  # Restar el #
    
    print(f"   Columnas en header: {n_columnas_header}")
    print(f"   Columnas en datos: {n_columnas_datos}")
    
    if n_columnas_datos != n_columnas_header:
        print("❌ ERROR: Número de columnas no coincide")
        return False
    
    print("✅ Número de columnas coincide")
    
    # Verificar tipos de datos
    try:
        # Saltar la primera columna (id) que puede ser string
        for i, valor in enumerate(datos_linea[1:], 1):  # Empezar desde la segunda columna
            float(valor)  # Intentar convertir a float
        print("✅ Todos los valores numéricos son válidos")
    except ValueError:
        print("❌ ERROR: Valores no numéricos donde se esperaban números")
        return False
    
    return True

def crear_configuracion_minima(output_file, mapeo_filtros):
    """Crea un archivo de configuración mínima para CIGALE"""
    
    config_content = f"""# CIGALE minimal configuration for S-PLUS globular clusters
# Generated automatically

data_file = {output_file}

sed_modules = sfhdelayed, bc03, redshifting

analysis_method = pdf_analysis
cores = 8

bands = {", ".join([f"{name}, {name}_err" for name in mapeo_filtros.values()])}

[sed_modules_params]
  [[sfhdelayed]]
    tau_main = 100, 1000
    age_main = 1000, 5000, 10000
    sfr_A = 1.0
    normalise = True
  [[bc03]]
    imf = 1
    metallicity = 0.02
  [[redshifting]]
    redshift = 0.001825

[analysis_params]
  variables = stellar.metallicity
  save_best_sed = True
"""
    
    with open('pcigale_minimal.ini', 'w') as f:
        f.write(config_content)
    
    print("✅ Configuración mínima creada: pcigale_minimal.ini")

if __name__ == "__main__":
    input_file = "../Results_Corrected/all_fields_photometry_COMPLETE_high_quality.csv"
    output_file = "gc_splus_cigale_custom.txt"
    
    print("🔧 PREPARADOR DE DATOS CIGALE - FORMATO CORREGIDO")
    print("=" * 70)
    
    # Preparar datos
    datos_cigale, mapeo = preparar_cigale_custom_filters(input_file, output_file, n_objects=100)
    
    # Verificar formato
    formato_ok = verificar_formato_archivo(output_file)
    
    # Crear configuración mínima
    crear_configuracion_minima(output_file, mapeo)
    
    # Generar configuración para pcigale.ini
    print(f"\n{'='*60}")
    print("🎯 CONFIGURACIÓN PARA pcigale.ini")
    print(f"{'='*60}")
    
    bands_list = []
    for name in mapeo.values():
        bands_list.extend([name, f"{name}_err"])
    bands_str = ", ".join(bands_list)
    
    print("Copia esto en tu pcigale.ini:")
    print(f"bands = {bands_str}")
    
    print(f"\n💡 Comando para ejecutar CIGALE:")
    print(f"   pcigale run -c pcigale_minimal.ini")
    
    if formato_ok:
        print(f"\n✅ ARCHIVO LISTO para CIGALE")
        print(f"   Ejecuta: pcigale run -c pcigale_minimal.ini")
    else:
        print(f"\n❌ PROBLEMAS con el formato del archivo")
