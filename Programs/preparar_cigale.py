# preparar_cigale.py
# VERSIÓN CORREGIDA: Coloca 'redshift' antes de 'id' en el archivo de salida
# para garantizar que CIGALE lo detecte correctamente como el primer valor numérico.

import pandas as pd
import numpy as np
import os

def abmag_to_mjy(mag_ab):
    """Convierte magnitud AB a flujo en mJy"""
    if mag_ab >= 90 or np.isnan(mag_ab):
        return 0.0
    # Conversión: F_mJy = 3631 Jy * 10^(-mag/2.5) * 1000 mJy/Jy
    return 3631.0 * 10**(-mag_ab / 2.5) * 1000

def preparar_datos_para_cigale(input_file, output_file, n_objects=100):
    """
    Crea archivo perfecto para CIGALE con redshift numérico.
    Coloca la columna 'redshift' antes que 'id' para evitar el error de tipo de dato
    cuando se usa redshift = from_file en CIGALE.
    """
    
    print("🔧 CREANDO ARCHIVO PARA CIGALE (Redshift primero)")
    print("=" * 78)
    
    if not os.path.exists(input_file):
        print(f"❌ ERROR: {input_file} no encontrado")
        return False
    
    try:
        # Asumiendo que el archivo de entrada es un CSV
        df = pd.read_csv(input_file)
        print(f"📊 Datos cargados: {len(df)} objetos")
    except Exception as e:
        print(f"❌ Error leyendo {input_file}: {e}")
        return False
    
    if n_objects and n_objects < len(df):
        df = df.head(n_objects).copy()
        print(f"🔬 Usando muestra: {len(df)} objetos")
    
    filter_map = {
        'F378': 'F0378', 'F395': 'F0395', 'F410': 'F0410',
        'F430': 'F0430', 'F515': 'F0515', 'F660': 'F0660', 'F861': 'F0861'
    }
    
    data_rows = []
    
    print("🔄 Procesando objetos...")
    for idx, row in df.iterrows():
        # Obtener ID (usando T17ID, recno o un ID genérico)
        obj_id = str(row.get('T17ID', row.get('recno', f"obj_{idx+1}")))
        redshift = 0.001825  # VALOR NUMÉRICO FIJO
        
        # --- CAMBIO CLAVE ---
        # El orden es [redshift, id, F0378, F0378_err, ...]
        values = [redshift, obj_id] 
        # --------------------

        for internal_name, cigale_name in filter_map.items():
            mag_col = f'MAG_{internal_name}_3'
            err_col = f'MAGERR_{internal_name}_3'
            
            mag = row.get(mag_col, 99.0)
            if pd.isna(mag): mag = 99.0
                
            err = row.get(err_col, 0.1)
            if pd.isna(err): err = 0.1
                
            if mag >= 90.0:
                flux = 0.0
                flux_err = 99.0
            else:
                flux = abmag_to_mjy(mag)
                if err > 0:
                    # Fórmula estándar para error de flujo a partir de error de magnitud
                    flux_err = flux * (np.log(10)/2.5) * err
                else:
                    # Asignar un error por defecto (ej: 5%)
                    flux_err = flux * 0.05
            
            values.append(flux)
            values.append(flux_err)
            
        data_rows.append(values)
    
    if not data_rows:
        print("❌ No se generaron datos válidos")
        return False
    
    # --- CAMBIO CLAVE: Orden del HEADER ---
    header_list = ['redshift', 'id', 'F0378', 'F0378_err', 'F0395', 'F0395_err', 
                   'F0410', 'F0410_err', 'F0430', 'F0430_err', 'F0515', 'F0515_err', 
                   'F0660', 'F0660_err', 'F0861', 'F0861_err']
    # --------------------------------------
    
    df_final = pd.DataFrame(data_rows, columns=header_list)
    
    # Escribir encabezado comentado
    header_line = '# ' + ' '.join(header_list) + '\n'
    
    with open(output_file, 'w') as f:
        f.write(header_line)
        # Escribir datos con formato controlado
        for row in data_rows:
            # Formato: redshift (float), id (string), flujos (notación científica)
            
            # row[0] es redshift, row[1] es id
            line = f"{row[0]:.6f} {row[1]} " 
            line += " ".join([f"{val:.6e}" for val in row[2:]])
            f.write(line + "\n")
    
    print(f"\n✅ ARCHIVO CREADO: {output_file}")
    print(f"     • {len(data_rows)} objetos")
    print(f"     • Redshift fijo: {redshift:.6f}")
    print(f"     • Columnas: {len(header_list)} (¡REDSHIFT PRIMERO!)")
    
    # Mostrar muestra
    print(f"\n📋 EJEMPLO (primer objeto):")
    print("     " + header_line.strip())
    with open(output_file, 'r') as f:
        # Leer la segunda línea (el primer objeto de datos)
        print("     " + f.readlines()[1].strip())
    
    return True

def verificar_archivo_cigale(archivo):
    """Verifica que el archivo sea perfecto para CIGALE"""
    
    if not os.path.exists(archivo):
        print(f"❌ {archivo} no existe")
        return False
    
    print(f"\n🔍 VERIFICACIÓN FINAL: {archivo}")
    print("-" * 60)
    
    # Leer manualmente la primera línea
    with open(archivo, 'r') as f:
        header_line = f.readline().strip()
    
    print(f"📋 Cabecera: {header_line}")
    
    # Verificar que empiece con #
    if not header_line.startswith('# '):
        print("⚠️  ADVERTENCIA: La cabecera no comienza con '# '")
    
    # Leer los datos con pandas
    try:
        # Extraer nombres de columnas del encabezado
        column_names = header_line[2:].split()  # Quitar '# ' y dividir
        
        # Leer datos
        # Usamos '\s+' para manejar cualquier separación (espacio, tabulador)
        df = pd.read_csv(archivo, sep=r'\s+', skiprows=1, names=column_names)
        
        print(f"✅ Datos leídos: {len(df)} filas, {len(df.columns)} columnas")
        
        # Verificar columnas esenciales
        if 'redshift' not in df.columns:
            print("❌ ERROR: No hay columna 'redshift'")
            return False
        
        # Verificar si 'redshift' es la primera columna
        if df.columns[0] != 'redshift':
            print(f"⚠️  ADVERTENCIA: La columna 'redshift' no es la primera columna (es '{df.columns[0]}').")
        else:
            print("✅ 'redshift' es la primera columna, ¡excelente!")

        
        # Verificar si el tipo de dato es numérico
        if not pd.api.types.is_numeric_dtype(df['redshift']):
            print("❌ ERROR: 'redshift' no es numérico")
            print(f"  Tipo: {df['redshift'].dtype}")
            print(f"  Primer valor: {df['redshift'].iloc[0]}")
            return False
        
        print(f"✅ 'redshift' es numérico: {df['redshift'].dtype}")
        print(f"📊 Valores: min={df['redshift'].min():.6f}, max={df['redshift'].max():.6f}")
        
        # Verificar que todos los redshifts sean iguales (fijo)
        if df['redshift'].nunique() == 1:
            print(f"✅ Todos los objetos tienen el mismo redshift: {df['redshift'].iloc[0]}")
        else:
            print(f"⚠️  ADVERTENCIA: Los redshifts no son todos iguales")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al verificar: {e}")
        return False

if __name__ == "__main__":
    # RUTA DE ARCHIVOS: AJÚSTALA SI ES NECESARIO
    input_csv = "../Results_Corrected/all_fields_photometry_COMPLETE_high_quality.csv"
    output_txt = "gc_splus_cigale_custom.txt"
    
    # Número de objetos a procesar (para pruebas rápidas)
    N_OBJECTS_TEST = 10 
    
    print("🚀 PREPARANDO DATOS PARA CIGALE")
    print("=" * 78)
    
    # 1. Crear archivo
    if preparar_datos_para_cigale(input_csv, output_txt, n_objects=N_OBJECTS_TEST):
        # 2. Verificar
        if verificar_archivo_cigale(output_txt):
            print("\n" + "=" * 78)
            print("🎉 ¡ARCHIVO LISTO PARA CIGALE!")
            print("🎉 ¡EL ERROR DE TIPO DE DATO DEBE ESTAR SOLUCIONADO!")
            print("=" * 78)
            
            print("\n📋 RESUMEN:")
            print(f"  1. Archivo creado: {output_txt}")
            print(f"  2. {N_OBJECTS_TEST} objetos procesados")
            print(f"  3. Redshift numérico: 0.001825 (¡POSICIÓN CORREGIDA!)")
            
            print("\n🚀 PASOS A SEGUIR:")
            print("  1. ASEGÚRATE DE QUE TU pcigale.ini TENGA:")
            print("     [[redshifting]]")
            print("       redshift = from_file")
            print("\n  2. EJECUTA:")
            print("     pcigale check  # ¡Debería funcionar!")
            print("     pcigale run    # ¡Debería completar sin errores!")
