#!/usr/bin/env python3
"""
Script CORREGIDO para PREPARAR DATOS CIGALE
Mejorado con sugerencias de robustez, claridad y experiencia de usuario.
"""

import pandas as pd
import numpy as np
import os
import subprocess
import traceback

def fix_splus_filters():
    """Arreglar los filtros SPLUS existentes añadiendo la cabecera requerida"""
    print("🔧 ARREGLANDO FILTROS SPLUS EXISTENTES")
    print("=" * 60)
    
    # Mapeo de archivos originales a nombres CIGALE
    splus_mapping = {
        'F0378.dat': 'splus.F378',
        'F0395.dat': 'splus.F395', 
        'F0410.dat': 'splus.F410',
        'F0430.dat': 'splus.F430',
        'F0515.dat': 'splus.F515',
        'F0660.dat': 'splus.F660',
        'F0861.dat': 'splus.F861'
    }
    
    fixed_count = 0
    for old_name, cigale_name in splus_mapping.items():
        if os.path.exists(old_name):
            try:
                # Leer el archivo original
                with open(old_name, 'r') as f:
                    lines = f.readlines()
                
                # Crear nuevo archivo con cabecera correcta
                new_filename = f"{cigale_name.replace('.', '_')}.dat"
                with open(new_filename, 'w') as f:
                    f.write(f"# {cigale_name}\n")
                    f.write("# photon\n")
                    f.write(f"# Fixed SPLUS filter from {old_name}\n")
                    # Copiar todos los datos originales
                    for line in lines:
                        f.write(line)
                
                print(f"   ✅ {old_name} → {new_filename}")
                fixed_count += 1
                
            except Exception as e:
                print(f"   ❌ Error arreglando {old_name}: {e}")
                traceback.print_exc()
        else:
            print(f"   ⚠️  {old_name} no encontrado")
    
    return fixed_count

def convert_decam_filters():
    """Convertir filtros DECam del archivo scidoc0472.txt"""
    print(f"\n🔄 CONVIRTIENDO FILTROS DECam")
    print("=" * 60)
    
    if not os.path.exists('scidoc0472.txt'):
        print("❌ scidoc0472.txt no encontrado")
        return 0
    
    try:
        # Leer el archivo DECam (usando pandas para manejar mejor el formato)
        data = pd.read_csv('scidoc0472.txt', delim_whitespace=True, comment='#', header=None)
        
        # Los datos están en nm, convertir a Angstroms
        wavelength_nm = data.iloc[:, 0].values
        wavelength_A = wavelength_nm * 10.0
        
        # Columnas: wavelength(nm), u, g, r, i, z, Y, VR
        decam_filters = {
            'decam.u': data.iloc[:, 1].values,
            'decam.g': data.iloc[:, 2].values, 
            'decam.r': data.iloc[:, 3].values,
            'decam.i': data.iloc[:, 4].values,
            'decam.z': data.iloc[:, 5].values
        }
        
        converted_count = 0
        for filter_name, transmission in decam_filters.items():
            filename = f"{filter_name.replace('.', '_')}.dat"
            
            with open(filename, 'w') as f:
                f.write(f"# {filter_name}\n")
                f.write("# photon\n")
                f.write(f"# DECam filter from scidoc0472.txt\n")
                
                for wl, tr in zip(wavelength_A, transmission):
                    if tr > 0:  # Solo escribir puntos con transmisión positiva
                        f.write(f"{wl:.1f} {tr:.6f}\n")
            
            print(f"   ✅ {filter_name} → {filename}")
            converted_count += 1
        
        return converted_count
        
    except Exception as e:
        print(f"❌ Error procesando scidoc0472.txt: {e}")
        traceback.print_exc()
        return 0

def create_cigale_input_without_duplicates():
    """Crear archivo de entrada para CIGALE sin IDs duplicados"""
    print(f"\n📝 CREANDO ARCHIVO DE ENTRADA SIN DUPLICADOS")
    print("=" * 60)
    
    try:
        # Cargar el archivo original
        file_path = "../Results/gc_photometry_final_high_quality_preliminar_teste_aperture3_only.csv"
        df = pd.read_csv(file_path)
        
        print(f"📊 DATOS ORIGINALES:")
        print(f"   • Filas totales: {len(df)}")
        print(f"   • IDs únicos: {df['T17ID'].nunique()}")
        
        # Encontrar y mostrar duplicados
        duplicates = df[df.duplicated(['T17ID'], keep=False)]
        if len(duplicates) > 0:
            duplicate_ids = duplicates['T17ID'].unique()
            print(f"⚠️  ENCONTRADOS {len(duplicate_ids)} IDs DUPLICADOS:")
            for dup_id in duplicate_ids[:10]:  # Mostrar solo los primeros 10
                dup_rows = duplicates[duplicates['T17ID'] == dup_id]
                print(f"   • {dup_id}: {len(dup_rows)} ocurrencias")
            
            if len(duplicate_ids) > 10:
                print(f"   ... y {len(duplicate_ids) - 10} más")
        
        # Eliminar duplicados, manteniendo la primera ocurrencia
        df_clean = df.drop_duplicates(subset=['T17ID'], keep='first')
        removed_count = len(df) - len(df_clean)
        
        print(f"\n🔧 LIMPIEZA DE DATOS:")
        print(f"   • Filas eliminadas: {removed_count}")
        print(f"   • Filas finales: {len(df_clean)}")
        
        # Crear DataFrame para CIGALE
        cigale_df = pd.DataFrame()
        cigale_df['id'] = df_clean['T17ID']
        cigale_df['redshift'] = 0.0
        
        print(f"\n🔄 CONVIRTIENDO MAGNITUDES A FLUJOS:")
        
        # Conversión de magnitudes AB a mJy
        def mag_AB_to_mjy(mag, wavelength_A):
            return 10**((23.9 - mag) / 2.5)
        
        # Procesar filtros DECam
        decam_info = {
            'umag': ('decam.u', 3543),
            'gmag': ('decam.g', 4770),
            'rmag': ('decam.r', 6231),
            'imag': ('decam.i', 7625),
            'zmag': ('decam.z', 9134)
        }
        
        for mag_col, (filter_name, wavelength) in decam_info.items():
            if mag_col in df_clean.columns:
                mag_values = df_clean[mag_col].copy()
                valid_mask = (mag_values > 0) & (mag_values < 30) & (~mag_values.isna())
                
                # Convertir a flujo
                fluxes = np.where(valid_mask, 
                                mag_AB_to_mjy(mag_values, wavelength), 
                                1e-10)
                
                # Calcular errores
                error_col = f"e_{mag_col}"
                if error_col in df_clean.columns:
                    mag_errors = df_clean[error_col].fillna(0.05)
                    flux_errors = np.where(valid_mask, fluxes * mag_errors, fluxes)
                else:
                    flux_errors = np.where(valid_mask, 0.05 * fluxes, fluxes)
                
                cigale_df[filter_name] = fluxes
                cigale_df[f"{filter_name}_err"] = flux_errors
                
                valid_count = valid_mask.sum()
                print(f"   ✅ {mag_col} → {filter_name} ({valid_count} válidos)")
        
        # Procesar filtros SPLUS
        splus_info = {
            'F378': 3785, 'F395': 3950, 'F410': 4100, 'F430': 4300,
            'F515': 5150, 'F660': 6600, 'F861': 8610
        }
        
        for filter_code, wavelength in splus_info.items():
            mag_col = f'MAG_{filter_code}_3'
            err_col = f'MAGERR_{filter_code}_3'
            
            if mag_col in df_clean.columns:
                filter_name = f'splus.{filter_code}'
                
                # Verificar datos válidos
                valid_data = (df_clean[mag_col].notna() & 
                            ~np.isinf(df_clean[mag_col]) & 
                            (df_clean[mag_col] > 0) & 
                            (df_clean[mag_col] < 30))
                valid_count = valid_data.sum()
                
                if valid_count > 0:
                    # Convertir a flujo
                    mag_values = df_clean[mag_col].where(valid_data, 99.0)
                    fluxes = np.where(valid_data, 
                                    mag_AB_to_mjy(mag_values, wavelength), 
                                    1e-10)
                    
                    # Calcular errores
                    if err_col in df_clean.columns:
                        mag_errors = df_clean[err_col].fillna(0.05)
                        flux_errors = np.where(valid_data, fluxes * mag_errors, fluxes)
                    else:
                        flux_errors = np.where(valid_data, 0.05 * fluxes, fluxes)
                    
                    cigale_df[filter_name] = fluxes
                    cigale_df[f"{filter_name}_err"] = flux_errors
                    
                    print(f"   ✅ {mag_col} → {filter_name} ({valid_count} válidos)")
                else:
                    print(f"   ⚠️  {mag_col} → Sin datos válidos")
        
        # Guardar archivo
        output_file = "cigale_input.txt"
        cigale_df.to_csv(output_file, sep=' ', index=False, float_format='%.6e')
        
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   • Objetos en archivo: {len(cigale_df)}")
        print(f"   • Filtros incluidos: {len([col for col in cigale_df.columns if not col.endswith('_err') and col not in ['id', 'redshift']])}")
        print(f"   • Archivo guardado: {output_file}")
        
        return cigale_df
        
    except Exception as e:
        print(f"❌ Error creando archivo de entrada: {e}")
        traceback.print_exc()
        return None

def setup_cigale_configuration():
    """Configurar CIGALE usando el flujo oficial"""
    print(f"\n⚙️  CONFIGURANDO CIGALE")
    print("=" * 60)
    
    try:
        # Eliminar archivos de configuración existentes
        for config_file in ['pcigale.ini', 'pcigale.ini.spec']:
            if os.path.exists(config_file):
                os.remove(config_file)
                print(f"   🔄 Eliminado {config_file} existente")
        
        # Paso 1: Inicializar configuración
        print("📝 Inicializando configuración...")
        result = subprocess.run(['pcigale', 'init'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error al inicializar: {result.stderr}")
            return False
        
        # Paso 2: Crear configuración básica
        config_content = """# CIGALE configuration for globular clusters
# Optimized for old stellar populations

data_file = cigale_input.txt
parameters_file = 
sed_modules = sfhdelayed, bc03, dustatt_powerlaw, redshifting
analysis_method = pdf_analysis
cores = 4

# Additional error (default 10%)
additionalerror = 0.05
save_best_sed = True
"""
        with open("pcigale.ini", "w") as f:
            f.write(config_content)
        
        # Paso 3: Generar configuración completa
        print("📝 Generando configuración completa...")
        result = subprocess.run(['pcigale', 'genconf'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error al generar configuración: {result.stderr}")
            return False
        
        print("✅ Configuración CIGALE completada")
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 PREPARACIÓN DE DATOS CIGALE - CORREGIDO")
    print("=" * 70)
    
    # Paso 1: Arreglar filtros SPLUS
    splus_fixed = fix_splus_filters()
    
    # Paso 2: Convertir filtros DECam  
    decam_converted = convert_decam_filters()
    
    # Paso 3: Crear archivo de entrada sin duplicados
    cigale_df = create_cigale_input_without_duplicates()
    if cigale_df is None:
        print("❌ No se pudo crear el archivo de entrada")
        return
    
    # Paso 4: Configurar CIGALE
    config_success = setup_cigale_configuration()
    if not config_success:
        print("❌ No se pudo configurar CIGALE")
        return
    
    # Crear script de ejecución mejorado
    run_script = """#!/bin/bash

echo "🚀 CIGALE - ANÁLISIS CORREGIDO"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Verificar archivos necesarios
if [ ! -f "cigale_input.txt" ]; then
    echo "❌ Error: No se encuentra cigale_input.txt"
    exit 1
fi

if [ ! -f "pcigale.ini" ]; then
    echo "❌ Error: No se encuentra pcigale.ini"
    exit 1
fi

if [ ! -f "pcigale.ini.spec" ]; then
    echo "❌ Error: No se encuentra pcigale.ini.spec"
    exit 1
fi

# Solo añadir filtros con formato correcto (los que empiezan con splus_ o decam_)
echo "🎛️  Añadiendo filtros corregidos..."
for filter_file in splus_*.dat decam_*.dat; do
    if [ -f "$filter_file" ]; then
        echo "   - Procesando $filter_file"
        pcigale-filters add "$filter_file"
        if [ $? -eq 0 ]; then
            echo "   ✅ $filter_file añadido"
        else
            echo "   ❌ Error con $filter_file"
        fi
    fi
done

echo ""
echo "📋 Configuración:"
echo "   - Objetos: $(wc -l < cigale_input.txt)"
echo "   - Filtros DECam: 5" 
echo "   - Filtros SPLUS: 7"
echo "   - Módulos: sfhdelayed, bc03, dustatt_powerlaw, redshifting"
echo ""

# Verificar configuración
echo "🔍 Verificando configuración..."
if pcigale check; then
    echo "✅ Configuración válida"
    
    echo ""
    # Ejecutar CIGALE
    echo "🔄 Ejecutando análisis..."
    pcigale run

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 ¡ANÁLISIS COMPLETADO!"
        echo "📁 Resultados en: out/"
        echo "⏰ Finalizado: $(date)"
    else
        echo "❌ Error en ejecución"
        exit 1
    fi
else
    echo "❌ Error en configuración"
    exit 1
fi
"""
    
    with open("run_cigale_fixed.sh", "w") as f:
        f.write(run_script)
    
    os.chmod("run_cigale_fixed.sh", 0o755)
    
    print(f"\n" + "="*70)
    print("🎯 PREPARACIÓN COMPLETADA - CORREGIDA")
    print("="*70)
    print("📁 ARCHIVOS GENERADOS:")
    print("   • cigale_input.txt       - Datos sin duplicados")
    print("   • pcigale.ini            - Configuración principal") 
    print("   • pcigale.ini.spec       - Especificación")
    print("   • splus_F378.dat, etc.   - Filtros SPLUS corregidos")
    print("   • decam_u.dat, etc.      - Filtros DECam")
    print("   • run_cigale_fixed.sh    - Script de ejecución")
    
    print(f"\n🚀 PARA EJECUTAR:")
    print("   ./run_cigale_fixed.sh")
    
    print(f"\n💡 CORRECCIONES APLICADAS:")
    print("   • ✅ Filtros SPLUS: Añadida cabecera 'photon' requerida")
    print("   • ✅ Datos: Eliminados IDs duplicados automáticamente") 
    print("   • ✅ Script: Solo usa filtros con formato correcto")
    print("   • ✅ Configuración: Reinicio limpio de archivos .ini")

if __name__ == "__main__":
    main()
