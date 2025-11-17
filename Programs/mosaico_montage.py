#!/usr/bin/env python3
"""
Script para crear mosaicos con Montage - Versión Terminal
Corregido para crear directorios projected_
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # IMPORTANTE: Sin interfaz gráfica
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import gc
import sys

# Configurar para no mostrar plots
plt.ioff()

# Importar MontagePy
try:
    from MontagePy.main import mHdr, mImgtbl, mProjExec, mAdd
    MONTAGE_AVAILABLE = True
    print("✅ MontagePy está disponible")
except ImportError as e:
    print(f"❌ Error importando MontagePy: {e}")
    MONTAGE_AVAILABLE = False
    sys.exit(1)

def verificar_estructura_directorios():
    """Verifica que la estructura de directorios sea correcta"""
    print("🔍 VERIFICANDO ESTRUCTURA DE DIRECTORIOS...")
    
    # Directorio actual debería ser anac_data
    current_dir = os.getcwd()
    print(f"   Directorio actual: {current_dir}")
    
    # Verificar que existen los campos
    campos = [f'CenA{i:02d}' for i in range(1, 25)]
    campos_encontrados = []
    
    for campo in campos[:3]:  # Verificar solo los primeros 3 para prueba
        campo_path = os.path.join(current_dir, campo)
        if os.path.exists(campo_path):
            archivos = []
            for filtro in ['F861', 'F660', 'F515']:
                archivo_path = os.path.join(campo_path, f"{campo}_{filtro}.fits.fz")
                if os.path.exists(archivo_path):
                    archivos.append(f"{campo}_{filtro}.fits.fz")
            
            if len(archivos) == 3:
                campos_encontrados.append(campo)
                print(f"   ✅ {campo}: {len(archivos)}/3 archivos")
            else:
                print(f"   ⚠️  {campo}: {len(archivos)}/3 archivos")
        else:
            print(f"   ❌ {campo}: Directorio no encontrado")
    
    return len(campos_encontrados) > 0

def crear_mosaico_terminal():
    """Versión optimizada para terminal"""
    print("🛠️ INICIANDO MOSAICO EN TERMINAL")
    print("=" * 60)
    
    # Obtener directorio actual (debería ser anac_data)
    current_dir = os.getcwd()
    print(f"📁 Directorio base: {current_dir}")
    
    # Configuración de directorios RELATIVA al directorio actual
    work_dir = "montage_work"  # Se creará en anac_data/montage_work
    output_dir = "Figs-images"  # Se creará en anac_data/Figs-images
    
    # Campos a procesar
    campos = [f'CenA{i:02d}' for i in range(1, 25)]
    
    print(f"📁 Procesando {len(campos)} campos...")
    
    # Verificar estructura primero
    if not verificar_estructura_directorios():
        print("❌ Estructura de directorios incorrecta. Verifica que:")
        print("   - Estés en el directorio anac_data")
        print("   - Existan las carpetas CenA01, CenA02, etc.")
        print("   - Cada carpeta tenga los archivos .fits.fz")
        return False
    
    # Limpiar directorio de trabajo
    if os.path.exists(work_dir):
        print("🧹 Limpiando directorio de trabajo anterior...")
        shutil.rmtree(work_dir)
    
    os.makedirs(work_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar directorio original
    original_dir = os.getcwd()
    os.chdir(work_dir)
    
    try:
        mosaicos_finales = {}
        
        for filtro in ['F861', 'F660', 'F515']:
            print(f"\n🌈 PROCESANDO FILTRO {filtro}...")
            
            # Preparar datos
            archivos_filtro = []
            archivos_faltantes = []
            
            for campo in campos:
                # Ruta RELATIVA desde anac_data
                src_path = os.path.join("..", campo, f"{campo}_{filtro}.fits.fz")
                dst_path = f"{campo}_{filtro}.fits"
                
                if os.path.exists(src_path):
                    try:
                        with fits.open(src_path) as hdul:
                            if len(hdul) > 1:
                                data = hdul[1].data
                                header = hdul[1].header
                            else:
                                data = hdul[0].data
                                header = hdul[0].header
                            
                            primary_hdu = fits.PrimaryHDU(data=data, header=header)
                            primary_hdu.writeto(dst_path, overwrite=True)
                            archivos_filtro.append(dst_path)
                        print(f"   ✅ {campo}_{filtro}")
                    except Exception as e:
                        print(f"   ❌ Error procesando {campo}: {e}")
                else:
                    archivos_faltantes.append(src_path)
            
            # Mostrar archivos faltantes solo si hay muchos
            if archivos_faltantes and len(archivos_faltantes) > 5:
                print(f"   ⚠️  {len(archivos_faltantes)} archivos no encontrados")
            elif archivos_faltantes:
                for faltante in archivos_faltantes:
                    print(f"   ❌ No encontrado: {faltante}")
            
            if not archivos_filtro:
                print(f"❌ No hay archivos para {filtro}")
                continue
            
            print(f"   📊 {len(archivos_filtro)} archivos listos para procesar")
            
            # Header de referencia
            ra_center = 201.3651
            dec_center = -43.0191
            size = 6.0
            
            rtn = mHdr(f"{ra_center} {dec_center}", size, size, f"region_{filtro}.hdr")
            if rtn['status'] != '0':
                crear_header_simple(ra_center, dec_center, size, f"region_{filtro}.hdr")
                print(f"   ✅ Header creado para {filtro}")
            
            # Procesar con Montage
            print("   🔄 Creando tabla de imágenes...")
            rtn_tbl = mImgtbl(".", f"rimages_{filtro}.tbl")
            if rtn_tbl['status'] != '0':
                print(f"❌ Error en mImgtbl: {rtn_tbl}")
                continue
            
            print(f"   📋 Tabla creada con {rtn_tbl.get('count', 0)} imágenes")
            
            # CORRECCIÓN: Crear directorio projected manualmente
            projected_dir = f"projected_{filtro}"
            if not os.path.exists(projected_dir):
                print(f"   📁 Creando directorio: {projected_dir}")
                os.makedirs(projected_dir)
            
            print("   🔄 Reprojectando imágenes...")
            rtn_proj = mProjExec(".", f"rimages_{filtro}.tbl", f"region_{filtro}.hdr", 
                                projdir=projected_dir, quickMode=True)
            if rtn_proj['status'] != '0':
                print(f"❌ Error en mProjExec: {rtn_proj}")
                continue
            
            print(f"   ✅ {rtn_proj.get('count', 0)} imágenes proyectadas")
            
            print("   🔄 Creando tabla proyectada...")
            rtn_ptbl = mImgtbl(projected_dir, f"pimages_{filtro}.tbl")
            if rtn_ptbl['status'] != '0':
                print(f"❌ Error en mImgtbl (proyectado): {rtn_ptbl}")
                continue
            
            # Crear mosaico
            print("   🖼️  Creando mosaico...")
            mosaic_file = f"mosaic_{filtro}.fits"
            rtn_add = mAdd(projected_dir, f"pimages_{filtro}.tbl", f"region_{filtro}.hdr", mosaic_file)
            
            if rtn_add['status'] == '0':
                # Verificar que el mosaico se creó correctamente
                if os.path.exists(mosaic_file):
                    with fits.open(mosaic_file) as hdul:
                        data = hdul[0].data
                        shape = data.shape
                        non_zero = np.sum(data > 0)
                        coverage = (non_zero / data.size) * 100
                    
                    mosaicos_finales[filtro] = mosaic_file
                    print(f"   ✅ Mosaico {filtro} creado: {shape} (cobertura: {coverage:.1f}%)")
                else:
                    print(f"   ❌ Mosaico {filtro} no se creó correctamente")
            else:
                print(f"❌ Error en mAdd: {rtn_add}")
            
            # Limpiar archivos temporales
            print("   🧹 Limpiando archivos temporales...")
            for archivo in archivos_filtro:
                if os.path.exists(archivo):
                    os.remove(archivo)
            
            # Limpiar directorio proyectado
            try:
                shutil.rmtree(projected_dir)
            except:
                pass
            
            gc.collect()
        
        # Crear RGB si tenemos los 3 mosaicos
        if len(mosaicos_finales) == 3:
            print("\n🌈 CREANDO IMAGEN RGB...")
            success = crear_rgb_simple(mosaicos_finales, os.path.join("..", output_dir), len(campos))
            
            if success:
                # Guardar mosaicos
                for filtro, archivo in mosaicos_finales.items():
                    dest_path = os.path.join("..", output_dir, f"mosaic_{filtro}.fits")
                    shutil.copy(archivo, dest_path)
                    print(f"✅ Mosaico {filtro} guardado en {dest_path}")
                
                return True
            else:
                print("❌ Falló la creación del RGB")
        else:
            print(f"❌ Faltan mosaicos: {len(mosaicos_finales)}/3")
        
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_dir)

def crear_header_simple(ra, dec, size, output_file):
    """Header simple"""
    naxis = 1500
    
    header_content = f"""SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -64 / number of bits per data pixel
NAXIS   =                    2 / number of data axes
NAXIS1  =                {naxis} / length of data axis 1
NAXIS2  =                {naxis} / length of data axis 2
EXTEND  =                    T / FITS dataset may contain extensions
CTYPE1  = 'RA---TAN'           / Right Ascension, gnomonic projection
CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection
CRVAL1  = {ra:20.10f} / [deg] Reference coordinate on axis 1
CRVAL2  = {dec:20.10f} / [deg] Reference coordinate on axis 2
CRPIX1  =              {naxis/2:.1f} / [pixel] Reference pixel on axis 1
CRPIX2  =              {naxis/2:.1f} / [pixel] Reference pixel on axis 2
CDELT1  = {-(size/naxis):20.10f} / [deg/pixel] Coordinate increment
CDELT2  = { (size/naxis):20.10f} / [deg/pixel] Coordinate increment
CROTA2  =                  0.0 / [deg] Rotation angle
EQUINOX =               2000.0 / Equinox of celestial coordinate system
"""
    
    with open(output_file, 'w') as f:
        f.write(header_content)

def crear_rgb_simple(mosaicos_dict, output_dir, total_campos):
    """RGB simple para terminal"""
    try:
        print("🎨 Creando imagen RGB...")
        
        # Cargar datos
        print("   📥 Cargando datos de mosaicos...")
        data_r = fits.getdata(mosaicos_dict['F861'])
        data_g = fits.getdata(mosaicos_dict['F660'])
        data_b = fits.getdata(mosaicos_dict['F515'])
        
        print(f"   📊 Datos cargados: R={data_r.shape}, G={data_g.shape}, B={data_b.shape}")
        
        # Crear RGB con Lupton (más eficiente)
        print("   🎨 Generando imagen RGB con método Lupton...")
        rgb_image = make_lupton_rgb(data_r, data_g, data_b, stretch=0.8, Q=10)
        
        # Crear figura
        print("   🖼️  Creando figura...")
        fig, ax = plt.subplots(figsize=(16, 14), facecolor='black')
        ax.imshow(rgb_image, origin='lower')
        ax.axis('off')
        
        # Información
        with fits.open(mosaicos_dict['F861']) as hdul:
            header = hdul[0].header
            ra = header.get('CRVAL1', 'N/A')
            dec = header.get('CRVAL2', 'N/A')
        
        ax.set_title(f'Centaurus A - Mosaico ({total_campos} campos)\nF861(R) + F660(G) + F515(B)', 
                    color='white', size=16, pad=20)
        
        info_text = f'''Coordinates: RA={ra:.4f}°, DEC={dec:.4f}°
Fields: {total_campos} campos
Image Size: {rgb_image.shape[1]}×{rgb_image.shape[0]} pixels'''
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='white', 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Barra de escala
        altura = rgb_image.shape[0]
        ancho = rgb_image.shape[1]
        pixel_scale = 0.55
        scale_arcmin = 20
        scale_pixels = int(scale_arcmin * 60 / pixel_scale)
        
        bar_y = altura * 0.05
        bar_x = ancho * 0.05
        
        ax.plot([bar_x, bar_x + scale_pixels], [bar_y, bar_y], 
                color='yellow', linewidth=4)
        ax.text(bar_x + scale_pixels/2, bar_y - altura * 0.02, f'{scale_arcmin} arcmin', 
                color='yellow', ha='center', va='top', fontsize=12, weight='bold')
        
        # Guardar
        output_path = os.path.join(output_dir, f"mosaico_final_{total_campos}campos.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
        print(f"✅ Imagen RGB guardada: {output_path}")
        
        # Liberar memoria
        plt.close(fig)
        del data_r, data_g, data_b, rgb_image
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en RGB: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 EJECUTANDO EN TERMINAL")
    print("💾 Versión optimizada sin interfaz gráfica")
    print("📁 Directorio base: anac_data")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("CenA01"):
        print("❌ ERROR: No se encuentra el directorio CenA01")
        print("   Asegúrate de ejecutar este script desde el directorio anac_data")
        print("   Estructura esperada:")
        print("   anac_data/")
        print("   ├── CenA01/")
        print("   ├── CenA02/")
        print("   ├── ...")
        print("   └── Este script")
        sys.exit(1)
    
    # Limpiar memoria
    gc.collect()
    
    # Ejecutar
    success = crear_mosaico_terminal()
    
    if success:
        print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("📁 Resultados en: Figs-images/")
        print("   - mosaico_final_24campos.png (imagen RGB)")
        print("   - mosaic_F861.fits, mosaic_F660.fits, mosaic_F515.fits (mosaicos individuales)")
    else:
        print(f"\n💥 EL PROCESO FALLÓ")
        print("   Revisa los mensajes de error arriba")
    
    gc.collect()
