#!/usr/bin/env python3
"""
Script para crear mosaicos con Montage - Versión Terminal
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

def crear_mosaico_terminal():
    """Versión optimizada para terminal"""
    print("🛠️ INICIANDO MOSAICO EN TERMINAL")
    print("=" * 60)
    
    # Configuración
    work_dir = os.path.abspath("../anac_data/montage_work")
    output_dir = os.path.abspath("../anac_data/Figs-images")
    data_base_dir = os.path.abspath("../anac_data")
    
    # Campos a procesar
    campos = [f'CenA{i:02d}' for i in range(1, 25)]
    
    print(f"📁 Procesando {len(campos)} campos...")
    
    # Limpiar directorio de trabajo
    if os.path.exists(work_dir):
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
            for campo in campos:
                src_path = f"{data_base_dir}/{campo}/{campo}_{filtro}.fits.fz"
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
                        print(f"   ❌ Error {campo}: {e}")
            
            if not archivos_filtro:
                continue
            
            # Header de referencia
            ra_center = 201.3651
            dec_center = -43.0191
            size = 6.0
            
            rtn = mHdr(f"{ra_center} {dec_center}", size, size, f"region_{filtro}.hdr")
            if rtn['status'] != '0':
                crear_header_simple(ra_center, dec_center, size, f"region_{filtro}.hdr")
            
            # Procesar con Montage
            rtn_tbl = mImgtbl(".", f"rimages_{filtro}.tbl")
            if rtn_tbl['status'] != '0':
                print(f"❌ Error tabla: {rtn_tbl}")
                continue
            
            rtn_proj = mProjExec(".", f"rimages_{filtro}.tbl", f"region_{filtro}.hdr", 
                                projdir=f"projected_{filtro}", quickMode=True)
            if rtn_proj['status'] != '0':
                print(f"❌ Error proyección: {rtn_proj}")
                continue
            
            rtn_ptbl = mImgtbl(f"projected_{filtro}", f"pimages_{filtro}.tbl")
            if rtn_ptbl['status'] != '0':
                print(f"❌ Error tabla proyectada: {rtn_ptbl}")
                continue
            
            # Crear mosaico
            mosaic_file = f"mosaic_{filtro}.fits"
            rtn_add = mAdd(f"projected_{filtro}", f"pimages_{filtro}.tbl", f"region_{filtro}.hdr", mosaic_file)
            
            if rtn_add['status'] == '0':
                mosaicos_finales[filtro] = mosaic_file
                print(f"✅ Mosaico {filtro} creado")
            
            # Limpiar archivos temporales
            for archivo in archivos_filtro:
                if os.path.exists(archivo):
                    os.remove(archivo)
            
            gc.collect()
        
        # Crear RGB
        if len(mosaicos_finales) == 3:
            print("\n🌈 CREANDO IMAGEN RGB...")
            crear_rgb_simple(mosaicos_finales, output_dir, len(campos))
            
            # Guardar mosaicos
            for filtro, archivo in mosaicos_finales.items():
                shutil.copy(archivo, f"{output_dir}/mosaic_{filtro}.fits")
                print(f"✅ Mosaico {filtro} guardado")
            
            return True
        
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
        data_r = fits.getdata(mosaicos_dict['F861'])
        data_g = fits.getdata(mosaicos_dict['F660'])
        data_b = fits.getdata(mosaicos_dict['F515'])
        
        print(f"📊 Datos cargados: R={data_r.shape}, G={data_g.shape}, B={data_b.shape}")
        
        # Crear RGB con Lupton (más eficiente)
        rgb_image = make_lupton_rgb(data_r, data_g, data_b, stretch=0.8, Q=10)
        
        # Crear figura
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
        
        # Guardar
        output_path = f"{output_dir}/mosaico_final_{total_campos}campos.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
        print(f"✅ Imagen RGB guardada: {output_path}")
        
        # Liberar memoria
        plt.close(fig)
        del data_r, data_g, data_b, rgb_image
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en RGB: {e}")
        return False

if __name__ == "__main__":
    print("🚀 EJECUTANDO EN TERMINAL")
    print("💾 Versión optimizada sin interfaz gráfica")
    
    # Limpiar memoria
    gc.collect()
    
    # Ejecutar
    success = crear_mosaico_terminal()
    
    if success:
        print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("📁 Resultados en: ../anac_data/Figs-images/")
    else:
        print(f"\n💥 EL PROCESO FALLÓ")
    
    gc.collect()
