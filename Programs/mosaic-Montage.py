#!/usr/bin/env python3
"""
Script para crear mosaicos con Montage - Versión Terminal
Con cálculo automático del header de referencia y manejo eficiente de memoria
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import gc
import sys

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

def calcular_header_automatico(campos, data_base_dir):
    """Calcula automáticamente el header óptimo basado en las posiciones reales de los campos"""
    print("🗺️  Calculando header automático basado en las posiciones de los campos...")
    
    todas_ras = []
    todas_decs = []
    
    for campo in campos:
        # Usar cualquier filtro para obtener las coordenadas
        for filtro in ['F861', 'F660', 'F515']:
            src_path = os.path.join(data_base_dir, campo, f"{campo}_{filtro}.fits.fz")
            if os.path.exists(src_path):
                try:
                    with fits.open(src_path) as hdul:
                        if len(hdul) > 1:
                            header = hdul[1].header
                        else:
                            header = hdul[0].header
                        
                        # Obtener coordenadas del centro
                        ra = header.get('CRVAL1')
                        dec = header.get('CRVAL2')
                        
                        if ra is not None and dec is not None:
                            todas_ras.append(ra)
                            todas_decs.append(dec)
                            print(f"   📍 {campo}: RA={ra:.4f}, DEC={dec:.4f}")
                            break
                except Exception as e:
                    print(f"   ❌ Error leyendo {campo}: {e}")
    
    if not todas_ras:
        print("❌ No se pudieron leer coordenadas de ningún campo.")
        return 201.3651, -43.0191, 8.0
    
    # Calcular centroide de todos los campos
    ra_center = np.mean(todas_ras)
    dec_center = np.mean(todas_decs)
    
    # Calcular el tamaño necesario para cubrir todos los campos
    # con un margen adicional del 20%
    ra_range = max(todas_ras) - min(todas_ras)
    dec_range = max(todas_decs) - min(todas_decs)
    size = max(ra_range, dec_range) * 1.2  # 20% de margen
    
    # Tamaño mínimo de 6 grados, máximo de 12 grados
    size = max(6.0, min(size, 12.0))
    
    print(f"   🎯 Centro calculado: RA={ra_center:.4f}, DEC={dec_center:.4f}")
    print(f"   📏 Tamaño calculado: {size:.2f} grados")
    print(f"   📊 Rango RA: {min(todas_ras):.4f} - {max(todas_ras):.4f}")
    print(f"   📊 Rango DEC: {min(todas_decs):.4f} - {max(todas_decs):.4f}")
    
    return ra_center, dec_center, size

def crear_mosaico_con_header_automatico():
    """Versión con header automático calculado"""
    print("🛠️ INICIANDO MOSAICO CON HEADER AUTOMÁTICO")
    print("=" * 60)
    
    current_dir = os.getcwd()
    print(f"📁 Directorio base: {current_dir}")
    
    work_dir = "montage_work"
    output_dir = "Figs-images"
    
    campos = [f'CenA{i:02d}' for i in range(1, 25)]
    
    print(f"📁 Procesando {len(campos)} campos...")
    
    if os.path.exists(work_dir):
        print("🧹 Limpiando directorio de trabajo anterior...")
        shutil.rmtree(work_dir)
    
    os.makedirs(work_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    original_dir = os.getcwd()
    os.chdir(work_dir)
    
    try:
        # Calcular header automático basado en las posiciones reales
        ra_center, dec_center, size = calcular_header_automatico(campos, "..")
        
        mosaicos_finales = {}
        
        for filtro in ['F861', 'F660', 'F515']:
            print(f"\n🌈 PROCESANDO FILTRO {filtro}...")
            
            archivos_filtro = []
            
            for campo in campos:
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
            
            if not archivos_filtro:
                print(f"❌ No hay archivos para {filtro}")
                continue
            
            print(f"   📊 {len(archivos_filtro)} archivos listos para procesar")
            
            # Usar el header calculado automáticamente
            rtn = mHdr(f"{ra_center} {dec_center}", size, size, f"region_{filtro}.hdr")
            if rtn['status'] != '0':
                crear_header_automatico_mejorado(ra_center, dec_center, size, f"region_{filtro}.hdr")
                print(f"   ✅ Header automático creado para {filtro}")
            
            # Procesar con Montage
            print("   🔄 Creando tabla de imágenes...")
            rtn_tbl = mImgtbl(".", f"rimages_{filtro}.tbl")
            if rtn_tbl['status'] != '0':
                print(f"❌ Error en mImgtbl: {rtn_tbl}")
                continue
            
            print(f"   📋 Tabla creada con {rtn_tbl.get('count', 0)} imágenes")
            
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
                if os.path.exists(mosaic_file):
                    with fits.open(mosaic_file) as hdul:
                        data = hdul[0].data
                        shape = data.shape
                        non_zero = np.sum(data > 0)
                        coverage = (non_zero / data.size) * 100
                    
                    mosaicos_finales[filtro] = mosaic_file
                    print(f"   ✅ Mosaico {filtro} creado: {shape} (cobertura: {coverage:.1f}%)")
                    
                    # Verificar si hay áreas sin datos
                    if coverage < 30:
                        print(f"   ⚠️  Cobertura baja - posiblemente faltan campos")
                else:
                    print(f"   ❌ Mosaico {filtro} no se creó correctamente")
            else:
                print(f"❌ Error en mAdd: {rtn_add}")
            
            # Limpiar archivos temporales
            print("   🧹 Limpiando archivos temporales...")
            for archivo in archivos_filtro:
                if os.path.exists(archivo):
                    os.remove(archivo)
            
            try:
                shutil.rmtree(projected_dir)
            except:
                pass
            
            gc.collect()
        
        # Crear RGB
        if len(mosaicos_finales) == 3:
            print("\n🌈 CREANDO IMAGEN RGB...")
            success = crear_rgb_eficiente_memoria(mosaicos_finales, os.path.join("..", output_dir), len(campos))
            
            if success:
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

def crear_header_automatico_mejorado(ra, dec, size, output_file):
    """Header mejorado basado en cálculo automático"""
    # Ajustar resolución según el tamaño para evitar mosaicos demasiado grandes
    if size <= 6:
        naxis = 1200
    elif size <= 8:
        naxis = 1000
    elif size <= 10:
        naxis = 800
    else:  # size <= 12
        naxis = 600  # Reducido para mosaicos grandes
    
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
    print(f"   📐 Resolución: {naxis}×{naxis} píxeles")

def crear_rgb_eficiente_memoria(mosaicos_dict, output_dir, total_campos):
    """Versión optimizada para memoria del creador de RGB"""
    try:
        print("🎨 Creando imagen RGB optimizada para memoria...")
        
        # Primero analizar los archivos sin cargarlos completamente
        print("   📊 Analizando estructura de archivos...")
        
        # Obtener dimensiones y WCS de un archivo de referencia
        ref_file = list(mosaicos_dict.values())[0]
        with fits.open(ref_file, memmap=True) as hdul:
            header = hdul[0].header
            shape = hdul[0].shape
            ra = header.get('CRVAL1', 'N/A')
            dec = header.get('CRVAL2', 'N/A')
        
        print(f"   📏 Tamaño del mosaico: {shape}")
        
        # Si el mosaico es muy grande, usar muestreo
        max_size = 8000  # Tamaño máximo manejable
        if max(shape) > max_size:
            sample_factor = max(1, max(shape) // max_size)
            print(f"   🔻 Muestreo: 1 de cada {sample_factor} píxeles")
        else:
            sample_factor = 1
        
        # Cargar datos con muestreo para reducir uso de memoria
        print("   📥 Cargando datos con muestreo...")
        
        def cargar_con_muestreo(archivo, sample_factor):
            with fits.open(archivo, memmap=True) as hdul:
                data = hdul[0].data
                if sample_factor > 1:
                    # Muestreo simple para reducir tamaño
                    data = data[::sample_factor, ::sample_factor]
                return data.astype(np.float32)
        
        data_r = cargar_con_muestreo(mosaicos_dict['F861'], sample_factor)
        data_g = cargar_con_muestreo(mosaicos_dict['F660'], sample_factor) 
        data_b = cargar_con_muestreo(mosaicos_dict['F515'], sample_factor)
        
        print(f"   📊 Datos cargados: R={data_r.shape}, G={data_g.shape}, B={data_b.shape}")
        
        # Encontrar región con datos válidos
        print("   🔍 Encontrando región con datos...")
        
        # Usar máscaras booleanas eficientes en memoria
        mask_r = data_r > np.percentile(data_r[data_r > 0], 5) if np.any(data_r > 0) else np.zeros_like(data_r, dtype=bool)
        mask_g = data_g > np.percentile(data_g[data_g > 0], 5) if np.any(data_g > 0) else np.zeros_like(data_g, dtype=bool)
        mask_b = data_b > np.percentile(data_b[data_b > 0], 5) if np.any(data_b > 0) else np.zeros_like(data_b, dtype=bool)
        
        mask_combined = mask_r | mask_g | mask_b
        
        if not np.any(mask_combined):
            print("   ❌ No hay datos válidos en los mosaicos")
            return False
        
        # Encontrar límites de la región con datos
        rows, cols = np.where(mask_combined)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        
        print(f"   📐 Región con datos: filas {min_row}-{max_row}, columnas {min_col}-{max_col}")
        
        # Añadir margen (reducido para ahorrar memoria)
        margin = int(min(data_r.shape) * 0.02)  # Solo 2% de margen
        min_row = max(0, min_row - margin)
        max_row = min(data_r.shape[0], max_row + margin)
        min_col = max(0, min_col - margin)
        max_col = min(data_r.shape[1], max_col + margin)
        
        # Recortar datos
        print("   ✂️  Recortando datos...")
        data_r_crop = data_r[min_row:max_row, min_col:max_col]
        data_g_crop = data_g[min_row:max_row, min_col:max_col]
        data_b_crop = data_b[min_row:max_row, min_col:max_col]
        
        print(f"   📏 Datos recortados: {data_r_crop.shape}")
        
        # Liberar memoria de los arrays grandes inmediatamente
        del data_r, data_g, data_b, mask_r, mask_g, mask_b, mask_combined
        gc.collect()
        
        # Normalización por partes para ahorrar memoria
        print("   📈 Normalizando canales...")
        
        def normalizar_por_partes(data, low_percent=2, high_percent=98, chunk_size=1000):
            """Normaliza en chunks para ahorrar memoria"""
            data_pos = data[data > 0]
            if len(data_pos) == 0:
                return np.zeros_like(data)
            
            vmin = np.percentile(data_pos, low_percent)
            vmax = np.percentile(data_pos, high_percent)
            
            # Normalizar en chunks
            result = np.empty_like(data, dtype=np.float32)
            for i in range(0, data.shape[0], chunk_size):
                i_end = min(i + chunk_size, data.shape[0])
                chunk = data[i:i_end]
                chunk_norm = np.arcsinh((chunk - vmin) / max(vmax - vmin, 1e-10) * 10) / 3
                chunk_norm = np.clip(chunk_norm, 0, 1)
                result[i:i_end] = chunk_norm
            
            return result
        
        r_norm = normalizar_por_partes(data_r_crop, 1, 99)
        g_norm = normalizar_por_partes(data_g_crop, 1, 98) 
        b_norm = normalizar_por_partes(data_b_crop, 2, 97)
        
        # Liberar más memoria
        del data_r_crop, data_g_crop, data_b_crop
        gc.collect()
        
        # Crear imagen RGB
        print("   🎨 Combinando canales RGB...")
        rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)
        
        # Crear figura
        print("   🖼️  Creando figura...")
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
        ax.imshow(rgb_image, origin='lower')
        ax.axis('off')
        
        # Información
        ax.set_title(f'Centaurus A - Mosaico Completo ({total_campos} campos)\nF861(R) + F660(G) + F515(B)', 
                    color='white', size=14, pad=20)
        
        info_text = f'''Coordinates: RA={ra:.4f}°, DEC={dec:.4f}°
Fields: {total_campos} campos
Size: {rgb_image.shape[1]}×{rgb_image.shape[0]} pixels
Method: Memory-optimized'''
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='white', 
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Barra de escala
        altura, ancho = rgb_image.shape[:2]
        scale_arcmin = 20
        scale_pixels = int(scale_arcmin * 60 / (12.0 / ancho))  # Aproximación
        
        bar_y = altura * 0.05
        bar_x = ancho * 0.05
        
        ax.plot([bar_x, bar_x + scale_pixels], [bar_y, bar_y], 
                color='yellow', linewidth=3)
        ax.text(bar_x + scale_pixels/2, bar_y - altura * 0.02, f'{scale_arcmin} arcmin', 
                color='yellow', ha='center', va='top', fontsize=10, weight='bold')
        
        # Guardar
        output_path = os.path.join(output_dir, f"mosaico_automatico_{total_campos}campos.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"✅ Imagen RGB guardada: {output_path}")
        
        # Liberar memoria final
        plt.close(fig)
        del r_norm, g_norm, b_norm, rgb_image
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Error en RGB optimizado: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 EJECUTANDO MOSAICO CON HEADER AUTOMÁTICO OPTIMIZADO")
    print("🎯 Calculando posición y tamaño óptimos")
    print("📁 Directorio base: anac_data")
    
    if not os.path.exists("CenA01"):
        print("❌ ERROR: No se encuentra el directorio CenA01")
        print("   Asegúrate de ejecutar este script desde el directorio anac_data")
        sys.exit(1)
    
    # Configurar para usar menos memoria
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    gc.collect()
    
    success = crear_mosaico_con_header_automatico()
    
    if success:
        print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("📁 Resultados en: Figs-images/")
        print("   - mosaico_automatico_24campos.png")
        print("   - mosaic_F861.fits, mosaic_F660.fits, mosaic_F515.fits")
    else:
        print(f"\n💥 EL PROCESO FALLÓ")
    
    gc.collect()
