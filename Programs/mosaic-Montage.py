#!/usr/bin/env python3
"""
Script para crear mosaicos con Montage - Versión Corregida con Análisis Mejorado
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
from scipy.ndimage import gaussian_filter
from astropy.visualization import ZScaleInterval, AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# Importar APLpy para mejor calidad en RGB
try:
    import aplpy
    APLPY_AVAILABLE = True
    print("✅ APLpy está disponible para mejor calidad RGB")
except ImportError:
    APLPY_AVAILABLE = False
    print("⚠️  APLpy no disponible, usando método manual para RGB")

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

def analizar_estadisticas_archivos(mosaicos_dict):
    """Analiza las estadísticas reales de los archivos FITS para ajustar correctamente la visualización"""
    print("📊 ANALIZANDO ESTADÍSTICAS DE LOS DATOS...")
    
    estadisticas = {}
    # Calcular más percentiles para tener mejor control
    percentiles = [1, 2, 5, 10, 15, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]
    
    for filtro, archivo in mosaicos_dict.items():
        print(f"   🔍 Analizando {filtro}...")
        with fits.open(archivo) as hdul:
            data = hdul[0].data
            
            # Filtrar solo píxeles con datos válidos (mayores que 0)
            data_valida = data[data > 0]
            
            if len(data_valida) > 0:
                # Calcular percentiles
                perc_values = np.percentile(data_valida, percentiles)
                stats = {
                    'min': np.min(data_valida),
                    'max': np.max(data_valida),
                    'mean': np.mean(data_valida),
                    'median': np.median(data_valida),
                    'std': np.std(data_valida),
                    'total_pixels': data.size,
                    'valid_pixels': len(data_valida),
                    'coverage': (len(data_valida) / data.size) * 100
                }
                # Añadir percentiles al diccionario
                for i, p in enumerate(percentiles):
                    # Usar nombres sin puntos para las claves
                    perc_name = f'percentile_{int(p) if p == int(p) else str(p).replace(".", "_")}'
                    stats[perc_name] = perc_values[i]
                
                estadisticas[filtro] = stats
                
                print(f"   📈 {filtro}:")
                print(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                print(f"      Mediana: {stats['median']:.4f}, Media: {stats['mean']:.4f}")
                print(f"      P1: {stats['percentile_1']:.4f}, P50: {stats['percentile_50']:.4f}, P99: {stats['percentile_99']:.4f}")
                print(f"      P99.5: {stats['percentile_99_5']:.4f}, P99.9: {stats['percentile_99_9']:.4f}")
                print(f"      Cobertura: {stats['coverage']:.1f}%")
                
                # Análisis adicional sobre el rango dinámico
                dynamic_range = stats['max'] / stats['percentile_50'] if stats['percentile_50'] > 0 else 0
                print(f"      Rango dinámico (max/mediana): {dynamic_range:.1f}")
                
            else:
                print(f"   ❌ {filtro}: No hay datos válidos")
                estadisticas[filtro] = None
    
    return estadisticas

def calcular_rango_visualizacion_optimo(estadisticas):
    """Calcula los rangos óptimos de visualización basado en las estadísticas reales"""
    print("🎯 CALCULANDO RANGOS ÓPTIMOS DE VISUALIZACIÓN...")
    
    rangos = {}
    
    for filtro, stats in estadisticas.items():
        if stats is None:
            continue
            
        # ESTRATEGIA MEJORADA: Basada en el análisis de tus datos
        # Tus datos tienen valores máximos altos pero percentiles bajos
        # Esto indica que hay pocos píxeles brillantes y muchos oscuros
        if filtro == 'F861':  # Rojo 
            # Usar un rango más amplio para capturar tanto áreas oscuras como brillantes
            vmin = max(stats['percentile_1'], 0.001)  # Evitar cero
            vmax = stats['percentile_99_9']  # Usar percentil más alto para capturar estructuras brillantes
            print(f"   🔴 {filtro}: vmin={vmin:.4f} (P1), vmax={vmax:.4f} (P99.9)")
            
        elif filtro == 'F660':  # Verde
            vmin = max(stats['percentile_2'], 0.001)
            vmax = stats['percentile_99_5']
            print(f"   🟢 {filtro}: vmin={vmin:.4f} (P2), vmax={vmax:.4f} (P99.5)")
            
        elif filtro == 'F515':  # Azul
            vmin = max(stats['percentile_5'], 0.001)
            vmax = stats['percentile_99']
            print(f"   🔵 {filtro}: vmin={vmin:.4f} (P5), vmax={vmax:.4f} (P99)")
        
        # Ajustes adicionales basados en las características de tus datos
        # Si el rango es muy pequeño, expandirlo
        if vmax - vmin < 0.1:
            vmax = vmin + 1.0
        
        # Asegurar que vmax sea mayor que vmin
        if vmax <= vmin:
            vmax = vmin + 0.1
        
        rangos[filtro] = {'vmin': vmin, 'vmax': vmax}
    
    return rangos

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
            
            # Crear header con resolución optimizada
            crear_header_calidad_mejorada(ra_center, dec_center, size, f"region_{filtro}.hdr")
            print(f"   ✅ Header creado para {filtro}")
            
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
        
        # ANALIZAR ESTADÍSTICAS ANTES DE CREAR RGB
        if len(mosaicos_finales) == 3:
            print("\n📊 ANALIZANDO DATOS PARA OPTIMIZAR VISUALIZACIÓN...")
            estadisticas = analizar_estadisticas_archivos(mosaicos_finales)
            rangos_optimos = calcular_rango_visualizacion_optimo(estadisticas)
            
            print("\n🌈 CREANDO IMAGEN RGB CON AJUSTES MEJORADOS...")
            
            # Primero intentar con método manual mejorado que nos da más control
            success = crear_rgb_manual_mejorado(mosaicos_finales, estadisticas, os.path.join("..", output_dir), len(campos))
            
            # Luego intentar con APLpy si está disponible
            if APLPY_AVAILABLE and success:
                print("\n🔄 INTENTANDO VERSIÓN APLPY...")
                success_aplpy = crear_rgb_aplpy_mejorado(mosaicos_finales, estadisticas, os.path.join("..", output_dir), len(campos))
                if not success_aplpy:
                    print("⚠️  APLpy falló, pero tenemos versión manual")
            
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

def crear_rgb_manual_mejorado(mosaicos_dict, estadisticas, output_dir, total_campos):
    """Método manual mejorado con ajustes específicos para datos de bajo contraste"""
    try:
        print("🎨 Creando RGB con método manual MEJORADO...")
        
        # Cargar datos
        data_r = fits.getdata(mosaicos_dict['F861'])
        data_g = fits.getdata(mosaicos_dict['F660'])
        data_b = fits.getdata(mosaicos_dict['F515'])
        
        stats_r = estadisticas['F861']
        stats_g = estadisticas['F660'] 
        stats_b = estadisticas['F515']
        
        print("   📈 ANALIZANDO CARACTERÍSTICAS ESPECÍFICAS...")
        print(f"   🔴 F861 - Rango: {stats_r['min']:.4f} a {stats_r['max']:.4f}")
        print(f"   🟢 F660 - Rango: {stats_g['min']:.4f} a {stats_g['max']:.4f}")
        print(f"   🔵 F515 - Rango: {stats_b['min']:.4f} a {stats_b['max']:.4f}")
        
        def procesar_canal_mejorado(data, stats, canal_nombre, boost_factor=1.0):
            """Procesamiento mejorado para datos de bajo contraste"""
            data_valida = data[data > 0]
            
            if len(data_valida) == 0:
                return np.zeros_like(data)
            
            # ESTRATEGIA MEJORADA: Usar transformación no lineal para realzar estructuras débiles
            vmin = stats['percentile_5']
            vmax = stats['percentile_99_9']  # Usar percentil alto para capturar detalles brillantes
            
            print(f"   📊 {canal_nombre}: vmin={vmin:.4f}, vmax={vmax:.4f}")
            
            # Normalizar con clip suave
            data_norm = (data - vmin) / (vmax - vmin + 1e-10)
            
            # Aplicar transformación no lineal (raíz cuadrada) para realzar áreas oscuras
            data_norm = np.sqrt(np.clip(data_norm, 0, 1))
            
            # Boost para mejorar visibilidad
            data_norm = np.clip(data_norm * boost_factor, 0, 1)
            
            # Suavizado muy ligero
            data_smooth = gaussian_filter(data_norm, sigma=0.5)
            
            return data_smooth
        
        # Procesar canales con boosts diferentes para balance de color
        print("   🎛️  Procesando canales con ajustes no lineales...")
        r_norm = procesar_canal_mejorado(data_r, stats_r, 'rojo', boost_factor=1.2)
        g_norm = procesar_canal_mejorado(data_g, stats_g, 'verde', boost_factor=1.1)
        b_norm = procesar_canal_mejorado(data_b, stats_b, 'azul', boost_factor=1.3)
        
        # Crear múltiples versiones con diferentes balances
        print("   🖼️  Creando múltiples versiones con diferentes balances...")
        
        balances = [
            ("balance1", (1.0, 1.0, 1.0)),
            ("balance2", (1.0, 1.1, 1.4)),
            ("balance3", (0.9, 1.0, 1.6)),
            ("balance4", (1.2, 1.1, 1.8)),
        ]
        
        for balance_name, (r_boost, g_boost, b_boost) in balances:
            r_final = np.clip(r_norm * r_boost, 0, 1)
            g_final = np.clip(g_norm * g_boost, 0, 1)
            b_final = np.clip(b_norm * b_boost, 0, 1)
            
            rgb_image = np.stack([r_final, g_final, b_final], axis=-1)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(15, 12), facecolor='black')
            ax.imshow(rgb_image, origin='lower', aspect='auto')
            ax.axis('off')
            
            # Título informativo
            ax.set_title(f'Centaurus A - Mosaico {total_campos} campos\nF861 (R) + F660 (G) + F515 (B)', 
                        color='white', size=18, pad=20, weight='bold')
            
            # Información de ajustes
            info_text = f'''VERSIÓN: {balance_name}
Balance: R×{r_boost}, G×{g_boost}, B×{b_boost}
Procesamiento: sqrt-stretch + boost
Cobertura: {stats_r["coverage"]:.1f}%'''
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='white', 
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            
            # Barra de escala
            altura, ancho = rgb_image.shape[:2]
            scale_pixels = (30 / 60) * (ancho / 12.0)  # 30 arcmin
            
            bar_y = altura * 0.08
            bar_x = ancho * 0.08
            
            ax.plot([bar_x, bar_x + scale_pixels], [bar_y, bar_y], 
                    color='yellow', linewidth=4)
            ax.text(bar_x + scale_pixels/2, bar_y - altura * 0.02, '30 arcmin', 
                    color='yellow', ha='center', va='top', fontsize=14, weight='bold')
            
            # Guardar
            output_path = os.path.join(output_dir, f"mosaico_manual_{balance_name}_{total_campos}campos.png")
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
            print(f"   ✅ {output_path}")
            plt.close()
        
        # Versión adicional con stretch asinh (mejor para astronomía)
        print("   🖼️  Creando versión con stretch asinh...")
        crear_version_asinh(data_r, data_g, data_b, stats_r, stats_g, stats_b, output_dir, total_campos)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en método manual mejorado: {e}")
        import traceback
        traceback.print_exc()
        return False

def crear_version_asinh(data_r, data_g, data_b, stats_r, stats_g, stats_b, output_dir, total_campos):
    """Crea versión con stretch asinh que es mejor para datos astronómicos"""
    try:
        # Usar stretch asinh para mejor rango dinámico
        stretch = AsinhStretch()
        
        def aplicar_asinh(data, stats):
            data_valida = data[data > 0]
            if len(data_valida) == 0:
                return np.zeros_like(data)
            
            vmin = stats['percentile_10']
            vmax = stats['percentile_99_9']
            
            # Normalizar
            data_norm = (data - vmin) / (vmax - vmin + 1e-10)
            data_norm = np.clip(data_norm, 0, 1)
            
            # Aplicar asinh stretch
            data_stretched = stretch(data_norm)
            
            return data_stretched
        
        r_asinh = aplicar_asinh(data_r, stats_r)
        g_asinh = aplicar_asinh(data_g, stats_g)
        b_asinh = aplicar_asinh(data_b, stats_b)
        
        # Balance para asinh
        r_final = np.clip(r_asinh * 1.1, 0, 1)
        g_final = np.clip(g_asinh * 1.0, 0, 1)
        b_final = np.clip(b_asinh * 1.3, 0, 1)
        
        rgb_asinh = np.stack([r_final, g_final, b_final], axis=-1)
        
        fig, ax = plt.subplots(figsize=(15, 12), facecolor='black')
        ax.imshow(rgb_asinh, origin='lower', aspect='auto')
        ax.axis('off')
        
        ax.set_title(f'Centaurus A - Stretch Asinh\n{total_campos} campos - F861 + F660 + F515', 
                    color='white', size=18, pad=20, weight='bold')
        
        info_text = f'''STRETCH ASINH
Mejor para rango dinámico astronómico
Cobertura: {stats_r["coverage"]:.1f}%'''
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='white', 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Barra de escala
        altura, ancho = rgb_asinh.shape[:2]
        scale_pixels = (30 / 60) * (ancho / 12.0)
        
        bar_y = altura * 0.08
        bar_x = ancho * 0.08
        
        ax.plot([bar_x, bar_x + scale_pixels], [bar_y, bar_y], 
                color='yellow', linewidth=4)
        ax.text(bar_x + scale_pixels/2, bar_y - altura * 0.02, '30 arcmin', 
                color='yellow', ha='center', va='top', fontsize=14, weight='bold')
        
        output_path = os.path.join(output_dir, f"mosaico_asinh_{total_campos}campos.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
        print(f"   ✅ {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️  Error en versión asinh: {e}")

def crear_rgb_aplpy_mejorado(mosaicos_dict, estadisticas, output_dir, total_campos):
    """Versión APLpy mejorada"""
    try:
        print("🎨 Creando RGB con APLpy mejorado...")
        
        r_file = mosaicos_dict['F861']
        g_file = mosaicos_dict['F660']
        b_file = mosaicos_dict['F515']
        
        stats_r = estadisticas['F861']
        stats_g = estadisticas['F660']
        stats_b = estadisticas['F515']
        
        # Crear cubo RGB
        cube_file = "rgb_cube.fits"
        aplpy.make_rgb_cube([r_file, g_file, b_file], cube_file)
        
        # Usar zscale para ajuste automático (más robusto)
        zscale = ZScaleInterval()
        
        with fits.open(r_file) as hdul:
            r_data = hdul[0].data
            r_zmin, r_zmax = zscale.get_limits(r_data[r_data > 0])
        
        with fits.open(g_file) as hdul:
            g_data = hdul[0].data
            g_zmin, g_zmax = zscale.get_limits(g_data[g_data > 0])
        
        with fits.open(b_file) as hdul:
            b_data = hdul[0].data
            b_zmin, b_zmax = zscale.get_limits(b_data[b_data > 0])
        
        print(f"   📊 ZScale - R: ({r_zmin:.4f}, {r_zmax:.4f}), G: ({g_zmin:.4f}, {g_zmax:.4f}), B: ({b_zmin:.4f}, {b_zmax:.4f})")
        
        # Crear imagen con zscale
        rgb_zscale = "rgb_zscale.png"
        aplpy.make_rgb_image(cube_file, rgb_zscale,
                           vmin_r=r_zmin, vmax_r=r_zmax,
                           vmin_g=g_zmin, vmax_g=g_zmax,
                           vmin_b=b_zmin, vmax_b=b_zmax)
        
        # Crear figura APLpy
        cube_2d_file = cube_file.replace('.fits', '_2d.fits')
        if os.path.exists(cube_2d_file):
            fig = aplpy.FITSFigure(cube_2d_file)
            fig.show_rgb(rgb_zscale)
            
            fig.axis_labels.set_font(size=14)
            fig.tick_labels.set_font(size=12)
            fig.tick_labels.set_xformat('hh:mm:ss')
            fig.tick_labels.set_yformat('dd:mm')
            
            fig.add_scalebar(0.5)
            fig.scalebar.set_label('30 arcmin')
            fig.scalebar.set_color('white')
            fig.scalebar.set_font_size(14)
            
            fig.add_label(0.05, 0.95, 
                         f'Centaurus A - {total_campos} campos\nF861 (R) + F660 (G) + F515 (B)',
                         relative=True, color='white', size=16,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            
            fig.add_label(0.05, 0.05, 'Ajuste: ZScale',
                         relative=True, color='white', size=12)
            
            output_path = os.path.join(output_dir, f"mosaico_aplpy_zscale_{total_campos}campos.png")
            fig.save(output_path, dpi=200)
            print(f"   ✅ {output_path}")
            plt.close('all')
        
        # Limpiar
        for temp_file in [cube_file, cube_2d_file, rgb_zscale]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en APLpy mejorado: {e}")
        return False

# [Mantener las funciones calcular_header_automatico y crear_header_calidad_mejorada igual]
def calcular_header_automatico(campos, data_base_dir):
    """Calcula automáticamente el header óptimo basado en las posiciones reales de los campos"""
    print("🗺️  Calculando header automático basado en las posiciones de los campos...")
    
    todas_ras = []
    todas_decs = []
    
    for campo in campos:
        for filtro in ['F861', 'F660', 'F515']:
            src_path = os.path.join(data_base_dir, campo, f"{campo}_{filtro}.fits.fz")
            if os.path.exists(src_path):
                try:
                    with fits.open(src_path) as hdul:
                        if len(hdul) > 1:
                            header = hdul[1].header
                        else:
                            header = hdul[0].header
                        
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
        return 201.3651, -43.0191, 8.0
    
    ra_center = np.mean(todas_ras)
    dec_center = np.mean(todas_decs)
    
    ra_range = max(todas_ras) - min(todas_ras)
    dec_range = max(todas_decs) - min(todas_decs)
    size = max(ra_range, dec_range) * 1.2
    size = max(6.0, min(size, 12.0))
    
    print(f"   🎯 Centro calculado: RA={ra_center:.4f}, DEC={dec_center:.4f}")
    print(f"   📏 Tamaño calculado: {size:.2f} grados")
    
    return ra_center, dec_center, size

def crear_header_calidad_mejorada(ra, dec, size, output_file):
    """Header optimizado para mejor calidad"""
    naxis = 2500
    
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
    print(f"   📐 Resolución alta: {naxis}×{naxis} píxeles")

if __name__ == "__main__":
    print("🚀 EJECUTANDO MOSAICO CON PROCESAMIENTO MEJORADO")
    print("🎯 Enfocado en realzar estructuras débiles en datos de bajo contraste")
    
    if not os.path.exists("CenA01"):
        print("❌ ERROR: No se encuentra el directorio CenA01")
        sys.exit(1)
    
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    gc.collect()
    
    success = crear_mosaico_con_header_automatico()
    
    if success:
        print(f"\n🎉 ¡PROCESO COMPLETADO!")
        print("📁 Resultados en: Figs-images/")
        print("   Versiones creadas:")
        print("   - mosaico_manual_balance[1-4]_24campos.png (4 balances diferentes)")
        print("   - mosaico_asinh_24campos.png (stretch asinh)")
        print("   - mosaico_aplpy_zscale_24campos.png (versión APLpy)")
        print("   - mosaic_F861.fits, mosaic_F660.fits, mosaic_F515.fits")
        print("\n💡 Recomendación: Revisa todas las versiones y elige la que mejor muestre Centaurus A")
    else:
        print(f"\n💥 EL PROCESO FALLÓ")
    
    gc.collect()
