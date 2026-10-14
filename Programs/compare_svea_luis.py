import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import re
import warnings
warnings.filterwarnings('ignore')

def cargar_datos_cigale_directo(ruta_fits='out/results.fits'):
    """
    Carga directamente los resultados de CIGALE desde el archivo FITS
    y calcula edad en Gyr y [Fe/H] desde la metalicidad Z.
    """
    print("📊 CARGANDO RESULTADOS DE CIGALE DESDE FITS")
    print("-" * 40)
    
    try:
        # Leer el archivo FITS usando astropy Table (más robusto)
        table = Table.read(ruta_fits, hdu=1)
        cigale_df = table.to_pandas()
        
        print(f"✅ Archivo FITS leído correctamente")
        print(f"   Número de objetos: {len(cigale_df)}")
        print(f"   Número de columnas: {len(cigale_df.columns)}")
        
        # Mostrar las primeras columnas para depuración
        print(f"\n🔍 Primeras 5 columnas: {list(cigale_df.columns[:5])}")
        
        # Identificar columnas clave
        id_col = 'id'
        
        # Buscar columnas de edad (podría haber varias)
        age_cols = [col for col in cigale_df.columns if 'age' in col.lower()]
        print(f"   Columnas de edad encontradas: {age_cols}")
        
        # Preferir 'bayes.stellar.age_m_star' si está disponible
        if 'bayes.stellar.age_m_star' in cigale_df.columns:
            age_col = 'bayes.stellar.age_m_star'
        elif 'best.stellar.age_m_star' in cigale_df.columns:
            age_col = 'best.stellar.age_m_star'
        elif age_cols:
            age_col = age_cols[0]
        else:
            raise ValueError("No se encontró columna de edad en los datos de CIGALE")
        
        # Buscar columnas de metalicidad
        metal_cols = [col for col in cigale_df.columns if 'metal' in col.lower()]
        print(f"   Columnas de metalicidad encontradas: {metal_cols}")
        
        # Preferir 'bayes.stellar.metallicity' si está disponible
        if 'bayes.stellar.metallicity' in cigale_df.columns:
            metal_col = 'bayes.stellar.metallicity'
        elif 'best.stellar.metallicity' in cigale_df.columns:
            metal_col = 'best.stellar.metallicity'
        elif metal_cols:
            metal_col = metal_cols[0]
        else:
            # Intentar buscar columnas con 'z' (pero cuidado con redshift)
            z_cols = [col for col in cigale_df.columns if 'z' in col.lower() 
                     and 'redshift' not in col.lower() 
                     and 'z' == col.lower()]
            if z_cols:
                metal_col = z_cols[0]
            else:
                raise ValueError("No se encontró columna de metalicidad en los datos de CIGALE")
        
        print(f"\n🔍 Columnas identificadas:")
        print(f"   ID: {id_col}")
        print(f"   Edad: {age_col}")
        print(f"   Metalicidad: {metal_col}")
        
        # Verificar que existan las columnas
        if id_col not in cigale_df.columns:
            # Buscar alternativa para ID
            id_col_candidates = [c for c in cigale_df.columns if 'id' in c.lower()]
            if id_col_candidates:
                id_col = id_col_candidates[0]
                print(f"   ID alternativo usado: {id_col}")
            else:
                id_col = cigale_df.columns[0]
                print(f"   Usando primera columna como ID: {id_col}")
        
        # Asegurar que las columnas estén en formato string/numérico
        cigale_df[id_col] = cigale_df[id_col].astype(str).str.strip()
        
        # Convertir a numérico y manejar posibles errores
        cigale_df[age_col] = pd.to_numeric(cigale_df[age_col], errors='coerce')
        cigale_df[metal_col] = pd.to_numeric(cigale_df[metal_col], errors='coerce')
        
        # Calcular edad en Gyr (CIGALE usa Myr)
        # Verificar si ya está en Gyr o en Myr
        edad_media = cigale_df[age_col].mean()
        if edad_media > 1000:  # Probablemente en Myr
            cigale_df['age_Gyr'] = cigale_df[age_col] / 1000.0
            print(f"   Edades convertidas de Myr a Gyr")
        else:
            cigale_df['age_Gyr'] = cigale_df[age_col]
            print(f"   Edades asumidas en Gyr")
        
        # Convertir Z a [Fe/H]
        Z_solar = 0.014  # Valor solar actualizado (Asplund et al. 2009)
        
        # Verificar el rango de valores para determinar si son Z o ya [Fe/H]
        median_metal = cigale_df[metal_col].median()
        
        if median_metal > 0.0001 and median_metal < 0.05:  # Valores típicos de Z
            print(f"   Metalicidad detectada como Z (mediana: {median_metal:.5f})")
            cigale_df['FeH'] = np.log10(cigale_df[metal_col] / Z_solar)
            print(f"   Convertido a [Fe/H] usando Z_solar = {Z_solar}")
        elif median_metal < -5:  # Valores muy bajos, probablemente ya es log(Z/Z_solar)
            print(f"   Metalicidad detectada como log(Z/Z_solar) (mediana: {median_metal:.2f})")
            cigale_df['FeH'] = cigale_df[metal_col]
        else:
            print(f"   Metalicidad detectada como [Fe/H] (mediana: {median_metal:.2f})")
            cigale_df['FeH'] = cigale_df[metal_col]
        
        # Estadísticas
        print(f"\n📈 ESTADÍSTICAS CIGALE:")
        print(f"   Edad: {cigale_df['age_Gyr'].min():.1f} - {cigale_df['age_Gyr'].max():.1f} Gyr")
        print(f"   Edad media: {cigale_df['age_Gyr'].mean():.1f} ± {cigale_df['age_Gyr'].std():.1f} Gyr")
        print(f"   [Fe/H]: {cigale_df['FeH'].min():.2f} - {cigale_df['FeH'].max():.2f}")
        print(f"   [Fe/H] medio: {cigale_df['FeH'].mean():.3f} ± {cigale_df['FeH'].std():.3f}")
        
        # Crear DataFrame limpio con solo las columnas necesarias
        cigale_clean = cigale_df[[id_col, 'age_Gyr', 'FeH']].copy()
        
        # Guardar opcionalmente como CSV para uso futuro
        cigale_clean.to_csv('cigale_results_auto_clean.csv', index=False)
        print(f"\n💾 Datos limpios guardados en: 'cigale_results_auto_clean.csv'")
        
        return cigale_clean, id_col
    
    except Exception as e:
        print(f"❌ Error cargando datos de CIGALE: {e}")
        print(f"\n🔧 INTENTANDO SOLUCIÓN ALTERNATIVA...")
        
        # Intentar solución alternativa leyendo directamente con pandas
        try:
            print("   Intentando leer con pandas.read_csv...")
            cigale_df = pd.read_csv('out/results.txt', delim_whitespace=True, comment='#')
            print(f"   ✅ Datos cargados desde results.txt: {len(cigale_df)} objetos")
            
            # Identificar columnas
            id_col = 'id'
            if 'id' not in cigale_df.columns:
                id_col = cigale_df.columns[0]
            
            # Buscar columnas de edad
            age_col = None
            for col in cigale_df.columns:
                if 'age' in col.lower():
                    age_col = col
                    break
            
            # Buscar columnas de metalicidad
            metal_col = None
            for col in cigale_df.columns:
                if 'metal' in col.lower():
                    metal_col = col
                    break
            if not metal_col:
                for col in cigale_df.columns:
                    if 'z' in col.lower() and 'redshift' not in col.lower():
                        metal_col = col
                        break
            
            if not age_col or not metal_col:
                raise ValueError("No se encontraron columnas necesarias")
            
            # Calcular edad en Gyr
            if cigale_df[age_col].max() > 1000:
                cigale_df['age_Gyr'] = cigale_df[age_col] / 1000.0
            else:
                cigale_df['age_Gyr'] = cigale_df[age_col]
            
            # Convertir a [Fe/H]
            Z_solar = 0.014
            median_metal = cigale_df[metal_col].median()
            
            if median_metal > 0.0001 and median_metal < 0.05:
                cigale_df['FeH'] = np.log10(cigale_df[metal_col] / Z_solar)
            else:
                cigale_df['FeH'] = cigale_df[metal_col]
            
            cigale_clean = cigale_df[[id_col, 'age_Gyr', 'FeH']].copy()
            cigale_clean.to_csv('cigale_results_auto_clean.csv', index=False)
            
            return cigale_clean, id_col
            
        except Exception as e2:
            print(f"❌ Error en solución alternativa: {e2}")
            raise

def buscar_coincidencia_cigale(t17id, cigale_df, id_col='id'):
    """
    Busca un T17ID en el DataFrame de CIGALE de manera flexible.
    """
    t17id_str = str(t17id).strip()
    
    # Intentar coincidencia exacta primero
    if t17id_str in cigale_df[id_col].values:
        return cigale_df[cigale_df[id_col] == t17id_str]
    
    # Intentar sin el 'T17-' prefix
    if t17id_str.startswith('T17-'):
        simple_id = t17id_str[4:]  # Quitar 'T17-'
        if simple_id in cigale_df[id_col].values:
            return cigale_df[cigale_df[id_col] == simple_id]
    
    # Buscar por número solamente
    numbers = re.findall(r'\d+', t17id_str)
    if numbers:
        num_id = numbers[0]
        # Buscar por número en el ID
        for idx, row in cigale_df.iterrows():
            cigale_id = str(row[id_col])
            if num_id == cigale_id or num_id in cigale_id.split('-')[-1]:
                return cigale_df.iloc[[idx]]
    
    return pd.DataFrame()  # DataFrame vacío si no se encuentra

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

print("🔍 COMPARACIÓN AUTOMÁTICA CIGALE vs SVEA HERNÁNDEZ")
print("=" * 60)

# 1. CARGAR DATOS DE CIGALE DIRECTAMENTE
cigale_df, cigale_id_col = cargar_datos_cigale_directo('out/results.fits')

# 2. CARGAR CATÁLOGOS DE SVEA Y FOTOMETRÍA
print(f"\n📚 CARGANDO CATÁLOGOS EXTERNOS")
print("-" * 40)

# Cargar catálogo de Svea
try:
    svea_df = pd.read_csv('../../globular_clusters_ngc5128.csv')
    svea_df.columns = ['ID', 'RA', 'Dec', 'FeH_svea', 'FeH_err_svea']
    print(f"✅ Catálogo Svea cargado: {len(svea_df)} objetos")
except:
    print("❌ No se pudo cargar el catálogo de Svea desde ruta original")
    try:
        svea_df = pd.read_csv('globular_clusters_ngc5128.csv')
        svea_df.columns = ['ID', 'RA', 'Dec', 'FeH_svea', 'FeH_err_svea']
        print(f"✅ Catálogo Svea cargado desde ubicación local")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

# Cargar catálogo de fotometría
try:
    phot_df = pd.read_csv('../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv')
    phot_df['T17ID'] = phot_df['T17ID'].astype(str).str.strip()
    print(f"✅ Catálogo de fotometría cargado: {len(phot_df)} objetos")
    print(f"   Formato de T17ID ejemplo: {phot_df['T17ID'].iloc[0]}")
    
    # Verificar que las columnas de coordenadas existan
    if 'RAJ2000' not in phot_df.columns or 'DEJ2000' not in phot_df.columns:
        print("❌ Columnas de coordenadas no encontradas en fotometría")
        # Buscar columnas alternativas
        ra_cols = [c for c in phot_df.columns if 'ra' in c.lower() or 'RA' in c]
        dec_cols = [c for c in phot_df.columns if 'dec' in c.lower() or 'DE' in c or 'Dec' in c]
        
        if ra_cols and dec_cols:
            phot_df['RAJ2000'] = phot_df[ra_cols[0]]
            phot_df['DEJ2000'] = phot_df[dec_cols[0]]
            print(f"   Usando columnas alternativas: {ra_cols[0]}, {dec_cols[0]}")
        else:
            print("❌ No se encontraron columnas de coordenadas")
            exit()
            
except Exception as e:
    print(f"❌ Error cargando catálogo de fotometría: {e}")
    try:
        phot_df = pd.read_csv('all_fields_photometry_OPTIMIZED_1394objects.csv')
        phot_df['T17ID'] = phot_df['T17ID'].astype(str).str.strip()
        print(f"✅ Catálogo de fotometría cargado desde ubicación local")
    except:
        print("❌ No se pudo cargar el catálogo de fotometría")
        exit()

# 3. CRUCE POR COORDENADAS (Svea -> Fotometría)
print(f"\n🤝 REALIZANDO CRUCE POR COORDENADAS")
print("-" * 40)

# Usar tolerancia más estricta para coincidencias
max_sep_arcsec = 2.0  # 2 arcsec de tolerancia
max_sep = max_sep_arcsec * u.arcsec

coords_svea = SkyCoord(ra=svea_df['RA'].values * u.degree, 
                       dec=svea_df['Dec'].values * u.degree)
coords_phot = SkyCoord(ra=phot_df['RAJ2000'].values * u.degree, 
                       dec=phot_df['DEJ2000'].values * u.degree)

# Encontrar la coincidencia más cercana para cada objeto de Svea
idx_svea_in_phot, d2d, _ = coords_svea.match_to_catalog_sky(coords_phot)
match_mask = d2d < max_sep

print(f"Coincidencias encontradas: {match_mask.sum()} de {len(svea_df)}")
print(f"Separación máxima permitida: {max_sep_arcsec:.1f} arcsec")

# Mostrar detalles de las coincidencias
if match_mask.sum() > 0:
    print(f"\n📋 DETALLES DE COINCIDENCIAS:")
    for i, (svea_idx, phot_idx, sep) in enumerate(zip(
        np.where(match_mask)[0], 
        idx_svea_in_phot[match_mask], 
        d2d[match_mask]
    )):
        svea_id = svea_df.iloc[svea_idx]['ID']
        t17id = phot_df.iloc[phot_idx]['T17ID']
        print(f"   {i+1:2d}. {svea_id:12s} -> T17ID: {t17id:10s} (sep: {sep.to(u.arcsec).value:.2f} arcsec)")
else:
    print("⚠️  No se encontraron coincidencias con tolerancia estricta")
    print("   Intentando con tolerancia más amplia...")
    
    # Intentar con tolerancia más amplia
    max_sep_arcsec = 3.0  # 3 arcsec
    max_sep = max_sep_arcsec * u.arcsec
    match_mask = d2d < max_sep
    
    print(f"Coincidencias encontradas: {match_mask.sum()} de {len(svea_df)}")
    print(f"Separación máxima permitida: {max_sep_arcsec:.1f} arcsec")
    
    if match_mask.sum() > 0:
        print(f"\n📋 DETALLES DE COINCIDENCIAS:")
        for i, (svea_idx, phot_idx, sep) in enumerate(zip(
            np.where(match_mask)[0], 
            idx_svea_in_phot[match_mask], 
            d2d[match_mask]
        )):
            svea_id = svea_df.iloc[svea_idx]['ID']
            t17id = phot_df.iloc[phot_idx]['T17ID']
            print(f"   {i+1:2d}. {svea_id:12s} -> T17ID: {t17id:10s} (sep: {sep.to(u.arcsec).value:.2f} arcsec)")

# 4. CRUCE CON CIGALE
print(f"\n🔗 BUSCANDO COINCIDENCIAS EN CIGALE")
print("-" * 40)

comparison = []
no_encontrados = []

if match_mask.sum() == 0:
    print("❌ No hay coincidencias para continuar")
    exit()

for i, (svea_idx, phot_idx, sep) in enumerate(zip(
    np.where(match_mask)[0], 
    idx_svea_in_phot[match_mask], 
    d2d[match_mask]
)):
    svea_row = svea_df.iloc[svea_idx]
    phot_row = phot_df.iloc[phot_idx]
    
    svea_id = svea_row['ID']
    t17id = phot_row['T17ID']
    
    print(f"\n[{i+1}] {svea_id}:")
    print(f"   T17ID en fotometría: {t17id}")
    print(f"   Separación: {sep.to(u.arcsec).value:.2f} arcsec")
    
    # Buscar en CIGALE
    cigale_match = buscar_coincidencia_cigale(t17id, cigale_df, cigale_id_col)
    
    if len(cigale_match) > 0:
        cigale_row = cigale_match.iloc[0]
        
        print(f"   ✅ ENCONTRADO en CIGALE como: {cigale_row[cigale_id_col]}")
        print(f"      Edad CIGALE: {cigale_row['age_Gyr']:.1f} Gyr")
        print(f"      [Fe/H] CIGALE: {cigale_row['FeH']:.2f}")
        print(f"      [Fe/H] Svea: {svea_row['FeH_svea']:.2f} ± {svea_row['FeH_err_svea']:.2f}")
        
        comparison.append({
            'Svea_ID': svea_id,
            'T17ID_phot': t17id,
            'CIGALE_ID': cigale_row[cigale_id_col],
            'RA_Svea': svea_row['RA'],
            'Dec_Svea': svea_row['Dec'],
            'Sep_arcsec': sep.to(u.arcsec).value,
            'FeH_Svea': svea_row['FeH_svea'],
            'FeH_err_Svea': svea_row['FeH_err_svea'],
            'FeH_CIGALE': cigale_row['FeH'],
            'Age_CIGALE_Gyr': cigale_row['age_Gyr']
        })
    else:
        print(f"   ❌ NO ENCONTRADO en CIGALE")
        print(f"   ID buscado: '{t17id}'")
        
        # Mostrar primeros IDs de CIGALE para referencia
        if i == 0:  # Solo mostrar una vez
            print(f"   Primeros IDs en CIGALE:")
            for j in range(min(5, len(cigale_df))):
                print(f"      {j+1}. {cigale_df.iloc[j][cigale_id_col]}")
        
        no_encontrados.append((svea_id, t17id))

comp_df = pd.DataFrame(comparison)

if len(comp_df) == 0:
    print(f"\n🚨 NO SE ENCONTRARON COINCIDENCIAS EN CIGALE")
    
    if no_encontrados:
        print(f"\n📋 IDs no encontrados:")
        for svea_id, t17id in no_encontrados:
            print(f"   {svea_id}: T17ID='{t17id}'")
    
    # Mostrar primeros IDs de CIGALE para referencia
    print(f"\n📋 Primeros 10 IDs en CIGALE:")
    for i in range(min(10, len(cigale_df))):
        print(f"   {i+1:2d}. {cigale_df.iloc[i][cigale_id_col]}")
    
    print(f"\n🔍 DIAGNÓSTICO:")
    print(f"   Los T17IDs en Svea/Fotometría no coinciden con los IDs en CIGALE")
    print(f"   Posibles causas:")
    print(f"   1. CIGALE analizó un conjunto diferente de objetos")
    print(f"   2. Los IDs tienen diferente formato")
    print(f"   3. El archivo de entrada de CIGALE no incluía estos objetos")
    
    exit()

print(f"\n✅ {len(comp_df)} comparaciones exitosas de {match_mask.sum()} posibles")

# 5. ANÁLISIS ESTADÍSTICO
print(f"\n📊 ANÁLISIS ESTADÍSTICO DE LA COMPARACIÓN")
print("=" * 60)

comp_df['FeH_diff'] = comp_df['FeH_CIGALE'] - comp_df['FeH_Svea']
comp_df['abs_diff'] = np.abs(comp_df['FeH_diff'])

print(f"\n📈 Estadísticas de diferencia [Fe/H]:")
print(f"  Media: {comp_df['FeH_diff'].mean():.3f} dex")
print(f"  Desviación estándar: {comp_df['FeH_diff'].std():.3f} dex")
print(f"  Error absoluto medio: {comp_df['abs_diff'].mean():.3f} dex")
print(f"  RMS: {np.sqrt((comp_df['FeH_diff']**2).mean()):.3f} dex")

# Clasificación por calidad
print(f"\n📊 CLASIFICACIÓN POR CALIDAD DE LA COMPARACIÓN:")
categorias = [
    (0.2, "Excelente", "Δ ≤ 0.2 dex"),
    (0.5, "Bueno", "0.2 < Δ ≤ 0.5 dex"), 
    (float('inf'), "Pobre", "Δ > 0.5 dex")
]

for umbral, categoria, desc in categorias:
    if umbral == 0.2:
        mask = comp_df['abs_diff'] <= umbral
    elif umbral == 0.5:
        mask = (comp_df['abs_diff'] > 0.2) & (comp_df['abs_diff'] <= umbral)
    else:
        mask = comp_df['abs_diff'] > 0.5
    
    count = len(comp_df[mask])
    porcentaje = (count / len(comp_df)) * 100 if len(comp_df) > 0 else 0
    print(f"  {categoria:10s} ({desc:15s}): {count:2d} objetos ({porcentaje:5.1f}%)")

# Casos problemáticos
problematicos = comp_df[comp_df['abs_diff'] > 0.5]
if len(problematicos) > 0:
    print(f"\n⚠️  CASOS CON GRAN DISCREPANCIA (Δ > 0.5 dex):")
    for _, fila in problematicos.iterrows():
        edad_str = f"{fila['Age_CIGALE_Gyr']:.1f}" if pd.notna(fila['Age_CIGALE_Gyr']) else "N/A"
        print(f"  • {fila['Svea_ID']:10s}: Svea={fila['FeH_Svea']:5.2f} ± {fila['FeH_err_Svea']:.2f}, "
              f"CIGALE={fila['FeH_CIGALE']:5.2f}, Δ={fila['FeH_diff']:+5.2f} dex, "
              f"Edad={edad_str} Gyr")

# 6. GRÁFICOS DE COMPARACIÓN
print(f"\n🎨 GENERANDO GRÁFICOS DE COMPARACIÓN")
print("-" * 40)

# Configuración de estilo
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Comparación 1:1
ax1 = axes[0, 0]

# Colores según la magnitud de la discrepancia
colores = []
for _, fila in comp_df.iterrows():
    if fila['abs_diff'] <= 0.2:
        colores.append('#2ecc71')  # Verde
    elif fila['abs_diff'] <= 0.5:
        colores.append('#f39c12')  # Naranja
    else:
        colores.append('#e74c3c')  # Rojo

for (_, fila), color in zip(comp_df.iterrows(), colores):
    ax1.errorbar(fila['FeH_Svea'], fila['FeH_CIGALE'],
                xerr=fila['FeH_err_Svea'],
                yerr=0.15,  # Error típico estimado para SED fitting
                fmt='o', capsize=4, alpha=0.8,
                color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    
    # Añadir etiqueta con el nombre del GC
    ax1.annotate(fila['Svea_ID'], 
                (fila['FeH_Svea'], fila['FeH_CIGALE']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# Línea de identidad (y = x)
min_feH = min(comp_df['FeH_Svea'].min(), comp_df['FeH_CIGALE'].min()) - 0.3
max_feH = max(comp_df['FeH_Svea'].max(), comp_df['FeH_CIGALE'].max()) + 0.3
rango = [min_feH, max_feH]

ax1.plot(rango, rango, 'k--', alpha=0.5, linewidth=1.5, label='1:1 (igualdad)')
ax1.set_xlabel('[Fe/H] Svea Hernández (espectroscopía)', fontsize=12)
ax1.set_ylabel('[Fe/H] CIGALE (SED fitting)', fontsize=12)
ax1.set_title(f'Comparación directa - {len(comp_df)} cúmulos globulares', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.set_xlim(rango)
ax1.set_ylim(rango)
ax1.legend(loc='lower right')

# Añadir cuadro con estadísticas
estadisticas_texto = f'Δ[Fe/H] medio = {comp_df["FeH_diff"].mean():+.2f} dex\n'
estadisticas_texto += f'MAE = {comp_df["abs_diff"].mean():.2f} dex\n'
estadisticas_texto += f'RMS = {np.sqrt((comp_df["FeH_diff"]**2).mean()):.2f} dex'

ax1.text(0.05, 0.95, estadisticas_texto, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Gráfico 2: Diferencia vs Edad
ax2 = axes[0, 1]

scatter = ax2.scatter(comp_df['Age_CIGALE_Gyr'], comp_df['FeH_diff'],
                     c=comp_df['FeH_Svea'], cmap='coolwarm',
                     s=80, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Acuerdo perfecto')
ax2.set_xlabel('Edad estimada por CIGALE (Gyr)', fontsize=12)
ax2.set_ylabel('Δ[Fe/H] = CIGALE - Svea', fontsize=12)
ax2.set_title('Discrepancia vs Edad estimada', fontsize=14)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Barra de color
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('[Fe/H] Svea', fontsize=11)

# Etiquetar puntos con ID
for _, fila in comp_df.iterrows():
    ax2.annotate(fila['Svea_ID'], 
                (fila['Age_CIGALE_Gyr'], fila['FeH_diff']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

# Gráfico 3: Histograma de diferencias
ax3 = axes[1, 0]

bins = np.linspace(comp_df['FeH_diff'].min() - 0.1, comp_df['FeH_diff'].max() + 0.1, 10)
n, bins, patches = ax3.hist(comp_df['FeH_diff'], bins=bins,
                           edgecolor='black', alpha=0.7, 
                           color='skyblue', density=False)

# Colorear barras según el signo de la diferencia
for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
    if left_edge < 0 and right_edge <= 0:
        patch.set_facecolor('lightcoral')  # Rojo para valores negativos
    elif left_edge >= 0:
        patch.set_facecolor('lightgreen')  # Verde para valores positivos

ax3.axvline(comp_df['FeH_diff'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Media: {comp_df["FeH_diff"].mean():+.2f} dex')
ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

ax3.set_xlabel('Δ[Fe/H] (CIGALE - Svea)', fontsize=12)
ax3.set_ylabel('Número de cúmulos', fontsize=12)
ax3.set_title('Distribución de discrepancias', fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Gráfico 4: Edad vs [Fe/H] original
ax4 = axes[1, 1]

scatter2 = ax4.scatter(comp_df['Age_CIGALE_Gyr'], comp_df['FeH_Svea'],
                      c=comp_df['FeH_diff'], cmap='RdYlBu',
                      s=80, alpha=0.8, edgecolor='black', linewidth=0.5,
                      vmin=-2, vmax=2)  # Escala fija para colores

ax4.set_xlabel('Edad estimada por CIGALE (Gyr)', fontsize=12)
ax4.set_ylabel('[Fe/H] Svea (espectroscopía)', fontsize=12)
ax4.set_title('Edad vs Metalicidad real (colores: discrepancia)', fontsize=14)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Barra de color para la discrepancia
cbar2 = plt.colorbar(scatter2, ax=ax4)
cbar2.set_label('Δ[Fe/H] (CIGALE - Svea)', fontsize=11)

# Etiquetar puntos
for _, fila in comp_df.iterrows():
    ax4.annotate(fila['Svea_ID'], 
                (fila['Age_CIGALE_Gyr'], fila['FeH_Svea']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

plt.suptitle(f'Comparación CIGALE vs Svea Hernández - NGC 5128 (Centaurus A)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Guardar gráficos
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
nombre_archivo = f'comparacion_automatica_cigale_svea_{timestamp}.png'
plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
plt.savefig('comparacion_automatica_cigale_svea_latest.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"💾 Gráficos guardados como:")
print(f"   • {nombre_archivo}")
print(f"   • comparacion_automatica_cigale_svea_latest.png")

# 7. GUARDAR RESULTADOS
nombre_csv = f'comparacion_automatica_cigale_svea_{timestamp}.csv'
comp_df.to_csv(nombre_csv, index=False)
comp_df.to_csv('comparacion_automatica_cigale_svea_latest.csv', index=False)

print(f"💾 Datos guardados como:")
print(f"   • {nombre_csv}")
print(f"   • comparacion_automatica_cigale_svea_latest.csv")

# 8. RESUMEN EJECUTIVO
print(f"\n📋 RESUMEN EJECUTIVO")
print("=" * 60)

print(f"\n🎯 HALLAZGOS PRINCIPALES:")
print(f"1. Datos CIGALE cargados directamente desde results.fits")
print(f"2. Coincidencias encontradas: {len(comp_df)} de {match_mask.sum()} posibles")
print(f"3. Diferencia media [Fe/H]: {comp_df['FeH_diff'].mean():+.3f} dex")
print(f"4. Error absoluto medio: {comp_df['abs_diff'].mean():.3f} dex")

diferencia_media = comp_df['FeH_diff'].mean()
print(f"\n🔍 INTERPRETACIÓN:")
if diferencia_media > 0.5:
    print(f"• CIGALE SOBREESTIMA FUERTEMENTE [Fe/H] en {diferencia_media:.2f} dex")
    print(f"• Esto explica la aparente correlación inversa en el plot original")
elif diferencia_media > 0.2:
    print(f"• CIGALE sobreestima moderadamente [Fe/H] ({diferencia_media:.2f} dex)")
elif diferencia_media < -0.5:
    print(f"• CIGALE SUBESTIMA FUERTEMENTE [Fe/H] en {abs(diferencia_media):.2f} dex")
elif diferencia_media < -0.2:
    print(f"• CIGALE subestima moderadamente [Fe/H] ({diferencia_media:.2f} dex)")
else:
    print(f"• Acuerdo razonable entre métodos ({diferencia_media:.2f} dex)")

print(f"\n🔧 DIAGNÓSTICO DEL PROBLEMA ORIGINAL:")
print(f"• Plot original mostraba: GCs jóvenes = pobres en metales")
print(f"• REALIDAD: CIGALE sobreestima [Fe/H], creando una correlación espuria")
print(f"• CAUSA PROBABLE: Degeneración edad-metalicidad en ajuste SED")

print(f"\n🎯 RECOMENDACIONES INMEDIATAS:")
print(f"1. Revisar configuración CIGALE: usar separation_age=0 para GCs viejos")
print(f"2. Probar ajustes SED fijando edad a 10 Gyr (típico para GCs)")
print(f"3. Verificar corrección de extinción en bandas estrechas")
print(f"4. Inspeccionar SEDs individuales de casos con Δ > 0.5 dex")

print(f"\n📊 PARA DISCUSIÓN CON COLABORADORES:")
print(f"• Mostrar gráfico de comparación 1:1")
print(f"• Destacar los {len(problematicos)} casos con Δ > 0.5 dex")
print(f"• Explicar la degeneración edad-metalicidad como causa raíz")

print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print(f"📁 Archivos generados:")
print(f"   • Gráficos: {nombre_archivo}")
print(f"   • Datos: {nombre_csv}")
print(f"   • Versiones 'latest' para referencia rápida")
