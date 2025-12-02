# configurar_cigale_GCs_MEJORADO_FUNCIONAL.py
# VERSIÓN MEJORADA BASADA EN TU SCRIPT FUNCIONAL
# Solo mejoras mínimas SIN cambiar módulos

import os
import shutil
from datetime import datetime

def crear_ini_mejorado():
    print("MEJORANDO TU VERSIÓN FUNCIONAL - Sin cambiar módulos")
    print("=" * 78)
    print("Mantiene EXACTAMENTE los mismos módulos que funcionan")
    print("Solo agrega más valores para mejor muestreo")
    print("-" * 78)

    config = """# NGC 5128 Globular Clusters – XSL 2024 (Arentsen+ 2024)
# VERSIÓN MEJORADA - Basada en tu configuración funcional

data_file = gc_splus_cigale_custom.txt
parameters_file = 

# MÓDULOS QUE SABEMOS FUNCIONAN (NO CAMBIAR):
sed_modules = sfhdelayed, xsl, redshifting

analysis_method = pdf_analysis
cores = 5

# Bands to consider. To consider uncertainties too, the name of the band
# must be indicated with the _err suffix. For instance: FUV, FUV_err.
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

additionalerror = 0.05

# Properties to be considered. All properties are to be given in the
# rest frame rather than the observed frame. This is the case for
# instance the equivalent widths and for luminosity densities.
properties = 


# Configuration of the SED creation modules.
[sed_modules_params]
  
  [[sfhdelayed]]
    # e-folding time of the main stellar population model in Myr.
    # Manteniendo tus valores (funcionan)
    tau_main = 1000.0, 2000.0, 5000.0
    # Age of the main stellar population in the galaxy in Myr. The precision
    # is 1 Myr.
    # MEJORA: Más valores para mejor muestreo (6 en lugar de 4)
    age_main = 8000, 9000, 10000, 11000, 12000, 13000
    # e-folding time of the late starburst population model in Myr.
    tau_burst = 50.0
    # Age of the late burst in Myr. The precision is 1 Myr.
    age_burst = 20
    # Mass fraction of the late burst population.
    f_burst = 0.0
    # Multiplicative factor controlling the SFR if normalise is False. For
    # instance without any burst: SFR(t)=sfr_A×t×exp(-t/τ)/τ²
    sfr_A = 1.0
    # Normalise the SFH to produce one solar mass.
    normalise = True
  
  [[xsl]]
    # Initial mass function: 1 (Kroupa)
    imf = 1
    # Metallicity. Possible values are: 0.0004, 0.004, 0.008, 0.02, 0.03.
    # Manteniendo TUS valores (sabemos que funcionan)
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.03
    # Age [Myr] of the separation between the young and the old star
    # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
    # differentiate ages (only an old population).
    separation_age = 10
  
  [[redshifting]]
    # Redshift of the objects. Leave empty to use the redshifts from the
    # input file.
    redshift = 0.001825


# Configuration of the statistical analysis method.
[analysis_params]
  # List of the physical properties to estimate. Leave empty to analyse
  # all the physical properties (not recommended when there are many
  # models).
  # MEJORA: Mantenemos tus variables (funcionan)
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star
  # List of bands for which to estimate the fluxes. Note that this is
  # independent from the fluxes actually fitted to estimate the physical
  # properties.
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  # If true, save the best SED for each observation to a file.
  save_best_sed = True
  # Save the raw chi2. It occupies ~15 MB/million models/variable. Allowed
  # values are 'all', 'none', 'properties', and 'fluxes'.
  save_chi2 = none
  # Take into account upper limits. If 'full', the exact computation is
  # done. If 'noscaling', the scaling of the models will not be adjusted
  # but the χ² will include the upper limits adequately. Waiving the
  # adjustment makes the fitting much faster compared to the 'full' option
  # while generally not affecting the results in any substantial manner.
  # This is the recommended option as it achieves a good balance between
  # speed and reliability. Finally, 'none' simply discards bands with
  # upper limits.
  lim_flag = noscaling
  # If true, for each object we create a mock object and analyse them.
  mock_flag = False
  # When redshifts are not given explicitly in the redshifting module,
  # number of decimals to round the observed redshifts to compute the grid
  # of models. To disable rounding give a negative value. Do not round if
  # you use narrow-band filters.
  redshift_decimals = 6
  # Number of blocks to compute the models and analyse the observations.
  # If there is enough memory, we strongly recommend this to be set to 1.
  blocks = 1
"""

    # 1. Hacer backup organizado
    backup_dir = "backups_cigale"
    os.makedirs(backup_dir, exist_ok=True)
    
    if os.path.exists("pcigale.ini"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{backup_dir}/pcigale.ini.backup_{timestamp}"
        shutil.copy2("pcigale.ini", backup)
        print(f"✅ Backup creado → {backup}")
        print(f"   (Backup guardado en {backup_dir}/)")
    else:
        print("✅ No existe pcigale.ini previo")

    # 2. Crear NUEVO archivo
    with open("pcigale.ini", "w") as f:
        f.write(config)

    print("\n✅ pcigale.ini creado con MEJORAS MÍNIMAS:")
    print("✓ Módulos INALTERADOS: sfhdelayed, xsl, redshifting")
    print("✓ Metalicidades INALTERADAS: 0.0004, 0.004, 0.008, 0.02, 0.03")
    print("✓ Filtros INALTERADOS: F0378 a F0861")
    print("✓ Variables INALTERADAS: masa, metalicidad, edad")
    
    print("\n📊 ÚNICA MEJORA:")
    print("   • Edades: 6 valores (8000, 9000, 10000, 11000, 12000, 13000)")
    print("     (antes: 4 valores: 8000, 10000, 12000, 13000)")
    
    # 3. Calcular número de modelos
    n_edades = 6
    n_tau = 3
    n_metal = 5
    total_modelos = n_edades * n_tau * n_metal
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   • Modelos totales: {total_modelos} (antes: 60)")
    print(f"   • Mejor muestreo en edad (+50%)")
    
    # 4. Crear resumen
    crear_resumen_mejoras(n_edades, n_tau, n_metal, total_modelos)
    
    return True

def crear_resumen_mejoras(n_edades, n_tau, n_metal, total_modelos):
    """Crear archivo con resumen de las mejoras"""
    
    resumen = f"""RESUMEN DE MEJORAS - CONFIGURACIÓN CIGALE GCs
================================================================
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MEJORAS RESPECTO A VERSIÓN ANTERIOR:
1. MÁS VALORES DE EDAD: 6 en lugar de 4
   - Antes: 8000, 10000, 12000, 13000 Myr
   - Ahora: 8000, 9000, 10000, 11000, 12000, 13000 Myr
   - Mejor muestreo: +50% de valores

2. TODO LO DEMÁS SE MANTIENE IGUAL:
   - Módulos: sfhdelayed, xsl, redshifting (FUNCIONAN)
   - Metalicidades: 0.0004, 0.004, 0.008, 0.02, 0.03 (FUNCIONAN)
   - Filtros: 7 filtros S-PLUS (F0378 a F0861) (FUNCIONAN)
   - Variables estimadas: masa, metalicidad, edad (FUNCIONAN)

ESTADÍSTICAS DE MODELOS:
  - Edades: {n_edades} valores
  - Tau: {n_tau} valores (1000, 2000, 5000)
  - Metalicidades: {n_metal} valores
  - Total modelos: {total_modelos}
  - Modelos anteriores: 60
  - Incremento: +{total_modelos - 60} modelos (+{(total_modelos-60)/60*100:.0f}%)

VALIDACIÓN ESPERADA:
  - Mismo comportamiento que versión anterior (ya probada)
  - Mejor resolución en edades (más opciones para fitting)
  - Tiempo de cómputo similar (poco incremento)

INSTRUCCIONES:
  1. Verificar: pcigale check  (debe pasar igual que antes)
  2. Ejecutar: pcigale run
  3. Comparar resultados con corridas anteriores

CONTACTO:
  Luis A. Gutiérrez Soto
  gsoto.angel@gmail.com
================================================================
"""
    
    with open("mejoras_resumen.txt", "w") as f:
        f.write(resumen)
    
    print(f"\n📄 Resumen creado: mejoras_resumen.txt")

if __name__ == "__main__":
    crear_ini_mejorado()
    
    print("\n" + "=" * 78)
    print("🚀 INSTRUCCIONES (IGUAL QUE ANTES):")
    print("=" * 78)
    print("\n1. Verificar que todo sigue funcionando:")
    print("   pcigale check")
    print("\n2. Si pasa la verificación, ejecutar:")
    print("   pcigale run")
    print("\n3. Los resultados serán compatibles con tus corridas anteriores")
    print("\n💡 RECUERDA: Esta versión usa los MISMOS módulos que ya funcionan")
    print("   Solo agregamos más valores de edad para mejor muestreo")
