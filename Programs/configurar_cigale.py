# configurar_cigale_final.py
import os
import shutil

def configurar_cigale_con_filtros():
    """Configura CIGALE con los filtros S-PLUS ahora registrados"""
    
    print("🎯 CONFIGURANDO CIGALE CON FILTROS S-PLUS REGISTRADOS")
    print("=" * 60)
    
    config_content = """# CIGALE configuration for NGC 5128 Globular Clusters
# Using registered S-PLUS narrow-band filters

data_file = gc_splus_cigale_custom.txt

parameters_file = 

sed_modules = sfhdelayed, bc03, redshifting

analysis_method = pdf_analysis

cores = 4

# S-PLUS narrow-band filters (now properly registered)
bands = F0378, F0378_err, F0395, F0395_err, F0410, F0410_err, F0430, F0430_err, F0515, F0515_err, F0660, F0660_err, F0861, F0861_err

properties = 

additionalerror = 0.05


[sed_modules_params]
  
  [[sfhdelayed]]
    # Delayed SFH optimized for old stellar populations
    tau_main = 100, 500, 1000, 2000
    age_main = 8000, 10000, 12000, 13000
    tau_burst = 50
    age_burst = 10
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True
  
  [[bc03]]
    # Bruzual & Charlot 2003 stellar populations
    imf = 1
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02
    separation_age = 10
  
  [[redshifting]]
    # Fixed redshift for NGC 5128 (Centaurus A)
    redshift = 0.0018


[analysis_params]
  # Physical properties to estimate
  variables = stellar.m_star, stellar.metallicity_mw, stellar.age_m_star
  # Bands for flux estimation
  bands = F0378, F0395, F0410, F0430, F0515, F0660, F0861
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 4
  blocks = 1
  best_weights = True
"""
    
    # Crear backup si existe
    if os.path.exists('pcigale.ini'):
        shutil.copy2('pcigale.ini', 'pcigale.ini.backup_pre_filtros')
    
    # Escribir configuración
    with open('pcigale.ini', 'w') as f:
        f.write(config_content)
    
    print("✅ pcigale.ini configurado con filtros S-PLUS")
    print("📋 Características de la configuración:")
    print("   - 7 filtros S-PLUS registrados: F0378, F0395, F0410, F0430, F0515, F0660, F0861")
    print("   - SFH: sfhdelayed (8-13 Gyr, optimizado para cúmulos globulares)")
    print("   - SSP: bc03 con rango completo de metalicidades para GCs")
    print("   - Redshift fijo: 0.0018 (NGC 5128)")

if __name__ == "__main__":
    configurar_cigale_con_filtros()
