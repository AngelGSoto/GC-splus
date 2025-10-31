#!/bin/bash

echo "🔧 USANDO MÓDULO ALTERNATIVO dustatt_calzleit"
echo "=========================================="

# Limpiar
rm -f pcigale.ini pcigale.ini.spec
rm -rf out

# Inicializar
pcigale init

# Configuración con dustatt_calzleit
cat > pcigale.ini << 'EOF'
data_file = cigale_input.txt
parameters_file = 
sed_modules = sfhdelayed, bc03, dustatt_calzleit, redshifting
analysis_method = pdf_analysis
cores = 4
additionalerror = 0.05
EOF

# Generar configuración
pcigale genconf

# Configuración completa
cat > pcigale.ini << 'EOF'
# CIGALE configuration for globular clusters
# Using dustatt_calzleit module

data_file = cigale_input.txt
parameters_file = 
sed_modules = sfhdelayed, bc03, dustatt_calzleit, redshifting
analysis_method = pdf_analysis
cores = 4

bands = decam.u, decam.u_err, decam.g, decam.g_err, decam.r, decam.r_err, decam.i, decam.i_err, decam.z, decam.z_err, splus.F378, splus.F378_err, splus.F395, splus.F395_err, splus.F410, splus.F410_err, splus.F430, splus.F430_err, splus.F515, splus.F515_err, splus.F660, splus.F660_err, splus.F861, splus.F861_err

properties = sfh.age, stellar.m_star, stellar.metallicity, attenuation.E_BVs

additionalerror = 0.05

[sed_modules_params]

  [[sfhdelayed]]
    tau_main = 100, 500, 1000, 2000, 5000
    age_main = 5000, 8000, 10000, 12000, 13000
    tau_burst = 50.0
    age_burst = 20
    f_burst = 0.0
    sfr_A = 1.0
    normalise = True

  [[bc03]]
    imf = 1
    metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02
    separation_age = 10

  [[dustatt_calzleit]]
    E_BVs_young = 0.01, 0.05, 0.1, 0.15, 0.2
    uv_bump_amplitude = 0.0, 1.5, 3.0
    powerlaw_slope = -0.5, -0.7, -1.0
    E_BVs_old_factor = 0.1, 0.25, 0.5
    filters = decam.u, decam.g, decam.r, decam.i, decam.z, splus.F378, splus.F395, splus.F410, splus.F430, splus.F515, splus.F660, splus.F861

  [[redshifting]]
    redshift = 0.0

[analysis_params]
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = False
  redshift_decimals = 2
  blocks = 1
EOF

pcigale check && pcigale run

if [ $? -eq 0 ]; then
    echo "🎉 ¡ÉXITO con dustatt_calzleit!"
    cd out
    pcigale-plots sed
    pcigale-plots pdf
    cd ..
fi
