#!/usr/bin/env python3
"""
cigale_setup_final.py -- version final de cigale_setup_optimizado.py

Configuracion del paper (Apendice E): misma plantilla que cigale_setup_optimizado.py,
con 14 edades (177,870 modelos) y additionalerror = 0.05. Es la que se obtiene con los
valores por defecto.

  * no es interactivo y no corre CIGALE (solo prepara la carpeta);
  * cada corrida va en su propia carpeta (no sobrescribe nada);
  * opciones para las variantes: --curves decam, --dust off, --sfh 2exp;
  * --test N usa solo los primeros N objetos (pruebas rapidas);
  * guarda setup_log.txt con la configuracion exacta usada.

Datos para reproducir la corrida del paper: gc_input_enero2026.txt
(recuperado de out/observations.fits de la corrida del 15-ene-2026).

Ejemplos:
  # reproducir la corrida del paper
  python cigale_setup_final.py --data gc_input_enero2026.txt --outdir CIGALE-paper
  # verificacion rapida con 30 objetos
  python cigale_setup_final.py --data gc_input_enero2026.txt --outdir t_check --test 30
  # variantes para el arbitro
  python cigale_setup_final.py --data gc_input_decam.txt     --outdir CIGALE-decamcurves --curves decam
  python cigale_setup_final.py --data gc_input_enero2026.txt --outdir CIGALE-nodust --dust off
  python cigale_setup_final.py --data gc_input_enero2026.txt --outdir CIGALE-burst --sfh 2exp --dust off

Luego:  cd <outdir> ; pcigale check ; pcigale run
"""
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

AGES = {
    # 16 edades: las del script cigale_setup_optimizado.py
    "script": "1000, 2000, 3000, 5000, 8000, 10000, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000",
    # 14 edades: las que aparecen en results.fits del paper (Apendice E: 177,870 modelos)
    "paper": "1000, 5000, 8000, 10000, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000",
}
TAU = "0.1, 1, 2, 3, 5, 50, 100"
METAL = "0.0004, 0.004, 0.008, 0.02, 0.03"
NB = ["F0378", "F0395", "F0410", "F0430", "F0515", "F0660", "F0861"]
BB = {"sdss": ["sdss.up", "sdss.gp", "sdss.rp", "sdss.ip", "sdss.zp"],
      "decam": ["DECam_u", "DECam_g", "DECam_r", "DECam_i", "DECam_z"]}


def sfh_block(a, ages):
    if a.sfh == "delayed":
        return f"""  [[sfhdelayed]]
    # e-folding time of the main stellar population model in Myr.
    tau_main = {TAU}
    # Age of the main stellar population in the galaxy in Myr.
    age_main = {ages}
    tau_burst = 1
    age_burst = 1
    f_burst = 0.0
    sfr_A = 1000000
    normalise = True
"""
    return f"""  [[sfh2exp]]
    # Main population (same tau and ages as the reference run) + a younger burst.
    tau_main = {TAU}
    tau_burst = {a.tau_burst}
    f_burst = {a.f_burst}
    age = {ages}
    burst_age = {a.burst_age}
    sfr_0 = 1.0
    normalise = True
"""


def dust_block(a, filters_amp):
    if a.dust == "on":
        ebv, fac, slope = ("0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7",
                           "0.25, 0.5, 0.75",
                           "-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0")
    else:
        ebv, fac, slope = "0.0", "0.25", "0.0"
    return f"""  [[dustatt_modified_starburst]]
    E_BV_lines = {ebv}
    E_BV_factor = {fac}
    uv_bump_wavelength = 217.5
    uv_bump_width = 35.0
    uv_bump_amplitude = 0.0
    powerlaw_slope = {slope}
    Ext_law_emission_lines = 1
    Rv = 3.1
    filters = {filters_amp}
"""


def n_models(ini_text):
    n = 1
    in_params = False
    for line in ini_text.splitlines():
        s = line.strip()
        if s.startswith("[sed_modules_params]"):
            in_params = True
        elif s.startswith("[analysis_params]"):
            in_params = False
        elif in_params and "=" in s and not s.startswith("#"):
            n *= len([v for v in s.split("=", 1)[1].split(",") if v.strip()]) or 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="archivo de entrada (con o sin '#')")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--ages", choices=AGES, default="paper")
    p.add_argument("--adderr", default="0.05", help="additionalerror")
    p.add_argument("--curves", choices=BB, default="sdss")
    p.add_argument("--dust", choices=("on", "off"), default="on")
    p.add_argument("--sfh", choices=("delayed", "2exp"), default="delayed")
    p.add_argument("--f-burst", default="0.0, 0.01, 0.05, 0.1")
    p.add_argument("--burst-age", default="50, 100, 300, 500")
    p.add_argument("--tau-burst", default="50")
    p.add_argument("--cores", type=int, default=12)
    p.add_argument("--test", type=int, default=0, help="usar solo los primeros N objetos")
    p.add_argument("--no-genconf", action="store_true", help="no crear pcigale.ini.spec")
    a = p.parse_args()

    ages = AGES[a.ages]
    if a.sfh == "2exp" and max(float(x) for x in a.burst_age.split(",")) >= \
            min(float(x) for x in ages.split(",")):
        raise SystemExit("ERROR: burst_age debe ser menor que la edad minima de la grilla")

    # --- datos: quitar '#' del encabezado (como limpiar_archivo) y opcional subset ----
    data_out = a.outdir / "gc_input_fixed.txt"
    with open(a.data) as fi:
        lines = [(l[1:] if l.startswith("#") else l).rstrip() for l in fi if l.strip()]
    if a.test:
        lines = lines[:a.test + 1]
    header = lines[0].split()

    bands_nb_bb = NB + BB[a.curves]
    missing = [b for b in bands_nb_bb if b not in header or f"{b}_err" not in header]
    if missing:
        raise SystemExit(f"ERROR: estas bandas no estan en {a.data}: {missing}\n"
                         "(usa --curves que coincida con los nombres del archivo de entrada)")

    a.outdir.mkdir(parents=True, exist_ok=False)     # recien ahora, si todo esta bien
    data_out.write_text("\n".join(lines) + "\n")
    bands = ", ".join(f"{b}, {b}_err" for b in bands_nb_bb)
    sfh_mod = "sfhdelayed" if a.sfh == "delayed" else "sfh2exp"

    ini = f"""# CIGALE v2022.1 -- GCs de NGC 5128 (S-PLUS + DECam)
# Generado por cigale_setup_final.py: {datetime.now():%Y-%m-%d %H:%M:%S}
# ages={a.ages} adderr={a.adderr} curves={a.curves} dust={a.dust} sfh={a.sfh} test={a.test or 'todos'}
# datos originales: {a.data}

data_file = {data_out.name}
parameters_file = 
redshift = from_file
sed_modules = {sfh_mod}, xsl, dustatt_modified_starburst, dale2014, redshifting
analysis_method = pdf_analysis
cores = {a.cores}
bands = {bands}
properties = 
additionalerror = {a.adderr}

[sed_modules_params]
{sfh_block(a, ages)}
  [[xsl]]
    imf = 1
    metallicity = {METAL}
    separation_age = 10

{dust_block(a, ' & '.join(bands_nb_bb))}
  [[dale2014]]
    fracAGN = 0.0
    alpha = 2.0

  [[redshifting]]
    redshift = 

[analysis_params]
  variables = stellar.m_star, stellar.metallicity, stellar.age_m_star
  bands = {', '.join(bands_nb_bb)}
  save_best_sed = True
  save_chi2 = none
  lim_flag = noscaling
  mock_flag = True
  redshift_decimals = -2
  blocks = 1
"""
    ini_path = a.outdir / "pcigale.ini"
    ini_path.write_text(ini)
    nmod = n_models(ini)

    # --- pcigale.ini.spec: se genera con pcigale init + genconf y se restaura nuestro .ini
    if not a.no_genconf and shutil.which("pcigale"):
        import re
        final = a.outdir / "pcigale.ini.final"
        shutil.move(ini_path, final)
        subprocess.run(["pcigale", "init"], cwd=a.outdir, capture_output=True, text=True)
        txt = ini_path.read_text() if ini_path.exists() else ""
        txt = re.sub(r"(?m)^data_file =.*$", f"data_file = {data_out.name}", txt)
        txt = re.sub(r"(?m)^sed_modules =.*$",
                     f"sed_modules = {sfh_mod}, xsl, dustatt_modified_starburst, dale2014, redshifting", txt)
        txt = re.sub(r"(?m)^analysis_method =.*$", "analysis_method = pdf_analysis", txt)
        ini_path.write_text(txt)
        subprocess.run(["pcigale", "genconf"], cwd=a.outdir, capture_output=True, text=True)
        shutil.move(final, ini_path)
        print("pcigale.ini.spec: creado" if (a.outdir / "pcigale.ini.spec").exists()
              else "AVISO: no se pudo crear pcigale.ini.spec")

    (a.outdir / "setup_log.txt").write_text(
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n" + "\n".join(f"{k} = {v}" for k, v in vars(a).items())
        + f"\nmodelos = {nmod}\nobjetos = {len(lines) - 1}\n")

    print(f"Carpeta: {a.outdir}/  | objetos: {len(lines) - 1} | modelos: {nmod:,}")
    print(f"  ages={a.ages} adderr={a.adderr} curves={a.curves} dust={a.dust} sfh={a.sfh}")
    print(f"Siguiente:\n  cd {a.outdir}\n  pcigale check   # debe dar {nmod:,} modelos\n  pcigale run")


if __name__ == "__main__":
    main()
