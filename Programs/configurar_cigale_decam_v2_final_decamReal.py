#!/usr/bin/env python3
"""
configurar_cigale_decam_v2_final_decamReal.py
Version final de cigale_setup_optimizado.py

Con los valores por defecto reproduce EXACTAMENTE la corrida del paper
(15-ene-2026), verificado con 30 objetos (diferencias ~0 en chi2, [Fe/H] y edad):
  * modulos: sfhdelayed, xsl, dustatt_modified_starburst, dale2014, redshifting
  * 14 edades, 7 tau, 5 metalicidades, 11x3x11 polvo -> 177,870 modelos
  * additionalerror = 0.05, imf = 1, mock_flag = True

Datos de entrada del paper: gc_input_enero2026_raw.txt
  (recuperado de out/observations.fits, restando el 5 % que CIGALE ya habia
   sumado a los errores; NO usar gc_input_enero2026.txt, que lo suma dos veces)

Que hace el script:
  * no es interactivo y no corre CIGALE (solo prepara la carpeta);
  * cada corrida va en su propia carpeta (nunca sobrescribe);
  * crea pcigale.ini y pcigale.ini.spec (este ultimo es obligatorio en CIGALE);
  * opciones para las variantes: --curves decam, --dust off, --sfh 2exp;
  * --test N usa solo los primeros N objetos (pruebas rapidas);
  * guarda setup_log.txt con la configuracion exacta usada.

Ejemplos:
  # verificacion rapida con 30 objetos
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw.txt --outdir t_check --test 30
  # variantes para el arbitro
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw_decam.txt --outdir CIGALE-decamcurves --curves decam
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw.txt --outdir CIGALE-nodust --dust off
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw.txt --outdir CIGALE-burst --sfh 2exp --dust off
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw.txt --outdir CIGALE-bc03 --ssp bc03
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw.txt --outdir CIGALE-bursty --sfh periodic
  python configurar_cigale_decam_v2_final_decamReal.py --data gc_input_enero2026_raw.txt --outdir CIGALE-broadonly --only-broad

Luego:  cd <outdir> ; pcigale check ; pcigale run
"""
import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

AGES = {
    # 14 edades: las de la corrida del paper (verificado; Apendice E: 177,870 modelos)
    "paper": "1000, 5000, 8000, 10000, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000",
    # 16 edades: version posterior de cigale_setup_optimizado.py (NO es la del paper)
    "script": "1000, 2000, 3000, 5000, 8000, 10000, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000",
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
    if a.sfh == "periodic":
        return f"""  [[sfhperiodic]]
    # Bursty SFH: periodic bursts over the same total ages as the reference run.
    type_bursts = 0
    delta_bursts = {a.delta_bursts}
    tau_bursts = {a.tau_bursts}
    age = {ages}
    sfr_A = 1.0
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


def ssp_block(a):
    if a.ssp == "xsl":
        return f"""  [[xsl]]
    imf = 1
    metallicity = {METAL}
    separation_age = 10
"""
    # BC03: imf 1 = Chabrier; 5 metallicities spanning the XSL range (BC03 has no Z = 0.03)
    return """  [[bc03]]
    imf = 1
    metallicity = 0.0004, 0.004, 0.008, 0.02, 0.05
    separation_age = 10
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


def make_spec(outdir, ini_path, data_name, sfh_mod, ssp="xsl"):
    """CIGALE exige pcigale.ini.spec. Se genera con 'pcigale init' + 'pcigale genconf'
    (como indica CIGALE) y despues se restaura nuestro pcigale.ini intacto."""
    final = outdir / "pcigale.ini.final"
    shutil.move(ini_path, final)
    try:
        subprocess.run(["pcigale", "init"], cwd=outdir, capture_output=True, text=True)
        txt = ini_path.read_text() if ini_path.exists() else ""
        txt = re.sub(r"(?m)^data_file =.*$", f"data_file = {data_name}", txt)
        txt = re.sub(r"(?m)^sed_modules =.*$",
                     f"sed_modules = {sfh_mod}, {ssp}, dustatt_modified_starburst, dale2014, redshifting", txt)
        txt = re.sub(r"(?m)^analysis_method =.*$", "analysis_method = pdf_analysis", txt)
        ini_path.write_text(txt)
        # genconf puede terminar con un error al imprimir su tabla, pero crea el .spec
        subprocess.run(["pcigale", "genconf"], cwd=outdir, capture_output=True, text=True)
    finally:
        shutil.move(final, ini_path)                      # nuestro .ini, siempre restaurado
    return (outdir / "pcigale.ini.spec").exists()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="archivo de entrada (con o sin '#')")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--ages", choices=AGES, default="paper")
    p.add_argument("--adderr", default="0.05", help="additionalerror (paper: 0.05)")
    p.add_argument("--curves", choices=BB, default="sdss")
    p.add_argument("--dust", choices=("on", "off"), default="on")
    p.add_argument("--sfh", choices=("delayed", "2exp", "periodic"), default="delayed")
    p.add_argument("--ssp", choices=("xsl", "bc03"), default="xsl")
    p.add_argument("--delta-bursts", default="50, 200, 500, 1000", help="Myr (sfhperiodic)")
    p.add_argument("--tau-bursts", default="10, 50", help="Myr (sfhperiodic)")
    p.add_argument("--f-burst", default="0.0, 0.01, 0.05, 0.1")
    p.add_argument("--burst-age", default="50, 100, 300, 500")
    p.add_argument("--tau-burst", default="50")
    p.add_argument("--only-broad", action="store_true",
                   help="ajustar solo las 5 bandas anchas (sin bandas angostas)")
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--test", type=int, default=0, help="usar solo los primeros N objetos")
    p.add_argument("--no-genconf", action="store_true", help="no crear pcigale.ini.spec")
    a = p.parse_args()

    ages = AGES[a.ages]
    if a.sfh == "2exp" and max(float(x) for x in a.burst_age.split(",")) >= \
            min(float(x) for x in ages.split(",")):
        raise SystemExit("ERROR: burst_age debe ser menor que la edad minima de la grilla")

    # --- datos: quitar '#' del encabezado (como limpiar_archivo) y opcional subset ----
    if not a.data.exists():
        raise SystemExit(f"ERROR: no existe {a.data}")
    with open(a.data) as fi:
        lines = [(l[1:] if l.startswith("#") else l).rstrip() for l in fi if l.strip()]
    if a.test:
        lines = lines[:a.test + 1]
    header = lines[0].split()

    bands_nb_bb = (BB[a.curves] if a.only_broad else NB + BB[a.curves])
    missing = [b for b in bands_nb_bb if b not in header or f"{b}_err" not in header]
    if missing:
        raise SystemExit(f"ERROR: estas bandas no estan en {a.data}: {missing}\n"
                         "(usa --curves que coincida con los nombres del archivo de entrada)")

    a.outdir.mkdir(parents=True, exist_ok=False)     # solo si todo lo anterior esta bien
    data_out = a.outdir / "gc_input_fixed.txt"
    data_out.write_text("\n".join(lines) + "\n")
    bands = ", ".join(f"{b}, {b}_err" for b in bands_nb_bb)
    sfh_mod = {"delayed": "sfhdelayed", "2exp": "sfh2exp", "periodic": "sfhperiodic"}[a.sfh]

    ini = f"""# CIGALE v2022.1 -- GCs de NGC 5128 (S-PLUS + DECam)
# Generado por configurar_cigale_decam_v2_final_decamReal.py: {datetime.now():%Y-%m-%d %H:%M:%S}
# ages={a.ages} adderr={a.adderr} curves={a.curves} dust={a.dust} sfh={a.sfh} ssp={a.ssp} only_broad={a.only_broad} test={a.test or 'todos'}
# datos originales: {a.data}

data_file = {data_out.name}
parameters_file = 
redshift = from_file
sed_modules = {sfh_mod}, {a.ssp}, dustatt_modified_starburst, dale2014, redshifting
analysis_method = pdf_analysis
cores = {a.cores}
bands = {bands}
properties = 
additionalerror = {a.adderr}

[sed_modules_params]
{sfh_block(a, ages)}
{ssp_block(a)}
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

    if not a.no_genconf:
        if shutil.which("pcigale"):
            ok = make_spec(a.outdir, ini_path, data_out.name, sfh_mod, a.ssp)
            print("pcigale.ini.spec: creado" if ok else "AVISO: no se pudo crear pcigale.ini.spec")
        else:
            print("AVISO: 'pcigale' no esta disponible (activa el entorno: conda activate cigale)")

    (a.outdir / "setup_log.txt").write_text(
        f"{datetime.now():%Y-%m-%d %H:%M:%S}\n" + "\n".join(f"{k} = {v}" for k, v in vars(a).items())
        + f"\nmodelos = {nmod}\nobjetos = {len(lines) - 1}\n")

    print(f"Carpeta: {a.outdir}/  | objetos: {len(lines) - 1} | modelos: {nmod:,} | cores: {a.cores}")
    print(f"  ages={a.ages} adderr={a.adderr} curves={a.curves} dust={a.dust} sfh={a.sfh} ssp={a.ssp} "
          f"bandas={'solo anchas (5)' if a.only_broad else 'todas (12)'}")
    print(f"Siguiente:\n  cd {a.outdir}\n  pcigale check   # debe dar {nmod:,} modelos\n  pcigale run")


if __name__ == "__main__":
    main()
