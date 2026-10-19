#!/usr/bin/env python3
"""
preparar_cigale_decam.py
Crea el archivo de entrada de CIGALE: 7 filtros S-PLUS + 5 filtros DECam (Taylor+2017),
con magnitudes corregidas por extincion. 'redshift' es la primera columna
(usar con redshift = from_file en pcigale.ini).

Cambios respecto de preparar_cigale_redshift_primero_decam_corr.py:
  * --curves sdss  -> nombres sdss.up ... (reproduce la entrada del paper)
    --curves decam -> nombres DECam_u ... (curvas DECam de SVO agregadas con pcigale-filters)
  * archivo de entrada y de salida por linea de comandos (por defecto, la muestra de 1394)
  * informa que columna de error se uso en cada banda y cuantos errores fueron
    inventados (NaN -> 0.1 mag, <= 0 -> 5 %), para que nada pase en silencio
  * --capped drop (opcional): marca como dato faltante las bandas angostas con el
    error recortado del pipeline (S/N = 2). Por defecto 'keep' = metodologia del paper.

Ejemplos:
  # reproducir la entrada del paper (debe ser identica a la original)
  python preparar_cigale_decam.py --curves sdss --output gc_input_sdss.txt
  # entrada con curvas DECam
  python preparar_cigale_decam.py --curves decam --output gc_input_decam.txt
"""
import argparse
import os

import numpy as np
import pandas as pd

SPLUS = {'F378': 'F0378', 'F395': 'F0395', 'F410': 'F0410', 'F430': 'F0430',
         'F515': 'F0515', 'F660': 'F0660', 'F861': 'F0861'}
CURVES = {
    "sdss":  {'u': 'sdss.up', 'g': 'sdss.gp', 'r': 'sdss.rp', 'i': 'sdss.ip', 'z': 'sdss.zp'},
    "decam": {'u': 'DECam_u', 'g': 'DECam_g', 'r': 'DECam_r', 'i': 'DECam_i', 'z': 'DECam_z'},
}
REDSHIFT = 0.001825
CAP = 1.0857362047581296 / 2.0          # magerr equivalente a S/N = 2


def abmag_to_mjy(mag_ab):
    if mag_ab >= 90 or np.isnan(mag_ab):
        return 0.0
    return 3631.0 * 10**(-mag_ab / 2.5) * 1000


def find_decam_cols(df, band):
    """Misma busqueda que el script original: primero corregidas, luego sin corregir."""
    mag_opts = [f'{band}mag_corr', f'{band}_corr', f'{band}mag_dered', f'{band}_dered',
                f'{band}mag_extcorr']
    err_opts = [f'e_{band}mag_corr', f'e_{band}_corr', f'err_{band}mag_corr',
                f'{band}magerr_corr']
    mag_col = next((c for c in mag_opts if c in df.columns and df[c].notna().any()), None)
    if mag_col is None:
        mag_col = next((c for c in (f'{band}mag', band)
                        if c in df.columns and df[c].notna().any()), None)
    err_col = next((c for c in err_opts if c in df.columns), None)
    if err_col is None:
        err_col = next((c for c in (f'e_{band}mag', f'err_{band}mag', f'{band}magerr',
                                    f'{band}err') if c in df.columns), None)
    return mag_col, err_col


def to_flux(mag, err, stats):
    """Conversion identica al script original, contando los casos inventados."""
    if pd.isna(mag):
        mag = 99.0
    if pd.isna(err):
        err = 0.1
        stats["err_nan"] += 1
    if mag >= 90.0:
        stats["no_data"] += 1
        return 0.0, 99.0
    flux = abmag_to_mjy(mag)
    if err > 0:
        return flux, flux * (np.log(10) / 2.5) * err
    stats["err_nonpos"] += 1
    return flux, flux * 0.05


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="../Results_Corrected/all_fields_photometry_OPTIMIZED_1394objects.csv")
    p.add_argument("--output", default="gc_splus_cigale_custom.txt")
    p.add_argument("--curves", choices=("sdss", "decam"), default="sdss")
    p.add_argument("--capped", choices=("keep", "drop"), default="keep")
    a = p.parse_args()

    if not os.path.exists(a.input):
        raise SystemExit(f"No se encuentra {a.input}")
    df = pd.read_csv(a.input)
    print(f"Entrada: {a.input} ({len(df)} objetos) | curvas: {a.curves} | recortados: {a.capped}")

    # --- columnas ----------------------------------------------------------------
    cols = {}
    print("\nS-PLUS:")
    for f, name in SPLUS.items():
        m, e = f'MAG_{f}_3_corr', f'MAGERR_{f}_3_corr'
        if not (m in df.columns and df[m].notna().any()):
            m, e = f'MAG_{f}_3', f'MAGERR_{f}_3'
            print(f"  AVISO {name}: usando {m} (SIN correccion de extincion)")
        cols[name] = (m if m in df.columns else None, e if e in df.columns else None)
        print(f"  {name}: mag={cols[name][0]}  err={cols[name][1]}")
    print("DECam:")
    for b, name in CURVES[a.curves].items():
        m, e = find_decam_cols(df, b)
        cols[name] = (m, e)
        flag = "" if (m and ("corr" in m or "dered" in m)) else "  <- SIN correccion de extincion"
        print(f"  {name}: mag={m}  err={e}{flag}")
        if e is None:
            print(f"  AVISO {name}: no se encontro columna de error -> se usara 0.1 mag para TODOS")

    # --- filas --------------------------------------------------------------------
    stats = {n: {"err_nan": 0, "err_nonpos": 0, "no_data": 0, "capped_dropped": 0} for n in cols}
    rows = []
    for idx, row in df.iterrows():
        obj_id = str(row.get('T17ID', row.get('recno', f"obj_{idx + 1}")))
        vals = [REDSHIFT, obj_id]
        for name, (m, e) in cols.items():
            if m is None:
                vals += [0.0, 99.0]
                stats[name]["no_data"] += 1
                continue
            mag = row.get(m, 99.0)
            err = row.get(e, 0.1) if e else 0.1
            if (a.capped == "drop" and name in SPLUS.values() and pd.notna(err)
                    and abs(err - CAP) < 2e-3):
                vals += [0.0, 99.0]                      # misma convencion que "sin dato"
                stats[name]["capped_dropped"] += 1
                continue
            vals += list(to_flux(mag, err, stats[name]))
        rows.append(vals)

    header = ['redshift', 'id']
    for name in cols:
        header += [name, f'{name}_err']
    with open(a.output, 'w') as fh:
        fh.write('# ' + ' '.join(header) + '\n')
        for r in rows:
            fh.write(' '.join([f"{r[0]:.6f}", r[1]] + [f"{v:.6e}" for v in r[2:]]) + "\n")

    print(f"\nArchivo creado: {a.output} ({len(rows)} objetos, {len(header)} columnas)")
    print("\nCasos especiales por banda (de", len(rows), "objetos):")
    print("  banda      sin_dato  err_NaN->0.1  err<=0->5%  recortados_quitados")
    for name, s in stats.items():
        print(f"  {name:9s} {s['no_data']:8d} {s['err_nan']:13d} {s['err_nonpos']:11d} "
              f"{s['capped_dropped']:20d}")
    print("\nEn pcigale.ini:  data_file =", a.output, " |  redshift = from_file")
    print("  bands =", ", ".join(f"{n}, {n}_err" for n in cols))


if __name__ == "__main__":
    main()
