#!/usr/bin/env python3
"""
extincion_enero.py -- que correccion por extincion tienen los flujos de la corrida del
paper (enero 2026)?

Compara, objeto por objeto, los flujos que uso CIGALE en enero (out/observations.fits,
que son los flujos de la entrada del paper) con las magnitudes SIN corregir del catalogo
fotometrico. La diferencia es la extincion aplicada en enero:

    A_enero = m_sin_corregir - m_enero ,  con  m_enero = -2.5 log10(F[mJy]) + 16.4 (AB)

y se ajusta A_enero = k * E(B-V)_SFD (y tambien A = a + k*E(B-V), para detectar
diferencias de punto cero que no sean extincion). Se compara k con los coeficientes
del CSV actual (columnas A_x / E_BV_SFD) y con valores de referencia.

Uso (desde anac_data):
  python extincion_enero.py
"""
import numpy as np
import pandas as pd
from astropy.table import Table

OBS = "CIGALE-DEcam-extinction/out/observations.fits"
CAT = "Results_Corrected/all_fields_photometry_COMPLETE_extinction.csv"

# banda en observations.fits -> (magnitud sin corregir, columna A_x del CSV actual)
BANDS = {
    "F0378":   ("MAG_F378_3", "A_F0378"),
    "F0395":   ("MAG_F395_3", "A_F0395"),
    "F0410":   ("MAG_F410_3", "A_F0410"),
    "F0430":   ("MAG_F430_3", "A_F0430"),
    "F0515":   ("MAG_F515_3", "A_F0515"),
    "F0660":   ("MAG_F660_3", "A_F0660"),
    "F0861":   ("MAG_F861_3", "A_F0861"),
    "sdss.up": ("umag", "A_u"),
    "sdss.gp": ("gmag", "A_g"),
    "sdss.rp": ("rmag", "A_r"),
    "sdss.ip": ("imag", "A_i"),
    "sdss.zp": ("zmag", "A_z"),
}
# Referencias: Schlafly & Finkbeiner (2011), Tabla 6, R_V = 3.1 (A/E(B-V)_SFD)
REF_SF11 = {"sdss.up": (4.239, 3.995), "sdss.gp": (3.303, 3.237), "sdss.rp": (2.285, 2.176),
            "sdss.ip": (1.698, 1.595), "sdss.zp": (1.263, 1.217)}   # (SDSS, DECam)


def main():
    obs = Table.read(OBS).to_pandas()
    obs["id"] = obs["id"].apply(lambda x: x.decode() if isinstance(x, bytes) else str(x)).str.strip()
    cat = pd.read_csv(CAT, low_memory=False)
    cat["T17ID"] = cat["T17ID"].astype(str).str.strip()
    cat = cat.drop_duplicates("T17ID", keep="first").set_index("T17ID")
    ids = obs["id"][obs["id"].isin(cat.index)]
    print(f"Objetos en observations.fits: {len(obs)} | con contraparte en el catalogo: {len(ids)}")
    o = obs.set_index("id").loc[ids]
    c = cat.loc[ids]
    ebv = c["E_BV_SFD"].astype(float).values

    print(f"\n{'banda':8s} {'N':>5s} {'k_enero':>8s} {'(a + k E):':>10s} {'a':>7s} {'k':>7s} "
          f"{'k_CSV_actual':>12s} {'mediana(A_enero-A_CSV)':>22s} {'SF11 SDSS/DECam':>16s}")
    for band, (magcol, acol) in BANDS.items():
        if band not in o.columns:
            print(f"{band:8s}  (no esta en observations.fits)")
            continue
        F = o[band].astype(float).values
        mraw = c[magcol].astype(float).values
        ok = (F > 0) & np.isfinite(F) & (mraw < 50) & np.isfinite(mraw) & np.isfinite(ebv) & (ebv > 0)
        if ok.sum() < 10:
            print(f"{band:8s} {ok.sum():5d}  (pocos objetos validos)")
            continue
        m_jan = -2.5 * np.log10(F[ok]) + 16.4
        A = mraw[ok] - m_jan
        x = ebv[ok]
        k0 = np.sum(A * x) / np.sum(x * x)                    # ajuste por el origen
        k1, a1 = np.polyfit(x, A, 1)                          # con ordenada
        k_csv = np.median(c[acol].astype(float).values[ok] / x)
        dA = np.median(A - c[acol].astype(float).values[ok])
        ref = REF_SF11.get(band)
        ref_txt = f"{ref[0]:.3f}/{ref[1]:.3f}" if ref else "—"
        print(f"{band:8s} {ok.sum():5d} {k0:8.3f} {'':>10s} {a1:+7.3f} {k1:7.3f} "
              f"{k_csv:12.3f} {dA:+22.3f} {ref_txt:>16s}")
    print("\nk_enero: A/E(B-V) que tienen los flujos de enero (ajuste por el origen).")
    print("a, k: ajuste con ordenada; |a| >> 0.01 indica un desplazamiento de punto cero, no de extincion.")
    print("k_CSV_actual: A/E(B-V) de las columnas A_x del CSV actual.")


if __name__ == "__main__":
    main()
