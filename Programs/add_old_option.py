# Agrega la opcion --old a spacial_distribution.py sin tocar los cambios de visualizacion
import sys
p = sys.argv[1]; s = open(p).read()
rep = [
 ('''    ap.add_argument("--outdir", default="distribucion_espacial")''',
  '''    ap.add_argument("--outdir", default="distribucion_espacial")
    ap.add_argument("--old", action="store_true",
                    help="restringir a GCs viejos (>8 Gyr): (a) clasificacion de la muestra completa "
                         "y (b) GMM re-ajustado sobre los viejos (como la Tabla 3 del paper)")'''),
 ('''    df["FeH"] = np.log10(df["bayes.stellar.metallicity"] / 0.02)
    q1, q3''',
  '''    df["FeH"] = np.log10(df["bayes.stellar.metallicity"] / 0.02)
    df["Age_Gyr"] = df["bayes.stellar.age_m_star"] / 1000.0
    q1, q3'''),
 ('''    print("Clasificacion del paper (muestra completa):")
    labels = get_labels_forced_n3(df["FeH"].values)[0]
    ok = df["RAJ2000"].notna().values''',
  '''    print("Clasificacion del paper (muestra completa):")
    labels = get_labels_forced_n3(df["FeH"].values)[0]
    ok = df["RAJ2000"].notna().values
    if a.old:
        old = (df["Age_Gyr"] > 8.0).values
        # (a) clasificacion de la muestra completa, restringida a los viejos
        Xa, Ya = tangent_xy_kpc(df.loc[ok & old, "RAJ2000"].values, df.loc[ok & old, "DEJ2000"].values)
        Ra, la_ = np.hypot(Xa, Ya), labels[ok & old]
        print(f"\\n(a) GCs viejos con la clasificacion de la muestra completa (N={len(Ra)}):")
        for i, p in enumerate(POPS):
            r = Ra[la_ == i]
            print(f"   {p:4s} N={len(r):4d}  R_mediano = {np.median(r):6.1f} kpc")
        for i in range(3):
            for j in range(i + 1, 3):
                r1, r2 = Ra[la_ == i], Ra[la_ == j]
                print(f"   {POPS[i]}-{POPS[j]}: KS p = {ks_2samp(r1, r2).pvalue:.3g}   "
                      f"AD p ~ {anderson_ksamp([r1, r2]).significance_level:.3g}")
        # (b) GMM re-ajustado sobre los viejos (el resto del analisis y la figura usan esta)
        print("\\n(b) GMM re-ajustado sobre los GCs viejos:")
        df = df[old].reset_index(drop=True)
        labels = get_labels_forced_n3(df["FeH"].values)[0]
        ok = df["RAJ2000"].notna().values
        a.outdir = a.outdir.rstrip("/") + "_old"
        os.makedirs(a.outdir, exist_ok=True)'''),
]
n = 0
for x, y in rep:
    if x in s:
        s = s.replace(x, y); n += 1
    else:
        print("NO ENCONTRADO:\n", x[:80])
open(p, "w").write(s)
print(f"Cambios aplicados: {n}/{len(rep)}")
