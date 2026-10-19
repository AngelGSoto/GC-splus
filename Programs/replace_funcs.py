# Reemplaza plot_bootstrap_slopes y plot_mdf_radial_bins en tu script por las versiones nuevas
import re, sys
p, newf = sys.argv[1], sys.argv[2]
s = open(p).read(); new = open(newf).read()
def cut(src, name):
    i = src.index(f"def {name}(")
    j = src.index("\ndef ", i + 1)
    # incluye hasta el siguiente 'def' o bloque de comentarios
    k = src.rfind("\n# ====", i, j)
    end = k if k != -1 else j
    return src[:i], src[end:]
for name in ["plot_bootstrap_slopes", "plot_mdf_radial_bins"]:
    a, b = cut(s, name)
    block = new[new.index(f"def {name}("):]
    nxt = block.find("\ndef ", 1)
    block = block if nxt == -1 else block[:nxt]
    block = block.split("\nMDF_YTOP")[0].rstrip() + "\n\n"
    s = a + block + b
s = s.replace("def main():", "MDF_YTOP = 1.45   # limite superior del eje Y de Fig9 (multiplo del maximo)\n\n\ndef main():", 1) \
    if "MDF_YTOP =" not in s else s
open(p, "w").write(s)
print("OK: funciones reemplazadas")
