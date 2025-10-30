import pandas as pd

# Verificar qué campos están en el archivo de offsets
df_offsets = pd.read_csv("plot_homogenization/final_offset_recommendations.csv")
campos_con_offsets = df_offsets['field'].unique()
print("Campos con offsets calculados:", sorted(campos_con_offsets))

# Verificar qué campos están en el catálogo GC
df_gc = pd.read_csv("Results/all_fields_gc_photometry_corrected_errors_v17.csv") 
campos_en_gc = df_gc['FIELD'].unique()
print("Campos en catálogo GC:", sorted(campos_en_gc))

# Encontrar campos faltantes
campos_faltantes = [campo for campo in campos_en_gc if campo not in campos_con_offsets]
print("Campos sin offsets:", campos_faltantes)
