import os

# Verificar el contenido del directorio de filtros
filters_path = '../filters'
print(f"Contenido de {filters_path}:")
if os.path.exists(filters_path):
    for item in os.listdir(filters_path):
        item_path = os.path.join(filters_path, item)
        if os.path.isfile(item_path):
            print(f"  Archivo: {item}")
        elif os.path.isdir(item_path):
            print(f"  Directorio: {item}")
else:
    print(f"El directorio {filters_path} no existe.")

# Verificar también el directorio actual
print(f"\nContenido del directorio actual ({os.getcwd()}):")
for item in os.listdir('.'):
    print(f"  {item}")
