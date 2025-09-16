from __future__ import print_function
import glob
import numpy as np

def combine_filters(input_pattern, output_file):
    filterss = []
    wll = []
    ress = []

    file_list = glob.glob(input_pattern)

    for file_name in sorted(file_list):
        # Cargar datos
        data = np.loadtxt(file_name)
        
        # Ordenar por longitud de onda (columna 0) de menor a mayor
        data = data[data[:, 0].argsort()]
        
        # Convertir unidades si es necesario (¡verificar!)
        # Si los datos originales están en nm y quieres Angstroms:
        # data[:, 0] = data[:, 0] * 10
        
        for row in data:
            filter_name = file_name.split('/')[-1].split('.txt')[0]
            wl = row[0] * 10 # Longitud de onda (sin multiplicar por 10)
            res = row[1]  # Transmisión
            
            filterss.append(filter_name)
            wll.append(wl)
            ress.append(res)

    # Escribir archivo en el formato requerido
    with open(output_file, 'w') as f:
        current_filter = None
        for name, wl, res in zip(filterss, wll, ress):
            f.write(f"{name}  {wl:.6f}  {res:.6f}\n")

# Configuración
input_pattern = "Carlos-escudero/*.txt"  # Ruta a tus archivos
output_file = "filters/carlos.filter"    # Archivo de salida

# Ejecutar
combine_filters(input_pattern, output_file)
