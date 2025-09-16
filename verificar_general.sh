#!/bin/bash

save_dir="anac_data"
log_file="verificacion_general.log"

echo "=== VERIFICACIÓN DE CARPETAS GENERAL ===" > "$log_file"

for num in {01..24}; do
    carpeta="CenA$num"
    general_dir="$save_dir/$carpeta/general"
    
    if [[ ! -d "$general_dir" ]]; then
        echo "❌ FALTANTE: $carpeta/general" | tee -a "$log_file"
    else
        file_count=$(find "$general_dir" -type f | wc -l)
        echo "✅ PRESENTE: $carpeta/general ($file_count archivos)" | tee -a "$log_file"
    fi
done

echo "Verificación completada. Resultados en: $log_file"
