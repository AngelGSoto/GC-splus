#!/bin/bash

save_dir="anac_data"
log_file="verificacion_integridad.log"
corrupt_log="archivos_corruptos.list"

echo "=== VERIFICACIÓN DE INTEGRIDAD FITS ===" > "$log_file"
> "$corrupt_log"  # Limpiar archivo de lista

# Función mejorada para manejar nombres de archivo correctamente
verificar_fits() {
    local archivo="$1"
    if ! fitsverify -q -e "$archivo" &> /dev/null; then
        echo "❌ CORRUPTO: $archivo" | tee -a "$log_file"
        echo "$archivo" >> "$corrupt_log"
        return 1
    else
        echo "✅ INTEGRO: $archivo" | tee -a "$log_file"
        return 0
    fi
}

export -f verificar_fits

# Verificación robusta con manejo de espacios en nombres
find "$save_dir" -name "*.fits.fz" -type f -print0 | while IFS= read -r -d $'\0' archivo; do
    verificar_fits "$archivo"
done

# Eliminar archivos corruptos si los hubiera
if [[ -s "$corrupt_log" ]]; then
    echo "=== ELIMINANDO ARCHIVOS CORRUPTOS ==="
    while IFS= read -r archivo; do
        echo "Eliminando: $archivo"
        rm -f "$archivo"
    done < "$corrupt_log"
else
    echo "No se encontraron archivos corruptos."
fi

echo "Verificación completada."
