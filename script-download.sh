#!/bin/bash

# Script para descargar datos astronómicos con capacidad de reanudación y barra de progreso
# Guardar como: descarga_anac.sh

base_url="https://adss.cbpf.br/dados/anac"
save_dir="anac_data"
log_file="descarga_anac.log"
state_file="descarga_anac.state"

echo "=== INICIANDO DESCARGA ===" | tee -a "$log_file"
mkdir -p "$save_dir"

# Cargar estado previo si existe
declare -A completed_folders
if [[ -f "$state_file" ]]; then
    source "$state_file"
fi

for num in {01..24}; do
    carpeta="CenA$num"
    
    # Saltar carpeta si ya fue completada
    if [[ "${completed_folders[$carpeta]}" == "done" ]]; then
        echo "↩️ Saltando carpeta previamente completada: $carpeta" | tee -a "$log_file"
        continue
    fi

    echo "📁 Descargando $carpeta..." | tee -a "$log_file"
    mkdir -p "$save_dir/$carpeta"
    incomplete=0

    # Descargar archivos FITS con reanudación y barra de progreso
    for filtro in F378 F395 F410 F430 F515 F660 F861; do
        # Archivo principal
        archivo_fits="CenA${num}_${filtro}.fits.fz"
        url_fits="$base_url/$carpeta/$archivo_fits"
        destino_fits="$save_dir/$carpeta/$archivo_fits"
        
        if [[ ! -f "$destino_fits" ]]; then
            echo "   ⬇️ Descargando: $archivo_fits" | tee -a "$log_file"
            # Usamos una tubería especial para mantener la barra de progreso
            (wget -c --show-progress --no-check-certificate --tries=0 \
                  --waitretry=5 --random-wait --limit-rate=400k \
                  -O "$destino_fits" "$url_fits" 2>&1 | awk '/\.fz/ {print "      " $0}') &
            pid=$!
            wait $pid
            exit_code=$?
            
            # Verificar si la descarga fue incompleta
            if [[ $exit_code -ne 0 ]]; then
                echo "   ❌ Error detectado en: $archivo_fits (código: $exit_code)" | tee -a "$log_file"
                incomplete=1
            fi
        else
            echo "   ✔️ Archivo existente: $archivo_fits" | tee -a "$log_file"
        fi

        # Archivo de peso (weight)
        archivo_weight="CenA${num}_${filtro}.weight.fits.fz"
        url_weight="$base_url/$carpeta/$archivo_weight"
        destino_weight="$save_dir/$carpeta/$archivo_weight"
        
        if [[ ! -f "$destino_weight" ]]; then
            echo "   ⬇️ Descargando: $archivo_weight" | tee -a "$log_file"
            (wget -c --show-progress --no-check-certificate --tries=0 \
                  --waitretry=5 --random-wait --limit-rate=400k \
                  -O "$destino_weight" "$url_weight" 2>&1 | awk '/\.fz/ {print "      " $0}') &
            pid=$!
            wait $pid
            exit_code=$?
            
            # Verificar si la descarga fue incompleta
            if [[ $exit_code -ne 0 ]]; then
                echo "   ❌ Error detectado en: $archivo_weight (código: $exit_code)" | tee -a "$log_file"
                incomplete=1
            fi
        else
            echo "   ✔️ Archivo existente: $archivo_weight" | tee -a "$log_file"
        fi
    done

    # Descargar carpeta general con reanudación (solo si no hubo errores previos)
    if [[ $incomplete -eq 0 ]]; then
        echo "   📂 Descargando carpeta general..." | tee -a "$log_file"
        wget -c -r -np -nH --cut-dirs=3 --no-check-certificate \
             --show-progress --progress=bar:force:noscroll \
             -P "$save_dir/$carpeta" "$base_url/$carpeta/general/" 2>&1 | tee -a "$log_file"
        
        # Capturar estado de salida de wget
        exit_code=${PIPESTATUS[0]}
        
        # Marcar carpeta como completada si todo fue exitoso
        if [[ $exit_code -eq 0 ]]; then
            echo "   ✅ $carpeta completada" | tee -a "$log_file"
            completed_folders["$carpeta"]="done"
            # Actualizar archivo de estado
            declare -p completed_folders > "$state_file"
        else
            echo "   ❌ Error en carpeta general: $carpeta (código: $exit_code)" | tee -a "$log_file"
        fi
    else
        echo "   ⚠️ Carpeta incompleta: $carpeta (no se descargó 'general')" | tee -a "$log_file"
    fi
done

echo "✅ Descarga completada! Directorio: $save_dir" | tee -a "$log_file"
echo "Total archivos: $(find "$save_dir" -type f | wc -l)" | tee -a "$log_file"
rm -f "$state_file"  # Limpiar estado al finalizar
