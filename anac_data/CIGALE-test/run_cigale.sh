#!/bin/bash

echo "🚀 CIGALE - ANÁLISIS DE CÚMULOS GLOBULARES"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Verificar que existen los archivos necesarios
if [ ! -f "cigale_input.txt" ]; then
    echo "❌ Error: No se encuentra cigale_input.txt"
    exit 1
fi

if [ ! -f "pcigale.ini" ]; then
    echo "❌ Error: No se encuentra pcigale.ini"
    exit 1
fi

if [ ! -f "pcigale.ini.spec" ]; then
    echo "❌ Error: No se encuentra pcigale.ini.spec"
    echo "💡 Ejecuta: pcigale genconf"
    exit 1
fi

# Añadir filtros personalizados si existen
if ls *.dat 1> /dev/null 2>&1; then
    echo "🎛️  Añadiendo filtros personalizados..."
    for filter_file in *.dat; do
        echo "   - Procesando $filter_file"
        pcigale-filters add "$filter_file"
        if [ $? -eq 0 ]; then
            echo "   ✅ $filter_file añadido correctamente"
        else
            echo "   ❌ Error con $filter_file"
            # Continuar con otros filtros
        fi
    done
else
    echo "ℹ️  No se encontraron archivos de filtros personalizados"
fi

echo ""
echo "📋 Configuración:"
echo "   - Objetos: $(wc -l < cigale_input.txt)"
echo "   - Filtros DECam: 5"
echo "   - Filtros SPLUS: 7" 
echo "   - Módulos: sfhdelayed, bc03, dustatt_powerlaw, redshifting"
echo ""

# Verificar configuración
echo "🔍 Verificando configuración..."
pcigale check

if [ $? -eq 0 ]; then
    echo "✅ Configuración válida"
    
    echo ""
    # Ejecutar CIGALE
    echo "🔄 Ejecutando análisis..."
    pcigale run

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 ¡ANÁLISIS COMPLETADO!"
        echo "📁 Resultados en: out/"
        echo "⏰ Finalizado: $(date)"
        
        # Crear gráficos si es posible
        echo ""
        echo "📊 Generando gráficos..."
        if command -v pcigale-plots &> /dev/null; then
            pcigale-plots sed
            echo "✅ Gráficos SED generados"
        else
            echo "ℹ️  pcigale-plots no disponible, omitiendo gráficos"
        fi
    else
        echo "❌ Error en ejecución"
        exit 1
    fi
else
    echo "❌ Error en configuración"
    echo "💡 Revisa el archivo pcigale.ini"
    exit 1
fi
