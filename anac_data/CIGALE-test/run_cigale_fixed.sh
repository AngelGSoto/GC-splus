#!/bin/bash

echo "🚀 CIGALE - ANÁLISIS CORREGIDO"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Verificar archivos necesarios
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
    exit 1
fi

# Solo añadir filtros con formato correcto (los que empiezan con splus_ o decam_)
echo "🎛️  Añadiendo filtros corregidos..."
for filter_file in splus_*.dat decam_*.dat; do
    if [ -f "$filter_file" ]; then
        echo "   - Procesando $filter_file"
        pcigale-filters add "$filter_file"
        if [ $? -eq 0 ]; then
            echo "   ✅ $filter_file añadido"
        else
            echo "   ❌ Error con $filter_file"
        fi
    fi
done

echo ""
echo "📋 Configuración:"
echo "   - Objetos: $(wc -l < cigale_input.txt)"
echo "   - Filtros DECam: 5" 
echo "   - Filtros SPLUS: 7"
echo "   - Módulos: sfhdelayed, bc03, dustatt_powerlaw, redshifting"
echo ""

# Verificar configuración
echo "🔍 Verificando configuración..."
if pcigale check; then
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
    else
        echo "❌ Error en ejecución"
        exit 1
    fi
else
    echo "❌ Error en configuración"
    exit 1
fi
