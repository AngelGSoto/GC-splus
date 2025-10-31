#!/bin/bash

echo "🚀 CIGALE SED FITTING - FILTROS SPLUS + DECam"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

echo "📋 Filtros configurados:"
echo "   - SPLUS: F378, F395, F410, F430, F515, F660, F861"
echo "   - DECam: u, g, r, i, z"
echo "   - Objetos: $(wc -l < input_cigale.csv)"
echo ""

# Verificar archivos
if [ ! -f "pcigale.ini" ]; then
    echo "❌ ERROR: No se encuentra pcigale.ini"
    exit 1
fi

if [ ! -f "input_cigale.csv" ]; then
    echo "❌ ERROR: No se encuentra input_cigale.csv" 
    exit 1
fi

echo "✅ Archivos verificados"
echo ""

# Verificar configuración
echo "🔍 Verificando configuración..."
pcigale check
echo ""

# Ejecutar CIGALE
echo "🔄 Iniciando análisis SED..."
echo "⏳ Esto puede tomar tiempo..."
echo ""

pcigale run

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!"
    echo "=========================================="
    echo "Finalizado: $(date)"
    echo ""
    echo "📁 Resultados en carpeta 'out/'"
else
    echo "❌ ERROR durante la ejecución"
    exit 1
fi
