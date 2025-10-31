#!/bin/bash

echo "🔧 CORRIGIENDO PROBLEMAS DE CIGALE"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Paso 1: Verificar que tenemos los archivos necesarios
echo "📁 Verificando archivos..."
if [ ! -f "cigale_input.txt" ]; then
    echo "❌ Error: No se encuentra cigale_input.txt"
    exit 1
fi

if [ ! -f "pcigale_proper.ini" ]; then
    echo "❌ Error: No se encuentra pcigale_proper.ini"
    echo "💡 Ejecuta primero el script de diagnóstico"
    exit 1
fi

# Paso 2: Limpiar configuración anterior
echo "🧹 Limpiando configuración anterior..."
rm -f pcigale.ini pcigale.ini.spec

# Paso 3: Usar configuración adecuada
echo "📝 Usando configuración adecuada..."
cp pcigale_proper.ini pcigale.ini

# Paso 4: Generar configuración completa
echo "🔧 Generando configuración completa..."
pcigale genconf

if [ $? -ne 0 ]; then
    echo "❌ Error al generar configuración"
    exit 1
fi

# Paso 5: Verificar la configuración
echo "🔍 Verificando configuración..."
pcigale check

if [ $? -eq 0 ]; then
    echo "✅ Configuración válida"
    
    # Paso 6: Crear directorio para nuevos resultados
    if [ -d "out_corrected" ]; then
        echo "📁 Eliminando resultados anteriores corregidos..."
        rm -rf out_corrected
    fi
    
    # Paso 7: Ejecutar CIGALE con configuración corregida
    echo "🔄 Ejecutando CIGALE con configuración corregida..."
    pcigale run
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 ¡ANÁLISIS CORREGIDO COMPLETADO!"
        echo "📁 Resultados en: out/"
        
        # Mover resultados a directorio nuevo
        mv out out_corrected
        echo "📁 Resultados movidos a: out_corrected/"
        
        # Generar gráficos
        echo "📊 Generando gráficos..."
        cd out_corrected
        pcigale-plots sed
        pcigale-plots pdf
        cd ..
        
        echo ""
        echo "⏰ Finalizado: $(date)"
    else
        echo "❌ Error en la ejecución"
        exit 1
    fi
else
    echo "❌ Error en la configuración"
    exit 1
fi
