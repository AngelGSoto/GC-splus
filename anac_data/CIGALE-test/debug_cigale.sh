#!/bin/bash
echo "🐛 SCRIPT DE DEPURACIÓN CIGALE"
echo "================================"

echo "1. Verificando archivos..."
ls -la pcigale.ini input_manual.csv 2>/dev/null || echo "Archivos no encontrados"

echo ""
echo "2. Verificando contenido de pcigale.ini..."
if [ -f "pcigale.ini" ]; then
    grep -E "(sed_modules|data_file)" pcigale.ini || echo "No se encontraron las claves necesarias"
else
    echo "pcigale.ini no existe"
fi

echo ""
echo "3. Probando pcigale genconf..."
pcigale genconf

echo ""
echo "4. Si genconf funciona, probando pcigale run..."
if [ $? -eq 0 ] && [ -f "pcigale.ini.spec" ]; then
    echo "✅ genconf exitoso, ejecutando run..."
    pcigale run
else
    echo "❌ genconf falló"
fi
