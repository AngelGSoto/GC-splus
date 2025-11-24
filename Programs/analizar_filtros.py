import numpy as np
import matplotlib.pyplot as plt
import os

def analizar_filtros_completo():
    """Análisis completo de los filtros S-PLUS para CIGALE"""
    
    print("🔬 ANÁLISIS COMPLETO FILTROS S-PLUS PARA CIGALE")
    print("=" * 60)
    
    filtros = {
        'F0378.dat': 'F0378 (u)',
        'F0395.dat': 'F0395 (u)', 
        'F0410.dat': 'F0410 (u)',
        'F0430.dat': 'F0430 (g)',
        'F0515.dat': 'F0515 (g)',
        'F0660.dat': 'F0660 (r)',
        'F0861.dat': 'F0861 (i)'
    }
    
    # Configuración de plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    resultados = {}
    
    for archivo, nombre in filtros.items():
        if os.path.exists(archivo):
            datos = np.loadtxt(archivo)
            wave = datos[:, 0]  # Å
            trans = datos[:, 1]
            
            # Cálculos detallados
            lambda_eff = np.sum(wave * trans) / np.sum(trans)
            max_trans = np.max(trans)
            fwhm = calcular_fwhm(wave, trans)
            
            # Ancho de banda equivalente
            eq_width = np.trapz(trans, wave) / max_trans
            
            # Guardar resultados
            resultados[nombre] = {
                'lambda_eff': lambda_eff,
                'fwhm': fwhm,
                'max_trans': max_trans,
                'eq_width': eq_width,
                'wave_range': (wave[0], wave[-1])
            }
            
            print(f"📊 {nombre}:")
            print(f"   λ_eff: {lambda_eff:.1f} Å")
            print(f"   FWHM: {fwhm:.1f} Å")
            print(f"   Ancho equivalente: {eq_width:.1f} Å")
            print(f"   Transmisión máxima: {max_trans:.3f}")
            print(f"   Rango: {wave[0]:.0f} - {wave[-1]:.0f} Å")
            print()
            
            # Plot curvas de transmisión
            ax1.plot(wave, trans, label=nombre, linewidth=2)
            
            # Plot densidad espectral (para CIGALE)
            ax2.plot(wave, trans/max_trans, label=nombre, linewidth=2)
    
    # Configurar plot 1: Curvas de transmisión
    ax1.set_xlabel('Longitud de Onda (Å)')
    ax1.set_ylabel('Transmisión')
    ax1.set_title('Curvas de Transmisión - Filtros S-PLUS')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configurar plot 2: Curvas normalizadas
    ax2.set_xlabel('Longitud de Onda (Å)')
    ax2.set_ylabel('Transmisión Normalizada')
    ax2.set_title('Curvas Normalizadas para CIGALE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_filtros_completo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Análisis para CIGALE
    print("🎯 RECOMENDACIONES PARA CIGALE:")
    print("=" * 50)
    
    # Verificar cobertura espectral
    lambdas_eff = [r['lambda_eff'] for r in resultados.values()]
    min_wave = min(lambdas_eff)
    max_wave = max(lambdas_eff)
    
    print(f"🌈 Cobertura espectral: {min_wave:.0f} - {max_wave:.0f} Å")
    print(f"📏 Rango total: {max_wave - min_wave:.0f} Å")
    
    # Espaciado entre filtros
    lambdas_sorted = sorted(lambdas_eff)
    separaciones = [lambdas_sorted[i+1] - lambdas_sorted[i] for i in range(len(lambdas_sorted)-1)]
    
    print(f"📐 Separación promedio entre filtros: {np.mean(separaciones):.1f} Å")
    print(f"📐 Separación mínima: {min(separaciones):.1f} Å (entre {lambdas_sorted[separaciones.index(min(separaciones))]:.0f}-{lambdas_sorted[separaciones.index(min(separaciones))+1]:.0f} Å)")
    
    # Para cúmulos globulares
    print(f"\n🌌 PARA CÚMULOS GLOBULARES:")
    print("   ✅ Excelente cobertura en UV-azul (sensibilidad a metalicidad)")
    print("   ✅ Buena cobertura en óptico (edad y masa)")
    print("   ✅ Múltiples filtros en u y g (redundancia para robustez)")
    
    return resultados

def calcular_fwhm(wavelength, transmission):
    """Calcula el FWHM de una curva de transmisión"""
    max_trans = np.max(transmission)
    half_max = max_trans / 2.0
    
    above_half = transmission >= half_max
    if np.any(above_half):
        left_idx = np.where(above_half)[0][0]
        right_idx = np.where(above_half)[0][-1]
        return wavelength[right_idx] - wavelength[left_idx]
    return 0.0

if __name__ == "__main__":
    resultados = analizar_filtros_completo()
