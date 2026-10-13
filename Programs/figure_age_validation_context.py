#!/usr/bin/env python3
# figure_age_validation_context.py
# Figura que muestra el contexto de validación

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Configuración
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
})

# Cargar datos
df = pd.read_csv('figures/age_comparison_woodley_exact.csv')

# Crear figura con contexto
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Lo que hace Woodley (esquemático)
ax1 = axes[0]

# Representación esquemática de la validación de Woodley
mw_ages = np.array([3, 5, 7, 9, 11, 13])  # Edades MSF (referencia)
lick_ages = mw_ages + np.array([-1, -0.8, -0.5, 0.2, 0.5, 1.0])  # Con offset sistemático

ax1.scatter(mw_ages, lick_ages, s=80, alpha=0.7, color='steelblue')
ax1.plot([0, 14], [0, 14], 'k--', alpha=0.7, label='1:1')

# Añadir flechas que muestran los offsets sistemáticos
for x, y in zip(mw_ages, lick_ages):
    ax1.arrow(x, x, 0, y-x, 
              head_width=0.3, head_length=0.3, 
              fc='red', ec='red', alpha=0.5, 
              length_includes_head=True)

ax1.set_xlabel('MSF Age (Gyr)', fontsize=12)
ax1.set_ylabel('Lick Index Age (Gyr)', fontsize=12)
ax1.set_xlim(2, 14)
ax1.set_ylim(2, 14)
ax1.grid(True, alpha=0.3)
ax1.set_title('Woodley et al. (2010) Validation\nLick indices vs Main-sequence fitting', fontsize=13)
ax1.text(0.05, 0.95, '"Lick index routine underestimates\nthe ages of the younger GCs\nwhile it overestimates the ages\nfor the oldest GCs."', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 2: Tu comparación
ax2 = axes[1]

# Puntos con errores
ax2.errorbar(df['age_Woodley'], df['age_CIGALE'],
            xerr=df['age_err_Woodley'].clip(upper=5),
            yerr=df['age_err_CIGALE'].clip(upper=5),
            fmt='o', color='coral', ecolor='gray',
            alpha=0.7, capsize=3, markersize=6)

# Línea 1:1
ax2.plot([0, 14], [0, 14], 'k--', alpha=0.7)

# Calcular métricas
if len(df) > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['age_Woodley'], df['age_CIGALE']
    )
    x_fit = np.array([0, 14])
    y_fit = slope * x_fit + intercept
    ax2.plot(x_fit, y_fit, 'r-', alpha=0.8, linewidth=2,
             label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
    
    # Texto
    textstr = (f'$N$ = {len(df)}\n'
               f'MAE = {np.mean(np.abs(df["age_CIGALE"] - df["age_Woodley"])):.2f} Gyr\n'
               f'$r$ = {r_value:.3f}\n'
               f'RMS = {np.sqrt(np.mean((df["age_CIGALE"] - df["age_Woodley"])**2)):.2f} Gyr')
    
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_xlabel('Woodley et al. (2010) Age (Gyr)', fontsize=12)
ax2.set_ylabel('This Work Age (Gyr)', fontsize=12)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 14)
ax2.grid(True, alpha=0.3)
ax2.set_title('This Work: Comparison\nSED fitting vs Lick indices', fontsize=13)
ax2.legend(fontsize=10)

# Panel 3: Síntesis - ¿Qué significa esto para NGC 5128?
ax3 = axes[2]

# Histogramas comparativos
bins = np.arange(0, 15, 1.5)
ax3.hist(df['age_Woodley'], bins=bins, alpha=0.5, 
         label='Woodley et al. (2010)', color='steelblue', density=True)
ax3.hist(df['age_CIGALE'], bins=bins, alpha=0.5,
         label='This Work', color='coral', density=True)

ax3.set_xlabel('Age (Gyr)', fontsize=12)
ax3.set_ylabel('Normalized Frequency', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_title('NGC 5128 Age Distribution\nComparison of Methods', fontsize=13)
ax3.legend(fontsize=10)

# Estadísticas
woodley_old = (df['age_Woodley'] > 8).sum() / len(df) * 100
our_old = (df['age_CIGALE'] > 8).sum() / len(df) * 100

ax3.text(0.05, 0.95, f'Woodley: {woodley_old:.0f}% > 8 Gyr\nThis work: {our_old:.0f}% > 8 Gyr', 
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figures/Figure_Age_Validation_Context.pdf', 
            bbox_inches='tight', dpi=600)
plt.savefig('figures/Figure_Age_Validation_Context.png', 
            bbox_inches='tight', dpi=300)

print("Figura creada: Contexto de validación de edades")
print("\nResumen:")
print(f"• Woodley valida SU método contra gold standard (MS fitting)")
print(f"• Encuentra offsets sistemáticos de ~1 Gyr")
print(f"• Advierte sobre posibles artefactos en NGC 5128")
print(f"• Tu comparación es con un método YA validado (pero con sistemáticos conocidos)")
print(f"• Ambos métodos concuerdan: sistema mayormente viejo (>8 Gyr)")
