#!/usr/bin/env python3
"""
splus_photometry_robust_unsharp.py
Fotometría con unsharp mask ROBUSTO y VALIDADO
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from astropy.stats import sigma_clipped_stats
import logging

def robust_galaxy_subtraction(data, field_name, filter_name, diagnostic_dir=None):
    """
    Unsharp mask ROBUSTO con validación automática
    """
    try:
        # ESTADÍSTICAS INICIALES PARA REFERENCIA
        initial_stats = sigma_clipped_stats(data, sigma=3.0)
        initial_median = initial_stats[1]
        
        # PARÁMETROS CONSERVADORES (Basados en Buzzo et al. pero mejorados)
        if field_name in ['CenA11', 'CenA12', 'CenA13']:  # Campos cerca del centro
            median_box = 35  # Más grande para no afectar cúmulos
            gaussian_sigma = 8  # Más suave
        else:
            median_box = 25  
            gaussian_sigma = 5
        
        # PASO 1: Filtro de mediana PARA PRESERVAR FUENTES
        logging.info(f"Unsharp mask: {field_name} {filter_name}, box={median_box}, sigma={gaussian_sigma}")
        
        # Usar solo píxeles de fondo para la mediana (excluir fuentes brillantes)
        background_mask = data < initial_median + 3 * initial_stats[2]  # Solo fondo
        if np.sum(background_mask) < 100:  # Si no hay suficiente fondo
            background_mask = np.ones_like(data, dtype=bool)
        
        median_filtered = median_filter(data, size=median_box)
        
        # PASO 2: Suavizado gaussiano CONSERVADOR
        galaxy_background = gaussian_filter(median_filtered, sigma=gaussian_sigma)
        
        # PASO 3: Resta CONTROLADA (evitar sobre-resta)
        residual = data - galaxy_background
        
        # VALIDACIÓN CRÍTICA
        residual_stats = sigma_clipped_stats(residual, sigma=3.0)
        residual_median = residual_stats[1]
        residual_std = residual_stats[2]
        
        # Porcentaje de píxeles negativos
        negative_pixels = np.sum(residual < 0)
        negative_fraction = negative_pixels / residual.size
        
        # Verificar que no hayamos creado artefactos
        if negative_fraction > 0.15:  # Límite conservador
            logging.warning(f"⚠️  Possible over-subtraction in {field_name} {filter_name}: "
                          f"{negative_fraction:.1%} negative pixels")
            
            # CORRECCIÓN: Mezclar con original para reducir sobre-resta
            blend_factor = min(0.7, 0.3 / negative_fraction)  # Factor adaptativo
            residual = blend_factor * residual + (1 - blend_factor) * data
            logging.info(f"   Applied blending correction: {blend_factor:.2f}")
        
        # Verificar que el fondo residual es razonable
        if abs(residual_median) > 2 * initial_std:
            logging.warning(f"⚠️  High residual background in {field_name} {filter_name}: "
                          f"median={residual_median:.4f}")
        
        logging.info(f"✅ Unsharp mask successful: "
                    f"neg_pixels={negative_fraction:.3f}, "
                    f"residual_median={residual_median:.4f}")
        
        # DIAGNÓSTICO OPCIONAL
        if diagnostic_dir:
            save_unsharp_diagnostic(data, galaxy_background, residual, 
                                  field_name, filter_name, diagnostic_dir)
        
        return residual, galaxy_background, True
        
    except Exception as e:
        logging.error(f"❌ Unsharp mask failed for {field_name} {filter_name}: {e}")
        return data, np.zeros_like(data), False

def save_unsharp_diagnostic(original, background, residual, field_name, filter_name, output_dir):
    """Guarda diagnóstico del unsharp mask"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Usar percentiles consistentes para visualización
        vmin, vmax = np.percentile(original, [10, 90])
        
        # Original
        im1 = axes[0, 0].imshow(original, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title(f'Original - {field_name} {filter_name}')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Fondo galáctico
        im2 = axes[0, 1].imshow(background, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title('Galaxy Background')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Residual
        residual_vmin, residual_vmax = np.percentile(residual, [10, 90])
        im3 = axes[1, 0].imshow(residual, cmap='viridis', vmin=residual_vmin, vmax=residual_vmax, origin='lower')
        axes[1, 0].set_title('Residual (Original - Background)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Histograma comparativo
        axes[1, 1].hist(original.flatten(), bins=100, alpha=0.7, label='Original', density=True)
        axes[1, 1].hist(residual.flatten(), bins=100, alpha=0.7, label='Residual', density=True)
        axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Pixel Value Distribution')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/{field_name}_{filter_name}_unsharp_diagnostic.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 Unsharp diagnostic saved: {plot_path}")
        
    except Exception as e:
        logging.warning(f"Could not save unsharp diagnostic: {e}")
