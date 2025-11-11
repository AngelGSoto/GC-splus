#!/usr/bin/env python3
"""
validate_clean_final.py
Valida la fotometría limpia final vs Taylor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def validate_final_photometry():
    """Valida la fotometría final vs Taylor"""
    
    clean_path = "Results/gc_photometry_CLEAN_FINAL.csv"
    taylor_path = "../TAP_1_J_MNRAS_3444_gc.csv"
    
    if not os.path.exists(clean_path):
        print("❌ Clean photometry not found")
        return
    
    clean_df = pd.read_csv(clean_path)
    taylor_df = pd.read_csv(taylor_path)
    
    filter_mapping = {
        'F378': 'umag', 'F395': 'umag', 'F410': 'gmag', 
        'F430': 'gmag', 'F515': 'gmag', 'F660': 'rmag', 'F861': 'imag'
    }
    
    print("🔍 VALIDATING CLEAN PHOTOMETRY vs TAYLOR")
    print("=" * 50)
    
    results = {}
    
    for splus_filt, taylor_filt in filter_mapping.items():
        mag_col = f'MAG_{splus_filt}_2'
        
        if mag_col in clean_df.columns and taylor_filt in taylor_df.columns:
            merged = pd.merge(
                clean_df[['T17ID', mag_col]],
                taylor_df[['T17ID', taylor_filt]],
                on='T17ID'
            )
            
            valid_mask = (
                (merged[mag_col] < 50) & (merged[taylor_filt] < 50) &
                np.isfinite(merged[mag_col]) & np.isfinite(merged[taylor_filt])
            )
            
            valid_data = merged[valid_mask]
            
            if len(valid_data) > 10:
                diff = valid_data[mag_col] - valid_data[taylor_filt]
                
                results[splus_filt] = {
                    'n': len(valid_data),
                    'median_diff': np.median(diff),
                    'std_diff': np.std(diff),
                    'data': valid_data
                }
                
                print(f"{splus_filt:5} vs {taylor_filt:4}: Δ = {results[splus_filt]['median_diff']:7.3f} ± {results[splus_filt]['std_diff']:5.3f} (n={results[splus_filt]['n']})")
    
    # Crear gráfico
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Offsets por filtro
        filters = list(results.keys())
        offsets = [results[f]['median_diff'] for f in filters]
        stds = [results[f]['std_diff'] for f in filters]
        
        bars = ax1.bar(filters, offsets, yerr=stds, capsize=5, alpha=0.7, color='lightblue')
        ax1.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax1.set_ylabel('Δmag (SPLUS - DECam)')
        ax1.set_title('Offsets por Filtro - CLEAN PHOTOMETRY')
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, offset in zip(bars, offsets):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{offset:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Distribución de diferencias
        for filt in filters:
            diff = results[filt]['data'][f'MAG_{filt}_2'] - results[filt]['data'][filter_mapping[filt]]
            ax2.hist(diff, bins=30, alpha=0.6, label=filt, density=True)
        
        ax2.axvline(0, color='black', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Δmag (SPLUS - DECam)')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribución de Diferencias')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('validation_clean_final.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Validation plot saved: validation_clean_final.png")

if __name__ == "__main__":
    validate_final_photometry()
