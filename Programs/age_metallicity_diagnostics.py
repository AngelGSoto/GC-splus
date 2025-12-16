#!/usr/bin/env python3
# age_metallicity_diagnostics.py
# Análisis detallado de la relación edad-metalicidad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class AgeMetallicityDiagnostics:
    """Diagnóstico completo de la relación edad-metalicidad."""
    
    def __init__(self, age, metallicity, age_err=None, metal_err=None):
        self.age = np.array(age)
        self.metallicity = np.array(metallicity)
        self.age_err = age_err
        self.metal_err = metal_err
        
        # Estadísticas básicas
        self.stats = self.calculate_statistics()
        
    def calculate_statistics(self):
        """Calcula estadísticas detalladas."""
        stats_dict = {
            'n_objects': len(self.age),
            'age_mean': np.mean(self.age),
            'age_median': np.median(self.age),
            'age_std': np.std(self.age),
            'metal_mean': np.mean(self.metallicity),
            'metal_median': np.median(self.metallicity),
            'metal_std': np.std(self.metallicity),
        }
        
        # Correlaciones
        if len(self.age) > 2:
            pearson_r, pearson_p = stats.pearsonr(self.age, self.metallicity)
            spearman_rho, spearman_p = stats.spearmanr(self.age, self.metallicity)
            
            stats_dict.update({
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_rho': spearman_rho,
                'spearman_p': spearman_p,
            })
            
            # Regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                self.age, self.metallicity)
            
            stats_dict.update({
                'slope': slope,
                'slope_err': std_err,
                'intercept': intercept,
                'r_squared': r_value**2,
                'regression_p': p_value,
            })
        
        return stats_dict
    
    def plot_comprehensive_diagnostic(self, figsize=(15, 12)):
        """Gráfico diagnóstico completo."""
        
        fig = plt.figure(figsize=figsize)
        
        # Layout: 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Diagrama principal edad vs metalicidad
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Scatter con barras de error si existen
        if self.age_err is not None and self.metal_err is not None:
            ax1.errorbar(self.age, self.metallicity, 
                        xerr=self.age_err, yerr=self.metal_err,
                        fmt='o', alpha=0.5, markersize=4,
                        ecolor='gray', elinewidth=0.5, capsize=2)
        else:
            ax1.scatter(self.age, self.metallicity, alpha=0.6, s=20)
        
        # Regresión lineal
        if len(self.age) > 2:
            slope = self.stats['slope']
            intercept = self.stats['intercept']
            x_fit = np.linspace(self.age.min(), self.age.max(), 100)
            y_fit = intercept + slope * x_fit
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Fit: [Fe/H] = {slope:.3f} × Age + {intercept:.3f}')
        
        ax1.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Histograma de edades
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(self.age, bins=20, orientation='horizontal', alpha=0.7, color='blue')
        ax2.set_ylabel('[Fe/H]')
        ax2.set_xlabel('Count')
        ax2.set_title('Age Distribution', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Histograma de metalicidades
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.hist(self.metallicity, bins=20, alpha=0.7, color='red')
        ax3.set_xlabel('[Fe/H]')
        ax3.set_ylabel('Count')
        ax3.set_title('Metallicity Distribution', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Diagrama de residuos
        ax4 = fig.add_subplot(gs[2, 0])
        if len(self.age) > 2:
            residuals = self.metallicity - (slope * self.age + intercept)
            ax4.scatter(self.age, residuals, alpha=0.6, s=15)
            ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Age (Gyr)')
            ax4.set_ylabel('Residuals [Fe/H]')
            ax4.set_title('Residuals from Linear Fit', fontsize=10)
            ax4.grid(True, alpha=0.3)
        
        # 5. QQ-plot para normalidad de residuos
        ax5 = fig.add_subplot(gs[2, 1])
        if len(self.age) > 2:
            from scipy.stats import probplot
            probplot(residuals, dist="norm", plot=ax5)
            ax5.set_title('QQ-Plot of Residuals', fontsize=10)
            ax5.grid(True, alpha=0.3)
        
        # 6. Texto con estadísticas
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_text = f'N = {self.stats["n_objects"]:,}\n'
        stats_text += f'⟨Age⟩ = {self.stats["age_mean"]:.2f} ± {self.stats["age_std"]:.2f} Gyr\n'
        stats_text += f'⟨[Fe/H]⟩ = {self.stats["metal_mean"]:.2f} ± {self.stats["metal_std"]:.2f}\n\n'
        
        if 'pearson_r' in self.stats:
            stats_text += f'Pearson r = {self.stats["pearson_r"]:.3f}\n'
            stats_text += f'p-value = {self.stats["pearson_p"]:.3e}\n\n'
            stats_text += f'Slope = {self.stats["slope"]:.3f} ± {self.stats["slope_err"]:.3f}\n'
            stats_text += f'R² = {self.stats["r_squared"]:.3f}\n'
        
        # Interpretación
        interpretation = "\nINTERPRETATION:\n"
        if 'slope' in self.stats:
            slope = self.stats['slope']
            if slope > 0.01:
                interpretation += "• POSITIVE correlation\n"
                interpretation += "• Older GCs are more metal-rich\n"
                interpretation += "• Check for: Sample bias\n"
                interpretation += "              Young metal-poor contamination\n"
                interpretation += "              Age-metallicity degeneracy\n"
            elif slope < -0.01:
                interpretation += "• NEGATIVE correlation\n"
                interpretation += "• Classical relation\n"
                interpretation += "• Older GCs are more metal-poor\n"
            else:
                interpretation += "• NO significant correlation\n"
                interpretation += "• Flat age-metallicity relation\n"
                interpretation += "• Multiple populations?\n"
        
        ax6.text(0.05, 0.95, stats_text + interpretation, 
                fontsize=9, family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Age-Metallicity Relation Diagnostic', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_populations(self):
        """Analiza posibles subpoblaciones."""
        
        print("\n" + "="*60)
        print("SUBPOPULATION ANALYSIS")
        print("="*60)
        
        # Dividir por metalicidad (rojo/azul típico)
        metal_median = np.median(self.metallicity)
        mask_metal_rich = self.metallicity > metal_median
        mask_metal_poor = self.metallicity <= metal_median
        
        print(f"\nMedian metallicity: {metal_median:.3f}")
        print(f"Metal-rich ([Fe/H] > {metal_median:.3f}): {np.sum(mask_metal_rich)} GCs")
        print(f"Metal-poor ([Fe/H] ≤ {metal_median:.3f}): {np.sum(mask_metal_poor)} GCs")
        
        # Estadísticas por subpoblación
        if np.sum(mask_metal_rich) > 5 and np.sum(mask_metal_poor) > 5:
            print("\n--- Metal-rich population ---")
            age_mr = self.age[mask_metal_rich]
            metal_mr = self.metallicity[mask_metal_rich]
            slope_mr, _, r_mr, _, _ = stats.linregress(age_mr, metal_mr)
            print(f"  Mean age: {age_mr.mean():.2f} ± {age_mr.std():.2f} Gyr")
            print(f"  Mean [Fe/H]: {metal_mr.mean():.2f} ± {metal_mr.std():.2f}")
            print(f"  Age-[Fe/H] slope: {slope_mr:.4f} (R²={r_mr**2:.3f})")
            
            print("\n--- Metal-poor population ---")
            age_mp = self.age[mask_metal_poor]
            metal_mp = self.metallicity[mask_metal_poor]
            slope_mp, _, r_mp, _, _ = stats.linregress(age_mp, metal_mp)
            print(f"  Mean age: {age_mp.mean():.2f} ± {age_mp.std():.2f} Gyr")
            print(f"  Mean [Fe/H]: {metal_mp.mean():.2f} ± {metal_mp.std():.2f}")
            print(f"  Age-[Fe/H] slope: {slope_mp:.4f} (R²={r_mp**2:.3f})")
            
            # Test de diferencias
            from scipy.stats import ttest_ind, ks_2samp
            t_stat, t_p = ttest_ind(age_mr, age_mp)
            ks_stat, ks_p = ks_2samp(age_mr, age_mp)
            
            print(f"\n--- Statistical tests ---")
            print(f"  T-test ages (rich vs poor): t={t_stat:.3f}, p={t_p:.4f}")
            if t_p < 0.05:
                print(f"  → Ages are SIGNIFICANTLY different")
            else:
                print(f"  → No significant age difference")
        
        # Buscar bimodalidad
        self.test_bimodality()
        
    def test_bimodality(self):
        """Test de bimodalidad en metalicidades."""
        from scipy.stats import gaussian_kde, norm
        from sklearn.mixture import GaussianMixture
        
        print("\n" + "="*60)
        print("BIMODALITY TEST")
        print("="*60)
        
        if len(self.metallicity) < 30:
            print("  Sample too small for reliable bimodality test")
            return
        
        # Test de Hartigan's Dip
        try:
            from dip import dipstat
            dip, pval = dipstat(self.metallicity)
            print(f"\nHartigan's Dip Test:")
            print(f"  DIP statistic: {dip:.4f}")
            print(f"  p-value: {pval:.4f}")
            if pval < 0.05:
                print(f"  → SIGNIFICANT bimodality detected!")
            else:
                print(f"  → No significant bimodality")
        except:
            print("  Dip test not available")
        
        # Gaussian Mixture Model
        print(f"\nGaussian Mixture Model:")
        X = self.metallicity.reshape(-1, 1)
        
        # Probar 1, 2 y 3 componentes
        best_bic = np.inf
        best_n = 1
        
        for n in range(1, 4):
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
            print(f"  {n} component(s): BIC={bic:.1f}, AIC={aic:.1f}")
            
            if bic < best_bic:
                best_bic = bic
                best_n = n
        
        print(f"\n  Best model: {best_n} component(s) (lowest BIC)")
        
        if best_n >= 2:
            # Fit GMM con mejor número
            gmm = GaussianMixture(n_components=best_n, random_state=42)
            gmm.fit(X)
            
            print(f"\n  GMM parameters ({best_n} components):")
            for i in range(best_n):
                mean = gmm.means_[i][0]
                std = np.sqrt(gmm.covariances_[i][0][0])
                weight = gmm.weights_[i]
                print(f"    Component {i+1}: μ={mean:.3f}, σ={std:.3f}, weight={weight:.3f}")
    
    def compare_with_models(self):
        """Compara con modelos teóricos."""
        
        print("\n" + "="*60)
        print("COMPARISON WITH THEORETICAL MODELS")
        print("="*60)
        
        # Modelo de enriquecimiento simple
        print("\nSimple Enrichment Model:")
        print("  Expected: Older → Lower [Fe/H] (negative slope)")
        print("  Your slope: positive (older → higher [Fe/H])")
        
        # Preguntas clave
        questions = [
            "1. Is your sample complete? (selection effects?)",
            "2. Are you including YMCs? (young massive clusters)",
            "3. Is there dust extinction affecting ages?",
            "4. Age-metallicity degeneracy in SSP fitting?",
            "5. Multiple formation epochs in NGC 5128?"
        ]
        
        print("\nKey questions to consider:")
        for q in questions:
            print(f"  {q}")
        
        # Referencias
        print("\nRelevant references for NGC 5128 GCs:")
        refs = [
            "• Woodley et al. (2010): Two GC populations in Cen A",
            "• Beasley et al. (2008): Extended GC formation",
            "• Harris et al. (2017): GC system properties",
            "• Rejkuba et al. (2011): Inner GCs more metal-rich"
        ]
        for ref in refs:
            print(f"  {ref}")

# ============================================================================
# EJEMPLO DE USO
# ============================================================================

def main():
    """Ejemplo de uso del diagnóstico."""
    
    # Simular datos (reemplazar con tus datos reales)
    np.random.seed(42)
    n_gcs = 500
    
    # Crear dos poblaciones
    # Población 1: Viejos y pobres en metales
    age_pop1 = np.random.normal(12.0, 1.0, n_gcs//2)
    metal_pop1 = np.random.normal(-1.5, 0.3, n_gcs//2)
    
    # Población 2: Jóvenes y ricos en metales
    age_pop2 = np.random.normal(3.0, 0.5, n_gcs//2)
    metal_pop2 = np.random.normal(-0.5, 0.2, n_gcs//2)
    
    # Combinar
    age = np.concatenate([age_pop1, age_pop2])
    metallicity = np.concatenate([metal_pop1, metal_pop2])
    
    # Añadir algo de ruido y mezclar
    noise_age = np.random.normal(0, 0.5, n_gcs)
    noise_metal = np.random.normal(0, 0.1, n_gcs)
    age += noise_age
    metallicity += noise_metal
    
    # Barajar
    idx = np.random.permutation(n_gcs)
    age = age[idx]
    metallicity = metallicity[idx]
    
    print("="*70)
    print("AGE-METALLICITY RELATION DIAGNOSTIC TOOL")
    print("="*70)
    
    # Crear diagnóstico
    diagnostic = AgeMetallicityDiagnostics(age, metallicity)
    
    # 1. Estadísticas básicas
    print("\n📊 BASIC STATISTICS:")
    print("-"*40)
    for key, value in diagnostic.stats.items():
        print(f"{key:20}: {value:.4f}")
    
    # 2. Gráfico diagnóstico
    print("\n🎨 CREATING DIAGNOSTIC PLOT...")
    fig = diagnostic.plot_comprehensive_diagnostic()
    
    # 3. Análisis de subpoblaciones
    print("\n🔍 ANALYZING SUBPOPULATIONS...")
    diagnostic.analyze_populations()
    
    # 4. Comparación con modelos
    print("\n📚 COMPARING WITH MODELS...")
    diagnostic.compare_with_models()
    
    # 5. Interpretación
    print("\n" + "="*70)
    print("SUMMARY INTERPRETATION")
    print("="*70)
    
    slope = diagnostic.stats.get('slope', 0)
    r_squared = diagnostic.stats.get('r_squared', 0)
    
    if slope > 0.02 and r_squared > 0.1:
        print("""
⚠️  WARNING: STRONG POSITIVE CORRELATION DETECTED

This means:
• Older globular clusters are MORE metal-rich
• This is COUNTER to classical galactic chemical evolution

Possible explanations for your data:
1. SAMPLE SELECTION BIAS:
   • You might be preferentially selecting red, metal-rich GCs
   • Or missing blue, metal-poor GCs at large galactocentric distances

2. AGE-METALLICITY DEGENERACY:
   • In photometric SSP fitting, age and metallicity are degenerate
   • Old metal-poor and young metal-rich can give similar colors
   • Check if your SED fitting properly breaks this degeneracy

3. COMPLEX FORMATION HISTORY:
   • NGC 5128 had multiple major mergers
   • Could have formed metal-rich GCs early via rapid enrichment
   • Then formed metal-poor GCs later from accreted satellites

4. CONTAMINATION:
   • Young massive clusters (YMCs) mistaken for old GCs
   • Background/foreground objects in sample

RECOMMENDATIONS:
1. Check your sample selection function
2. Verify SSP model assumptions in CIGALE
3. Add spectroscopic metallicities if available
4. Plot color-color diagrams to check for contamination
5. Compare with literature values for NGC 5128 GCs
""")
    elif abs(slope) < 0.01:
        print("""
✅ FLAT RELATION DETECTED

This suggests:
• No strong correlation between age and metallicity
• GCs formed over extended period with varied metallicities
• Multiple formation epochs with different enrichment histories

This is actually COMMON in massive galaxies like NGC 5128!
""")
    elif slope < -0.02:
        print("""
✅ NEGATIVE CORRELATION DETECTED

This is the CLASSICAL relation:
• Older clusters → lower metallicity
• Consistent with hierarchical galaxy formation
• Early GCs formed from pristine gas, later ones from enriched gas
""")

if __name__ == "__main__":
    main()
