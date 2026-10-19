def plot_bootstrap_slopes(rob_all, rob_old, output_dir):
    """Fig 8: distribucion bootstrap de las pendientes (fila 1: todos; fila 2: viejos)."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for row, (rob, lab) in enumerate([(rob_all, 'All GCs'), (rob_old, 'Old GCs >8 Gyr')]):
        for i, p in enumerate(POP_SHORT_3):
            ax = axes[row, i]
            if p not in rob:
                ax.axis('off'); continue
            r = rob[p]
            bins = np.linspace(min(np.nanmin(r['_bs']), np.nanmin(r['_bg'])),
                               max(np.nanmax(r['_bs']), np.nanmax(r['_bg'])), 40)
            ax.hist(r['_bs'], bins=bins, density=True, alpha=0.5, color=COLORS_3[i],
                    edgecolor='black', linewidth=0.6)
            ax.hist(r['_bg'][np.isfinite(r['_bg'])], bins=bins, density=True,
                    histtype='step', color='black', linewidth=2.2)
            ax.axvline(r['ols_slope'], color=COLORS_3[i], linewidth=3)
            ax.axvline(0, color='gray', linestyle='--', linewidth=2)
            b = r['bootstrap']
            ax.set_title(f"{POP_NAMES_3[i]} ({p}) – {lab}\n"
                         f"{r['ols_slope']:.3f} (+{b['p84']-b['median']:.3f}/"
                         f"−{b['median']-b['p16']:.3f}), N={r['N']}", fontsize=21)
            ax.set_xlabel('Gradient slope (dex dex$^{-1}$)', fontsize=26)
            if i == 0:
                ax.set_ylabel('Density', fontsize=28)
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.grid(True, alpha=0.2, linestyle='--')
    # Leyenda fuera de los paneles: no tapa ningun histograma
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.5, edgecolor='black'),
               Line2D([0], [0], color='black', linewidth=2.2),
               Line2D([0], [0], color='gray', linewidth=3),
               Line2D([0], [0], color='gray', linestyle='--', linewidth=2)]
    labels = ['Bootstrap (fixed membership)', 'Bootstrap (GMM re-fitted)',
              'Least-squares slope', 'Zero slope']
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=22,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    plt.tight_layout()
    base = os.path.join(output_dir, 'figures', 'Fig8_Bootstrap_Slopes')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")


def plot_mdf_radial_bins(bins_res, mu_g, sd_g, output_dir):
    """Fig 9: distribucion de [Fe/H] en bins radiales con las componentes globales."""
    nb = len(bins_res)
    fig, axes = plt.subplots(1, nb, figsize=(7*nb, 7), sharey=True)
    axes = np.atleast_1d(axes)
    allf = np.concatenate([b['_feh'] for b in bins_res])
    x = np.linspace(allf.min()-0.3, allf.max()+0.3, 1000)
    ymax = 0.0
    for ax, b in zip(axes, bins_res):
        h = ax.hist(b['_feh'], bins=20, density=True, alpha=0.3, color='lightgray',
                    edgecolor='black', linewidth=0.8, label=f"N={b['N']}")[0]
        tot = np.zeros_like(x)
        for i in range(3):
            pc = b['fixed_weights'][i] * norm.pdf(x, mu_g[i], sd_g[i])
            tot += pc
            ax.plot(x, pc, '--', linewidth=2.5, color=COLORS_3[i], alpha=0.85,
                    label=f"{POP_SHORT_3[i]}: f={b['fixed_weights'][i]:.2f}±{b['fixed_weights_err'][i]:.2f}")
        ax.plot(x, tot, 'k-', linewidth=3, alpha=0.9, label='GMM (global μ, σ)')
        ymax = max(ymax, h.max(), tot.max())
        ax.set_title(f"{b['R_min_kpc']:.0f} < R < {b['R_max_kpc']:.0f} kpc", fontsize=25)
        ax.set_xlabel('[Fe/H] (dex)', fontsize=27)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.legend(fontsize=17, loc='upper left', framealpha=0.9)
        means_txt = ', '.join(f'{v:.2f}' for v in b['free_n3_means'])
        ax.text(0.98, 0.98, f"Min. BIC: n={b['best_n_bic']}\nFree n=3 μ:\n{means_txt}",
                transform=ax.transAxes, fontsize=17, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_ylabel('Probability Density', fontsize=27)
    for ax in axes:
        ax.set_ylim(0, MDF_YTOP * ymax)
    plt.tight_layout()
    base = os.path.join(output_dir, 'figures', 'Fig9_MDF_Radial_Bins')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.savefig(f'{base}.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"OK figura: {base}.pdf")


MDF_YTOP = 1.45   # limite superior del eje Y de Fig9 (multiplo del maximo de los datos)
