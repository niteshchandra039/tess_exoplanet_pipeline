# Standalone script to recreate all TESS analysis plots
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Apply publication-ready style configurations (MNRAS-style STIX fonts, inward ticks)
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.frameon'] = False
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.major.size'] = 4.0
plt.rcParams['xtick.minor.size'] = 2.0
plt.rcParams['ytick.major.size'] = 4.0
plt.rcParams['ytick.minor.size'] = 2.0
plt.rcParams['xtick.major.width'] = 0.75
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.75
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Try to load target metadata
tic_id = "261136679"
sectors_str = "unknown"
try:
    with open("TIC261136679_metadata.json") as f:
        meta = json.load(f)
        secs = meta.get("metadata", {}).get("sectors_used", [])
        sectors_str = ", ".join(map(str, secs)) if secs else "unknown"
except Exception:
    pass

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Helper function to compute phase folded curve
def phase_fold(time, flux, period, epoch):
    t0 = epoch if epoch is not None else time[0]
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]

# Helper for binning
def bin_phase_curve(phase, flux, n_bins=80):
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_flux[i] = np.mean(flux[mask])
    return bin_centers, bin_flux

# ── Load Data ──
try:
    pg_data = np.load("data_for_plots/periodogram_data.npz", allow_pickle=True)
    lc_data = np.load("data_for_plots/lightcurve_model_data.npz", allow_pickle=True)
    print("Successfully loaded plotting data.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# ── 1. Recreate TLS/BLS Periodograms ──
for method in ("tls", "bls"):
    keys = [k for k in pg_data.keys() if k.startswith(f"{method}_periods_p")]
    if not keys:
        keys_broad = [k for k in pg_data.keys() if k.startswith(f"{method}_periods_broad_p")]
        if not keys_broad:
            continue
        n_panels = len(keys_broad)
    else:
        n_panels = len(keys)
        
    # Standard combined auto-mode periodogram
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), squeeze=False)
    for idx in range(n_panels):
        ax = axes[idx, 0]
        # Prefer broad/coarse if available, else fine (auto mode)
        periods = pg_data.get(f"{method}_periods_broad_p{idx}")
        if periods is None:
            periods = pg_data.get(f"{method}_periods_p{idx}")
        power = pg_data.get(f"{method}_power_broad_p{idx}")
        if power is None:
            power = pg_data.get(f"{method}_power_p{idx}")
            
        best_period = float(pg_data[f"period_p{idx}"])
        stat = float(pg_data[f"sde_p{idx}"])
        stat_name = "SDE" if method == "tls" else "SNR"
        
        if periods is not None and len(periods) > 0:
            color = "#2563eb" if method == "tls" else "#d97706"
            ax.plot(periods, power, color=color, linewidth=0.8)
            ax.set_xlim(np.min(periods), np.max(periods))
            ax.set_ylim(0, np.max(power) * 1.1)
            
        ax.axvline(best_period, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Best period: {best_period:.5f} d")
        if method == "tls":
            for n in (2, 3):
                ax.axvline(best_period / n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.axvline(best_period * n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
                
        ax.set_ylabel(f"{method.upper()} Power")
        ax.legend(fontsize=9, loc="upper right")
        title_suffix = f" (Planet {idx+1} Search)" if n_panels > 1 else ""
        ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | {method.upper()} Periodogram{title_suffix} ({stat_name} = {stat:.2f})", fontsize=10, fontweight="bold")
    axes[-1, 0].set_xlabel("Period (days)")
    fig.tight_layout()
    print(f"Generating plot: plots/03_{method}_periodogram.png")
    fig.savefig(f"plots/03_{method}_periodogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save individual coarse and fine periodograms for each planet candidate
    for idx in range(n_panels):
        stat = float(pg_data[f"sde_p{idx}"])
        stat_name = "SDE" if method == "tls" else "SNR"
        best_period = float(pg_data[f"period_p{idx}"])
        
        for mode in ("coarse", "fine"):
            if mode == "coarse":
                periods = pg_data.get(f"{method}_periods_broad_p{idx}")
                if periods is None:
                    periods = pg_data.get(f"{method}_periods_p{idx}")
                power = pg_data.get(f"{method}_power_broad_p{idx}")
                if power is None:
                    power = pg_data.get(f"{method}_power_p{idx}")
            else:
                periods = pg_data.get(f"{method}_periods_p{idx}")
                power = pg_data.get(f"{method}_power_p{idx}")
                
            if periods is None or len(periods) == 0:
                continue
                
            fig_ind, ax_ind = plt.subplots(figsize=(12, 3.5))
            color = "#2563eb" if method == "tls" else "#d97706"
            ax_ind.plot(periods, power, color=color, linewidth=0.8)
            ax_ind.set_xlim(np.min(periods), np.max(periods))
            ax_ind.set_ylim(0, np.max(power) * 1.1)
            
            ax_ind.axvline(best_period, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Best period: {best_period:.5f} d")
            if method == "tls":
                for n in (2, 3):
                    ax_ind.axvline(best_period / n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
                    ax_ind.axvline(best_period * n, color="#ea580c", linestyle=":", linewidth=0.8, alpha=0.6)
            
            ax_ind.set_ylabel(f"{method.upper()} Power")
            ax_ind.set_xlabel("Period (days)")
            ax_ind.legend(fontsize=9, loc="upper right")
            mode_label = " Coarse" if mode == "coarse" else " Fine"
            ax_ind.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | {method.upper()}{mode_label} Periodogram (Planet {idx+1} Search) ({stat_name} = {stat:.2f})", fontsize=10, fontweight="bold")
            fig_ind.tight_layout()
            print(f"Generating plot: plots/03_{method}_periodogram_{mode}_p{idx}.png")
            fig_ind.savefig(f"plots/03_{method}_periodogram_{mode}_p{idx}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ind)

# ── 2. Recreate Phase curves ──
if "time" in lc_data and "flux" in lc_data:
    time = lc_data["time"]
    flux = lc_data["flux"]
    
    keys = [k for k in lc_data.keys() if k.startswith("period_med_p")]
    n_planets = len(keys) if keys else 1
    
    for idx in range(n_planets):
        if f"period_med_p{idx}" not in lc_data:
            if idx == 0 and "gp_model" in lc_data:
                p_med = float(lc_data.get("period_p0") or 1.0)
                p_std = 0.0
                e_med = float(lc_data.get("epoch_p0") or time[0])
                t14_med = 0.15
            else:
                break
        else:
            p_med = float(lc_data[f"period_med_p{idx}"])
            p_std = float(lc_data.get(f"period_std_p{idx}") or 0.0)
            e_med = float(lc_data[f"epoch_med_p{idx}"])
            t14_med = float(lc_data.get(f"t14_med_p{idx}") or 0.15)
            
        gp_model = lc_data.get("gp_model")
        detrended = flux - gp_model if gp_model is not None else flux
        
        other_mod = np.zeros_like(flux)
        j = 0
        while True:
            key = f"light_curve_median_p{j}"
            if key in lc_data:
                if j != idx:
                    other_mod += lc_data[key]
                j += 1
            else:
                break
        detrended = detrended - other_mod
        
        phase, y_fold = phase_fold(time, detrended, p_med, e_med)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        
        axes[0].scatter(phase * p_med, y_fold, s=0.8, color="black", alpha=0.2, label="Data", rasterized=True, zorder=-1000)
        
        bin_phase, bin_flux = bin_phase_curve(phase, y_fold, n_bins=80)
        valid = np.isfinite(bin_flux)
        axes[0].errorbar(bin_phase[valid] * p_med, bin_flux[valid], fmt="o", ms=4, color="#16a34a", label="Binned", zorder=100)
        
        pred_key = f"pred_50_p{idx}"
        depth = 0.005
        transit_med = None
        if pred_key in lc_data:
            phase_grid = np.linspace(-0.3, 0.3, 200)
            axes[0].plot(phase_grid, lc_data[pred_key], color="#dc2626", linewidth=2.0, label="Transit Model", zorder=1000)
            depth = max(1.0 - np.min(lc_data[pred_key]), 1e-4)
        elif f"light_curve_median_p{idx}" in lc_data:
            transit_med = lc_data[f"light_curve_median_p{idx}"]
            _, t_fold = phase_fold(time, transit_med + 1.0, p_med, e_med)
            sort_idx = np.argsort(phase)
            x_sorted = phase[sort_idx] * p_med
            y_sorted = t_fold[sort_idx]
            grid_x = np.linspace(-0.3, 0.3, 1000)
            grid_y = np.interp(grid_x, x_sorted, y_sorted, left=1.0, right=1.0)
            axes[0].plot(grid_x, grid_y, color="#dc2626", linewidth=2.0, label="Transit Model", zorder=1000)
            depth = max(1.0 - np.min(grid_y), 1e-4)
            
        txt = f"P = {p_med:.5f} ± {p_std:.5f} d"
        axes[0].annotate(txt, (0.02, 0.05), xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"), fontsize=10)
        
        axes[0].set_xlim(-2.5 * t14_med, 2.5 * t14_med)
        axes[0].set_ylim(1.0 - 2.5 * depth, 1.0 + 1.5 * depth)
        axes[0].set_ylabel("De-trended Relative Flux")
        axes[0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Phased MCMC Fit - Planet {idx + 1}", fontsize=10, fontweight="bold")
        axes[0].legend(fontsize=9, loc="upper right")
        
        if transit_med is not None:
            res_val = detrended - (transit_med + 1.0)
        elif "residuals" in lc_data:
            res_val = lc_data["residuals"]
        else:
            res_val = np.zeros_like(flux)
            
        _, r_fold = phase_fold(time, res_val + 1.0, p_med, e_med)
        axes[1].scatter(phase * p_med, r_fold - 1.0, s=0.8, color="gray", alpha=0.3, label="Residuals", rasterized=True)
        
        bin_r_phase, bin_r_flux = bin_phase_curve(phase, r_fold - 1.0, n_bins=80)
        valid_r = np.isfinite(bin_r_flux)
        axes[1].errorbar(bin_r_phase[valid_r] * p_med, bin_r_flux[valid_r], fmt="o", ms=4, color="#dc2626", label="Binned Residuals", zorder=10)
        
        axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Residuals")
        axes[1].set_xlabel("Time since transit (days)")
        axes[1].legend(fontsize=9, loc="upper right")
        axes[1].set_ylim(-1.5 * depth, 1.5 * depth)
        
        fig.tight_layout()
        filename = f"04_mcmc_phase_p{idx}.png" if (idx > 0 or n_planets > 1) else "04_mcmc_phase.png"
        print(f"Generating plot: plots/{filename}")
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)

# ── 3. Recreate 2-Panel MCMC Fit plot ──
if "gp_model" in lc_data and "transit_model" in lc_data:
    gp_model = lc_data["gp_model"]
    transit_model = lc_data["transit_model"]
    flux_model = lc_data["flux_model"]
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    
    axes[0].scatter(time, flux, s=0.4, color="black", alpha=0.25, label="Data", rasterized=True)
    trend = flux_model - (transit_model - 1.0)
    axes[0].plot(time, trend, color="#22c55e", linewidth=1.2, label="GP Activity Model")
    axes[0].set_ylabel("Relative Flux")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].set_title(f"TIC {tic_id} | Sectors: {sectors_str} | Bayesian GP Detrending (Time Domain)", fontsize=10, fontweight="bold")
    
    detrended = flux - gp_model
    axes[1].scatter(time, detrended, s=0.4, color="black", alpha=0.25, label="Detrended Data", rasterized=True)
    axes[1].plot(time, transit_model, color="#dc2626", linewidth=1.2, label="Keplerian Fit")
    
    axes[1].set_ylabel("Relative Flux")
    axes[1].set_xlabel("Time (BTJD)")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_title("Detrended Light Curve & Best-Fit Transit Model", fontsize=10, fontweight="bold")
    
    fig.tight_layout()
    print("Generating plot: plots/06_bayesian_fit.png")
    fig.savefig("plots/06_bayesian_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 4. Recreate GP Autocorrelation Plot ──
if "gp_model" in lc_data:
    fig, ax = plt.subplots(figsize=(10, 4))
    res_before = flux - lc_data.get("transit_model", np.ones_like(flux))
    res_after = lc_data.get("residuals", np.zeros_like(flux))
    
    def get_acf(x, max_lag=150):
        x_centered = x - np.mean(x)
        n = len(x_centered)
        acf = np.correlate(x_centered, x_centered, mode='full')[n-1:n+max_lag]
        if acf[0] > 0:
            acf = acf / acf[0]
        return acf
        
    max_lag = min(len(time) // 4, 150)
    lags = np.arange(max_lag)
    acf_before = get_acf(res_before, max_lag=max_lag-1)
    acf_after = get_acf(res_after, max_lag=max_lag-1)
    
    ax.plot(lags, acf_before, color="#94a3b8", label="Raw Residuals (No GP)", linewidth=1.5)
    ax.plot(lags, acf_after, color="#0f766e", label="MCMC Residuals (After GP)", linewidth=1.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    
    conf = 1.96 / np.sqrt(len(time))
    ax.axhspan(-conf, conf, color="#0f766e", alpha=0.1, label="95% Confidence Interval")
    
    ax.set_xlabel("Lag (indices)")
    ax.set_ylabel("Autocorrelation Coeff")
    ax.set_title(f"TIC {tic_id} | Sectors: {sectors_str} | GP Residuals ACF", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    print("Generating plot: plots/10_gp_acf.png")
    fig.savefig("plots/10_gp_acf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 5. Recreate Stacked transits ──
if "time" in lc_data:
    keys = [k for k in lc_data.keys() if k.startswith("period_med_p")]
    n_planets = len(keys) if keys else 1
    
    for idx in range(n_planets):
        if f"period_med_p{idx}" not in lc_data:
            if idx == 0 and ("period_p0" in lc_data or "epoch_p0" in lc_data):
                p_med = float(lc_data.get("period_p0") or 1.0)
                e_med = float(lc_data.get("epoch_p0") or time[0])
                t14_med = float(lc_data.get("t14_med_p0") or 0.15)
            else:
                break
        else:
            p_med = float(lc_data[f"period_med_p{idx}"])
            e_med = float(lc_data[f"epoch_med_p{idx}"])
            t14_med = float(lc_data.get(f"t14_med_p{idx}") or 0.15)
            
        gp_model = lc_data.get("gp_model")
        detrended = flux - gp_model if gp_model is not None else flux / np.median(flux)
        
        # Clean out other planets' transits
        other_mod = np.zeros_like(flux)
        j = 0
        while True:
            key = f"light_curve_median_p{j}"
            if key in lc_data:
                if j != idx:
                    other_mod += lc_data[key]
                j += 1
            else:
                break
        detrended = detrended - other_mod
        
        # Select individual planet transit model if available
        transit_model = None
        if f"light_curve_median_p{idx}" in lc_data:
            transit_model = lc_data[f"light_curve_median_p{idx}"] + 1.0
        elif "transit_model" in lc_data:
            transit_model = lc_data["transit_model"]
            
        half_width = max(0.2, t14_med * 3.0)
        t_start = np.min(time)
        t_end = np.max(time)
        n_start = int(np.floor((t_start - e_med) / p_med))
        n_end = int(np.ceil((t_end - e_med) / p_med))
        epochs = [e_med + n * p_med for n in range(n_start, n_end + 1)]
        epochs = [ep for ep in epochs if t_start - half_width < ep < t_end + half_width]
        
        if len(epochs) > 0:
            max_to_plot = 8
            if len(epochs) > max_to_plot:
                indices = np.linspace(0, len(epochs) - 1, max_to_plot, dtype=int)
                epochs_to_plot = [epochs[i] for i in indices]
            else:
                epochs_to_plot = epochs
                
            fig, axes = plt.subplots(len(epochs_to_plot), 1, figsize=(8, 1.5 * len(epochs_to_plot)), sharex=True)
            if len(epochs_to_plot) == 1:
                axes = [axes]
                
            for s_idx, ep in enumerate(epochs_to_plot):
                ax = axes[s_idx]
                mask = (time >= ep - half_width) & (time <= ep + half_width)
                t_sub = time[mask] - ep
                f_sub = detrended[mask]
                
                if len(t_sub) > 0:
                    ax.scatter(t_sub, f_sub, s=1.5, color="black", alpha=0.4, rasterized=True, label="Data" if s_idx == 0 else None)
                    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.7)
                    
                    if transit_model is not None:
                        m_sub = transit_model[mask]
                        sort_idx = np.argsort(t_sub)
                        ax.plot(t_sub[sort_idx], m_sub[sort_idx], color="#dc2626", linewidth=1.5, label="Transit Model" if s_idx == 0 else None)
                    
                ax.set_ylabel(f"Transit {s_idx+1}")
                ax.set_xlim(-half_width, half_width)
                
                if len(f_sub) > 0:
                    min_val = np.min(f_sub)
                    if transit_model is not None and len(m_sub) > 0:
                        min_val = min(min_val, np.min(m_sub))
                    local_std = np.std(f_sub)
                    ax.set_ylim(min_val - 0.00001 * abs(min_val), 1.0 + 3.0 * local_std)
                    
            axes[-1].set_xlabel("Time since transit mid-time (days)")
            if transit_model is not None:
                axes[0].legend(fontsize=8, loc="upper right")
            
            title_suffix = f" - Planet {idx+1}" if n_planets > 1 else ""
            fig.suptitle(f"TIC {tic_id} | Sectors: {sectors_str} | Stacked Individual Transits{title_suffix}", y=0.99, fontsize=11, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.2)
            
            filename = f"11_transit_stack_p{idx}.png" if (idx > 0 or n_planets > 1) else "11_transit_stack.png"
            print(f"Generating plot: plots/{filename}")
            fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
            plt.close(fig)

print("All plots recreated successfully.")
