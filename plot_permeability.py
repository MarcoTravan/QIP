#!/usr/bin/env python3
"""
plot_permeability.py — QIP Paper I companion figure

Generates the permeability curve κ(ρ) = 1/(1 + 2αρ) for the
isotropic limit, with α = 2π − 1/2.

Key features annotated:
  • κ = 1 at ρ = 0 (vacuum: full coherence)
  • κ_sat = 1/(1 + 2α) ≈ 0.080 at ρ = 1 (maximum congestion)
  • ρ_crit = 7/8 (overflow threshold, from octonionic bath dimension)
  • Scattering rate γ = 1 − κ shown as complementary shading

Output: fig_permeability.pdf  (vector, publication-quality)

Repository: https://github.com/MarcoTravan/QIP
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── QIP parameters ──────────────────────────────────────────────────
alpha = 2 * np.pi - 0.5       # α = 2π − 1/2 ≈ 5.7832
rho_crit = 7.0 / 8.0          # overflow threshold

def kappa(rho):
    """Isotropic permeability: κ = 1/(1 + 2αρ)."""
    return 1.0 / (1.0 + 2.0 * alpha * rho)

# ── Evaluate ────────────────────────────────────────────────────────
rho = np.linspace(0, 1, 1000)
k = kappa(rho)
k_sat = kappa(1.0)
k_crit = kappa(rho_crit)

# ── Figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Shaded regions: coherent (blue) and scattered (red)
ax.fill_between(rho, 0, k, alpha=0.15, color="steelblue",
                label=r"coherent fraction $\kappa$")
ax.fill_between(rho, k, 1, alpha=0.10, color="indianred",
                label=r"scattered fraction $\gamma = 1 - \kappa$")

# Main curve
ax.plot(rho, k, color="navy", linewidth=2.0, zorder=5)

# ── Annotations ─────────────────────────────────────────────────────
# ρ_crit = 7/8  vertical marker
ax.axvline(rho_crit, color="gray", linestyle="--", linewidth=0.8)
ax.annotate(r"$\rho_{\mathrm{crit}} = \frac{7}{8}$",
            xy=(rho_crit, k_crit), xytext=(rho_crit + 0.04, k_crit + 0.18),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# κ_sat floor
ax.axhline(k_sat, color="gray", linestyle=":", linewidth=0.7)
ax.text(0.02, k_sat + 0.02,
        r"$\kappa_\mathrm{sat} \approx 0.080$",
        fontsize=9, color="gray")

# Vacuum label
ax.annotate(r"vacuum: $\kappa = 1$",
            xy=(0.0, 1.0), xytext=(0.12, 0.92),
            fontsize=9, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))

# ── Axes ────────────────────────────────────────────────────────────
ax.set_xlabel(r"Purity congestion $\rho$", fontsize=11)
ax.set_ylabel(r"Permeability $\kappa$", fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.05)
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.legend(loc="center right", fontsize=9, framealpha=0.9)
ax.set_title(r"QIP permeability: $\kappa(\rho) = 1/(1 + 2\alpha\rho)$,"
             r"  $\alpha = 2\pi - \frac{1}{2}$",
             fontsize=10, pad=8)

fig.tight_layout()

# ── Save ────────────────────────────────────────────────────────────
outpath = "fig_permeability.pdf"
fig.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved: {outpath}")
print(f"  α  = {alpha:.4f}")
print(f"  κ_sat = {k_sat:.4f}")
print(f"  κ(ρ_crit) = {k_crit:.4f}")
print(f"  ρ_crit = {rho_crit:.4f}")
