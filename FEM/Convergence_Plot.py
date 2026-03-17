#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("Data/New Data/mesh_convergence_summary.csv")

h = df["lc_tip"].to_numpy(dtype=float)
KI = df["KI_ref"].to_numpy(dtype=float)   # or "KI_selected" if that is your column name

# sort from coarse to fine
idx = np.argsort(h)[::-1]
h = h[idx]
KI = KI[idx]

# fit only refined meshes
mask = h <= 0.0020
h_fit = h[mask]
KI_fit = KI[mask]

def conv_model(h, KI_inf, C, p):
    return KI_inf + C * h**p

KI_inf_guess = np.mean(KI_fit[-2:])
C_guess = KI_fit[0] - KI_inf_guess
p_guess = 1.0

popt, _ = curve_fit(
    conv_model,
    h_fit,
    KI_fit,
    p0=[KI_inf_guess, C_guess, p_guess],
    maxfev=20000
)

KI_inf, C, p = popt

h_smooth = np.linspace(np.min(h), np.max(h), 300)
KI_smooth = conv_model(h_smooth, *popt)

error = np.abs(KI - KI_inf) / KI_inf


plt.figure(figsize=(8, 5))
plt.plot(h, KI, "o", label="FE results")
plt.plot(h_smooth, KI_smooth, "-", label="Asymptotic fit")
plt.axhline(KI_inf, linestyle="--", label=rf"$K_{{I,\infty}}={KI_inf:.3e}$")
plt.gca().invert_xaxis()
plt.xlabel("Tip element size, $h=lc_{tip}$ (m)")
plt.ylabel(r"Selected $K_I$ (Pa$\sqrt{m}$)")
plt.title(r"Asymptotic mesh convergence of $K_I$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Data/New Data/mesh_convergence_plot.png", dpi=220, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7,5))
plt.loglog(h, error, 'o-', linewidth=2)

plt.xlabel("Tip element size $h = lc_{tip}$ (m)")
plt.ylabel(r"$|K_I - K_{I,\infty}|$")
plt.title("Mesh convergence error")

plt.grid(True, which="both")
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig("Data/New Data/mesh_convergence_error_plot.png", dpi=220, bbox_inches="tight")

print(f"Extrapolated KI_inf = {KI_inf:.6e} Pa*sqrt(m)")
print(f"Observed convergence exponent p = {p:.4f}")