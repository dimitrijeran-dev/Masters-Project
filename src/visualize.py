#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

r_out = np.array([0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007])
KI = np.array([
    1546714.3726463828,
    1505712.7733211333,
    1495474.962785934,
    1494040.447465708,
    1491792.4977428948,
    1484361.9897479797,
    1473534.5708483662,
    1461833.366168616,
    1446942.7109881274,
    1425132.69115545,
    1392992.4417012604
])

# trusted plateau region
mask = (r_out >= 0.003) & (r_out <= 0.0045)
KI_plateau = KI[mask]
KI_mean = KI_plateau.mean()

# optional tolerance band from plateau spread
rel_band = (KI_plateau.max() - KI_plateau.min()) / (2 * KI_mean)
KI_upper = KI_mean * (1 + rel_band)
KI_lower = KI_mean * (1 - rel_band)

plt.figure(figsize=(8,5))
plt.plot(r_out, KI, marker="o", label=r"$K_I$ from J")
plt.axhline(KI_mean, linestyle="--", label=rf"Plateau mean = {KI_mean:.3e}")
plt.axhline(KI_upper, linestyle=":", linewidth=1)
plt.axhline(KI_lower, linestyle=":", linewidth=1)

# shade trusted contour window
plt.axvspan(0.003, 0.0045, alpha=0.2, label="Trusted contour window")

plt.xlabel(r"$r_{out}$ (m)")
plt.ylabel(r"$K_I$ (Pa$\sqrt{m}$)")
plt.title(r"$K_I$ vs. J-domain outer radius with plateau window")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("KI_vs_r_out.png", dpi=300)
plt.show()