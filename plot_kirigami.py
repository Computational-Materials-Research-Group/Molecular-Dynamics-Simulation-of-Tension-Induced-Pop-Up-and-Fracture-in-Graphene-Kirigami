#!/usr/bin/env python3
"""Stress-strain and out-of-plane amplitude from stress_strain.txt -> kirigami_curves.png"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = np.loadtxt("stress_strain.txt")
e, F, s, z, T = d.T
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].plot(e * 100, s, "-", lw=1.5); ax[0].set_ylabel("Stress [GPa] (t = 3.35 A)")
ax[0].set_title("Stress-strain")
ax[1].plot(e * 100, z, "-", color="C3"); ax[1].set_ylabel("Out-of-plane amplitude z_max - z_min [A]")
ax[1].set_title("Pop-up")
ax[2].plot(e * 100, T, "-", color="C2"); ax[2].set_ylabel("Temperature of free atoms [K]")
ax[2].set_title("Thermostat check")
for a in ax:
    a.set_xlabel("Nominal strain [%]"); a.grid(alpha=.3)
i = np.argmax(s)
print(f"peak stress {s[i]:.2f} GPa at strain {e[i]*100:.1f} %, max pop-up amplitude {z.max():.1f} A")
fig.tight_layout(); fig.savefig("kirigami_curves.png", dpi=140)
print("saved kirigami_curves.png")
