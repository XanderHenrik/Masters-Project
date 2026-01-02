import numpy as np
import matplotlib.pyplot as plt

"""
This script is meant to produce a single plot, comparing the scaling of the
different constituents of the Universe in one plot and show how they depend on
the scale factor.
"""

# Define the functions and scale factor domain we get from the theory
a = np.linspace(5e-5, 1, 100)

rho_crit_0 = (1.88 * (0.67)**2) * 10**(-29)
rho_m = (a**(-3))# + 0.3166 * rho_crit_0)
rho_r = (a**(-4))# + 9.4e-5 * rho_crit_0)
rho_de = (np.ones_like(a) * 0.68)

# Create the figure:
plt.figure()
plt.subplots_adjust(left=0.15)
# plt.loglog(a, (rho_m * 0.3166), color='g', label=r"$\rho_{m}$")
# plt.loglog(a, (rho_r * 9.4e-5), color='r', label=r"$\rho_{r}$")
# plt.loglog(a, (rho_de), color='b', label=r"$\rho_{\Lambda}$")

# equality scalefactors (determined by eye-balling)
plt.axvline(x=2.97e-4, linestyle=':', color='k', alpha=0.5, label=r"$a_{eq}$")
plt.axvline(x=7.75e-1, linestyle='--', color='k', alpha=0.5, label=r"$a_{\Lambda}$")

# Or in the color theme of the Neutrino hierarchy plot:
plt.loglog(a, (rho_m * 0.3166), color='royalblue', label=r"$\rho_{m}$")
plt.loglog(a, (rho_r * 9.4e-5), color='orangered', label=r"$\rho_{r}$")
plt.loglog(a, (rho_de), color='gold', label=r"$\rho_{\Lambda}$")


# plt.xscale('log')
plt.ylim(1e-2, 1e12)
plt.xlabel(r"$a(t)$", fontsize=14)
plt.ylabel(r"$\frac{\rho(t)}{\rho_{crit, 0}}$", fontsize=16)
plt.grid(which='both', alpha=0.25)
plt.legend(fontsize=12)
plt.savefig("rho_scaling_expansion_history.pdf")
plt.show()