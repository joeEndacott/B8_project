# pylint: disable=invalid-name
"""
This script calculates and plots the X-ray form factor for an atom using a Gaussian 
interpolation and a hard shell approximation, and compares this result to the form 
factor calculated using xraydb.
"""

import numpy as np
import matplotlib.pyplot as plt
import xraydb

from B8_project import file_reading, crystal

# Parameters.
WAVELENGTH = 100
ATOMIC_NUMBER = 31
ELEMENT = "Ga"
MAX_DEFLECTION_ANGLE = 180

# Calculate the X-ray form factors for both approximations.
form_factor_gaussian = file_reading.read_xray_form_factors()[ATOMIC_NUMBER]
form_factor_hard_shell = file_reading.read_x_ray_form_factors_hard_shell()[
    ATOMIC_NUMBER
]

# Generate a range of deflection angles
deflection_angles = np.linspace(0, MAX_DEFLECTION_ANGLE, 1000)

# Calculate the corresponding RLV magnitudes.
rlv_magnitudes = crystal.ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
    deflection_angles, WAVELENGTH
)

# Calculate the form factor values.
gaussian_values = form_factor_gaussian.evaluate_form_factors(rlv_magnitudes)
hard_shell_values = form_factor_hard_shell.evaluate_form_factors(rlv_magnitudes)

# Generate Q-values
q_values = np.sin(deflection_angles * np.pi / 360) / WAVELENGTH

# Calculate the form factor values using xraydb.
xraydb_values = xraydb.f0(ATOMIC_NUMBER, q_values)

# Plot graph of form factor against RLV magnitude.
plt.figure(figsize=(10, 6))
plt.plot(rlv_magnitudes, gaussian_values, label="Gaussian interpolation")
plt.plot(rlv_magnitudes, hard_shell_values, label="Hard shell approximation")
plt.plot(rlv_magnitudes, xraydb_values, label="xraydb calculation")
plt.xlabel(r"$|\mathbf{G}|$")
plt.ylabel(r"$f \, (\mathbf{G})$")
plt.title(f"X-ray form factor for {ELEMENT}")
plt.legend()
plt.grid(True)
plt.savefig(f"results/X_ray_form_factor_{ELEMENT}_1.pdf", format="pdf")

# Plot graph of form factor against deflection angle.
plt.figure(figsize=(10, 6))
plt.plot(deflection_angles, gaussian_values, label="Gaussian interpolation")
plt.plot(deflection_angles, hard_shell_values, label="Hard shell approximation")
plt.plot(deflection_angles, xraydb_values, label="xraydb calculation")
plt.xlabel(r"$2 \theta (^{\circ})$")
plt.ylabel(r"$f \, (\mathbf{G})$")
plt.title(f"X-ray form factor for {ELEMENT}")
plt.legend()
plt.grid(True)
plt.savefig(f"results/X_ray_form_factor_{ELEMENT}_2.pdf", format="pdf")
