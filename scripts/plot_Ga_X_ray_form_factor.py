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
WAVELENGTH = 0.1
ATOMIC_NUMBER = 49
MAX_DEFLECTION_ANGLE = 60

# Calculate the X-ray form factors for both approximations.
x_ray_form_factors_gaussian = file_reading.read_xray_form_factors()
x_ray_form_factors_hard_shell = file_reading.read_x_ray_form_factors_hard_shell()

# Get the form factor for Ga.
Ga_form_factor_gaussian = x_ray_form_factors_gaussian[ATOMIC_NUMBER]
Ga_form_factor_hard_shell = x_ray_form_factors_hard_shell[ATOMIC_NUMBER]

# Generate a range of deflection angles
deflection_angles = np.linspace(0, MAX_DEFLECTION_ANGLE, 1000)

# Calculate the corresponding RLV magnitudes.
rlv_magnitudes = crystal.ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
    deflection_angles, WAVELENGTH
)

# Calculate the form factor values.
gaussian_values = Ga_form_factor_gaussian.evaluate_form_factors(rlv_magnitudes)
hard_shell_values = Ga_form_factor_hard_shell.evaluate_form_factors(rlv_magnitudes)

# Generate Q-values
q_values = np.sin(deflection_angles * np.pi / 360) / WAVELENGTH

# Calculate the form factor values using xraydb.
xraydb_values = xraydb.f0(ATOMIC_NUMBER, q_values)

plt.figure(figsize=(10, 6))
plt.plot(rlv_magnitudes, gaussian_values, label="Gaussian Form Factor")
plt.plot(
    rlv_magnitudes,
    hard_shell_values,
    label="Hard Shell Form Factor",
)
plt.plot(rlv_magnitudes, xraydb_values, label="X-ray DB calculation.")
plt.xlabel("Reciprocal Lattice Vector Magnitude")
plt.ylabel("Form Factor")
plt.title("Ga X-ray Form Factor")
plt.legend()
plt.grid(True)
plt.show()
