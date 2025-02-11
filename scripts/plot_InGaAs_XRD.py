# pylint: disable=invalid-name
"""
This script calculates and plots the XRD pattern for the disordered alloy 
In(x)Ga(1-x)As, for a range of different concentrations (x).
"""

import numpy as np

from B8_project import file_reading, crystal, alloy, diffraction

# Read GaAs parameters from .csv files.
GaAs_basis = file_reading.read_basis("data/GaAs_basis.csv")
GaAs_lattice = file_reading.read_lattice("data/GaAs_lattice.csv")
GaAs_unit_cell = crystal.UnitCell.new_unit_cell(GaAs_basis, GaAs_lattice)

# Generate a pure GaAs super cell.
GaAs_super_cell = alloy.SuperCell.new_super_cell(GaAs_unit_cell, (4, 4, 4))

# Read InAs parameters from .csv files.
InAs_basis = file_reading.read_basis("data/InAs_basis.csv")
InAs_lattice = file_reading.read_lattice("data/InAs_lattice.csv")
InAs_unit_cell = crystal.UnitCell.new_unit_cell(InAs_basis, InAs_lattice)

# Read form factors from .csv files.
neutron_form_factors = file_reading.read_neutron_scattering_lengths()
x_ray_form_factors = file_reading.read_xray_form_factors()

# Plot the diffraction patterns of pure GaAs and InAs for comparison with later results.
diffraction.plot_diffraction_pattern(
    GaAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    wavelength=0.1,
    min_deflection_angle=10,
    max_deflection_angle=90,
    intensity_cutoff=5e-3,
    peak_width=0.1,
    y_axis_logarithmic=True,
    line_width=1,
    file_path="results/InGaAs/",
)

diffraction.plot_diffraction_pattern(
    InAs_unit_cell,
    "XRD",
    neutron_form_factors,
    x_ray_form_factors,
    wavelength=0.1,
    min_deflection_angle=10,
    max_deflection_angle=90,
    intensity_cutoff=5e-3,
    peak_width=0.1,
    y_axis_logarithmic=True,
    line_width=1,
    file_path="results/InGaAs/",
)

# Generate a range of concentrations.
concentrations = np.linspace(0, 1, 11)

# For every concentration, generate a disordered super cell and calculate + plot the
# XRD pattern.
for conc in concentrations:
    # Name of the alloy.
    alloy_name = f"In({conc:.3f})Ga({1-conc:.3f})As"

    # Generate an InGaAs super cell.
    disordered_GaAs_super_cell = alloy.SuperCell.apply_disorder(
        GaAs_super_cell,
        target_atomic_number=31,
        substitute_atomic_number=49,
        concentration=conc,
        lattice_constants_no_substitution=GaAs_unit_cell.lattice_constants,
        lattice_constants_full_substitution=InAs_unit_cell.lattice_constants,
        material_name=alloy_name,
    )

    # Plot the XRD pattern of the InGaAs super cell.
    diffraction.plot_diffraction_pattern(
        disordered_GaAs_super_cell,
        "XRD",
        neutron_form_factors,
        x_ray_form_factors,
        wavelength=0.1,
        min_deflection_angle=10,
        max_deflection_angle=90,
        intensity_cutoff=5e-3,
        peak_width=0.1,
        y_axis_logarithmic=True,
        line_width=1,
        file_path="results/InGaAs/",
    )
