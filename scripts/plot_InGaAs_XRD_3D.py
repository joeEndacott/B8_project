# pylint: disable=invalid-name
"""
This script calculates the ND pattern for the disordered alloy In(x)Ga(1-x)As for a 
range of different concentrations (x), and represents the data as a surface plot.
"""

import numpy as np
import plotly.graph_objects as go

from B8_project import file_reading, crystal, alloy, diffraction

# Concentration parameters
NUMBER_OF_CONCENTRATIONS = 2
MIN_CONCENTRATION = 0
MAX_CONCENTRATION = 1

# Diffraction parameters.
WAVELENGTH = 1
MIN_DEFLECTION_ANGLE = 20
MAX_DEFLECTION_ANGLE = 90
INTENSITY_CUTOFF = 0

# Plot parameters.
FILENAME = "results/InGaAs/InGaAs_3D_plot_4.html"
PEAK_WIDTH = 0.1
Z_AXIS_LOGARITHMIC = False
Z_AXIS_MIN = 0
Z_AXIS_MAX = 1

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

# Generate a range of concentrations.
concentrations = np.linspace(
    MIN_CONCENTRATION, MAX_CONCENTRATION, NUMBER_OF_CONCENTRATIONS
)

# Calculate the number of deflection angles using the same method as
# diffraction.get_diffraction_pattern().
num_deflection_angles = np.round(
    10 * (MAX_DEFLECTION_ANGLE - MIN_DEFLECTION_ANGLE) / PEAK_WIDTH
).astype(int)

# Generate an array of deflection angles.
deflection_angles = np.linspace(
    MIN_DEFLECTION_ANGLE, MAX_DEFLECTION_ANGLE, num_deflection_angles
)

# Initialise an array which stores the intensity data.
intensity_data = np.zeros(
    (NUMBER_OF_CONCENTRATIONS, num_deflection_angles),
)

for i, conc in enumerate(concentrations):
    # Name of the alloy.
    alloy_name = f"In({conc:.3f})Ga({1-conc:.3f})As"

    # Generate an InGaAs super cell.
    disordered_GaAs_super_cell = alloy.SuperCell.apply_disorder(
        super_cell=GaAs_super_cell,
        target_atomic_number=31,
        substitute_atomic_number=49,
        concentration=conc,
        lattice_constants_no_substitution=GaAs_unit_cell.lattice_constants,
        lattice_constants_full_substitution=InAs_unit_cell.lattice_constants,
        material_name=alloy_name,
    )

    # Get the diffraction pattern for the InGaAs cell.
    diffraction_pattern = diffraction.get_diffraction_pattern(
        unit_cell=disordered_GaAs_super_cell,
        diffraction_type="ND",
        neutron_form_factors=neutron_form_factors,
        x_ray_form_factors=x_ray_form_factors,
        wavelength=WAVELENGTH,
        min_deflection_angle=MIN_DEFLECTION_ANGLE,
        max_deflection_angle=MAX_DEFLECTION_ANGLE,
        peak_width=PEAK_WIDTH,
        intensity_cutoff=INTENSITY_CUTOFF,
    )

    intensity_data[i] = diffraction_pattern["intensities"]

    print(f"Computed intensities for x = {conc}")

# Create the figure.
fig = go.Figure()

# Plot the data as a series of 2D line plots displaced along the z-axis.
for i, conc in enumerate(concentrations):
    fig.add_trace(
        go.Scatter3d(
            x=np.full_like(deflection_angles, conc),
            y=deflection_angles,
            z=intensity_data[i],
            mode="lines",
            line=dict(color="blue"),
            opacity=0.5,
        )
    )

# Set axis labels.
fig.update_layout(
    scene=dict(
        xaxis_title="Concentration of In",
        yaxis_title="Deflection angle (°)",
        zaxis_title="Relative intensity",
    ),
    title="ND pattern for InGaAs",
    font=dict(size=15),
)

# Update the limits on the z-axis.
fig.update_layout(
    scene=dict(
        zaxis=dict(
            range=[Z_AXIS_MIN, Z_AXIS_MAX],
        ),
    )
)

fig.show()
# fig.write_html(FILENAME)
