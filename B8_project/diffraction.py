"""
Diffraction
===========

TODO: update module docstring.
"""

from datetime import datetime
from typing import Mapping
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from B8_project import utils
from B8_project.crystal import UnitCell, ReciprocalSpace
from B8_project.form_factor import FormFactorProtocol, NeutronFormFactor, XRayFormFactor
from B8_project.alloy import SuperCell


def _calculate_structure_factors(
    unit_cell: UnitCell,
    form_factors: Mapping[int, FormFactorProtocol],
    reciprocal_lattice_vectors: np.ndarray,
) -> np.ndarray:
    """
    Calculate structure factors
    ===========================

    Calculates the structure factor of a crystal for a specified range of
    reciprocal lattice vectors, and returns the structure factors as a NumPy array.
    """
    # Extract atomic numbers and positions.
    atomic_numbers = unit_cell.atoms["atomic_numbers"]
    positions = unit_cell.atoms["positions"]

    # Initialize the structure factors array.
    structure_factors = np.zeros(
        reciprocal_lattice_vectors.shape[0], dtype=np.complex128
    )

    # Iterate over unique atomic numbers to evaluate form factors
    for atomic_number in np.unique(atomic_numbers):
        try:
            form_factor = form_factors[atomic_number]
        except KeyError as exc:
            raise KeyError(f"Error reading form factor Mapping: {exc}") from exc

        # Evaluate the form factor for all RLVs.
        form_factor_values = form_factor.evaluate_form_factors(
            reciprocal_lattice_vectors["magnitudes"]
        )

        # Get the positions of all of the current atoms
        mask = atomic_numbers == atomic_number
        current_positions = positions[mask]

        # Calculate exponents for all current atoms and Miller indices at once
        exponents = (2 * np.pi * 1j) * np.dot(
            reciprocal_lattice_vectors["miller_indices"], current_positions.T
        )

        # Sum the contribution from all atoms in the UC with the current atomic number.
        structure_factors += np.sum(
            form_factor_values[:, np.newaxis] * np.exp(exponents), axis=1
        )

    return structure_factors


def _calculate_diffraction_peaks(
    unit_cell: UnitCell,
    form_factors: Mapping[int, FormFactorProtocol],
    wavelength: float,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    intensity_cutoff: float = 1e-6,
) -> np.ndarray:
    """
    Calculate diffraction peaks
    ===========================

    Calculates the miller indices, deflection angle and intensity of every peak in the
    diffraction pattern of a specified crystal, and returns this data as a structured
    NumPy array.

    Array format
    ------------
    The structured NumPy array representing the diffraction peaks has the following
    fields:
        - 'miller_indices': Ndarray representing the Miller indices (h, k, l) of the
        peak.
        - 'deflection_angle': Float equal to the deflection angle of the peak.
        - 'intensities': Float equal to the (normalized) intensity of the peak.
        - 'multiplicities': Int equal to the multiplicity value of a peak.
    """
    # Error handling.
    if not (min_deflection_angle >= 0 and max_deflection_angle > 0):
        raise ValueError(
            """min_deflection_angle and max_deflection_angle should be greater than
            or equal to 0."""
        )
    if max_deflection_angle <= min_deflection_angle:
        raise ValueError(
            "max_deflection_angle should be larger than min_deflection_angle"
        )

    # Calculate the minimum and maximum RLV magnitudes
    try:
        min_magnitude = ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
            np.array(min_deflection_angle), wavelength
        )
        max_magnitude = ReciprocalSpace.rlv_magnitudes_from_deflection_angles(
            np.array(max_deflection_angle), wavelength
        )
    except ValueError as exc:
        raise ValueError(
            f"Error calculating RLV max and min magnitudes: {exc}"
        ) from exc

    # Generate an array of all reciprocal lattice vectors with valid magnitudes.
    try:
        reciprocal_lattice_vectors = ReciprocalSpace.get_reciprocal_lattice_vectors(
            float(min_magnitude),
            float(max_magnitude),
            np.array(unit_cell.lattice_constants),
        )
    except ValueError as exc:
        raise ValueError(f"Error generating reciprocal lattice vectors: {exc}") from exc

    # Generate an array of deflection angles.
    deflection_angles = ReciprocalSpace.deflection_angles_from_rlv_magnitudes(
        reciprocal_lattice_vectors["magnitudes"], wavelength
    )

    # Calculate the structure factor for each reciprocal lattice vectors
    structure_factors = _calculate_structure_factors(
        unit_cell, form_factors, reciprocal_lattice_vectors
    )

    # Calculate the intensity of each peak and normalize the intensities
    intensities = np.abs(structure_factors) ** 2
    relative_intensities = intensities / np.max(intensities)

    # Define a custom datatype to represent intensity peaks.
    dtype = np.dtype(
        [
            ("miller_indices", "3i4"),
            ("deflection_angles", "f8"),
            ("intensities", "f8"),
            ("multiplicities", "i4"),
        ]
    )

    # Create a structured NumPy array to store the intensity peaks.
    diffraction_peaks = np.empty(relative_intensities.shape[0], dtype=dtype)
    diffraction_peaks["miller_indices"] = reciprocal_lattice_vectors["miller_indices"]
    diffraction_peaks["deflection_angles"] = deflection_angles
    diffraction_peaks["intensities"] = relative_intensities
    diffraction_peaks["multiplicities"] = np.ones(len(relative_intensities))

    # Remove duplicate angles and sum the intensities of duplicate peaks.
    diffraction_peaks = _merge_peaks(diffraction_peaks, intensity_cutoff)

    return diffraction_peaks


def _merge_peaks(
    diffraction_peaks: np.ndarray, intensity_cutoff: float = 1e-6
) -> np.ndarray:
    """
    Merge peaks
    ===========

    Filters an array of diffraction peaks, and returns the filtered array. The
    filtering is done in two steps. First, all peaks which occur at the same deflection
    angle (to within a given tolerance) are merged. Second, the remaining peaks are
    normalized. Finally, any peaks which have an intensity smaller than the intensity
    cutoff are removed.
    """
    # Relative tolerance for comparing deflection angles.
    angle_tolerance = 1e-10

    # Sort diffraction_peaks based on the deflection angle.
    diffraction_peaks.sort(order="deflection_angles")

    # Initialise a mask to keep track of duplicate peaks.
    mask = np.ones(len(diffraction_peaks), dtype=bool)

    # Iterate over all diffraction peaks.
    i = 0
    length = len(diffraction_peaks)
    while i < length - 1:
        # Find all peaks with similar deflection angles and merge them.
        j = i + 1
        while j < length and np.isclose(
            diffraction_peaks["deflection_angles"][i],
            diffraction_peaks["deflection_angles"][j],
            rtol=angle_tolerance,
        ):
            # Merge intensities.
            diffraction_peaks["intensities"][i] += diffraction_peaks["intensities"][j]

            # Update multiplicity values.
            diffraction_peaks["multiplicities"][i] += 1

            # Mark peak as a duplicate.
            mask[j] = False
            j += 1
        i = j

    # Remove duplicate intensities
    diffraction_peaks = diffraction_peaks[mask]

    # Normalize the intensities
    max_intensity = diffraction_peaks["intensities"].max()
    diffraction_peaks["intensities"] /= max_intensity

    # Remove peaks with intensities below the cutoff.
    diffraction_peaks = diffraction_peaks[
        diffraction_peaks["intensities"] >= intensity_cutoff
    ]

    # Take the absolute value of the miller indices for each peak and sort from largest
    # to smallest.
    for i in range(len(diffraction_peaks)):
        diffraction_peaks["miller_indices"][i] = np.sort(
            diffraction_peaks["miller_indices"][i]
        )[::-1]

    return diffraction_peaks


def get_miller_peaks(
    unit_cell: UnitCell,
    diffraction_type: str,
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    intensity_cutoff: float = 1e-6,
    print_peak_data: bool = False,
    save_to_csv: bool = False,
) -> np.ndarray:
    """
    Get miller peaks
    ================

    Tabulates and returns information about the intensity peaks of the diffraction
    pattern for a specified crystal. If desired, the peak data can be printed or saved
    to a .csv file.

    Parameters
    ----------
    unit_cell : UnitCell
        Represents the desired crystal.
    diffraction_type : str
        Should have a value of "ND" for neutron diffraction, or "XRD" for X-ray
        diffraction.
    neutron_form_factors : Mapping[int, NeutronFormFactor]
        A mapping from atomic numbers to a class which represents a neutron form
        factor.
    x_ray_form_factors : Mapping[int, XRayFormFactor]
        A mapping from atomic numbers to a class which represents an X-ray form
        factor.
    wavelength : float
        The wavelength of incident particles, given in angstroms (Å).
    min_deflection_angle : float, optional
        The minimum deflection angle to be plotted. Default is 10.
    max_deflection_angle : float, optional
        The maximum deflection angle to be plotted. Default is 170.
    intensity_cutoff : float, optional
        The minimum intensity required for a peak to be tabulated. Default is 1e-6.
    print_peak_data : bool, optional
        If True, print the peak data. Default is False.
    save_to_csv : bool, optional
        If True, save the peak data to a .csv file. Default is False.

    Returns
    -------
    np.ndarray
        A structured NumPy array that stores the peak data. The array has the following fields:
        - 'miller_indices': An np.ndarray representing the Miller indices (h, k, l) of
        the peak.
        - 'deflection_angle': The deflection angle of the peak.
        - 'intensities': The (normalized) intensity of the peak.
    """
    if diffraction_type == "ND":
        diffraction_peaks = _calculate_diffraction_peaks(
            unit_cell,
            neutron_form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
            intensity_cutoff,
        )
    elif diffraction_type == "XRD":
        diffraction_peaks = _calculate_diffraction_peaks(
            unit_cell,
            x_ray_form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
            intensity_cutoff,
        )
    else:
        raise ValueError("Invalid diffraction type")

    if print_peak_data is True:
        print(f"\n{unit_cell.material} diffraction peaks.")
        for i, peak in enumerate(diffraction_peaks):
            print(
                f"Peak {i+1}: "
                f"[h k l] = {peak[0]}; deflection angle = {np.round(peak[1], 2)}°; "
                f"relative intensity = {round(peak[2], 4)}"
            )

    if save_to_csv is True:
        raise ValueError("Saving peak data to .csv is not currently supported")

    return diffraction_peaks


def get_diffraction_pattern(
    unit_cell: UnitCell,
    diffraction_type: str,
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float = 1,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    peak_width: float = 0.1,
    intensity_cutoff: float = 1e-6,
) -> np.ndarray:
    """
    Get diffraction pattern
    =======================

    Calculates the diffraction pattern for a crystal, and returns a structured NumPy
    array containing deflection angles and intensities.

    Parameters
    ----------
    unit_cell : UnitCell
        The unit cell of the chosen crystal.
    diffraction_type : str
        The type of diffraction desired. Should be either `"ND"` for neutron
        diffraction, or `"XRD"` for X-ray diffraction.
    neutron_form_factors : Mapping[int, NeutronFormFactor]
        A mapping from atomic numbers to a class which represents a neutron form factor.
    x_ray_form_factors : Mapping[int, XRayFormFactor]
        A mapping from atomic numbers to a class which represents an X-ray form factor.
    wavelength : float
        The wavelength of incident particles, given in angstroms (Å). The default value
        is 1Å.
    min_deflection_angle, max_deflection_angle : float
        These parameters specify the range of deflection angles to be plotted.
    peak_width : float
        The width of the intensity peaks. This parameter is only used for plotting. A
        value should be chosen so that all diffraction peaks can be observed. The
        default value is 0.1°.
    intensity_cutoff : float
        The minimum intensity required for a peak to be registered. The default value
        is 1e-6.

    Returns
    -------
    np.ndarray
        A structured NumPy array that contains the diffraction pattern data. The array
        has the following fields:
            - 'deflection_angles': A list of all sampled deflection angles.
            - 'intensities': The intensity at each deflection angle.
    """
    # Find the diffraction peaks.
    if diffraction_type == "ND":
        try:
            diffraction_peaks = _calculate_diffraction_peaks(
                unit_cell,
                neutron_form_factors,
                wavelength,
                min_deflection_angle,
                max_deflection_angle,
                intensity_cutoff,
            )
        except Exception as exc:
            raise ValueError(f"Error finding diffraction peaks: {exc}") from exc

    elif diffraction_type == "XRD":
        try:
            diffraction_peaks = _calculate_diffraction_peaks(
                unit_cell,
                x_ray_form_factors,
                wavelength,
                min_deflection_angle,
                max_deflection_angle,
                intensity_cutoff,
            )
        except Exception as exc:
            raise ValueError(f"Error finding diffraction peaks: {exc}") from exc

    else:
        raise ValueError("Invalid diffraction type.")

    # Calculate a sensible number of points
    num_points = np.round(
        10 * (max_deflection_angle - min_deflection_angle) / peak_width
    ).astype(int)

    # Get x coordinates of plotted points.
    x_values = np.linspace(min_deflection_angle, max_deflection_angle, num_points)

    # Get y coordinates of plotted points.
    y_values = np.zeros_like(x_values)

    gaussian_peaks = utils.gaussian(
        x_values[:, np.newaxis],
        diffraction_peaks["deflection_angles"],
        peak_width,
        diffraction_peaks["intensities"],
    )

    y_values += np.sum(gaussian_peaks, axis=1)

    # Define a custom datatype to represent the diffraction data.
    dtype = np.dtype(
        [
            ("deflection_angles", "f8"),
            ("intensities", "f8"),
        ]
    )

    # Create a structured NumPy array to store the diffraction pattern.
    diffraction_pattern = np.empty(x_values.shape[0], dtype=dtype)
    diffraction_pattern["deflection_angles"] = x_values
    diffraction_pattern["intensities"] = y_values

    return diffraction_pattern


def plot_diffraction_pattern(
    unit_cell: UnitCell,
    diffraction_type: str,
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float = 1,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    intensity_cutoff: float = 1e-6,
    peak_width: float = 0.1,
    line_width: float = 1.0,
    y_axis_min: float = 0,
    y_axis_max: float = 1,
    file_path: str = "results/",
):
    """
    Plot diffraction pattern
    ========================

    Plots the diffraction pattern for a given crystal and saves the plot as a .pdf
    file in a specified directory.

    Name of .pdf file
    -----------------
    For neutron diffraction, the .pdf file has the following name:
    `"<material>_<ND>_<date>.pdf"`, where `"ND"` stands for Neutron Diffraction. For
    X-ray  diffraction, the .pdf file has the following name:
    `"<material>_<XRD>_<date>.pdf"`, where `"XRD"` stands for X-Ray Diffraction.

    Parameters
    ----------
    unit_cell : UnitCell
        The unit cell of the chosen crystal.
    diffraction_type : str
        The type of diffraction desired. Should be either `"ND"` for neutron
        diffraction, or `"XRD"` for X-ray diffraction.
    neutron_form_factors : Mapping[int, NeutronFormFactor]
        A mapping from atomic numbers to a class which represents a neutron form factor.
    x_ray_form_factors : Mapping[int, XRayFormFactor]
        A mapping from atomic numbers to a class which represents an X-ray form factor.
    wavelength : float
        The wavelength of incident particles, given in angstroms (Å). The default value
        is 1Å.
    min_deflection_angle, max_deflection_angle : float
        These parameters specify the range of deflection angles to be plotted.
    intensity_cutoff : float
        The minimum intensity needed for a peak to be registered. The default value is
        1e-6.
    peak_width : float
        The width of the intensity peaks. This parameter is only used for plotting. A
        value should be chosen so that all diffraction peaks can be observed. The
        default value is 0.1°.
    line_width : float
        The linewidth of the plot. Default value is 1.
    y_axis_min, y_axis_max : float
        The limits of the y axis of the plot.
    file_path : str
        The path to the directory where the plot will be stored. Default value is
        `"results/"`.

    Todos
    -----
    TODO: give the user the option to display the plot, and the option to not save the
    plot as a .pdf file.
    """
    # Get the diffraction pattern.
    try:
        diffraction_pattern = get_diffraction_pattern(
            unit_cell,
            diffraction_type,
            neutron_form_factors,
            x_ray_form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
            peak_width,
            intensity_cutoff,
        )
    except Exception as exc:
        raise ValueError(f"Error getting diffraction pattern: {exc}") from exc

    # Get today's date and format as a string.
    today = datetime.today()
    date_string = today.strftime("%d-%m-%Y")

    # Filename
    filename = f"{unit_cell.material}_{diffraction_type}_{date_string}"

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data.
    ax.plot(
        diffraction_pattern["deflection_angles"],
        diffraction_pattern["intensities"],
        label=f"{unit_cell.material}, {diffraction_type}, "
        f"λ = {round(wavelength, 4)}Å",
        linewidth=line_width,
    )

    # Set axis labels.
    ax.set_xlabel("Deflection angle (°)", fontsize=11)
    ax.set_ylabel("Relative intensity", fontsize=11)

    # Set y-axis limits.
    ax.set_ylim(y_axis_min, y_axis_max)

    # Set title.
    ax.set_title(
        f"Diffraction pattern for {unit_cell.material} ({diffraction_type}).",
        fontsize=15,
    )

    # Add legend.
    ax.legend()

    # Add grid lines.
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Customize the tick marks.
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="both", which="minor", length=4, color="gray")

    # Add minor ticks.
    ax.minorticks_on()

    # Adjust layout to prevent clipping.
    fig.tight_layout()

    # Save the figure.
    fig.savefig(f"{file_path}{filename}.pdf", format="pdf")

    # Print the path to the .pdf file.
    print(f"Plot created at {file_path}{filename}.pdf")


def plot_superimposed_diffraction_patterns(
    unit_cells_with_diffraction_types: list[tuple[UnitCell, str]],
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float = 1,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    intensity_cutoff: float = 1e-6,
    peak_width: float = 0.1,
    variable_wavelength: bool = False,
    line_width: float = 1.0,
    opacity: float = 0.5,
    y_axis_min: float = 0,
    y_axis_max: float = 1,
    file_path: str = "results/",
) -> None:
    """
    Plot superimposed diffraction patterns
    ======================================

    Plots the diffraction pattern of a list of crystals on the same plot. Each crystal
    is represented as an instance of `UnitCell`, and the diffraction type (neutron or
    X-ray) should be specified for each crystal.

    Name of .pdf file
    -----------------
    The filename consists of the chemical formula of each crystal followed by the
    diffraction type. For instance, suppose that we wanted to plot the ND pattern of
    NaCl and the XRD pattern of CsCl. The filename would then be
    `"NaCl_ND_CsCl_XRD_<date>"`.

    Parameters
    ----------
    unit_cells_with_diffraction_types : list[tuple[UnitCell, str]]
        Each element in the list is a tuple (`unit_cell`, `diffraction_type`).
        `unit_cell` is an instance of `UnitCell`, and represents a crystal.
        `diffraction_type` is a string. `diffraction_type` should be `"ND"` for neutron
        diffraction or `"XRD"` for X-ray diffraction.
    neutron_form_factors : Mapping[int, NeutronFormFactor]
        A mapping from atomic numbers to a class which represents a neutron form factor.
    x_ray_form_factors : Mapping[int, XRayFormFactor]
        A mapping from atomic numbers to a class which represents an X-ray form factor.
    wavelength : float
        The wavelength of incident particles, given in angstroms (Å). Default value
        is 1Å.
    min_deflection_angle, max_deflection_angle : float
        These parameters specify the range of deflection angles to be plotted. Default
        values are 10°, 170° respectively.
    intensity_cutoff : float
        The minimum intensity needed for a peak to be registered. The default value is
        1e-6.
    peak_width : float
        The width of the intensity peaks. This parameter is only used for plotting. A
        value should be chosen so that all diffraction peaks can be observed. The
        default value is 0.1°.
    variable_wavelength : bool
        False (default) -> Each plot uses the same wavelength. True -> the first plot
        uses the wavelength specified when the function is called, and the other plots
        use different wavelengths, such that the peaks for all of the plots overlap.
    line_width : float
        The linewidth of each curve. Default value is 1.
    opacity : float
        The opacity of each curve. Default value is 0.5.
    y_axis_min, y_axis_max : float
        The limits of the y axis of the plot.
    file_path : str
        The path to the directory where the plot will be stored. Default value is `"results/"`.

    Todos
    -----
    TODO: give the user the option to display the plot, and the option to not save the
    plot as a .pdf file.
    """
    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the diffraction pattern for each crystal.
    for unit_cell, diffraction_type in unit_cells_with_diffraction_types:
        # Calculate the wavelength for the current crystal.
        if variable_wavelength is True:
            current_wavelength = (
                wavelength
                * unit_cell.lattice_constants[0]
                / unit_cells_with_diffraction_types[0][0].lattice_constants[0]
            )
        else:
            current_wavelength = wavelength

        # Get points to plot for ND.
        if diffraction_type == "ND":
            try:
                diffraction_pattern = get_diffraction_pattern(
                    unit_cell,
                    diffraction_type,
                    neutron_form_factors,
                    x_ray_form_factors,
                    current_wavelength,
                    min_deflection_angle,
                    max_deflection_angle,
                    peak_width,
                    intensity_cutoff,
                )
            except Exception as exc:
                raise ValueError(f"Error getting points to plot: {exc}") from exc

        # Get points to plot for XRD
        elif diffraction_type == "XRD":
            try:
                diffraction_pattern = get_diffraction_pattern(
                    unit_cell,
                    diffraction_type,
                    neutron_form_factors,
                    x_ray_form_factors,
                    current_wavelength,
                    min_deflection_angle,
                    max_deflection_angle,
                    peak_width,
                    intensity_cutoff,
                )
            except Exception as exc:
                raise ValueError(f"Error getting points to plot: {exc}") from exc
        else:
            raise ValueError("Invalid diffraction type.")

        # Plot the points.
        try:
            ax.plot(
                diffraction_pattern["deflection_angles"],
                diffraction_pattern["intensities"],
                label=f"{unit_cell.material}, {diffraction_type}"
                f"λ = {round(current_wavelength, 4)}Å",
                linewidth=line_width,
                alpha=opacity,
            )
        except Exception as exc:
            raise ValueError(f"Error plotting points: {exc}") from exc

    # Get today's date and format as a string.
    today = datetime.today()
    date_string = today.strftime("%d-%m-%Y")

    # Create a string for the plot title.
    title_string = "Diffraction pattern for "
    for unit_cell, diffraction_type in unit_cells_with_diffraction_types:
        title_string += f"{unit_cell.material} ({diffraction_type}), "

    title_string = title_string[:-2]
    title_string += "."

    # Create a string for the filename.
    filename = ""
    for unit_cell, diffraction_type in unit_cells_with_diffraction_types:
        filename += f"{unit_cell.material}_{diffraction_type}_"

    filename += f"{date_string}"

    # Set title.
    ax.set_title(title_string, fontsize=15)

    # Set axis labels.
    ax.set_xlabel("Deflection angle (°)", fontsize=11)
    ax.set_ylabel("Relative intensity", fontsize=11)

    # Set y-axis limits.
    ax.set_ylim(y_axis_min, y_axis_max)

    # Add legend.
    plt.legend()

    # Add grid lines.
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Customize the tick marks.
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="both", which="minor", length=4, color="gray")

    # Add minor ticks.
    ax.minorticks_on()

    # Adjust layout to prevent clipping.
    fig.tight_layout()

    # Save the figure.
    fig.savefig(f"{file_path}{filename}.pdf", format="pdf")

    # Print the path to the .pdf file.
    print(f"Plot created at {file_path}{filename}.pdf")


def plot_disordered_diffraction_pattern_3d(
    unit_cell_no_substitution: UnitCell,
    unit_cell_full_substitution: UnitCell,
    target_atomic_number: int,
    substitute_atomic_number: int,
    concentrations: list[float],
    super_cell_side_lengths: tuple[int, int, int],
    alloy_name: str,
    diffraction_type: str,
    neutron_form_factors: Mapping[int, NeutronFormFactor],
    x_ray_form_factors: Mapping[int, XRayFormFactor],
    wavelength: float = 1,
    min_deflection_angle: float = 10,
    max_deflection_angle: float = 170,
    intensity_cutoff: float = 1e-6,
    peak_width: float = 0.1,
    line_width: float = 1.0,
    z_axis_min: float = 0,
    z_axis_max: float = 1,
    filename: str = "results/disordered_alloy_3D_plot.html",
):
    """
    Plot disordered diffraction pattern 3D
    ======================================

    TODO: add documentation.
    """
    # Error handling
    if max(concentrations) > 1 or min(concentrations) < 0:
        raise ValueError("Concentration must be between 0 and 1.")

    # Generate a pure super cell.
    pure_super_cell = SuperCell.new_super_cell(
        unit_cell_no_substitution,
        super_cell_side_lengths,
        unit_cell_no_substitution.material,
    )

    # Sort the concentrations from smallest to largest, so that the plot looks sensible.
    concentrations.sort()

    # Calculate the number of deflection angles using the same method as
    # diffraction.get_diffraction_pattern().
    num_deflection_angles = np.round(
        10 * (max_deflection_angle - min_deflection_angle) / peak_width
    ).astype(int)

    # Generate an array of deflection angles.
    deflection_angles = np.linspace(
        min_deflection_angle, max_deflection_angle, num_deflection_angles
    )

    # Initialise an array which stores the intensity data.
    intensity_data = np.zeros(
        (len(concentrations), num_deflection_angles),
    )

    for i, conc in enumerate(concentrations):
        # Name of the alloy is set equal to the concentration, for debugging purposes.
        alloy_name = f"conc={conc}"

        # Generate a disordered super cell.
        disordered_super_cell = SuperCell.apply_disorder(
            pure_super_cell,
            target_atomic_number,
            substitute_atomic_number,
            conc,
            unit_cell_no_substitution.lattice_constants,
            unit_cell_full_substitution.lattice_constants,
            alloy_name,
        )

        # Get the diffraction pattern for the super cell.
        diffraction_pattern = get_diffraction_pattern(
            disordered_super_cell,
            diffraction_type,
            neutron_form_factors,
            x_ray_form_factors,
            wavelength,
            min_deflection_angle,
            max_deflection_angle,
            peak_width,
            intensity_cutoff,
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
                linewidth=line_width,
            )
        )

    # Set axis labels.
    fig.update_layout(
        scene=dict(
            xaxis_title="Concentration of substitute atoms",
            yaxis_title="Deflection angle (°)",
            zaxis_title="Relative intensity",
        ),
        title=f"{diffraction_type} pattern for {alloy_name}",
        font=dict(size=15),
    )

    # Update the limits on the z-axis.
    fig.update_layout(
        scene=dict(
            zaxis=dict(
                range=[z_axis_min, z_axis_max],
            ),
        )
    )

    fig.show()
    fig.write_html(filename)
