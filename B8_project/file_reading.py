"""
File reading
============

This module contains functions which read crystal parameters and form factors from .csv 
files.

Functions
---------
    - read_basis: reads basis parameters from a .csv file and returns the atomic 
    numbers and positions of atoms in the basis.
    - read_lattice: reads lattice parameters from a .csv file and returns the 
    material, lattice type, and lattice constants.
    - get_neutron_scattering_lengths_from_csv: reads neutron scattering lengths from a 
    .csv file and returns a `Mapping` mapping atomic numbers to neutron scattering 
    lengths.
    - get_x_ray_form_factors_from_csv: reads X-ray form factors from a .csv file and 
    returns a `Mapping` mapping atomic numbers to X-ray form factors.
"""

from typing import Mapping
import pandas as pd

from B8_project.form_factor import (
    NeutronFormFactor,
    XRayFormFactor,
    XRayFormFactorHardShell,
)


def read_basis(
    filename: str = "data/basis.csv",
) -> tuple[list[int], list[tuple[float, float, float]]]:
    """
    Read basis parameters from a .csv file
    ======================================

    Reads basis parameters from a specified .csv file, and returns the atomic numbers
    and positions of atoms in the basis.

    Format of the .csv file
    -----------------------
    Each row of the .csv file corresponds to a different atom in the unit cell. The .csv
    file must contain the following columns:
        - "atomic_number" (int): The atomic number of the atom.
        - "x", "y", "z" (float): The position of the atom in the unit cell in terms of
        the lattice constants. E.g., (x, y, z) = (0.5, 0.5, 0.5) corresponds to a
        position (0.5*a, 0.5*b. 0.5*c), where a, b, c are the side lengths of the unit
        cell in the x, y, z directions respectively.

    Parameters
    ----------
    filename : str
        Filename of the .csv file containing the basis parameters. Default value is
        `"data/basis.csv"`
    """
    try:
        # Read the CSV file containing the lattice parameters into a DataFrame.
        basis_df = pd.read_csv(filename)

        # Expected columns of the DataFrame
        basis_columns = {"atomic_number", "x", "y", "z"}

        # Ensure the DataFrame contains the required columns.
        if not basis_columns.issubset(basis_df.columns):
            raise KeyError(
                f"The {filename} must contain the following columns: {basis_columns}"
            )

        # Read parameters from the DataFrame
        atomic_numbers = basis_df["atomic_number"].astype(int).tolist()
        atomic_positions = list(
            zip(
                basis_df["x"].astype(float).tolist(),
                basis_df["y"].astype(float).tolist(),
                basis_df["z"].astype(float).tolist(),
            )
        )

        # Check that atomic_numbers and atomic_positions have the same length
        if not len(atomic_numbers) == len(atomic_positions):
            raise ValueError(
                """atomic_numbers and atomic_positions must have the same length"""
            )

    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Error processing '{filename}': {exc}") from exc

    return atomic_numbers, atomic_positions


def read_lattice(
    filename: str = "data/lattice.csv",
) -> tuple[str, int, tuple[float, float, float]]:
    """
    Read lattice parameters from a .csv file
    ========================================

    Reads lattice parameters from a specified .csv file and returns the  material,
    lattice type, and lattice constants.

    The lattice can be any orthorhombic lattice (i.e. the side lengths of the unit cell
    can be different, but all angles must be 90 degrees).

    Format of the .csv file
    -----------------------
    The .csv file must contain the following columns:
        - "material" (str): Chemical formula of the crystal (e.g. "NaCl").
        - "lattice_type" (int): Integer (1 - 4 inclusive) that represents the Bravais lattice type.
            - 1 -> Simple.
            - 2 -> Body centred.
            - 3 -> Face centred.
            - 4 -> Base centred.
        - "a", "b", "c" (float): Side lengths of the unit cell in the x, y and z
        directions respectively in angstroms (Å).

    Parameters
    ----------
    filename : str
        Filename of the .csv file containing the lattice parameters. Default value is
        `"data/lattice.csv"`
    """
    try:
        # Read the CSV file containing the lattice parameters into a DataFrame.
        lattice_df = pd.read_csv(filename)

        # Expected columns of the DataFrame
        lattice_columns = {"material", "lattice_type", "a", "b", "c"}

        # Ensure the DataFrame contains the required columns.
        if not lattice_columns.issubset(lattice_df.columns):
            raise KeyError(
                f"The {filename} must contain the following columns: {lattice_columns}"
            )

        # Read parameters from DataFrame
        material = str(lattice_df.loc[0, "material"])
        lattice_type = int(pd.to_numeric(lattice_df.loc[0, "lattice_type"]))
        lattice_constants = (
            float(pd.to_numeric(lattice_df.loc[0, "a"])),
            float(pd.to_numeric(lattice_df.loc[0, "b"])),
            float(pd.to_numeric(lattice_df.loc[0, "c"])),
        )

    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Error processing '{filename}': {exc}") from exc

    return material, lattice_type, lattice_constants


def read_neutron_scattering_lengths(
    filename: str = "data/neutron_scattering_lengths.csv",
) -> Mapping[int, NeutronFormFactor]:
    """
    Read neutron scattering lengths from a .csv file
    ================================================

    Reads neutron scattering lengths from a specified .csv file and returns a
    `Mapping` mapping atomic numbers to neutron scattering lengths.

    The neutron scattering lengths are represented in the `Mapping` as instances of
    the `NeutronFormFactor` class.

    Format of the .csv file
    -----------------------
    Each row of the .csv file corresponds to a different atom. The .csv file must
    contain the following columns:
        - atomic_number (int): The atomic number of the atom.
        - neutron_scattering_length (float): The neutron scattering length of the
        atom, in femtometers (fm).

    Parameters
    ----------
    filename : str
        Filename of the .csv file containing the neutron scattering lengths. Default
        value is `"data/neutron_scattering_lengths.csv"`
    """
    try:
        # Read the CSV file containing the neutron scattering lengths into a DataFrame.
        neutron_df = pd.read_csv(filename)

        # Expected columns of the DataFrame.
        neutron_columns = {"atomic_number", "neutron_scattering_length"}

        # Ensure the DataFrame contains the required columns.
        if not neutron_columns.issubset(neutron_df.columns):
            raise ValueError(
                f"The {filename} must contain the following columns: {neutron_columns}"
            )

        # Read the atomic numbers.
        atomic_numbers = neutron_df["atomic_number"].astype(int).tolist()

        # Read the neutron scattering lengths, and store them as an instance of
        # NeutronFormFactor
        neutron_scattering_lengths = [
            NeutronFormFactor(x)
            for x in neutron_df["neutron_scattering_length"].astype(float).tolist()
        ]

        # Validate that atomic_numbers and neutron_scattering_lengths have the same
        # length.
        if not len(atomic_numbers) == len(neutron_scattering_lengths):
            raise ValueError(
                "atomic_numbers and neutron_scattering_lengths must have the same length"
            )

    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Error processing '{filename}': {exc}") from exc

    # Create a Mapping mapping the atomic numbers to the neutron scattering lengths
    length = len(atomic_numbers)
    return {atomic_numbers[i]: neutron_scattering_lengths[i] for i in range(length)}


def read_xray_form_factors(
    filename: str = "data/x_ray_form_factors.csv",
) -> Mapping[int, XRayFormFactor]:
    """
    Read X-ray form factors from a .csv file
    ========================================

    Reads X-ray form factors from a specified .csv file and returns a `Mapping`
    mapping atomic numbers to X-ray form factors.

    The X-ray form factors are represented in the `Mapping` as instances of the
    `XRayFormFactor` class.

    Format of the .csv file
    -----------------------
    Each row of the .csv file corresponds to a different atom. The .csv file must
    contain the following columns:
        - "atomic_number" (int): The atomic number of the atom.
        - "a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4", "c" (float): Parameters which
        specify the X-ray form factor of the atom. These correspond to the attributes
        of the `XRayFormFactor` class.

    Parameters
    ----------
    filename : str
        Filename of the .csv file containing the X-ray form factors. Default value is
        `"data/x_ray_form_factors.csv"`
    """
    try:
        # Read the CSV file containing the X-ray form factors into a DataFrame.
        xray_df = pd.read_csv(filename)

        # Expected columns of the CSV file
        xray_columns = {
            "atomic_number",
            "a1",
            "b1",
            "a2",
            "b2",
            "a3",
            "b3",
            "a4",
            "b4",
            "c",
        }

        # Ensure the CSV file contains the required columns.
        if not xray_columns.issubset(xray_df.columns):
            raise ValueError(
                f"The {filename} must contain the following columns: {xray_columns}"
            )

        # Read the atomic numbers
        atomic_numbers = xray_df["atomic_number"].astype(int).tolist()

        # Read the X-ray form factors
        xray_form_factors = [
            XRayFormFactor(a1, b1, a2, b2, a3, b3, a4, b4, c)
            for a1, b1, a2, b2, a3, b3, a4, b4, c in zip(
                xray_df["a1"].astype(float).tolist(),
                xray_df["b1"].astype(float).tolist(),
                xray_df["a2"].astype(float).tolist(),
                xray_df["b2"].astype(float).tolist(),
                xray_df["a3"].astype(float).tolist(),
                xray_df["b3"].astype(float).tolist(),
                xray_df["a4"].astype(float).tolist(),
                xray_df["b4"].astype(float).tolist(),
                xray_df["c"].astype(float).tolist(),
            )
        ]

        # Validate that atomic_numbers and xray_form_factors have the same length
        if not len(atomic_numbers) == len(xray_form_factors):
            raise ValueError(
                "atomic_numbers and form_factors must have the same length"
            )

    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Error processing '{filename}': {exc}") from exc

    # Create a Mapping mapping the atomic numbers to the X-ray form factors.
    length = len(atomic_numbers)
    return {atomic_numbers[i]: xray_form_factors[i] for i in range(length)}


def read_x_ray_form_factors_hard_shell(filename: str = "data/atomic_radii.csv"):
    """
    Read X-ray form factors hard shell
    ==================================

    TODO: add documentation.
    """
    try:
        # Read the CSV file containing the X-ray form factors into a DataFrame.
        xray_df = pd.read_csv(filename)

        # Expected columns of the CSV file
        xray_columns = {
            "atomic_number",
            "vdw_radius",
        }

        # Ensure the CSV file contains the required columns.
        if not xray_columns.issubset(xray_df.columns):
            raise ValueError(
                f"The {filename} must contain the following columns: {xray_columns}"
            )

        # Read the atomic numbers.
        atomic_numbers = xray_df["atomic_number"].astype(int).tolist()

        # Read the atomic radii.
        atomic_radii = xray_df["vdw_radius"].astype(float).tolist()

        # Validate that atomic_numbers and xray_form_factors have the same length
        if not len(atomic_numbers) == len(atomic_radii):
            raise ValueError(
                "atomic_numbers and atomic_radii must have the same length"
            )

        # Make a list of X-ray form factor objects.
        x_ray_form_factors = [
            XRayFormFactorHardShell(atomic_number, atomic_radius)
            for atomic_number, atomic_radius in zip(atomic_numbers, atomic_radii)
        ]

    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"Error processing '{filename}': {exc}") from exc

    # Create a Mapping mapping the atomic numbers to the X-ray form factors.
    length = len(atomic_numbers)
    return {atomic_numbers[i]: x_ray_form_factors[i] for i in range(length)}
