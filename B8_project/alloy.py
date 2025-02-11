"""
Alloy
=====

This module contains code related to generating and representing alloys.
"""

from dataclasses import dataclass
from copy import deepcopy
import numpy as np

from B8_project.crystal import UnitCell


@dataclass
class SuperCell:
    """
    Super cell
    ==========

    A class that groups functions related to generating super cells from unit cells.

    Methods
    -------
    _get_lattice_vectors
        Returns the position vector of each unit cell in the super cell.
    new_super_cell
        Generates a new super cell.
    """

    @staticmethod
    def _get_lattice_vectors(side_lengths: tuple[int, int, int]):
        """
        Get lattice vectors
        ===================

        Returns the position vector of each unit cell in the super cell, in terms of
        the lattice constants.
        """
        x_length, y_length, z_length = side_lengths

        # Error handling for if any of the lengths are 0 or negative.
        if x_length <= 0 or y_length <= 0 or z_length <= 0:
            raise ValueError("All entries in side_lengths must be greater than 0.")

        # lattice_vectors stores the position vectors of the corners of the unit cells
        # comprising the super cell.
        lattice_vectors = np.array(
            [
                [x, y, z]
                for x in range(x_length)
                for y in range(y_length)
                for z in range(z_length)
            ]
        )

        sorted_lattice_vectors = lattice_vectors[np.lexsort(lattice_vectors.T)]

        return sorted_lattice_vectors

    @classmethod
    def new_super_cell(
        cls,
        unit_cell: UnitCell,
        side_lengths: tuple[int, int, int],
        material: str = "",
    ):
        """
        New super cell
        ==============

        Creates a new super cell, given a material, a unit cell and the side lengths of
        the super cell.

        Parameters
        ----------
        unit_cell : UnitCell
            Represents the unit cell of the crystal that is to be converted to a super
            cell. The `UnitCell` object stores a list of atoms and atomic positions in
            the unit cell of the crystal.
        side_lengths : tuple[int, int, int]
            The desired side lengths in the x, y and z directions of the super cell,
            specified in terms of the lattice constants of the unit cell.
        material : str
            The name of the material. The default value is the same name as the original
            crystal.
        """
        if material == "":
            material = unit_cell.material

        # Error handling for if any of the side lengths are less than or equal to 0.
        if side_lengths[0] <= 0 or side_lengths[1] <= 0 or side_lengths[2] <= 0:
            raise ValueError("All entries in side_lengths must be greater than 0.")

        lattice_vectors = SuperCell._get_lattice_vectors(side_lengths)

        lattice_constants = np.array(side_lengths) * unit_cell.lattice_constants

        atomic_numbers = unit_cell.atoms["atomic_numbers"]
        atomic_positions = unit_cell.atoms["positions"]

        # For each unit cell copy, duplicates all of the atoms and shifts their
        # positions.
        atomic_numbers = np.repeat(atomic_numbers, len(lattice_vectors))
        atomic_positions = (
            atomic_positions[:, np.newaxis, :] + lattice_vectors
        ) / np.array(side_lengths)

        # Define a custom datatype to represent atoms.
        dtype = np.dtype([("atomic_numbers", "i4"), ("positions", "3f8")])

        # Create a structured NumPy array to store the atoms.
        atoms = np.empty(len(atomic_numbers), dtype=dtype)
        atoms["atomic_numbers"] = atomic_numbers
        atoms["positions"] = atomic_positions.reshape(-1, 3)

        return UnitCell(material, lattice_constants, atoms)

    @staticmethod
    def apply_disorder(
        super_cell: UnitCell,
        target_atomic_number: int,
        substitute_atomic_number: int,
        concentration: float,
        lattice_constants_no_substitution: np.ndarray,
        lattice_constants_full_substitution: np.ndarray,
        material_name: str,
    ):
        """
        Apply disorder
        ==============

        Randomly replaces target atoms in a super cell with substitute atoms until a
        specified concentration is reached. The new disordered super cell is returned.

        The lattice constants of the ordered crystal with 0% substitution and the
        ordered crystal with 100% substitution must be specified. For example, if you
        want to produce a super cell for the alloy In(x)Ga(1-x)As, you should input the
        lattice constants of GaAs for the `lattice_constants_no_substitution`
        parameter, and the lattice constants of InAs for the
        `lattice_constants_full_substitution` parameter. A linear interpolation is then
        used to estimate the lattice constants of In(x)Ga(1-x)As.

        Parameters
        ----------
        TODO: add parameters.

        Returns
        -------
        TODO: add returns.
        """
        # Error handling.
        if concentration < 0 or concentration > 1:
            raise ValueError("concentration must be between 0 and 1")

        # Get a list of target atoms, and shuffle the list.
        atoms = deepcopy(super_cell.atoms)
        target_atoms = atoms[atoms["atomic_numbers"] == target_atomic_number]
        np.random.shuffle(target_atoms)

        # Calculate the number of substitute atoms.
        num_substitute_atoms = int(np.ceil(concentration * len(target_atoms)))

        # Replace the correct number of atoms in target_atoms with the substitute atoms.
        for i in range(num_substitute_atoms):
            target_atoms["atomic_numbers"][i] = substitute_atomic_number

        # Add the substituted atoms back into atoms
        atoms[atoms["atomic_numbers"] == target_atomic_number] = target_atoms

        # Calculate the side lengths of the super cell.
        side_lengths = super_cell.lattice_constants / lattice_constants_no_substitution

        # Calculate the concentration of substitute atoms.
        actual_concentration = float(num_substitute_atoms) / float(len(target_atoms))

        # Use a linear interpolation to calculate the lattice constants of the
        # disordered super cell.
        lattice_constants = (
            super_cell.lattice_constants
            + actual_concentration
            * side_lengths
            * (lattice_constants_full_substitution - lattice_constants_no_substitution)
        )

        return UnitCell(material_name, lattice_constants, atoms)
