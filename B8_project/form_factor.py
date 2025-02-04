"""
Form factor
===========

This module contains classes which represent atomic form factors. Classes which 
represent form factors should implement the interface defined by FormFactorProtocol.

Classes
-------
    - FormFactorProtocol: This protocol defines the interface for any class that 
    represents a form factor.
    - NeutronFormFactor: A class to represent the neutron form factor of an atom.
    - XRayFormFactor: A class to represent the X-ray form factor of an atom.

TODO: update this documentation.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class FormFactorProtocol(Protocol):
    """
    Form factor protocol
    ====================

    This protocol defines the interface for any class that represents a form factor.
    Form factor classes must implement the `evaluate_form_factor` method.
    """

    def evaluate_form_factors(
        self, reciprocal_lattice_vector_magnitudes: np.ndarray
    ) -> np.ndarray:
        """
                Evaluate form factor
                ====================

                Calculates the form factors for a range of reciprocal lattice vectors The
        method used to calculate the form varies depending on the class that implements
                the form factor interface.
        """
        ...  # pylint: disable=W2301


@dataclass
class NeutronFormFactor:
    """
    Neutron form factor
    ===================

    A class to represent the neutron form factor of an atom.

    The neutron form factor is proportional to the neutron scattering length. Since we
    are only interested in relative intensities, we do not make a distinction
    between the neutron form factor and the neutron scattering length of an atom.

    Attributes
    ----------
        - neutron_scattering_length (float): The neutron scattering length of an atom.

    Methods
    -------
    """

    neutron_scattering_length: float

    def evaluate_form_factors(
        self,
        reciprocal_lattice_vector_magnitudes: np.ndarray,  # pylint: disable=W0613
    ) -> np.ndarray:
        """
        Evaluate neutron form factor
        ============================

        Evaluates a neutron form factor.
        """
        # Create a NumPy array to store the neutron form factors.
        form_factors = self.neutron_scattering_length * np.ones(
            reciprocal_lattice_vector_magnitudes.shape[0], dtype=float
        )

        return form_factors


@dataclass
class XRayFormFactor:
    """
    X-ray form factor
    ==================

    A class to represent the X-ray form factor of an atom.

    The X-ray form factor of an atom can be approximated by a sum of four Gaussian
    functions and a constant term.
    Each Gaussian has a height and a width, which gives us nine total parameters.
    An instance of `XRayFormFactor` stores these nine parameters, allowing the form
    factor to be calculated.

    For more information, see this website:
    TODO: add link.

    Attributes
    ----------
        - a1, a2, a3, a4 (float): The height of Gaussian 1, 2, 3, 4 respectively.
        - b1, b2, b3, b4 (float): Inversely proportional to the to the squared width of
        Gaussian 1, 2, 3, 4 respectively.
        - c (float): The constant term.

    Methods
    -------
        - evaluate_form_factor:
    """

    a1: float
    b1: float
    a2: float
    b2: float
    a3: float
    b3: float
    a4: float
    b4: float
    c: float

    def evaluate_form_factors(
        self, reciprocal_lattice_vector_magnitudes: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate X-ray form factor
        ==========================

        Evaluates an X-ray form factor.
        """
        a = [self.a1, self.a2, self.a3, self.a4]
        b = [self.b1, self.b2, self.b3, self.b4]
        c = self.c

        # Create an empty NumPy array to store the form factors.
        form_factors = np.zeros(
            reciprocal_lattice_vector_magnitudes.shape[0], dtype=float
        )

        # Evaluate the form factor for each reciprocal lattice vector
        for i in range(4):
            form_factors += a[i] * np.exp(
                -b[i] * (reciprocal_lattice_vector_magnitudes / (4 * np.pi)) ** 2
            )
        form_factors += c

        return form_factors
