"""
This module contains unit tests for the form_factor module.
"""

import numpy as np
from B8_project import form_factor


class TestFormFactorProtocol:
    """
    Unit tests for the form factor protocol
    """

    @staticmethod
    def test_neutron_form_factor_implements_protocol():
        """
        A unit test to check if the `NeutronFormFactor` class implements the form
        factor protocol
        """
        neutron_form_factor = form_factor.NeutronFormFactor(
            neutron_scattering_length=1.0
        )
        assert isinstance(neutron_form_factor, form_factor.FormFactorProtocol)

    @staticmethod
    def test_x_ray_form_factor_implements_protocol():
        """
        A unit test to check if the `XRayFormFactor` class implements the form
        factor protocol
        """
        x_ray_form_factor = form_factor.XRayFormFactor(1, 1, 1, 1, 1, 1, 1, 1, 1)
        assert isinstance(x_ray_form_factor, form_factor.FormFactorProtocol)


class TestNeutronFormFactor:
    """
    Unit tests for the `NeutronFormFactor` class.
    """

    @staticmethod
    def test_neutron_form_factor_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `NeutronFormFactor` instance.
        This unit test tests initialization with normal attributes.
        """
        neutron_form_factor = form_factor.NeutronFormFactor(1)
        assert neutron_form_factor.neutron_scattering_length == 1

    @staticmethod
    def test_form_factor_normal_operation():
        """
        A unit test for the form_factor function. This unit test tests normal
        operation of the function.
        """
        neutron_form_factor = form_factor.NeutronFormFactor(1)
        rlv_magnitudes = np.linspace(1, 10, 100)
        assert np.array_equal(
            neutron_form_factor.evaluate_form_factors(rlv_magnitudes), np.full(100, 1)
        )


class TestXRayFormFactor:
    """
    Unit tests for the `XRayFormFactor` class.
    """

    @staticmethod
    def test_xray_form_factor_initialization_normal_operation():
        """
        A unit test that tests the initialization of a `XRayFormFactor` instance.
        This unit test tests initialization with normal attributes.
        """
        xray_form_factor = form_factor.XRayFormFactor(1, 2, 3, 4, 5, 6, 7, 8, 9)

        assert xray_form_factor.a1 == 1
        assert xray_form_factor.b1 == 2
        assert xray_form_factor.a2 == 3
        assert xray_form_factor.b2 == 4
        assert xray_form_factor.a3 == 5
        assert xray_form_factor.b3 == 6
        assert xray_form_factor.a4 == 7
        assert xray_form_factor.b4 == 8
        assert xray_form_factor.c == 9

    @staticmethod
    def test_form_factor():
        """
        A unit test for the form_factor function. This unit test tests normal
        operation of the function.
        """
        x_ray_form_factor = form_factor.XRayFormFactor(1, 2, 3, 4, 5, 6, 7, 8, 9)
        rlv_magnitude = np.array([2 * np.pi])

        result = x_ray_form_factor.evaluate_form_factors(rlv_magnitude)

        a = [1, 3, 5, 7]
        b = [2, 4, 6, 8]
        c = 9

        expected_result = 0

        for i in range(4):
            expected_result += a[i] * np.exp(-b[i] / 4)

        expected_result += c

        assert np.isclose(result, expected_result, rtol=1e-6)
