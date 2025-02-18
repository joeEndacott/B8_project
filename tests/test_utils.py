"""
This module contains unit tests for the utils.py module.
"""

from typing import cast
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from B8_project import utils


def test_duplicate_elements_normal_operation():
    """
    A unit test for the duplicate_elements function. This unit test tests normal
    operation of the function.
    """
    assert utils.duplicate_elements([0], 1) == [0]
    assert utils.duplicate_elements(["a"], 2) == ["a", "a"]
    assert utils.duplicate_elements([1, 2, 3], 1) == [1, 2, 3]
    assert utils.duplicate_elements([4, 5, 6], 2) == [4, 4, 5, 5, 6, 6]
    assert utils.duplicate_elements(["a", "b", "c"], 3) == [
        "a",
        "a",
        "a",
        "b",
        "b",
        "b",
        "c",
        "c",
        "c",
    ]


def test_gaussian_normal_operation():
    """
    A unit test for the gaussian function. This unit test tests normal
    operation of the function.
    """
    x = 5
    mean = 1
    width = 0.5
    amplitude = 2

    assert np.isclose(
        utils.gaussian(x, mean, width, amplitude),
        amplitude * np.exp(-0.5 * ((x - mean) / width) ** 2),
        rtol=1e-6,
    )


def test_random_uniform_unit_vector_is_unit_vector():
    """
    Add test docstring here.
    """

    n = 500
    for dim in range(2, 5):
        vectors = [utils.random_uniform_unit_vector(dim) for _ in range(n)]
        for vector in vectors:
            mag_squared = 0.0
            for e in vector:
                mag_squared += e**2
            nptest.assert_allclose(mag_squared, 1.0)


def test_random_uniform_unit_vectors_are_unit_vectors():
    """
    Add test docstring here.
    """

    n = 500
    for dim in range(2, 5):
        vectors = utils.random_uniform_unit_vectors(n, dim)
        for vector in vectors:
            mag_squared = 0.0
            for e in vector:
                mag_squared += e**2
            nptest.assert_allclose(mag_squared, 1.0)


def plot_3d_unit_vectors(vectors, title):
    """
    Plots 3D unit vectors using matplotlib.

    Parameters:
    vectors (np.ndarray): Array of 3D vectors to plot.
    title (str): Title of the plot.
    """

    fig = plt.figure()
    ax = cast(Axes3D, fig.add_subplot(projection="3d"))
    xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z", fontsize=12)
    ax.set_aspect("auto")
    plt.title(title)
    plt.show()


# These two tests are merely visual tests - check if the generated random unit
# vectors seem spherically uniform
def test_random_uniform_3d_unit_vector():
    """
    Add test docstring here.
    """
    n = 500
    vectors = [utils.random_uniform_unit_vector(3) for _ in range(n)]
    vectors = np.array(vectors)
    plot_3d_unit_vectors(
        vectors, "Visual check for spherical uniformity: random_uniform_unit_vector"
    )


def test_random_uniform_3d_unit_vectors():
    """
    Add test docstring here.
    """
    n = 500
    vectors = utils.random_uniform_unit_vectors(n, 3)
    plot_3d_unit_vectors(
        vectors, "Visual check for spherical uniformity: random_uniform_unit_vectors"
    )
