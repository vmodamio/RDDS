"""
=======
splines
=======

Provides an interface to calculate splines and their derivative for a
given order and knot placement, and implements specific spline bases.

Spline implementations
----------------------
I-spline
    Monotonically increasing spline basis.
M-spline
    Non-negative spline basis, used in the computation of I-splines.
"""

from numba import jit
import numpy as np
from numpy.typing import NDArray, ArrayLike

def evaluate(
        kind: str,
        x: ArrayLike,
        degree: int,
        inner_knots: ArrayLike,
        derivative: int = 0
        ) -> NDArray:
    """Build and evaluate the spline basis at the given points.

    A more convenient way of calling the spline functions, with input
    validation.

    Parameters
    ----------
    kind : {'I', 'M'}
        The kind of spline to evaluate.
    x : ArrayLike
        Points at which to evaluate the spline basis.
    degree : int
        Degree of the spline basis
    knots : ArrayLike
        Positions of the spline knots. The multiplicity of the first
        and last knots is determined from the spline degree.
    derivative : int, default 0
        Order of the derivative to compute. The default value 0 computes
        the spline basis.

    Returns
    -------
    array
        Spline basis or its derivative evaluated at `x`.
    """
    # validate input parameters
    if not np.all(inner_knots[:-1] < inner_knots[1:]):
        raise ValueError(
            "The sequence of knot positions must be strictly increasing."
            )
    if min(x) < inner_knots[0] or max(x) > inner_knots[-1]:
        raise ValueError(
            "x cannot contain values outside the knot range "
            f"[{inner_knots[0]}; {inner_knots[-1]}]."
            )
    knots = close_knot_vector(inner_knots, degree)

    if derivative != 0:
        raise NotImplementedError("I haven't done this yet. -jsh")

    if kind == 'I':
        basis = compute_ispline_basis(x, degree, knots)
    elif kind == 'M':
        basis = compute_mspline_basis(x, degree, knots)
    else:
        raise ValueError(f"Spline kind must be one of {'I', 'M'}, not {kind}.")
    return basis

def close_knot_vector(knots: ArrayLike, degree: int):
    """Add knots at the start and end to match desired spline degree.

    Parameters
    ----------
    knot_positions : array_like
        Desired knot positions including endpoints.
    degree : int
        Degree of the spline

    Returns
    -------
    array of float
        sequence of knots with added multiplicity of the first and last
        knot to match the spline degree.

    Notes
    -----
    A spline of degree :math:`k` has a closed knot vector :math:`t` if

    - :math:`t_i = ... = t_{k+1}` and :math:`t_{n+1} = ... = t_{n+k+1}`,
    - all remaining knots have multiplicity at most :math:`k`.
    """
    knots = np.concatenate(
        [[knots[0]]*degree, knots[1:-1], [knots[-1]]*degree],
        dtype=float
        )
    return knots

@jit
def compute_ispline_basis(x: ArrayLike, degree: int, knots: ArrayLike
        ) -> NDArray:
    """Compute an I-spline basis

    Parameters
    ----------
    x : array_like of float
        Points at which to evaluate the spline basis functions.
    degree : int
        Degree of the spline.
    knots : array_like of float
        Spline knot values, including repeated knots.

    Returns
    -------
    array
        Evaluated I-spline basis with each row corresponding to one
        basis function.
    """
    m_knots = np.zeros(len(knots)+2)
    m_knots[1:-1] = knots
    m_knots[0] = knots[0]
    m_knots[-1] = knots[-1]
    m_spline = compute_mspline_basis(x, degree + 1, m_knots)

    n = len(knots) - degree
    basis = np.zeros((n, len(x)), dtype=float)

    for i in range(n):
        below = (x < m_knots[i])
        above = (x > m_knots[i + degree + 1])
        basis[i] = np.sum(
            np.vstack((
#                   below *np.zeros_like(x),  # unnecessary
                above*np.ones_like(x),
                (~below & ~above)*np.sum(
                    (m_knots[i+degree+2:] - m_knots[i+1:-degree-1])
                    *m_spline[i+1:].T
                    /(degree + 1),
                    axis=1
                    ),
                )),
            axis=0
            )
    return basis

@jit
def compute_mspline_basis(x: ArrayLike, degree: int, knots: ArrayLike
        ) -> NDArray:
    """Compute an M-spline basis

    Parameters
    ----------
    x : array_like of float
        Points at which to evaluate the spline basis functions.
    degree : int
        Degree of the spline.
    knots : array_like of float
        Spline knot values, including repeated knots.

    Returns
    -------
    array
        Evaluated M-spline basis with each row corresponding to one
        basis function.
    """
    n = len(knots) - degree
    basis = np.zeros((n+1, len(x)), dtype=float)

    k = 1
    for i in range(n):
        if knots[i] == knots[i+k]:
            basis[i] = np.zeros_like(x)
        else:
            basis[i] = (
                ((knots[i] <= x) & (x < knots[i+1]))/(knots[i+1] - knots[i])
                )
    basis[n-1] += (knots[n] == x)/(knots[n-1] - knots[n-2])  # boundary

    while k < degree:
        k += 1
        for i in range(n):
            if knots[i] == knots[i+k]:
                basis[i] *= 0
            else:
                basis[i] = (
                    k*((x - knots[i])*basis[i] + (knots[i+k] - x)*basis[i+1])
                    /((k - 1)*(knots[i+k] - knots[i]))
                    )
    return basis[:-1]  # the last element is zero

