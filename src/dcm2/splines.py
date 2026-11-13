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
from dataclasses import dataclass

from numba import jit
import numpy as np
from numpy.typing import NDArray, ArrayLike

@dataclass
class EvaluatedSpline:
    """Dataclass for an evaluated spline.

    Attributes
    ----------
    knots : array of float
        Knot vector.
    degree : int
        Spline degree.
    order : int
        Spline order.
    x : array of float
        Poins at which the spline has been evaluated
    basis : array of float
        Spline basis evaluated at `x`.
    derivative : array of float
        Derivative of spline basis, evaluated at `x`.
    """
    knots: NDArray[float]
    degree: int
    order: int
    x: NDArray[float]
    basis: NDArray[float]
    derivative: NDArray[float]


def evaluate(
        kind: str,
        x: ArrayLike,
        degree: int,
        inner_knots: ArrayLike,
        ) -> EvaluatedSpline:
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

    Returns
    -------
    EvaluatedSpline
        Spline basis and derivative evaluated at `x`.
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

    if kind == 'I':
        order = degree
        knots = close_knot_vector(inner_knots, order)
        basis, derivative = compute_ispline(x, order, knots)
    elif kind == 'M':
        order = degree + 1
        knots = close_knot_vector(inner_knots, order)
        basis, derivative = compute_mspline(x, order, knots)
    else:
        raise ValueError(f"Spline kind must be one of {'I', 'M'}, not {kind}.")
    return EvaluatedSpline(knots, degree, order, x, basis, derivative)

def close_knot_vector(knots: ArrayLike, order: int):
    """Add knots at the start and end to match desired spline order.

    Parameters
    ----------
    knot_positions : array_like
        Desired knot positions including endpoints.
    order : int
        Order of the spline

    Returns
    -------
    array of float
        sequence of knots with added multiplicity of the first and last
        knot to match the spline order.

    Notes
    -----
    A spline of order :math:`k` has a closed knot vector :math:`t` if

    - :math:`t_i = ... = t_{k+1}` and :math:`t_{n+1} = ... = t_{n+k+1}`,
    - all remaining knots have multiplicity at most :math:`k`.
    """
    knots = np.concatenate(
        [[knots[0]]*order, knots[1:-1], [knots[-1]]*order],
        dtype=float
        )
    return knots

@jit
def compute_ispline(x: ArrayLike, order: int, knots: ArrayLike
        ) -> NDArray:
    """Compute an I-spline basis

    Parameters
    ----------
    x : array_like of float
        Points at which to evaluate the spline basis functions.
    order : int
        Order of the spline.
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
    m_basis, m_derivative = compute_mspline(x, order + 1, m_knots)

    n = len(knots) - order
    basis, derivative = np.zeros((2, n, len(x)), dtype=float)

    for i in range(n):
        below = (x < m_knots[i])
        above = (x > m_knots[i + order + 1])
        basis[i] = np.sum(
            np.vstack((
#                   below *np.zeros_like(x),  # unnecessary
                above*np.ones_like(x),
                (~below & ~above)*np.sum(
                    (m_knots[i+order+2:] - m_knots[i+1:-order-1])
                    *m_basis[i+1:].T
                    /(order + 1),
                    axis=1
                    ),
                )),
            axis=0
            )
        derivative[i] = np.sum(
            np.vstack((
                (~below & ~above)*np.sum(
                    (m_knots[i+order+2:] - m_knots[i+1:-order-1])
                    *m_derivative[i+1:].T
                    /(order + 1),
                    axis=1
                    ),
                )),
            axis=0
            )
    return np.stack((basis, derivative))

@jit
def compute_mspline(x: ArrayLike, order: int, knots: ArrayLike
        ) -> NDArray:
    """Compute an M-spline basis

    Parameters
    ----------
    x : array_like of float
        Points at which to evaluate the spline basis functions.
    order : int
        Order of the spline.
    knots : array_like of float
        Spline knot values, including repeated knots.

    Returns
    -------
    array
        Evaluated M-spline basis with each row corresponding to one
        basis function.
    """
    n = len(knots) - order
    basis, derivative = np.zeros((2, n+1, len(x)), dtype=float)

    k = 1
    for i in range(n):
        if knots[i] == knots[i+k]:  # avoid zerodivision
            basis[i] = np.zeros_like(x)
        else:
            basis[i] = (
                ((knots[i] <= x) & (x < knots[i+1]))/(knots[i+1] - knots[i])
                )
#        derivative[i] = np.zeros_like(x)  # unnecessary
    basis[n-1] += (knots[n] == x)/(knots[n] - knots[n-1])  # boundary
    k += 1
    while k <= order:
        for i in range(n):
            if knots[i] == knots[i+k]:  # avoid zerodivision
                basis[i] *= 0
                derivative[i] *= 0
            else:
                # derivative must be updated before basis
                derivative[i] = (
                    (
                        ((knots[i] <= x) & (x < knots[i+k]))
                        | (x == knots[n])  # boundary
                        )
                    *k*(
                        (x - knots[i])*derivative[i] + basis[i]
                        + (knots[i+k] - x)*derivative[i+1] - basis[i+1]
                        )
                    /((k - 1)*(knots[i+k] - knots[i]))
                    )
                basis[i] = (
                    (
                        ((knots[i] <= x) & (x < knots[i+k]))
                        | (x == knots[n])  # boundary
                        )
                    *k*((x - knots[i])*basis[i] + (knots[i+k] - x)*basis[i+1])
                    /((k - 1)*(knots[i+k] - knots[i]))
                    )
        k += 1

    return np.stack((basis[:-1], derivative[:-1]))  # the last element is zero
