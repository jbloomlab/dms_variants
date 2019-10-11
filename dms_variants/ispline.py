"""
=================
ispline
=================

Implements I-splines, which are monotonic spline functions.
See `Ramsay (1988)`_ for details about I-splines.

.. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395

"""


import numpy


class Msplines:
    r"""Implements M-splines (see `Ramsay (1988)`_).

    Parameters
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_.
        Polynomials are of degree :math:`k - 1`.
    mesh : array-like
        Mesh sequence, :math:`\lambda_1 < \ldots < \lambda_q` in the notation
        of `Ramsay (1988)`_.

    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_.
        Polynomials are of degree :math:`k - 1`.
    mesh : numpy.ndarray
        Mesh sequence, :math:`\lambda_1 < \ldots < \lambda_q` in the notation
        of `Ramsay (1988)`_. This class implements **fixed** mesh sequences.
    n : int
        Number of members in spline, denoted as :math:`n` in `Ramsay (1988)`_.
        Related to number of points :math:`q` in the mesh and the order
        :math:`k` by :math:`n = q - 2 + k`.
    knots : numpy.ndarray
        The knot sequence, :math:`t_1, \ldots, t_{n + k}` in the notation of
        `Ramsay (1988)`_.
    lower : float
        Lower end of interval spanned by the splines (first point in mesh).
    upper : float
        Upper end of interval spanned by the splines (last point in mesh).

    Example
    -------
    Demonstrate using the example in Fig. 1 of `Ramsay (1988)`_:

    .. plot::

       >>> import numpy
       >>> import pandas as pd
       >>> from dms_variants.ispline import Msplines

       >>> msplines = Msplines(3, [0.0, 0.3, 0.5, 0.6, 1.0])
       >>> msplines.order
       3
       >>> msplines.mesh
       array([0. , 0.3, 0.5, 0.6, 1. ])
       >>> msplines.n
       6
       >>> msplines.knots
       array([0. , 0. , 0. , 0.3, 0.5, 0.6, 1. , 1. , 1. ])
       >>> msplines.lower
       0.0
       >>> msplines.upper
       1.0

       Evaluate the M-splines at some selected points:

       >>> x = numpy.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
       >>> for i in range(1, msplines.n + 1):
       ...     print(f"M{i}: {numpy.round(msplines.m(x, i), 2)}")
       ... # doctest: +NORMALIZE_WHITESPACE
       M1: [10. 1.11 0.  0.   0.   0.  ]
       M2: [0.  3.73 2.4 0.6  0.   0.  ]
       M3: [0.  1.33 3.  3.67 0.   0.  ]
       M4: [0.  0.   0.  0.71 0.86 0.  ]
       M5: [0.  0.   0.  0.   3.3  0.  ]
       M6: [0.  0.   0.  0.   1.88 7.5 ]

       Plot the M-splines:

       >>> xplot = numpy.linspace(0, 1, 1000, endpoint=False)
       >>> data = {'x': xplot}
       >>> for i in range(1, msplines.n + 1):
       ...     data[f"M{i}"] = msplines.m(xplot, i)
       >>> df = pd.DataFrame(data)
       >>> _ = df.plot(x='x',
       ...             y=[f"M{i}" for i in range(1, msplines.n + 1)],
       ...             kind='line')

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395

    """

    def __init__(self, order, mesh):
        """See main class docstring."""
        if not (isinstance(order, int) and order >= 1):
            raise ValueError(f"`order` not int >= 1: {order}")
        self.order = order

        self.mesh = numpy.array(mesh, dtype='float')
        if self.mesh.ndim != 1:
            raise ValueError(f"`mesh` not array-like of dimension 1: {mesh}")
        if len(self.mesh) < 2:
            raise ValueError(f"`mesh` not length >= 2: {mesh}")
        if not numpy.array_equal(self.mesh, numpy.unique(self.mesh)):
            raise ValueError(f"`mesh` elements not unique and sorted: {mesh}")
        self.lower = self.mesh[0]
        self.upper = self.mesh[-1]
        assert self.lower < self.upper

        self.knots = numpy.array(
                        [self.lower] * self.order +
                        list(self.mesh[1: -1]) +
                        [self.upper] * self.order,
                        dtype='float')

        self.n = len(self.knots) - self.order
        assert self.n == len(self.mesh) - 2 + self.order

    def m(self, x, i, k=None, invalid_i='raise'):
        r"""Evaluate spline :math:`M_i` at point(s) `x`.

        Parameters
        ----------
        x : numpy.ndarray
            One or more points in the range covered by the splines.
        i : int
            The spline member :math:`M_i`, where :math:`1 \le i \le n`.
        k : int or None
            Order of spline. If `None`, assumed to be overall `order`
            of the M-splines family.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?

        Returns
        -------
        numpy.ndarray
            The value of the M-splines at each point in `x`.

        Note
        ----
        The splines are evaluated using the recursive relationship given by
        `Ramsay (1988) <https://www.jstor.org/stable/2245395>`_:

        .. math::

           M_i\left(x \mid k=1\right)
           &=&
           \begin{cases}
           1 / \left(t_{i+1} - t_i\right), & \rm{if\;} t_i \le x < t_{i+1} \\
           0, & \rm{otherwise}
           \end{cases} \\
           M_i\left(x \mid k > 1\right) &=&
           \begin{cases}
           \frac{k\left[\left(x - t_i\right) M_i\left(x \mid k-1\right) +
                        \left(t_{i+k} -x\right) M_{i+1}\left(x \mid k-1\right)
                        \right]}
           {\left(k - 1\right)\left(t_{i + k} - t_i\right)},
           & \rm{if\;} t_i \le x < t_{i+1} \\
           0, & \rm{otherwise}
           \end{cases}

        """
        if not (1 <= i <= self.n):
            if invalid_i == 'raise':
                raise ValueError(f"invalid spline member `i` of {i}")
            elif invalid_i == 'zero':
                return 0
            else:
                raise ValueError(f"invalid `invalid_i` of {invalid_i}")
        if k is None:
            k = self.order
        if not 1 <= k <= self.order:
            raise ValueError(f"invalid spline order `k` of {k}")
        if not isinstance(x, numpy.ndarray):
            raise ValueError('`x` not numpy.ndarray')
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` not between {self.lower} and {self.upper}")

        tiplusk = self.knots[i + k - 1]
        ti = self.knots[i - 1]
        if tiplusk == ti:
            return numpy.zeros(x.shape, dtype='float')

        if k == 1:
            return numpy.where((ti <= x) & (x < tiplusk),
                               1.0 / (tiplusk - ti),
                               0.0)
        else:
            assert k > 1
            return numpy.where(
                        (ti <= x) & (x < tiplusk),
                        (k * ((x - ti) * self.m(x, i, k - 1) +
                         (tiplusk - x) * self.m(x, i + 1, k - 1,
                                                invalid_i='zero')
                         ) / ((k - 1) * (tiplusk - ti))),
                        0.0)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
