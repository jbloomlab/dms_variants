"""
=================
ispline
=================

Implements :class:`Isplines`, which are monotonic spline functions that are
defined in terms of :class:`Msplines`.

See `Ramsay (1988)`_ for details about these splines, and also note the
corrections in the `Praat manual`_ to the errors in the I-spline formula
by `Ramsay (1988)`_.

.. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
.. _`Praat manual`: http://www.fon.hum.uva.nl/praat/manual/spline.html

"""


import numpy


class Isplines:
    r"""Implements I-splines (see `Ramsay (1988)`_).

    Parameters
    ----------
    order : int
        Sets :attr:`Isplines.order`.
    mesh : array-like
        Sets :attr:`Isplines.mesh`.

    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_. Note that
        the degree of the I-spline is equal to :math:`k`, while the
        associated M-spline has order :math:`k` but degree :math:`k - 1`.
    mesh : numpy.ndarray
        Mesh sequence, :math:`\xi_1 < \ldots < \xi_q` in the notation
        of `Ramsay (1988)`_. This class implements **fixed** mesh sequences.
    n : int
        Number of members in spline, denoted as :math:`n` in `Ramsay (1988)`_.
        Related to number of points :math:`q` in the mesh and the order
        :math:`k` by :math:`n = q - 2 + k`.
    lower : float
        Lower end of interval spanned by the splines (first point in mesh).
    upper : float
        Upper end of interval spanned by the splines (last point in mesh).

    Example
    -------
    Demonstrate using the example in Fig. 1 of `Ramsay (1988)`_:

    .. plot::
       :context: reset

       >>> import functools
       >>> import itertools
       >>> import numpy
       >>> import pandas as pd
       >>> import scipy.optimize
       >>> from dms_variants.ispline import Isplines

       >>> isplines = Isplines(3, [0.0, 0.3, 0.5, 0.6, 1.0])
       >>> isplines.order
       3
       >>> isplines.mesh
       array([0. , 0.3, 0.5, 0.6, 1. ])
       >>> isplines.n
       6
       >>> isplines.lower
       0.0
       >>> isplines.upper
       1.0

       Evaluate the I-splines at some selected points:

       >>> x = numpy.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
       >>> for i in range(1, isplines.n + 1):
       ...     print(f"I{i}: {numpy.round(isplines.I(x, i), 2)}")
       ... # doctest: +NORMALIZE_WHITESPACE
       I1: [0.   0.96 1.   1.   1.   1.  ]
       I2: [0.   0.52 0.84 0.98 1.   1.  ]
       I3: [0.   0.09 0.3  0.66 1.   1.  ]
       I4: [0.   0.   0.   0.02 0.94 1.  ]
       I5: [0.   0.   0.   0.   0.58 1.  ]
       I6: [0.   0.   0.   0.   0.13 1.  ]

       Plot the I-splines:

       >>> xplot = numpy.linspace(0, 1, 1000)
       >>> data = {'x': xplot}
       >>> for i in range(1, isplines.n + 1):
       ...     data[f"I{i}"] = isplines.I(xplot, i)
       >>> df = pd.DataFrame(data)
       >>> _ = df.plot(x='x')

       Check that the gradients are correct:

       >>> for i, xval in itertools.product(range(1, isplines.n + 1), x):
       ...     xval = numpy.array([xval])
       ...     ifunc = functools.partial(isplines.I, i=i)
       ...     difunc = functools.partial(isplines.dI_dx, i=i)
       ...     err = scipy.optimize.check_grad(ifunc, difunc, xval)
       ...     if err > 1e-5:
       ...         raise ValueError(f"excess err {err} for {i}, {xval}")

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

        self.n = len(self.mesh) - 2 + self.order

        self._msplines = Msplines(order + 1, mesh)

    def I(self, x, i):  # noqa: E743
        r"""Evaluate spline :math:`I_i` at point(s) :math:`x`.

        Parameters
        ----------
        x : numpy.ndarray
            One or more points in range covered by the spline.
        i : int
            Spline member :math:`I_i`, where :math:`1 \le i \le`
            :attr:`Isplines.n`.

        Returns
        -------
        numpy.ndarray
            The values of the I-spline at each point in `x`.

        Note
        ----
        The spline is evaluated using the formula given in the
        `Praat manual`_, which corrects some errors in the formula
        provided by `Ramsay (1988)`_:

        .. math::

           I_i\left(x\right)
           =
           \begin{cases}
           0 & \rm{if\;} i > j, \\
           1 & \rm{if\;} i < j - k, \\
           \sum_{m=i+1}^j \left(t_{m+k+1} - t_m\right)
                          M_m\left(x \mid k + 1\right) / \left(k + 1 \right)
             & \rm{otherwise},
           \end{cases}

        where :math:`j` is the index such that :math:`t_j \le x < t_{j+1}`
        (the :math:`\left\{t_j\right\}` are the :attr:`Msplines.knots` for a
        M-spline of order :math:`k + 1`) and :math:`k` is
        :attr:`Isplines.order`.

        .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
        .. _`Praat manual`: http://www.fon.hum.uva.nl/praat/manual/spline.html

        """
        return self._calculate_I_or_dI(x, i, 'I')

    def _calculate_I_or_dI(self, x, i, quantity):
        """Calculate :meth:`Isplines.I` or :meth:`Isplines.dI_dx`.

        Parameters
        ----------
        x : numpy.ndarray
            Same meaning as for :meth:`Isplines.I`.
        i : int
            Same meaning as for :meth:`Isplines.I`.
        quantity : {'I', 'dI'}
            Calculate :meth:`Isplines.I` or :meth:`Isplines.dI_dx`?

        Returns
        -------
        numpy.ndarray
            The return value of :meth:`Isplines.I` or :meth:`Isplines.dI_dx`.

        Note
        ----
        Most calculations for :meth:`Isplines.I` and :meth:`Isplines.dI_dx`
        are the same, so this method implements both.

        """
        if quantity == 'I':
            func = self._msplines.M
            i_lt_jminusk = 1.0
        elif quantity == 'dI':
            func = self._msplines.dM_dx
            i_lt_jminusk = 0.0
        else:
            raise ValueError(f"invalid `quantity` {quantity}")

        if not (1 <= i <= self.n):
            raise ValueError(f"invalid spline member `i` of {i}")

        if not isinstance(x, numpy.ndarray):
            raise ValueError('`x` is not numpy.ndarray')
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

        k = self.order

        # create `sum_terms`, where row m - 1 has the summation term for m
        sum_terms = numpy.vstack(
                [(self._msplines.knots[m + k] - self._msplines.knots[m - 1]) *
                 func(x, m, k + 1) / (k + 1)
                 for m in range(1, self._msplines.n + 1)])
        assert sum_terms.shape == (self._msplines.n, len(x))

        # calculate j for all entries in x
        j = numpy.searchsorted(self._msplines.knots, x, 'right')
        assert all(1 <= j) and all(j <= len(self._msplines.knots))
        assert x.shape == j.shape

        # create `binary_terms` where entry (m - 1, x) is 1 if and only if
        # the corresponding `sum_terms` entry is part of the sum.
        binary_terms = numpy.vstack(
                [numpy.zeros(len(x)) if m < i + 1 else (m <= j).astype(int)
                 for m in range(1, self._msplines.n + 1)])
        assert binary_terms.shape == sum_terms.shape

        # compute sums from `sum_terms` and `binary_terms`
        sums = numpy.sum(sum_terms * binary_terms, axis=0)
        assert sums.shape == x.shape

        # return value with sums, 0, or 1
        return numpy.where(i > j, 0.0,
                           numpy.where(i < j - k, i_lt_jminusk,
                                       sums))

    def dI_dx(self, x, i):
        r"""Get derivative of :meth:`Isplines.I` with respect to `x`.

        Parameters
        ----------
        x : numpy.ndarray
            Same meaning as for :meth:`Isplines.I`.
        i : int
            Same meaning as for :meth:`Isplines.I`.

        Returns
        -------
        numpy.ndarray
            Derivative of I-spline with respect to `x`.

        Note
        ----
        The derivative is calculated from the equation in :meth:`Isplines.I`:

        .. math::

           \frac{\partial I_i\left(x\right)}{\partial x}
           =
           \begin{cases}
           0 & \rm{if\;} i > j \rm{\; or \;} i < j - k, \\
           \sum_{m=i+1}^j\left(t_{m+k+1} - t_m\right)
                         \frac{\partial M_m\left(x \mid k+1\right)}{\partial x}
                         \frac{1}{k + 1}
             & \rm{otherwise}.
           \end{cases}

        """
        return self._calculate_I_or_dI(x, i, 'dI')


class Msplines:
    r"""Implements M-splines (see `Ramsay (1988)`_).

    Parameters
    ----------
    order : int
        Sets :attr:`Msplines.order`.
    mesh : array-like
        Sets :attr:`Msplines.mesh`.

    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_.
        Polynomials are of degree :math:`k - 1`.
    mesh : numpy.ndarray
        Mesh sequence, :math:`\xi_1 < \ldots < \xi_q` in the notation
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
       :context: reset

       >>> import functools
       >>> import itertools
       >>> import numpy
       >>> import pandas as pd
       >>> import scipy.optimize
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
       ...     print(f"M{i}: {numpy.round(msplines.M(x, i), 2)}")
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
       ...     data[f"M{i}"] = msplines.M(xplot, i)
       >>> df = pd.DataFrame(data)
       >>> _ = df.plot(x='x')

       Check that the gradients are correct:

       >>> for i, xval in itertools.product(range(1, msplines.n + 1), x):
       ...     xval = numpy.array([xval])
       ...     mfunc = functools.partial(msplines.M, i=i)
       ...     dmfunc = functools.partial(msplines.dM_dx, i=i)
       ...     err = scipy.optimize.check_grad(mfunc, dmfunc, xval)
       ...     if err > 1e-5:
       ...         raise ValueError(f"excess err {err} for {i}, {xval}")

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

    def M(self, x, i, k=None, invalid_i='raise'):
        r"""Evaluate spline :math:`M_i` at point(s) `x`.

        Parameters
        ----------
        x : numpy.ndarray
            One or more points in the range covered by the spline.
        i : int
            Spline member :math:`M_i`, where :math:`1 \le i \le`
            :attr:`Msplines.n`.
        k : int or None
            Order of spline. If `None`, assumed to be :attr:`Msplines.order`.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?

        Returns
        -------
        numpy.ndarray
            The values of the M-spline at each point in `x`.

        Note
        ----
        The spline is evaluated using the recursive relationship given by
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
            raise ValueError('`x` is not numpy.ndarray')
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

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
                        (k * ((x - ti) * self.M(x, i, k - 1) +
                         (tiplusk - x) * self.M(x, i + 1, k - 1,
                                                invalid_i='zero')
                         ) / ((k - 1) * (tiplusk - ti))),
                        0.0)

    def dM_dx(self, x, i, k=None, invalid_i='raise'):
        r"""Get derivative of :meth:`Msplines.M` with respect to `x`.

        Parameters
        ----------
        x : numpy.ndarray
            Same as for :meth:`Msplines.M`.
        i : int
            Same as for :meth:`Msplines.M`.
        k : int or None
            Same as for :meth:`Msplines.M`.
        invalid_i : {'raise', 'zero'}
            Same as for :meth:`Msplines.M`.

        Returns
        -------
        numpy.ndarray
            Derivative of M-spline with respect to `x`.

        Note
        ----
        The derivative is calculated from the equation in :meth:`Msplines.M`:

        .. math::

           \frac{\partial M_i\left(x \mid k=1\right)}{\partial x} &=& 0
           \\
           \frac{\partial M_i\left(x \mid k > 1\right)}{\partial x}
           &=&
           \begin{cases}
           \frac{k\left[\left(x - t_i\right)
                        \frac{\partial M_i\left(x \mid k-1\right)}{\partial x}
                        +
                        M_i\left(x \mid k-1\right)
                        +
                        \left(t_{i+k} -x\right)
                        \frac{\partial M_{i+1}\left(x \mid k-1\right)}
                             {\partial x}
                        -
                        M_{i+1}\left(x \mid k-1\right)
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
            raise ValueError('`x` is not numpy.ndarray')
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

        tiplusk = self.knots[i + k - 1]
        ti = self.knots[i - 1]
        if tiplusk == ti or k == 1:
            return numpy.zeros(x.shape, dtype='float')
        else:
            assert k > 1
            return numpy.where(
                        (ti <= x) & (x < tiplusk),
                        (k * ((x - ti) * self.dM_dx(x, i, k - 1) +
                              self.M(x, i, k - 1) +
                              (tiplusk - x) * self.dM_dx(x, i + 1, k - 1,
                                                         invalid_i='zero') -
                              self.M(x, i + 1, k - 1, invalid_i='zero')
                              ) / ((k - 1) * (tiplusk - ti))
                         ),
                        0.0)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
