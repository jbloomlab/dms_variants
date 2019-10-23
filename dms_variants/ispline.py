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


import methodtools

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

       Now calculate the weighted sum of the I-spline family, using the
       same weights as in Fig. 1 of `Ramsay (1988)`_:

       >>> weights=numpy.array([1.2, 2, 1.2, 1.2, 3, 0]) / 6
       >>> numpy.round(isplines.Itotal(x, weights, w_lower=0), 2)
       array([0.  , 0.38, 0.54, 0.66, 1.21, 1.43])

       Now calculate using some points that require linear extrapolation
       outside the mesh and also have a nonzero `w_lower`:

       >>> numpy.round(isplines.Itotal(
       ...      numpy.array([-0.5, -0.25, 0, 0.01, 1.0, 1.5]),
       ...      weights=weights,
       ...      w_lower=1), 3)
       array([0.   , 0.5  , 1.   , 1.02 , 2.433, 2.433])

       Check that gradients are correct. First test :meth:`Isplines.dI_dx`:

       >>> for i, xval in itertools.product(range(1, isplines.n + 1), x):
       ...     xval = numpy.array([xval])
       ...     ifunc = functools.partial(isplines.I, i=i)
       ...     difunc = functools.partial(isplines.dI_dx, i=i)
       ...     err = scipy.optimize.check_grad(ifunc, difunc, xval)
       ...     if err > 1e-5:
       ...         raise ValueError(f"excess err {err} for {i}, {xval}")

       Test :meth:`Isplines.dItotal_dx`:

       >>> x_deriv = numpy.array([-0.5, -0.25, 0, 0.01, 0.5, 0.7, 1.0, 1.5])
       >>> for xval in x_deriv:
       ...     xval = numpy.array([xval])
       ...     itotfunc = functools.partial(isplines.Itotal, weights=weights,
       ...                                  w_lower=0)
       ...     ditotfunc = functools.partial(isplines.dItotal_dx,
       ...                                   weights=weights)
       ...     err = scipy.optimize.check_grad(itotfunc, ditotfunc, xval)
       ...     if err > 1e-5:
       ...         raise ValueError(f"excess err {err} for {xval}")

       >>> (isplines.dItotal_dw_lower(x) == numpy.ones(x.shape)).all()
       True

       Test :meth:`Isplines.dItotal_dweights`:

       >>> wl = 1.5
       >>> (isplines.dItotal_dweights(x_deriv, weights, wl).shape ==
       ...  (len(x_deriv), len(weights)))
       True
       >>> weightslist = list(weights)
       >>> for xval, iw in itertools.product(x_deriv, range(len(weights))):
       ...     xval = numpy.array([xval])
       ...     w = numpy.array([weightslist[iw]])
       ...     def func(w):
       ...         iweights = numpy.array(weightslist[: iw] +
       ...                                list(w) +
       ...                                weightslist[iw + 1:])
       ...         return isplines.Itotal(xval, iweights, wl)
       ...     def dfunc(w):
       ...         iweights = numpy.array(weightslist[: iw] +
       ...                                list(w) +
       ...                                weightslist[iw + 1:])
       ...         return isplines.dItotal_dweights(xval, iweights, wl)[0, iw]
       ...     err = scipy.optimize.check_grad(func, dfunc, w)
       ...     if err > 1e-6:
       ...         raise ValueError(f"excess err {err} for {iw, xval}")

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

    def Itotal(self, x, weights, w_lower, linear_extrapolate=True):
        r"""Evaluate weighted sum of spline family at points :math:`x`.

        Note
        ----
        Evaluates the full interpolating curve from the I-splines. When
        :math:`x` falls within the lower :math:`L` and upper :math:`U`
        bounds of the range covered by the I-splines (:math:`L \le x \le U`),
        then this curve is defined as:

        .. math::

           I_{\rm{total}}\left(x\right)
           =
           w_{\rm{lower}} + \sum_i w_i I_i\left(x\right).

        When :math:`x` is outside the range of the mesh covered by the splines,
        the values are linearly extrapolated from first derivative at the
        bounds. Specifically, if :math:`x < L` then:

        .. math::

           I_{\rm{total}}\left(x\right)
           =
           I_{\rm{total}}\left(L\right) +
           \left(x - L\right)
           \left.\frac{\partial I_{\rm{total}}\left(y\right)}
                      {\partial y}\right\rvert_{y=L},

        and if :math:`x > U` then:

        .. math::

           I_{\rm{total}}\left(x\right)
           =
           I_{\rm{total}}\left(U\right) +
           \left(x - U\right)
           \left.\frac{\partial I_{\rm{total}}\left(y\right)}
                      {\partial y}\right\rvert_{y=U}.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the spline family.
        weights : numpy.ndarray
            Nonnegative weights :math:`w_i` of members :math:`I_i` of spline
            family, should be of length equal to :attr:`Isplines.n`.
        w_lower : float
            The value at the lower bound :math:`L` of the spline range,
            :math:`w_{\rm{lower}}`.
        linear_extrapolate : bool
            Linearly extrapolate values outside mesh covered by spline. If
            `False`, raise an error if any points in `x` outside mesh range.

        Returns
        -------
        numpy.ndarray
            Value of :math:`I_{\rm{total}}` for each point in `x`.

        """
        return self._calculate_Itotal_or_dItotal(x, weights, w_lower,
                                                 linear_extrapolate,
                                                 'Itotal')

    def _calculate_Itotal_or_dItotal(self, x, weights, w_lower,
                                     linear_extrapolate, quantity):
        """Calculate :meth:`Isplines.Itotal` or derivatives.

        All parameters have same meaning as for :meth:`Isplines.Itotal`
        except for `quantity`, which should be

          - 'Itotal' to compute :meth:`Isplines.Itotal`
          - 'dItotal_dx' to compute :meth:`Isplines.dItotal_dx`
          - 'dItotal_dweights` to compute :meth:`Isplines.dItotal_dweights`

        """
        if not isinstance(x, numpy.ndarray):
            raise ValueError(f"`x` is not a numpy array: {type(x)}")

        # get indices of `x` in, above, or below I-spline range
        index = {'below': numpy.flatnonzero(x < self.lower),
                 'above': numpy.flatnonzero(x > self.upper),
                 'in': numpy.flatnonzero((x >= self.lower) & (x <= self.upper))
                 }
        if (not linear_extrapolate) and (index['below'] or index['above']):
            raise ValueError('`x` out of range, `linear_extrapolate` is False')

        # check validity of `weights`
        if weights.shape != (self.n,):
            raise ValueError(f"invalid shape of `weights`: {weights.shape}")
        if any(weights < 0):
            raise ValueError(f"`weights` not all non-negative: {weights}")

        # get spline limits in array form
        limits = [('below', numpy.array([self.lower])),
                  ('above', numpy.array([self.upper]))]

        # compute return values for each category of indices
        returnvals = {}
        if quantity == 'Itotal':
            returnshape = len(x)
            if len(index['in']):
                returnvals['in'] = numpy.sum([self.I(x[index['in']], i) *
                                              weights[i - 1]
                                              for i in range(1, self.n + 1)],
                                             axis=0) + w_lower
            for name, limit in limits:
                if not len(index[name]):
                    continue
                returnvals[name] = (self.Itotal(limit, weights, w_lower) +
                                    (x[index[name]] - limit) *
                                    self.dItotal_dx(limit, weights)
                                    )
        elif quantity == 'dItotal_dx':
            returnshape = len(x)
            if len(index['in']):
                returnvals['in'] = numpy.sum([self.dI_dx(x[index['in']], i) *
                                              weights[i - 1]
                                              for i in range(1, self.n + 1)],
                                             axis=0)
            for name, limit in limits:
                if not len(index[name]):
                    continue
                returnvals[name] = self.dItotal_dx(limit, weights)
        elif quantity == 'dItotal_dweights':
            returnshape = (len(x), len(weights))
            if len(index['in']):
                returnvals['in'] = (numpy.vstack([self.I(x[index['in']], i) for
                                                  i in range(1, self.n + 1)])
                                    ).transpose()
            for name, limit in limits:
                if not len(index[name]):
                    continue
                returnvals[name] = (numpy.vstack([self.I(limit, i) +
                                                  (x[index[name]] - limit) *
                                                  self.dI_dx(limit, i) for
                                                  i in range(1, self.n + 1)])
                                    ).transpose()
        else:
            raise ValueError(f"invalid `quantity` {quantity}")

        # reconstruct single return value from indices and returnvalues
        returnval = numpy.full(returnshape, fill_value=numpy.nan)
        for name, name_index in index.items():
            if len(name_index):
                returnval[name_index] = returnvals[name]
        assert not numpy.isnan(returnval).any()
        return returnval

    def dItotal_dx(self, x, weights, linear_extrapolate=True):
        r"""Get derivative of :meth:`Isplines.Itotal` with respect to :math:`x`.

        Note
        ----
        Derivatives calculated from equations in :meth:`Isplines.Itotal` as:

        .. math::

           \frac{\partial I_{\rm{total}}\left(x\right)}{\partial x}
           =
           \begin{cases}
           \sum_i w_i \frac{\partial I_i\left(x\right)}{\partial x}
             & \rm{if\;} L \le x \le U, \\
           \left.\frac{\partial I_{\rm{total}}\left(y\right)}
                      {\partial y}\right\rvert_{y=L}
             & \rm{if\;} x < L, \\
           \left.\frac{\partial I_{\rm{total}}\left(y\right)}
                      {\partial y}\right\rvert_{y=U}
             & \rm{otherwise}.
           \end{cases}

        Parameters
        ----------
        x : numpy.ndarray
            Same meaning as for :meth:`Isplines.Itotal`.
        weights : numpy.ndarray
            Same meaning as for :meth:`Isplines.Itotal`.
        linear_extrapolate : bool
            Same meaning as for :meth:`Isplines.Itotal`.

        Returns
        -------
        numpy.ndarray
            Derivative :math:`\frac{\partial I_{\rm{total}}}{\partial x}`
            for each point in `x`.

        """
        return self._calculate_Itotal_or_dItotal(x, weights, None,
                                                 linear_extrapolate,
                                                 'dItotal_dx')

    def dItotal_dweights(self, x, weights, w_lower, linear_extrapolate=True):
        r"""Derivative of :meth:`Isplines.Itotal` by :math:`w_i`.

        Parameters
        ----------
        x : numpy.ndarray
            Same meaning as for :meth:`Isplines.Itotal`.
        weights : numpy.ndarray
            Same meaning as for :meth:`Isplines.Itotal`.
        w_lower : float
            Same meaning as for :meth:`Isplines.Itotal`.
        linear_extrapolate : bool
            Same meaning as for :meth:`Isplines.Itotal`.

        Returns
        -------
        numpy.ndarray
            The array is of shape `(len(x), len(weights))`, and element
            `ix, iweight` gives the derivative with respect to weight
            `weights[iweight]` evaluated at `x[ix]`.

        Note
        ----
        The derivative is:

        .. math::

           \frac{\partial I_{\rm{total}}\left(x\right)}{\partial w_i}
           =
           \begin{cases}
           I_i\left(x\right)
            & \rm{if\;} L \le x \le U, \\
           I_i\left(L\right) + \left(x-L\right)
           \left.\frac{\partial I_i\left(y\right)}{\partial y}\right\vert_{y=L}
            & \rm{if\;} x < L, \\
           I_i\left(U\right) + \left(x-U\right)
           \left.\frac{\partial I_i\left(y\right)}{\partial y}\right\vert_{y=U}
            & \rm{if\;} x > U.
           \end{cases}

        """
        return self._calculate_Itotal_or_dItotal(x, weights, w_lower,
                                                 linear_extrapolate,
                                                 'dItotal_dweights')

    def dItotal_dw_lower(self, x):
        r"""Derivative of :meth:`Isplines.Itotal` by :math:`w_{\rm{lower}}`.

        Parameters
        ----------
        x : numpy.ndarray
            Same meaning as for :meth:`Isplines.Itotal`.

        Returns
        -------
        numpy.ndarray
            :math:`\frac{\partial{I_{\rm{total}}}}{\partial w_{\rm{lower}}}`,
            which is just one for all `x`.

        """
        return numpy.ones(x.shape, dtype='float')

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


class Msplines_fixed_x:
    """Implementation of :class:`Msplines` with fixed `x` parameter.

    Note
    ----
    This class is like :class:`Msplines` **except** that the `x` parameter
    that is the first argument to methods of :class:`Msplines` is fixed to
    the value set at initialization. This enables the results to be cached,
    and so makes :class:`Msplines_fixed_x` faster if making repeated calls
    with same `x`.

    The other difference from :class:`Msplines` is that the returned
    arrays are **not** writeable.

    Parameters
    ----------
    order : int
        Same as for :class:`Msplines`.
    mesh : array-like
        Same as for :class:`Msplines`.
    x : numpy.ndarray
        The fixed `x` value for evaluating the splines.

    Example
    -------
    Show how :class:`Msplines_fixed_x` returns the same results as calling
    :class:`Msplines` with the fixed value of `x`:

    >>> order = 3
    >>> mesh = [0.0, 0.3, 0.5, 0.6, 1.0]
    >>> x = numpy.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
    >>> msplines = Msplines(order, mesh)
    >>> msplines_fixed_x = Msplines_fixed_x(order, mesh, x)
    >>> all(all(msplines.M(x, i) == msplines_fixed_x.M(i))
    ...     for i in range(1, msplines.n + 1))
    True
    >>> all(all(msplines.dM_dx(x, i) == msplines_fixed_x.dM_dx(i))
    ...     for i in range(1, msplines.n + 1))
    True

    Confirm that :class:`Msplines` and :class:`Msplines_fixed_x` have the
    same methods:

    >>> (set(key for key in Msplines.__dict__ if key[0] != '_') ==
    ...  set(key for key in Msplines_fixed_x.__dict__ if key[0] != '_'))
    True

    """

    def __init__(self, order, mesh, x):
        """See main class docstring."""
        self._x = x.copy()
        self._x.flags.writeable = False
        self._msplines = Msplines(order, mesh)

    @methodtools.lru_cache(maxsize=65536)
    def M(self, i, k=None, invalid_i='raise'):
        """Same as :meth:`Msplines.M` for the fixed value of `x`."""
        returnval = self._msplines.M(self._x, i, k, invalid_i)
        returnval.flags.writeable = False
        return returnval

    @methodtools.lru_cache(maxsize=65536)
    def dM_dx(self, i, k=None, invalid_i='raise'):
        """Same as :meth:`Msplines.dM_dx` for the fixed value of `x`."""
        returnval = self._msplines.dM_dx(self._x, i, k, invalid_i)
        returnval.flags.writeable = False
        return returnval


if __name__ == '__main__':
    import doctest
    doctest.testmod()
