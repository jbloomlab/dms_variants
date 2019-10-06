r"""
=================
globalepistasis
=================

Implements global epistasis models based on `Otwinoski et al (2018)`_.

.. contents:: Contents
   :local:
   :depth: 1

Definition of models
---------------------

The models are defined as follows. Let :math:`v` be a variant. We convert
:math:`v` into a binary representation with respect to some wildtype
sequence. This representation is a vector :math:`\mathbf{b}\left(v\right)`
with element :math:`b\left(v\right)_m` equal to 1 if the variant has mutation
:math:`m` and 0 otherwise, and :math:`m` ranging over all :math:`M` mutations
observed in the overall set of variants (so :math:`\mathbf{b}\left(v\right)`
is of length :math:`M`). Variants can be converted into this binary form
using :class:`dms_variants.binarymap.BinaryMap`.

We define a *latent effect* for each mutation :math:`m`, which we denote as
:math:`\beta_m`. The latent effects of mutations contribute additively to the
*latent phenotype*, and the latent phenotype of the wildtype sequence is
:math:`\beta_{\rm{wt}}`. So the *latent phenotype* of variant :math:`v` is:

.. math::
   :label: latent_phenotype

   \phi\left(v\right) = \beta_{\rm{wt}} +
                        \sum_{m=1}^M \beta_m b\left(v\right)_m.

The predicted *observed phenotype* :math:`p\left(v\right)` is a function of the
latent phenotype:

.. math::
   :label: observed_phenotype

   p\left(v\right) = g\left(\phi\left(v\right)\right)

where :math:`g` is the *global epistasis function*.

We define epistasis models with the following global epistasis functions:

 - **Non-epistatic model**:
   This model has no epistasis, so the observed phenotype is just the latent
   phenotype. In other words:

   .. math::
      :label: noepistasis

      g\left(x\right) = x.

   This model is implemented as :class:`NoEpistasis`.

Fitting of models
-------------------
For each variant :math:`v`, we have an experimentally measured functional
score :math:`y_v` and optionally an estimate of the error (variance)
:math:`\sigma^2_{y_v}` in this functional score measurement. If no error
estimates are available, then we set :math:`\sigma^2_{y_v} = 0`.

The goal of the fitting is to parameterize the model so the observed phenotype
:math:`p\left(v\right)` predicted by the model is as close as possible to the
measured functional score :math:`y_v`. Following `Otwinoski et al (2018)`_,
we assume the likelihood of measuring a functional score :math:`y_v` is
normally distributed around the model prediction :math:`p\left(v\right)`
with variance :math:`\sigma^2_{y_v} + \sigma^2_{\rm{HOC}}`, where
:math:`\sigma^2_{\rm{HOC}}` is the un-modeled *house-of-cards epistasis*
(although in practice it could also represent experimental noise not
capture in the variance estimates). So the overall log likelihood of
the model is

.. math::
   :label: loglik

   \mathcal{L} = \sum_{v=1}^V \ln\left[N\left(y_v \mid p\left(v\right),
                 \sigma^2_{y_v} + \sigma^2_{\rm{HOC}}\right)\right]

where :math:`V` is the number of variants and :math:`N` is the normal
distribution defined by

.. math::
   :label: normaldist

   N\left(y \mid \mu, \sigma^2\right) = \frac{1}{\sigma \sqrt{2\pi}} \exp
                    \left(-\frac{\left(y - \mu\right)^2}{2 \sigma^2}\right).

To fit the model, we maximize the log likelihood in Eq. :eq:`loglik` with
respect to all model parameters: the latent effects :math:`\beta_m` of all
mutations, the latent phenotype :math:`\beta_{\rm{wt}}` of the wildtype
sequence, the house-of-cards epistasis :math:`\sigma^2_{\rm{HOC}}`,
and any parameters that define the global epistasis function :math:`g`.

Details of optimization
-------------------------

Vector representation of :math:`\beta_{\rm{wt}}`
+++++++++++++++++++++++++++++++++++++++++++++++++
For the purposes of the optimization (and in the equations below), we change
how :math:`\beta_{\rm{wt}}` is represented to simplify the calculations.
Specifically, to the binary encoding :math:`\mathbf{b}\left(v\right)` of
each variant, we append a 1 so that the encodings are now of length
:math:`M + 1`. We then define :math:`\beta_{M + 1} = \beta_{\rm{wt}}`.
Then Eq. :eq:`latent_phenotype` can be rewritten as

.. math::
   :label: latent_phenotype_wt_vec

   \phi\left(v\right) = \sum_{m=1}^{M+1} \beta_m b\left(v\right)_m

enabling :math:`\beta_{\rm{wt}}` to just be handled like the other
:math:`\beta_m` parameters.

Gradients
+++++++++++

For the optimization, we use the following gradients:

.. math::
   :label: dlatent_phenotype_dlatent_effect

   \frac{\partial \phi\left(v\right)}{\partial \beta_m} =
   b\left(v_m\right)

.. math::
   :label: dobserved_phenotype_dlatent_effect

   \frac{\partial p\left(v\right)}{\partial \beta_m}
   &=& \left.\frac{\partial g\left(x\right)}{\partial x}
       \right\rvert_{x = \phi\left(v\right)} \times
       \frac{\partial \phi\left(v\right)}{\partial \beta_m} \\
   &=& \left.\frac{\partial g\left(x\right)}{\partial x}
       \right\rvert_{x = \phi\left(v\right)} \times b\left(v_m\right)

.. math::
   :label: dnormaldist

   \frac{\partial \ln\left[N\left(y \mid \mu, \sigma^2\right)\right]}
        {\partial \mu} =
   \frac{y - \mu}{\sigma^2}

.. math::
   :label: dloglik_dlatent_effect

   \frac{\partial \mathcal{L}}{\partial \beta_m}
   &=& \sum_{v=1}^V \frac{\partial \ln\left[N\left(y_v \mid p\left(v\right),
                          \sigma_{y_v}^2 + \sigma^2_{\rm{HOC}}\right)\right]}
                         {\partial p\left(v\right)} \times
                    \frac{\partial p\left(v\right)}{\partial \beta_m} \\
   &=& \sum_{v=1}^V \frac{y_v - p\left(v\right)}
                         {\sigma_{y_v}^2 + \sigma^2_{\rm{HOC}}} \times
                    \left.\frac{\partial g\left(x\right)}{\partial x}
                    \right\rvert_{x = \phi\left(v\right)} \times
                    b\left(v_m\right)


API implementing models
--------------------------

.. _`Otwinoski et al (2018)`: https://www.pnas.org/content/115/32/E7550

"""


import abc
import collections
import inspect

import numpy

import scipy.optimize
import scipy.stats


class EpistasisFittingError(Exception):
    """Error fitting an epistasis model."""

    pass


class AbstractEpistasis(abc.ABC):
    """Abstract base class for epistasis models.

    Parameters
    ----------
    binarymap : :class:`dms_variants.binarymap.BinaryMap`
        Contains the variants, their functional scores, and score variances.

    Note
    ----
    This is an abstract base class. It implements most of the epistasis model
    functionality, but does not define the actual functional form of
    the global epistasis function :meth:`AbstractEpistasis.epistasis_func`.

    """

    _NEARLY_ZERO = 1e-10
    """float: lower bound for parameters that should be > 0."""

    def __init__(self,
                 binarymap,
                 ):
        """See main class docstring."""
        self._binarymap = binarymap
        self._nlatent = self.binarymap.binarylength  # number latent effects
        self._cache = {}  # cache computed values

        # initialize params
        self._latenteffects = numpy.zeros(self._nlatent + 1, dtype='float')
        self.epistasis_HOC = 1.0
        self._epistasis_func_params = numpy.zeros(
                    len(self._epistasis_func_param_names), dtype='float')

    # ------------------------------------------------------------------------
    # Methods / properties to set and get model parameters that are fit.
    # The setters must clear appropriate elements from the cache.
    # ------------------------------------------------------------------------
    @property
    def _latenteffects(self):
        r"""numpy.ndarray: Latent effects of mutations and wildtype.

        The :math:`\beta_m` values followed by :math:`\beta_{\rm{wt}}` for
        the representation in Eq. :eq:`latent_phenotype_wt_vec`.

        """
        return self._latenteffects_val

    @_latenteffects.setter
    def _latenteffects(self, val):
        self._cache = {}
        if not (isinstance(val, numpy.ndarray) and
                len(val) == self._nlatent + 1):
            raise ValueError(f"invalid value for `_latenteffects`: {val}")
        self._latenteffects_val = val.copy()
        self._latenteffects_val.flags.writeable = False

    @property
    def epistasis_HOC(self):
        r"""float: House of cards epistasis, :math:`\sigma^2_{rm{HOC}}`."""
        return self._epistasis_HOC_val

    @epistasis_HOC.setter
    def epistasis_HOC(self, val):
        for key in list(self._cache.keys()):
            if key not in {'_latent_phenotypes', '_observed_phenotypes'}:
                del self._cache[key]
        if val <= 0:
            raise ValueError(f"`epistasis_HOC` must be > 0: {val}")
        self._epistasis_HOC_val = val

    @property
    def _epistasis_func_params(self):
        """numpy.ndarray: :meth:`AbstractEpistasis.epistasis_func` params."""
        return self._epistasis_func_params_val

    @_epistasis_func_params.setter
    def _epistasis_func_params(self, val):
        for key in list(self._cache.keys()):
            if key not in {'_latent_phenotypes', '_variances'}:
                del self._cache[key]
        if len(val) != len(self._epistasis_func_param_names):
            raise ValueError('invalid length for `_epistasis_func_params`')
        self._epistasis_func_params_val = val.copy()
        self._epistasis_func_params_val.flags.writeable = False

    # ------------------------------------------------------------------------
    # Methods / properties to get model parameters in useful formats
    # ------------------------------------------------------------------------
    @property
    def binarymap(self):
        """:class:`dms_variants.binarymap.BinaryMap`: Variants to model.

        The binary map is set during initialization of the model.

        """
        return self._binarymap

    @property
    def _binary_variants(self):
        r"""scipy.sparse.csr.csr_matrix: Binary variants with 1 in last column.

        As in Eq. :eq:`latent_phenotype_wt_vec` with :math:`\beta_{M+1}`.

        """
        if not hasattr(self, '_binary_variants_val'):
            # add column as here: https://stackoverflow.com/a/41947378
            self._binary_variants_val = scipy.sparse.hstack(
                [self.binarymap.binary_variants,
                 numpy.ones(self.binarymap.nvariants, dtype='int8')[:, None],
                 ],
                format='csr',
                )
        return self._binary_variants_val

    @property
    def nparams(self):
        """int: Total number of parameters in model."""
        return (len(self._latenteffects) +  # latent effects, wt latent pheno
                1 +  # HOC epistasis
                len(self._epistasis_func_params)  # params of epistasic func
                )

    @property
    def latent_phenotype_wt(self):
        r"""float: latent phenotype of wildtype.

        :math:`\beta_{\rm{wt}}` in Eq. :eq:`latent_phenotype`.

        """
        return self._latenteffects[self._nlatent]

    @property
    def _epistasis_func_param_names(self):
        """list: Names of :meth:`AbstractEpistasis.epistasis_func_params`."""
        if not hasattr(self, '_epistasis_func_param_names_val'):
            sig_params = inspect.signature(self.epistasis_func).parameters
            if 'latent_phenotype' not in sig_params:
                raise ValueError('`epistasis_func` signature lacks '
                                 f"'latent_phenotype':\n{sig_params}")
            self._epistasis_func_param_names_val = [x for x in sig_params
                                                    if x != 'latent_phenotype']
        return self._epistasis_func_param_names_val

    @property
    def epistasis_func_params_dict(self):
        """OrderedDict: :meth:`AbstractEpistasis.epistasis_func` param values.

        Maps names of parameters defining the global epistasis function to
        current values. These parameters are all arguments to
        :meth:`AbstractEpistasis.epistasis_func` **except** `latent_phenotype`
        (which is the function input, not a parameter defining the function).

        """
        if not self._epistasis_func_param_names:
            return collections.OrderedDict()
        assert (len(self._epistasis_func_params) ==
                len(self._epistasis_func_param_names))
        return collections.OrderedDict(zip(self._epistasis_func_param_names,
                                           self._epistasis_func_params))

    # ------------------------------------------------------------------------
    # Methods to calculate phenotypes given current model state
    # ------------------------------------------------------------------------
    def latent_phenotypes_frombinary(self,
                                     binary_variants,
                                     *,
                                     wt_col=False,
                                     ):
        """Latent phenotypes from binary variant representations.

        Parameters
        ----------
        binary_variants : scipy.sparse.csr.csr_matrix or numpy.ndarray
            Binary variants in form used by
            :class:`dms_variants.binarymap.BinaryMap`.
        wt_col : bool
            Set to `True` if `binary_variants` contains a terminal
            column of ones to enable calculations in the form given
            by Eq. :eq:`latent_phenotype_wt_vec`.

        Returns
        --------
        numpy.ndarray
            Latent phenotypes calculated using Eq. :eq:`latent_phenotype`.

        """
        if len(binary_variants.shape) != 2:
            raise ValueError(f"`binary_variants` not 2D:\n{binary_variants}")
        if binary_variants.shape[1] != self._nlatent + int(wt_col):
            raise ValueError(f"variants wrong length: {binary_variants.shape}")

        if wt_col:
            assert len(self._latenteffects) == binary_variants.shape[1]
            return binary_variants.dot(self._latenteffects)
        else:
            assert (len(self._latenteffects) - 1) == binary_variants.shape[1]
            return (binary_variants.dot(self._latenteffects[: -1]) +
                    self.latent_phenotype_wt)

    def observed_phenotypes_fromlatent(self, latent_phenotypes):
        """Observed phenotypes from latent ones.

        Parameters
        ----------
        latent_phenotypes : numpy.ndarray
            Latent phenotypes.

        Returns
        -------
        numpy.ndarray
            Observed phenotypes calculated using Eq. :eq:`latent_phenotype`.

        """
        observed = self.epistasis_func(latent_phenotypes)
        assert observed.shape == latent_phenotypes.shape
        return observed

    # ------------------------------------------------------------------------
    # Methods / properties used for model fitting. Many of these are properties
    # that store the current state for the variants we are fitting, using the
    # cache so that they don't have to be re-computed needlessly.
    # ------------------------------------------------------------------------
    def fit(self):
        """Fit all model params to maximum likelihood values."""
        # least squares fit of latent effects for reasonable initial values
        self._fit_latent_leastsquares()

        # initial parameter values and bounds
        params = numpy.array(list(self._latenteffects) +
                             [self.epistasis_HOC] +
                             list(self._epistasis_func_params)
                             )
        bounds = ([(None, None)] * len(self._latenteffects) +
                  [(self._NEARLY_ZERO, None)] +  # HOC epistasis must be > 0
                  [(None, None)] * len(self._epistasis_func_params)
                  )

        # define function to optimize
        def func(params):
            self._latenteffects = params[: len(self._latenteffects)]
            self.epistasis_HOC = params[len(self._latenteffects)]
            if self._epistasis_func_params:
                self._epistasis_func_params = params[len(self._latenteffects)
                                                     + 1:]
            return -self.loglik

        # optimize model
        optres = scipy.optimize.minimize(
                        fun=func,
                        x0=params,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'ftol': 1e-8,
                                 'maxfun': 100000,
                                 },
                        )
        if not optres.success:
            raise EpistasisFittingError(
                    f"Fitting of {self.__class__.__name__} failed after "
                    f"{optres.nit} iterations. Message:\n{optres.message}")

    @property
    def loglik(self):
        """float: Current log likelihood as defined in Eq. :eq:`loglik`."""
        key = 'loglik'
        if key not in self._cache:
            self._cache[key] = sum(self._loglik_by_variant)
        return self._cache[key]

    @property
    def _loglik_by_variant(self):
        """numpy.ndarray: Log likelihoods per variant (Eq. :eq:`loglik`)."""
        key = '_loglik_by_variant'
        if key not in self._cache:
            standard_devs = numpy.sqrt(self._variances)
            if not all(standard_devs > 0):
                raise ValueError('standard deviations not all > 0')
            self._cache[key] = scipy.stats.norm.logpdf(
                                    self.binarymap.func_scores,
                                    loc=self._observed_phenotypes,
                                    scale=standard_devs)
        return self._cache[key]

    @property
    def _latent_phenotypes(self):
        """numpy.ndarray: Latent phenotypes, Eq. :eq:`latent_phenotype`."""
        key = '_latent_phenotypes'
        if key not in self._cache:
            self._cache[key] = self.latent_phenotypes_frombinary(
                                binary_variants=self._binary_variants,
                                wt_col=True,
                                )
        return self._cache[key]

    @property
    def _observed_phenotypes(self):
        """numpy.ndarray: Observed phenotypes, Eq. :eq:`observed_phenotype`."""
        key = '_observed_phenotypes'
        if key not in self._cache:
            self._cache[key] = self.observed_phenotypes_fromlatent(
                                    self._latent_phenotypes)
        return self._cache[key]

    @property
    def _variances(self):
        r"""numpy.ndarray: Functional score variance plus HOC epistasis.

        :math:`\sigma_{y_v}^2 + \sigma_{\rm{HOC}}^2` in Eq. :eq:`loglik`.

        """
        key = '_variances'
        if key not in self._cache:
            if self.binarymap.func_scores_var is not None:
                var = self.binarymap.func_scores_var + self.epistasis_HOC
            else:
                var = numpy.full(self.binarymap.nvariants, self.epistasis_HOC)
            self._cache[key] = var
        return self._cache[key]

    def _fit_latent_leastsquares(self):
        """Fit latent effects and HOC epistasis by least squares.

        Note
        ----
        This is a useful way to quickly get "reasonable" initial values for
        `_latenteffects` and `epistasis_HOC`.

        """
        # fit by least squares
        fitres = scipy.sparse.linalg.lsqr(
                    A=self._binary_variants,
                    b=self.binarymap.func_scores,
                    x0=self._latenteffects,
                    )

        # use fit result to update latenteffects
        self._latenteffects = fitres[0]

        # estimate HOC epistasis as residuals not from func_score variance
        residuals2 = fitres[3]**2
        if self.binarymap.func_scores_var is None:
            self.epistasis_HOC = max(residuals2 / self.binarymap.nvariants,
                                     self._NEARLY_ZERO)
        else:
            self.epistasis_HOC = max((residuals2 -
                                      sum(self.binarymap.func_scores_var)
                                      ) / self.binarymap.nvariants,
                                     self._NEARLY_ZERO)

    # ------------------------------------------------------------------------
    # Abstract methods for global epistasis func, must implement in subclasses
    # ------------------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def epistasis_func(self, latent_phenotype):
        """Global epistasis function :math:`g` in Eq. :eq:`observed_phenotype`.

        Note
        ----
        This is an abstract method for the :class:`AbstractEpistasis` class.
        The actual functional forms for specific models are defined in
        concrete subclasses. Those concrete implementations of the function
        will typically also have additional parameters used by the function.

        """
        return NotImplementedError


class NoEpistasis(AbstractEpistasis):
    """Non-epistatic model (see Eq. :eq:`noepistasis`).

    See docs for the base class :class:`AbstractEpistasis` for details on
    most attributes and methods.

    """

    @classmethod
    def epistasis_func(cls, latent_phenotype):
        """Global epistasis function :math:`g` in Eq. :eq:`noepistasis`.

        Parameters
        -----------
        latent_phenotype : float or numpy.ndarray
            Latent phenotype(s) of one or more variants.

        Returns
        -------
        float or numpy.ndarray
            Observed phenotype(s) after transforming the latent phenotypes
            using the global epistasis function.

        """
        return latent_phenotype


if __name__ == '__main__':
    import doctest
    doctest.testmod()
