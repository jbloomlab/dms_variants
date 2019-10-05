r"""
=================
globalepistasis
=================

Implements global epistasis models based on `Otwinoski et al (2018)`_.

.. contents:: Contents
   :local:

Definition of models
---------------------

The models are defined as follows. Let :math:`v` be a variant. We convert
:math:`v` into a binary representation with respect to some wildtype
sequence. This representation is a vector :math:`\mathbf{b}\left(v\right)`
with element :math:`b\left(v\right)_m` equal to 1 if the variant has mutation
:math:`m` and 0 otherwise, and :math:`m` ranging over all mutations observed in
the overall set of variants. Variants can be converted to this binary form
using :class:`dms_variants.binarymap.BinaryMap`.

We define a *latent effect* for each mutation :math:`m`, which we denote as
:math:`\beta_m`. The latent effects of mutations contribute additively to the
*latent phenotype*, and the latent phenotype of the wildtype sequence is
:math:`\beta_{\rm{wt}}`. So the *latent phenotype* of variant :math:`v` is:

.. math::
   :label: latent_phenotype

   \phi\left(v\right) = \beta_{\rm{wt}} + \sum_m \beta_m b\left(v\right)_m.

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

   \mathcal{L} = \sum_v \ln\left[N\left(y_v \mid p\left(v\right),
                 \sigma^2_{y_v} + \sigma^2_{\rm{HOC}}\right)\right]

where :math:`N` is the normal distribution defined by

.. math::
   :label: normaldist

   N\left(y \mid \mu, \sigma^2\right) = \frac{1}{\sigma \sqrt{2\pi}} \exp
                    \left(-\frac{\left(y - \mu\right)^2}{2 \sigma^2}\right).

To fit the model, we maximize the log likelihood in Eq. :eq:`loglik` with
respect to all model parameters: the latent effects :math:`\beta_m` of all
mutations, the latent phenotype :math:`\beta_{\rm{wt}}` of the wildtype
sequence, the house-of-cards epistasis :math:`\sigma^2_{\rm{HOC}}`,
and any parameters that define the global epistasis function :math:`g`.

For this optimization, we use the following gradients:

.. math::
   :label: dlatent_phenotype_dlatent_effect

   \frac{\partial \phi\left(v\right)}{\partial \beta_m} =
   b\left(v_m\right)

.. math::
   :label: dlatent_phenotype_dlatent_wt

   \frac{\partial \phi\left(v\right)}{\partial \beta_{\rm{wt}}} = 1

.. math::
   :label: dobserved_phenotype_dlatent_effect

   \frac{\partial p\left(v\right)}{\partial \beta_m}
   &=& \left.\frac{\partial g\left(x\right)}{\partial x}
       \right\rvert_{x = \phi\left(v\right)} \times
       \frac{\partial \phi\left(v\right)}{\partial \beta_m} \\
   &=& \left.\frac{\partial g\left(x\right)}{\partial x}
       \right\rvert_{x = \phi\left(v\right)} \times b\left(v_m\right)


.. math::
   :label: dobserved_phenotype_dlatent_wt

   \frac{\partial p\left(v\right)}{\partial \beta_{\rm{wt}}}
   &=& \left.\frac{\partial g\left(x\right)}{\partial x}
       \right\rvert_{x = \phi\left(v\right)} \times
       \frac{\partial \phi\left(v\right)}{\partial \beta_{\rm{wt}}} \\
   &=& \left.\frac{\partial g\left(x\right)}{\partial x}
       \right\rvert_{x = \phi\left(v\right)}

.. math::
   :label: dnormaldist

   \frac{\partial \ln\left[N\left(y \mid \mu, \sigma^2\right)\right]}
        {\partial \mu} =
   \frac{y - \mu}{\sigma^2}

.. math::
   :label: dloglik_dlatent_effect

   \frac{\partial \mathcal{L}}{\partial \beta_m}
   &=& \sum_v \frac{\partial \ln\left[N\left(y_v \mid p\left(v\right),
                    \sigma_{y_v}^2 + \sigma^2_{\rm{HOC}}\right)\right]}
                   {\partial p\left(v\right)} \times
              \frac{\partial p\left(v\right)}{\partial \beta_m} \\
   &=& \sum_v \frac{y_v - p\left(v\right)}
                   {\sigma_{y_v}^2 + \sigma^2_{\rm{HOC}}} \times
              \left.\frac{\partial g\left(x\right)}{\partial x}
              \right\rvert_{x = \phi\left(v\right)} \times b\left(v_m\right)

.. math::
   :label: dloglik_dlatent_wt

   \frac{\partial \mathcal{L}}{\partial \beta_{\rm{wt}}}
   = \sum_v \frac{y_v - p\left(v\right)}
                 {\sigma_{y_v}^2 + \sigma^2_{\rm{HOC}}} \times
            \left.\frac{\partial g\left(x\right)}{\partial x}
            \right\rvert_{x = \phi\left(v\right)}

API implementing models
--------------------------

.. _`Otwinoski et al (2018)`: https://www.pnas.org/content/115/32/E7550

"""


import numpy

import scipy.optimize
import scipy.stats


class EpistasisFittingError(Exception):
    """Error fitting an epistasis model."""

    pass


class NoEpistasis:
    """Non-epistatic (linear) model.

    Note
    ----
    The :meth:`NoEpistasis.epistasis_func` is defined in Eq. :eq:`noepistasis`.

    Parameters
    ----------
    binarymap : :class:`dms_variants.binarymap.BinaryMap`
        Contains the variants, their functional scores, and score variances.

    """

    _EPISTASIS_FUNC_PARAMS = ()
    """tuple: names of parameters used by `epistasis_func`."""

    _NEARLY_ZERO = 1e-10
    """float: lower bound for parameters that should be > 0."""

    def __init__(self,
                 binarymap,
                 ):
        """See main class docstring."""
        self._binarymap = binarymap
        self._nlatent = self.binarymap.binarylength  # number latent effects

        # initialize params
        self.params = numpy.array(
                        [0.0] * self._nlatent +  # latent effects
                        [0.0] +  # latent phenotype of wildtype
                        [1.0] +  # initial HOC epistasis
                        [0.0] * len(self._EPISTASIS_FUNC_PARAMS),
                        dtype='float')

    @property
    def binarymap(self):
        """:class:`dms_variants.binarymap.BinaryMap`: variants to model."""
        return self._binarymap

    @property
    def params(self):
        """numpy.ndarray: all model parameters.

        Note
        ----
        The order of parameters in the array is:
          - latent effects
          - latent phenotype of wildtype
          - epistasis_HOC
          - `epistasis_func` params

        Updating `params` is the only way you should alter the parameters
        of the model.

        """
        return self._params

    @params.setter
    def params(self, val):
        if not isinstance(val, numpy.ndarray):
            raise ValueError(f"`params` is invalid type of {type(val)}")
        if hasattr(self, '_params') and val.shape != self.params.shape:
            raise ValueError('trying to set `params` to new shape')
        self._params = val

    @property
    def latenteffects_array(self):
        r"""numpy.ndarray: Latent effects of mutations.

        These are the :math:`\beta_m` values in Eq. :eq:`latent_phenotype`.

        """
        assert len(self.params) >= self._nlatent
        return self.params[: self._nlatent]

    @property
    def latent_phenotype_wt(self):
        r"""float: latent phenotype of wildtype.

        This is :math:`\beta_{rm{wt}}` in Eq. :eq:`latent_phenotype`.

        """
        return self.params[self._nlatent]

    def latent_phenotype_frombinary(self, binary_variants):
        """Latent phenotypes from binary variant representations.

        Parameters
        ----------
        binary_variants : scipy.sparse.csr.csr_matrix or numpy.ndarray
            Binary variants in form used by
            :class:`dms_variants.binarymap.BinaryMap`.

        Returns
        --------
        numpy.ndarray
            Latent phenotypes calculated using Eq. :eq:`latent_phenotype`.

        """
        if len(binary_variants.shape) != 2:
            raise ValueError(f"`binary_variants` not 2D:\n{binary_variants}")
        if binary_variants.shape[1] != self._nlatent:
            raise ValueError(f"variants not length {self._nlatent}")

        return (binary_variants.dot(self.latenteffects_array) +
                self.latent_phenotype_wt)

    def observed_phenotype_frombinary(self, binary_variants):
        """Observed phenotypes from binary variant representations.

        Parameters
        ----------
        binary_variants
            Same as for :meth:`NoEpistasis.latent_phenotype_frombinary`.

        Returns
        --------
        numpy.ndarray
            Observed phenotypes calculated using Eq. :eq:`observed_phenotype`.

        """
        latent = self.latent_phenotype_frombinary(binary_variants)
        return self.epistasis_func(latent)

    @property
    def loglik(self):
        """float: Current log likelihood as defined in Eq. :eq:`loglik`."""
        return self._loglik_func(self.params)

    def _loglik_func(self, params, negative=False):
        """Calculate log likelihood from array of parameters.

        Parameters
        ----------
        params : numpy.ndarray
            Parameter values used to set the `params` attribute.
        negative : bool
            Return negative log likelihood rather than log likelihood.

        Returns
        -------
        float
            Log likelihood.

        Note
        ----
        Calling this method updates the `params` attribute (and so
        all of the model parameters) to whatever is specified by
        `params`. So do not call this method unless you understand
        what you are doing!

        """
        self.params = params
        predicted = self.observed_phenotype_frombinary(
                    self.binarymap.binary_variants)
        actual = self.binarymap.func_scores
        if self.binarymap.func_scores_var is not None:
            var = self.binarymap.func_scores_var + self.epistasis_HOC
        else:
            var = numpy.full(self.binarymap.nvariants, self.epistasis_HOC)
        sd = numpy.sqrt(var)
        if not all(sd > 0):
            raise ValueError('standard deviations not all > 0')
        logliks_by_variant = scipy.stats.norm.logpdf(
                                        actual,
                                        loc=predicted,
                                        scale=sd)
        if negative:
            return -sum(logliks_by_variant)
        else:
            return sum(logliks_by_variant)

    @property
    def nparams(self):
        """int: Total number of parameters in model."""
        return len(self.params)

    @property
    def epistasis_HOC(self):
        r"""float: House of cards epistasis, :math:`\sigma^2_{rm{HOC}}`."""
        return self.params[self._nlatent + 1]

    def epistasis_func(self, latent_phenotype):
        """Global epistasis function :math:`g` in Eq. :eq:`observed_phenotype`.

        Parameters
        -----------
        latent_phenotype : float or numpy.ndarray
            Latent phenotype(s) of one or more variants.

        Returns
        -------
        float or numpy.ndarray
            Observed phenotype(s) after transforming latent phenotype(s)
            using global epistasis function.

        """
        return latent_phenotype

    @property
    def epistasis_func_params(self):
        """dict: Parameters of global epistasis function."""
        if not self._EPISTASIS_FUNC_PARAMS:
            return {}
        offset = self._nlatent + 2
        assert len(self.params[offset:]) >= len(self._EPISTASIS_FUNC_PARAMS)
        return {key: val for key, val in
                zip(self._EPISTASIS_FUNC_PARAMS, self.params[offset:])}

    def fit(self):
        """Fit all model params to maximum likelihood values."""
        # least squares fit of latent effects for reasonable initial values
        self._fit_latent_leastsquares()

        # set parameter bounds
        bounds = [(None, None)] * self.nparams
        # HOC epistasis must be > 0
        bounds[self._nlatent + 1] = (self._NEARLY_ZERO, None)

        # optimize model
        optres = scipy.optimize.minimize(
                        fun=self._loglik_func,
                        x0=self.params,
                        args=(True,),  # get negative of loglik
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
        self.params == optres.x

    def _fit_latent_leastsquares(self):
        """Fit latent effects, phenotype, and HOC epistasis by least squares.

        Note
        ----
        This is a useful way to quickly get "reasonable" initial values in
        `params` for subsequent likelihood-based fitting.

        """
        # To fit the wt latent phenotype (intercept) as well as the latent
        # effects, add a column of all ones to the end of the binary_variants
        # sparse matrix as here: https://stackoverflow.com/a/41947378
        binary_variants = scipy.sparse.hstack(
                [self.binarymap.binary_variants,
                 numpy.ones(self.binarymap.nvariants, dtype='int8')[:, None],
                 ],
                format='csr',
                )
        ncol = self._nlatent + 1  # columns after adding 1
        assert binary_variants.shape == (self.binarymap.nvariants, ncol)

        # fit by least squares
        fitres = scipy.sparse.linalg.lsqr(
                    A=binary_variants,
                    b=self.binarymap.func_scores,
                    x0=self.params[: ncol],
                    )
        assert len(fitres[0]) == ncol

        # use fit result to update params
        newparams = scipy.append(fitres[0], self.params[ncol:])

        # estimate HOC epistasis as residuals not from func_score variance
        residuals2 = fitres[3]**2
        if self.binarymap.func_scores_var is None:
            epistasis_HOC = residuals2 / self.binarymap.nvariants
        else:
            epistasis_HOC = (residuals2 - sum(self.binarymap.func_scores_var)
                             ) / self.binarymap.nvariants
        newparams[ncol] = max(epistasis_HOC, self._NEARLY_ZERO)

        # update the params with least squares estimate
        self.params = newparams


if __name__ == '__main__':
    import doctest
    doctest.testmod()
