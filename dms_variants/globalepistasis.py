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

API implementing models
--------------------------

.. _`Otwinoski et al (2018)`: https://www.pnas.org/content/115/32/E7550

"""


import numpy

import scipy.stats


class NoEpistasis:
    """Non-epistatic (linear) model.

    Note
    ----
    The :meth:`NoEpistasis.epistasis_func` is defined in Eq. :eq:`noepistasis`.

    Parameters
    ----------
    binarymap : :class:`dms_variants.binarymap.BinaryMap`
        Contains the variants, their functional scores, and score variances.

    Attributes
    ----------
    binarymap : :class:`dms_variants.binarymap.BinaryMap`
        Contains variants, functional scores, and (optionally) score variances.

    """

    _EPISTASIS_FUNC_PARAMS = ()
    """tuple: names of parameters used by `epistasis_func`."""

    def __init__(self,
                 binarymap,
                 ):
        """See main class docstring."""
        self.binarymap = binarymap
        self._fit_complete = False

        # parameter order: latent effects, epistasis_HOC, epistasis_func params
        nparams = (self.binarymap.binarylength + 1 +
                   len(self._EPISTASIS_FUNC_PARAMS))
        self._params = numpy.zeros(nparams, dtype='float')

    @property
    def latenteffects_array(self):
        r"""numpy.ndarray of floats : Latent effects of mutations.

        These are the :math:`\beta_m` values in Eq. :eq:`latent_phenotype`.

        """
        assert len(self._params) >= self.binarymap.binarylength
        return self._params[: self.binarymap.binarylength]

    def latent_phenotype_frombinary(self, binary_variants):
        """Latent phenotypes from binary variant representations.

        Parameters
        ----------
        binary_variants : 2D numpy.ndarray or scipy.sparse.csr.csr_matrix
            Binary variants in form used by
            :class:`dms_variants.binarymap.BinaryMap`, with each row
            giving a different variant.

        Returns
        --------
        numpy.ndarray
            Latent phenotypes calculated using Eq. :eq:`latent_phenotype`.

        """
        if len(binary_variants.shape) != 2:
            raise ValueError(f"`binary_variants` not 2D:\n{binary_variants}")
        nvariants, binarylength = binary_variants.shape
        if binarylength != self.binarymap.binarylength:
            raise ValueError(f"variants not length {binarylength}")
        return binary_variants.dot(self.latenteffects_array)

    def observed_phenotype_frombinary(self, binary_variants):
        """Observed phenotypes from binary variant representations.

        Parameters
        ----------
        binary_variants : 2D numpy.ndarray or scipy.sparse.csr.csr_matrix
            Binary variants in form used by
            :class:`dms_variants.binarymap.BinaryMap`, with each row

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
        return self._loglik_func(self._params)

    def _loglik_func(self, paramsarray):
        """Calculate log likelihood from array of parameters.

        Parameters
        ----------
        paramsarray : numpy.ndarray
            Parameters values in form of the `_params` attribute.

        Returns
        -------
        float
            Log likelihood.

        Note
        ----
        Calling this method updates the `_params` attribute (and so
        all of the model parameters) to whatever is specified by
        `paramsarray`. So do not call this methd unless you understand
        what you are doing!

        """
        assert self._params.shape == paramsarray.shape
        self._params = paramsarray
        predicted = self.observed_phenotype_frombinary(
                        self.binarymap.binary_variants)
        actual = self.binarymap.func_scores
        if self.binarymap.func_scores_var is not None:
            var = self.binarymap.func_scores_var + self.epistasis_HOC
        else:
            var = numpy.full(self.binarymap.binarylength, self.epistasis_HOC)
        sd = numpy.sqrt(var)
        if not all(sd > 0):
            raise ValueError('standard deviations not all > 0')
        logliks_by_var = scipy.stats.norm.logpdf(actual, predicted, sd)
        return sum(logliks_by_var)

    @property
    def nparams(self):
        """int: Total number of parameters in model."""
        return len(self._params)

    @property
    def epistasis_HOC(self):
        r"""float: House of cards epistasis, :math:`\sigma^2_{rm{HOC}}`."""
        return self._params[self.binarymap.binarylength]

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
        offset = self.binarymap.binarylength + 1
        assert len(self._params[offset:]) >= len(self._EPISTASIS_FUNC_PARAMS)
        return {key: val for key, val in
                zip(self._EPISTASIS_FUNC_PARAMS, self._params[offset:])}

    def fit(self):
        """Fit all model params to maximum likelihood values."""
        if not self._fit_complete:
            raise RuntimeError('not yet implemented')

            self._fit_complete = True


if __name__ == '__main__':
    import doctest
    doctest.testmod()
