r"""
=================
globalepistasis
=================

Implements global epistasis models that are based on (but extend in some
ways) those described in `Otwinoski et al (2018)`_.
See also `Sailer and Harms (2017)`_ and `Otwinoski (2018)`_.

.. contents:: Contents
   :local:
   :depth: 2

.. _global_epistasis_function:

Global epistasis function
---------------------------

The global epistasis function is defined as follows.
Let :math:`v` be a variant. We convert
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

We define the following global epistasis functions:

.. _no_epistasis_function:

No epistasis function
+++++++++++++++++++++++
No epistasis, so the observed phenotype is just the latent phenotype:

.. math::
   :label: noepistasis

   g\left(x\right) = x.

This function is implemented as :class:`NoEpistasis`.

.. _monotonic_spline_epistasis_function:

Monotonic spline epistasis function
++++++++++++++++++++++++++++++++++++
This is the function used in `Otwinoski et al (2018)`_. It transforms
the latent phenotype to the observed phenotype using monotonic I-splines with
linear extrapolation outside the spline boundaries:

.. math::
   :label: monotonicspline

   g\left(x\right)
   =
   \begin{cases}
   c_{\alpha} + \sum_{m=1}^M \alpha_{m} I_m\left(x\right)
     & \rm{if\;} L \le x \le U, \\
   c_{\alpha} + \sum_{m=1}^M \alpha_m
     \left[I_m\left(L\right) + \left(x - L\right)
           \left.\frac{\partial I_m\left(y\right)}
                      {\partial y}\right\rvert_{y=L}
     \right]
     & \rm{if\;} x < L, \\
   c_{\alpha} + \sum_{m=1}^M \alpha_m
     \left[I_m\left(U\right) + \left(x - U\right)
           \left.\frac{\partial I_m\left(y\right)}
                      {\partial y}\right\rvert_{y=U}
     \right]
     & \rm{if\;} x > U,
   \end{cases}

where :math:`c_{\alpha}` is an arbitrary number giving the *minimum*
observed phenotype, the :math:`\alpha_m` coefficients are all :math:`\ge 0`,
:math:`I_m` indicates a family of I-splines defined via
:class:`dms_variants.ispline.Isplines_total`, and :math:`L` and :math:`U` are
the lower and upper bounds on the regions over which the I-splines are defined.
Note how when :math:`x` is outside the range of the I-splines, we linearly
extrapolate :math:`g` from its range boundaries to calculate.

This function is implemented as :class:`MonotonicSplineEpistasis`. By default,
the I-splines are of order 3 and are defined on a mesh of four evenly spaced
points such that the total number of I-splines is :math:`M=5` (although these
options can be adjusted when initializing a :class:`MonotonicSplineEpistasis`
model).

The latent effects are scaled so that their mean absolute value is one,
and the latent phenotype of the wildtype is set to zero.

.. _multi_latent:

Multiple latent phenotypes
+++++++++++++++++++++++++++
Although this package allows multiple latent phenotypes, we do **not**
recommend using them as the models generally do not seem to converge
in fitting in a useful way with multiple latent phenotypes.

Equations :eq:`latent_phenotype` and :eq:`observed_phenotype` can be
generalized to the case where multiple latent phenotypes contribute
to the observed phenotype. Specifically, let there be
:math:`k = 1, \ldots, K` different latent phenotypes, and let
:math:`\beta_m^k` denote the effect of mutation :math:`m` on latent phenotype
:math:`k`. Then we generalize Equation :eq:`latent_phenotype` to

.. math::
   :label: latent_phenotype_multi

   \phi_k\left(v\right) = \beta_{\rm{wt}}^k +
                          \sum_{m=1}^M \beta_m^k b\left(v\right)_m,

and Equation :eq:`observed_phenotype` to

.. math::
   :label: observed_phenotype_multi

   p\left(v\right) = \sum_{k=1}^K g_k\left(\phi_k\left(v\right)\right),

where :math:`\phi_k\left(v\right)` is the :math:`k`-th latent phenotype of
variant :math:`v`, and :math:`g_k` is the :math:`k`-th global epistasis
function.

Note that it does **not** make sense to fit multiple latent phenotypes
to a non-epistatic model as a linear combination of linear effects
reduces to a simple linear model (in other words, a multi-latent
phenotype non-epistatic model is no differen than a one-latent
phenotype non-epistatic model).

.. _likelihood_calculation:

Likelihood calculation
---------------------------------------
We defined a *likelihood* capturing how well the model describes the
actual data, and then fit the models by finding the parameters that
maximize this likelihood. This means that different epistasis functions
(as described in `Global epistasis function`_) can be compared via
their likelihoods after correcting for the number of parameters
(e.g. by `AIC <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_).

We consider several different forms for calculating the likelihood.
Note that epistasis functions can only be compared within the same form of
the likelihood on the same dataset--you **cannot** compare likelihoods
calculated using different methods, or on different datasets.

.. _gaussian_likelihood:

Gaussian likelihood
++++++++++++++++++++
This is the form of the likelihood used in `Otwinoski et al (2018)`_. It is
most appropriate when we have functional scores for variants along with
good estimates of normally distributed errors on these functional scores.

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
   :label: loglik_gaussian

   \mathcal{L} = \sum_{v=1}^V \ln\left[N\left(y_v \mid p\left(v\right),
                 \sigma^2_{y_v} + \sigma^2_{\rm{HOC}}\right)\right]

where :math:`V` is the number of variants and :math:`N` is the normal
distribution defined by

.. math::
   :label: normaldist

   N\left(y \mid \mu, \sigma^2\right) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp
                    \left(-\frac{\left(y - \mu\right)^2}{2 \sigma^2}\right).

This likelihood calculation is implemented as :class:`GaussianLikelihood`.

.. _cauchy_likelihood:

Cauchy likelihood
+++++++++++++++++++
This form of the likelihood assumes that the difference between the
measured functional scores and the model's observed phenotypes follows a
`Cauchy distribution <https://en.wikipedia.org/wiki/Cauchy_distribution>`_.

A potential advantage over the :ref:`gaussian_likelihood` is the
`fatter tails <https://en.wikipedia.org/wiki/Fat-tailed_distribution>`_
of the Cauchy distribution relative to the Gaussian distribution.
This could be advantageous if some measurements are very large outliers
(either due to un-modeled epistasis or experimental factors such as
mis-calling of variant sequences) in ways that are not
capture in the functional score variance estimates :math:`\sigma^2_{y_v}`.
Such outlier measurements will have less influence on the overall model
fit for the Cauchy likelihood relative to the :ref:`gaussian_likelihood`.

Specifically, we compute the overall log likelihood as

.. math::
   :label: loglik_cauchy

   \mathcal{L} = -\sum_{v=1}^V
                 \ln\left[\pi \sqrt{\gamma^2 + \sigma^2_{y_v}}
                          \left(1 + \frac{\left[y_v - p\left(v\right)\right]^2}
                                         {\gamma^2 + \sigma^2_{y_v}}
                          \right)
                     \right]

where :math:`\gamma` is the scale parameters, and the functional score
variance estimates :math:`\sigma^2_{y_v}` are incorporated in heuristic way
(with no real theoretical basis)
that qualitatively captures the fact that larger variance estimates give a
broader distribution. If variance estimates are not available then
:math:`\sigma^2_{y_v}` is set to zero.

This likelihood calculation is implemented as :class:`CauchyLikelihood`.

.. _bottleneck_likelihood:

Bottleneck likelihood
++++++++++++++++++++++
This form of the likelihood is appropriate when most noise in the experiment
comes from a bottleneck when passaging the library from the pre-selection to
post-selection condition. This will be the case when the total pre- and post-
selection sequencing depths greatly exceed the number of variants
that were physically passaged from the pre-selection library to the post-
selection condition. At least in Bloom lab viral deep mutational scanning
experiments, this situation is quite common.

A full derivation of the log likelihood in this situation is given in
:doc:`bottleneck_likelihood`. As explained in those calculations, the
likelihood is computed from the experimental observables
:math:`f_v^{\text{pre}}` and :math:`f_v^{\text{post}}`, which represent
the pre- and post-selection frequencies of variant :math:`v`. They are
computed from the pre- and post-selection counts :math:`n_v^{\text{pre}}`
and :math:`n_v^{\text{post}}` of the variants as

.. math::
   :label: f_v_pre_post

   f_v^{\text{pre}}
   &=& \frac{n_v^{\text{pre}} + C}
            {\sum_{v'=1}^V \left(n_{v'}^{\text{pre}} + C \right)} \\
   f_v^{\text{post}}
   &=& \frac{n_v^{\text{post}} + C}
            {\sum_{v'=1}^V \left(n_{v'}^{\text{post}} + C \right)}

where :math:`C` is a pseudocount which by default is 0.5.

Using the bottleneck likelihood also requires an estimation of the experimental
bottleneck :math:`N_{\rm{bottle}}` when passaging the library from the pre-
to post-selection conditions. A smaller bottleneck will correspond to more
"noise" in the experiment, since random bottlenecking changes the frequencies
of variants unpredictably. You can either estimate :math:`N_{\rm{bottle}}`
experimentally or from fluctuations in the relative frequencies of
wildtype or synonymous variants, such as via
:func:`dms_variants.bottlenecks.estimateBottleneck`.

Given these experimentally measured parameters, the overall log likelihood is:

.. math::
   :label: loglik_bottleneck

   \mathcal{L}
   =
   \sum_{v=1}^V
   \left[ n_v^{\rm{bottle}} \ln\left(N_{\rm{bottle}} f_v^{\text{pre}}\right)
          - \ln \Gamma \left(n_v^{\rm{bottle}} + 1\right)
        \right] - N_{\rm{bottle}}

where :math:`\Gamma` is the
`gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_
and :math:`n_v^{\rm{bottle}}` is the estimated number of copies of variant
:math:`v` that made it through the bottleneck, which is defined in terms
of the phenotype :math:`p\left(v\right)` as

.. math::
   :label: n_v_bottle

   n_v^{\rm{bottle}}
   =
   \frac{f_v^{\rm{post}} N_{\rm{bottle}}
         \sum_{v'=1}^V f_{v'}^{\text{pre}} 2^{p\left(v'\right)}}
        {2^{p\left(v\right)}}.

The free parameters are therefore the :math:`p\left(v\right)` values
(the :math:`n_v^{\rm{bottle}}` values are hidden variables that are
not explicitly estimated). Note that Eq. :eq:`n_v_bottle`
uses an exponent base of 2, but it can be set to arbitrary positive values
in the actual implementation.

After fitting the observed phenotypes, the parameters are re-scaled
so that the observed phenotype of wildtype is zero
(i.e., :math:`p\left(\rm{wt}\right) = 0`).

This likelihood calculation is implemented as :class:`BottleneckLikelihood`.

The model classes
------------------
The epistasis models are defined in a set of classes. All these classes
inherit their main functionality from the :class:`AbstractEpistasis`
abstract base class.

There are subclasses of :class:`AbstractEpistasis` that implement the
global epistasis functions and likelihood calculation methods. Specifically,
the following classes implement a :ref:`global_epistasis_function`:

  - :class:`NoEpistasis`
  - :class:`MonotonicSplineEpistasis`

and the following classes each implement a :ref:`likelihood_calculation`:
  - :class:`GaussianLikelihood`
  - :class:`CauchyLikelihood`
  - :class:`BottleneckLikelihood`

However, those classes can still not be directly instantianted, as a fully
concrete model subclass must have **both** a global epistasis function and
a likelihood calculation method.
The following classes implement both, and so can be directly instantiated
for use in analyses:

  - :class:`NoEpistasisGaussianLikelihood`
  - :class:`NoEpistasisCauchyLikelihood`
  - :class:`NoEpistasisBottleneckLikelihood`
  - :class:`MonotonicSplineEpistasisGaussianLikelihood`
  - :class:`MonotonicSplineEpistasisCauchyLikelihood`
  - :class:`MonotonicSplineEpistasisBottleneckLikelihood`

Details of fitting
-------------------------

.. _fitting_workflow:

Fitting workflow
+++++++++++++++++
The fitting workflow for a single latent phenotype is similar to that described
in `Otwinoski et al (2018)`_:

 1. The latent effects are fit under an additive (non-epistatic) model
    using least squares. The residuals from this fit are then used to
    estimate :math:`\sigma^2_{\rm{HOC}}` for :ref:`gaussian_likelihood`,
    or :math:`\gamma^2` for :ref:`cauchy_likelihood`.
 2. If there are any parameters in the epistasis function, they are set
    to reasonable initial values. For :class:`MonotonicSplineEpistasis`
    this involves setting the mesh to go from 0 to 1, scaling the latent
    effects so that the latent phenotypes range from 0 to 1, setting
    :math:`c_{\alpha}` to the minimum functional score and setting the
    weights :math:`\alpha_m` to equal values such that the max of the
    epistasis function is the same as the max functional score.
 3. The overall model is fit by maximum likelihood.
 4. For :class:`MonotonicSplineEpistasis`, the latent effects and wildtype
    latent phenotype are rescaled so that the mean absolute value latent
    effect is one and the wildtype latent phenotype is zero.


.. _fitting_multi_latent:

Fitting multiple latent phenotypes
++++++++++++++++++++++++++++++++++
When there are multiple latent phenotypes (see :ref:`multi_latent`), the
fitting workflow changes. To fit a model with :math:`K > 1` latent phenotypes,
first fit a model of the same type to the same data with :math:`K - 1` latent
phenotype. The values from that fit for the first :math:`K - 1` latent
phenotypes are used to initialize all parameters relevant to those first
:math:`K - 1` latent phenotypes and the associated global epistasis
functions and likelihood calculations.

Then parameters relevant to latent phenotype :math:`K` are set so the initial
contribution of this phenotype to the overall observed phenotype is zero.
Specifically the latent effects :math:`\beta_m^K` for phenotype :math:`K` are
all set to zero and :math:`\beta_{\rm{wt}}^K` is chosen so that
:math:`0 = g_K \left(\beta_{\rm{wt}}^K\right)`. For a
:class:`MonotonicSplineEpistasis` model, initial parameters for :math:`g_K` are
chosen so that over the mesh, :math:`g_K` spans +/- the absolute value of
the largest residual of the model with :math:`K - 1` latent phenotypes.

After initializing the paramters in this way, the entire model (all latent
phenotypes) is fit by maximum likelihood.

Conveniently fitting and comparing several models
+++++++++++++++++++++++++++++++++++++++++++++++++
To conveniently fit and compare several models, use :func:`fit_models`.
This is especially useful when you are including models with multiple
latent phenotypes.

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

Optimization
++++++++++++
The optimization is performed by :meth:`AbstractEpistasis.fit`.
There are several options to that method about how to do the optimization;
by default it uses a L-BFGS-B algorithm with exact gradients
calculated as below.

Gradients used in optimization
+++++++++++++++++++++++++++++++

For the optimization, we use the following gradients:

Gradient of latent phenotype with respect to latent effects:

.. math::
   :label: dlatent_phenotype_dlatent_effect

   \frac{\partial \phi_j\left(v\right)}{\partial \beta_m^k} =
   \begin{cases}
   b\left(v_m\right) & \rm{if\;} j = k, \\
   0 & \rm{otherwise.} \\
   \end{cases}

Gradient of observed phenotype with respect to latent phenotypes:

.. math::
   :label: dobserved_phenotype_dlatent_effect

   \frac{\partial p\left(v\right)}{\partial \beta_m^k}
   &=& \left.\frac{\partial g_k\left(x\right)}{\partial x}
       \right\rvert_{x = \phi_k\left(v\right)} \times
       \frac{\partial \phi_k\left(v\right)}{\partial \beta_m^k} \\
   &=& \left.\frac{\partial g_k\left(x\right)}{\partial x}
       \right\rvert_{x = \phi_k\left(v\right)} \times b\left(v_m\right)

Derivative of the likelihood with respect to latent effects:

.. math::
   :label: dloglik_dlatent_effect

   \frac{\partial \mathcal{L}}{\partial \beta_m^k}
   = \sum_{v=1}^V \frac{\mathcal{L}}
                       {\partial p\left(v\right)} \times
                  \frac{\partial p\left(v\right)}{\partial \beta_m^k}.

Derivative of :ref:`gaussian_likelihood` (Eq. :eq:`loglik_gaussian`) with
respect to observed phenotype:

.. math::
   :label: dloglik_gaussian_dobserved_phenotype

   \frac{\partial \mathcal{L}}{\partial p\left(v\right)}
   = \frac{y_v - p\left(v\right)}
          {\sigma_{y_v}^2 + \sigma^2_{\rm{HOC}}}.

Derivative of :ref:`gaussian_likelihood` (Eq. :eq:`loglik_gaussian`) with
respect to house-of-cards epistasis:

.. math::
   :label: dloglik_gaussian_depistasis_HOC

   \frac{\partial \mathcal{L}}{\partial \sigma^2_{\rm{HOC}}} = \sum_{v=1}^V
   \frac{1}{2} \left[\left(\frac{\partial \mathcal{L}}
                                {\partial p\left(v\right)}\right)^2
                     - \frac{1}{\sigma_{y_v}^2 + \sigma_{\rm{HOC}}^2} \right].

Derivative of :ref:`cauchy_likelihood` (Eq. :eq:`loglik_cauchy`) with
respect to observed phenotype:

.. math::
   :label: dloglik_cauchy_dobserved_phenotype

   \frac{\partial \mathcal{L}}{\partial p\left(v\right)}
   =
   \frac{2\left[y_v - p\left(v\right)\right]}
        {\gamma^2 + \sigma^2_{y_v} +
         \left[y_v - p\left(v\right)\right]^2}.

Derivative of :ref:`cauchy_likelihood` (Eq. :eq:`loglik_cauchy`) with
respect to scale parameter:

.. math::
   :label: dloglik_cauchy_dscale_parameter

   \frac{\partial \mathcal{L}}{\partial \gamma}
   =
   \sum_{v=1}^{V} \frac{\gamma\left(\left[y_v - p\left(v\right)\right]^2 -
                                    \gamma^2 - \sigma^2_{y_v}
                              \right)}
                       {\left(\gamma^2 + \sigma^2_{y_v}\right)
                        \left(\gamma^2 + \sigma^2_{y_v} +
                              \left[y_v - p\left(v\right)\right]^2\right)
                        }

Derivative of :ref:`bottleneck_likelihood` with respect to
:math:`p\left(v\right)`:

.. math::

   \frac{\partial n_v^{\text{bottle}}}
        {\partial p\left(v'\right)}
   &=&
   \begin{cases}
   \frac{\left(\ln 2\right) f_v^{\text{post}} N_{\text{bottle}}
         f_{v}^{\text{pre}} 2^{p\left(v\right)}}
        {2^{p\left(v\right)}}
   - \left(\ln 2\right) n_v^{\text{bottle}}
   & \rm{if\;} v = v', \\
   \frac{\left(\ln 2\right) f_v^{\text{post}} N_{\text{bottle}}
         f_{v'}^{\text{pre}} 2^{p\left(v'\right)}}
        {2^{p\left(v\right)}}
   & \text{otherwise}
   \end{cases} \\
   &=&
   \frac{\left(\ln 2\right) f_v^{\text{post}} N_{\text{bottle}}
         f_{v'}^{\text{pre}} 2^{p\left(v'\right)}}
        {2^{p\left(v\right)}}
   - \delta_{v,v'} \left(\ln 2\right) n_{v'}^{\text{bottle}}


.. math::
   :label: dloglik_bottleneck_dobserved_phenotype

   \frac{\partial \mathcal{L}}
        {\partial p\left(v'\right)}
   &=&
   \sum_{v=1}^V
   \frac{\partial n_v^{\text{bottle}}}{\partial p\left(v'\right)}
   \ln\left( N_{\text{bottle}} f_v^{\text{pre}} \right)
   -
   \frac{\partial n_v^{\text{bottle}}}{\partial p\left(v'\right)}
   \psi_0\left(n_v^{\text{bottle}} + 1\right)
   \\
   &=&
   \left(\ln 2\right) f_{v'}^{\text{pre}} 2^{p\left(v'\right)}
   N_{\text{bottle}}
   \left(\sum_{v=1}^V
         \frac{f_v^{\text{post}}}
              {2^{p\left(v\right)}}
         \left[\ln\left(N_{\text{bottle}} f_v^{\text{pre}}\right) -
               \psi_0\left(n_v^{\text{bottle}} + 1\right)
               \right]
         \right)
   - \left(\ln 2\right) n_{v'}^{\text{bottle}}
     \left[\ln\left(N_{\text{bottle}} f_{v'}^{\text{pre}}\right) -
           \psi_0\left(n_{v'}^{\text{bottle}} + 1\right)
           \right]

where :math:`\psi_0` is the
`digamma function <https://en.wikipedia.org/wiki/Digamma_function>`_.

Derivative of :ref:`monotonic_spline_epistasis_function` with respect to its
parameters:

.. math::
   :label: dspline_epistasis_dcalpha

   \frac{\partial g_k\left(x\right)}{\partial c_{\alpha}^k} = 1

.. math::
   :label: dspline_epistasis_dalpham

   \frac{\partial g_k\left(x\right)}{\partial \alpha^k_m}
   = I^k_m\left(x\right)


Detailed documentation of models
---------------------------------

.. _`Otwinoski et al (2018)`: https://www.pnas.org/content/115/32/E7550
.. _`Sailer and Harms (2017)`: https://www.genetics.org/content/205/3/1079
.. _`Otwinoski (2018)`: https://doi.org/10.1093/molbev/msy141

"""


import abc
import collections
import re
import time
import warnings

import numpy

import pandas as pd

import scipy.optimize
import scipy.sparse
import scipy.special
import scipy.stats

import dms_variants.ispline
import dms_variants.utils


class EpistasisFittingError(Exception):
    """Error fitting an epistasis model."""

    pass


class EpistasisFittingWarning(Warning):
    """Warning when fitting epistasis model."""

    pass


class AbstractEpistasis(abc.ABC):
    """Abstract base class for epistasis models.

    Parameters
    ----------
    binarymap : :class:`dms_variants.binarymap.BinaryMap`
        Contains the variants, their functional scores, and score variances.
    n_latent_phenotypes : int
        Number of distinct latent phenotypes. See :ref:`multi_latent`.
    model_one_less_latent : None or :class:`AbstractEpistasis`
        If `n_latent_phenotypes` > 1, should be a fit model of the same
        type fit the same `binarymap` for one less latent phenotype. This
        is used to initialize the parameters. See :ref:`fitting_multi_latent`.

    Note
    ----
    This is an abstract base class. It implements most of the epistasis model
    functionality, but requires subclasses to define the actual
    :ref:`global_epistasis_function` and :ref:`likelihood_calculation`.

    """

    _NEARLY_ZERO = 1e-8
    """float: lower bound for parameters that should be > 0."""

    def __init__(self,
                 binarymap,
                 *,
                 n_latent_phenotypes=1,
                 model_one_less_latent=None,
                 ):
        """See main class docstring."""
        self._binarymap = binarymap
        if not (isinstance(n_latent_phenotypes, int) and
                n_latent_phenotypes >= 1):
            raise ValueError('`n_latent_phenotypes` must be integer >= 1')
        self._n_latent_phenotypes = n_latent_phenotypes
        self._n_latent_effects = self.binarymap.binarylength
        self._cache = {}  # cache computed values

        # initialize params
        self._latenteffects = numpy.zeros(
                        (self.n_latent_phenotypes,
                         self._n_latent_effects + 1),
                        dtype='float')
        self._likelihood_calc_params = self._init_likelihood_calc_params
        self._epistasis_func_params = numpy.zeros(
                        shape=(self.n_latent_phenotypes,
                               len(self._epistasis_func_param_names)),
                        dtype='float')

        if self.n_latent_phenotypes > 1:
            self._set_lower_latent_phenotype_params(model_one_less_latent)
        elif model_one_less_latent is not None:
            raise ValueError('`n_latent_phenotypes` is 1, but '
                             '`model_one_less_latent` is not `None`.')

    def _set_lower_latent_phenotype_params(self, model_one_less_latent):
        """Set parameters for lower-order latent phenotypes.

        Parameters
        ----------
        model_one_less_latent : :class:`AbstractEpistasis`
            Model like `self` but fit with one less latent phenotype.

        Initializes all parameters relevant to the first :math:`K - 1`
        latent phenotypes as described in :ref:`fitting_multi_latent`.

        """
        assert self.n_latent_phenotypes > 1, 'calling with only 1 latent pheno'
        if model_one_less_latent is None:
            raise ValueError('`model_one_less_latent` cannot be `None` when '
                             'fitting multiple latent phenotypes')
        if type(self) != type(model_one_less_latent):
            raise ValueError('`model_one_less_latent` not same type as current'
                             f" object: {type(self)} versus "
                             f"{type(model_one_less_latent)}")
        if self.binarymap != model_one_less_latent.binarymap:
            raise ValueError('`model_one_less_latent` has different '
                             '`binarymap` than current object.')
        if self.n_latent_phenotypes - 1 != (model_one_less_latent
                                            .n_latent_phenotypes):
            raise ValueError('`model_one_less_latent` does not have 1 fewer '
                             'latent phenotype than current object.')

        self._likelihood_calc_params = (model_one_less_latent
                                        ._likelihood_calc_params)
        new_latenteffects = self._latenteffects.copy()
        assert new_latenteffects.shape == (self.n_latent_phenotypes,
                                           self._n_latent_effects + 1)
        new_epistasis_func_params = self._epistasis_func_params.copy()
        assert (new_epistasis_func_params.shape ==
                (self.n_latent_phenotypes,
                 len(self._epistasis_func_param_names)
                 ))
        for k in range(1, self.n_latent_phenotypes):
            ki = k - 1
            new_latenteffects[ki] = model_one_less_latent._latenteffects[ki]
            new_epistasis_func_params[ki] = (model_one_less_latent
                                             ._epistasis_func_params[ki])
        self._latenteffects = new_latenteffects
        self._epistasis_func_params = new_epistasis_func_params

    def __getstate__(self):
        """Clears the internal `_cache` before pickling.

        See: https://docs.python.org/3/library/pickle.html#object.__getstate__

        """
        self._cache = {}
        return self.__dict__

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
        if val.shape != (self.n_latent_phenotypes, self._n_latent_effects + 1):
            raise ValueError('invalid value for `_latenteffects`:\n'
                             f"shape should be: {self.n_latent_phenotypes} by "
                             f"{self._n_latent_effects + 1}\n"
                             f"but trying to set to: {val.shape}")
        if (not hasattr(self, '_latenteffects_val')) or (self._latenteffects
                                                         != val).any():
            self._cache = {}
            self._latenteffects_val = val.copy()
            self._latenteffects_val.flags.writeable = False

    @property
    def _likelihood_calc_params(self):
        """numpy.ndarray: Parameters for likelihood calculation."""
        return self._likelihood_calc_params_val

    @_likelihood_calc_params.setter
    def _likelihood_calc_params(self, val):
        if val.shape != (len(self._likelihood_calc_param_names),):
            raise ValueError('invalid length for `_likelihood_calc_params`')
        if ((not hasattr(self, '_likelihood_calc_params_val')) or
                (val != self._likelihood_calc_params).any()):
            self._cache = {}
            self._likelihood_calc_params_val = val.copy()
            self._likelihood_calc_params_val.flags.writeable = False

    @property
    def _epistasis_func_params(self):
        """numpy.ndarray: :meth:`AbstractEpistasis.epistasis_func` params."""
        return self._epistasis_func_params_val

    @_epistasis_func_params.setter
    def _epistasis_func_params(self, val):
        if val.shape != (self.n_latent_phenotypes,
                         len(self._epistasis_func_param_names)):
            raise ValueError('invalid value for `_epistasis_func_params`')
        if ((not hasattr(self, '_epistasis_func_params_val')) or
                (val != self._epistasis_func_params).any()):
            self._cache = {}
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
    def n_latent_phenotypes(self):
        """int: number of latent phenotypes, see :ref:`multi_latent`."""
        return self._n_latent_phenotypes

    @property
    def _binary_variants(self):
        r"""scipy.sparse.csr.csr_matrix: Binary variants with 1 in last column.

        As in Eq. :eq:`latent_phenotype_wt_vec` with :math:`\beta_{M+1}`.
        So this is a :math:`V` by :math:`M + 1` matrix.

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
    def aic(self):
        """float: Aikake Information Criterion given current log likelihood."""
        return 2 * self.nparams - 2 * self.loglik

    @property
    def nparams(self):
        """int: Total number of parameters in model."""
        return len(self._allparams)

    def _process_latent_phenotype_k(self, k):
        """Process latent phenotype number to 0-based index.

        Parameters
        ----------
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`)
            or `None`.

        Returns
        -------
        int
            If `k` is valid latent phenotype number, return `k - 1`. If `k`
            is `None` and there is just one latent phenotype, return
            0. If `k` is `None` and there are multiple latent phenotypes,
            raise an errror.

        """
        if k is None:
            if self.n_latent_phenotypes == 1:
                return 0
            else:
                raise ValueError('must set numerical value for `k` when '
                                 'there are multiple latent phenotypes')
        elif (1 <= k <= self.n_latent_phenotypes) and isinstance(k, int):
            return k - 1
        else:
            raise ValueError('`k` must be >= 1 and <= `n_latent_phenotypes`')

    def latent_phenotype_wt(self, k=None):
        r"""Latent phenotype of wildtype.

        Parameters
        ----------
        k : int or None
            Which latent phenotype to get (1 <= k <=
            :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`.

        Returns
        ---------
        float
            Wildtype latent phenotype, which is :math:`\beta_{\rm{wt}}` in
            Eq. :eq:`latent_phenotype` or :math:`\beta_{\rm{wt}}^k` in
            Eq. :eq:`latent_phenotype_multi`.

        """
        return self._latenteffects[self._process_latent_phenotype_k(k),
                                   self._n_latent_effects]

    @property
    def epistasis_func_params_dict(self):
        """dict: Parameters for the :ref:`global_epistasis_function`.

        Maps names of parameters defining the global epistasis function to
        their current values.

        """
        assert (self._epistasis_func_params.shape ==
                (self.n_latent_phenotypes,
                 len(self._epistasis_func_param_names)))
        if self.n_latent_phenotypes == 1:
            suffixed_names = self._epistasis_func_param_names
        else:
            suffixed_names = []
            for k in range(1, self.n_latent_phenotypes + 1):
                for name in self._epistasis_func_param_names:
                    suffixed_names.append(f"{name}_{k}")
        assert len(suffixed_names) == len(set(suffixed_names))
        assert len(suffixed_names) == self._epistasis_func_params.size
        return dict(zip(suffixed_names, self._epistasis_func_params.ravel()))

    @property
    def likelihood_calc_params_dict(self):
        """dict: Parameters for the :ref:`likelihood_calculation`.

        Maps names of parameters defining the likelihood calculation to
        their current values.

        """
        assert (len(self._likelihood_calc_params) ==
                len(self._likelihood_calc_param_names))
        return dict(zip(self._likelihood_calc_param_names,
                        self._likelihood_calc_params))

    # ------------------------------------------------------------------------
    # Methods to get phenotypes / mutational effects given current model state
    # ------------------------------------------------------------------------
    def phenotypes_frombinary(self,
                              binary_variants,
                              phenotype,
                              *,
                              wt_col=False,
                              k=None,
                              ):
        """Phenotypes from binary variant representations.

        Parameters
        ----------
        binary_variants : scipy.sparse.csr.csr_matrix or numpy.ndarray
            Binary variants in form used by
            :class:`dms_variants.binarymap.BinaryMap`.
        phenotype : {'latent', 'observed'}
            Calculate the latent or observed phenotype.
        wt_col : bool
            Set to `True` if `binary_variants` contains a terminal
            column of ones to enable calculations in the form given
            by Eq. :eq:`latent_phenotype_wt_vec`.
        k : int or None
            If `phenotype` is 'latent', which latent phenotype to use (1 <= k
            (<= :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`. Has no meaning
            if `phenotype` is 'observed'.

        Returns
        --------
        numpy.ndarray
            Latent phenotypes calculated using Eq. :eq:`latent_phenotype` or
            observed phenotypes calculated using Eq. :eq:`observed_phenotype`
            (or Eqs. :eq:`latent_phenotype_multi` or
            :eq:`observed_phenotype_multi`).

        """
        if len(binary_variants.shape) != 2:
            raise ValueError(f"`binary_variants` not 2D:\n{binary_variants}")
        if binary_variants.shape[1] != self._n_latent_effects + int(wt_col):
            raise ValueError(f"variants wrong length: {binary_variants.shape}")

        if phenotype == 'latent':
            ki = self._process_latent_phenotype_k(k)
            if wt_col:
                return binary_variants.dot(self._latenteffects[ki])
            else:
                return (binary_variants.dot(self._latenteffects[ki][: -1]) +
                        self.latent_phenotype_wt(k))
        elif phenotype == 'observed':
            if wt_col:
                latents = self._latenteffects.transpose()
                assert latents.shape[0] == binary_variants.shape[1]
                latent_phenos = binary_variants.dot(latents).transpose()
            else:
                latents = self._latenteffects.transpose()[:-1, ]
                assert latents.shape[0] == binary_variants.shape[1]
                latent_phenos = (binary_variants.dot(latents) +
                                 [self.latent_phenotype_wt(kj + 1) for kj in
                                  range(self.n_latent_phenotypes)]
                                 ).transpose()
            assert latent_phenos.shape == (self.n_latent_phenotypes,
                                           binary_variants.shape[0])
            observed_phenos = numpy.zeros(binary_variants.shape[0],
                                          dtype='float')
            for kj in range(self.n_latent_phenotypes):
                observed_phenos += self.epistasis_func(latent_phenos[kj],
                                                       kj + 1)
            return observed_phenos

    @property
    def latent_effects_df(self):
        """pandas.DataFrame: Latent effects of mutations.

        For each single mutation in :attr:`AbstractEpistasis.binarymap`, gives
        current predicted latent effect of that mutation. If there are multiple
        latent phenotypes (:attr:`AbstractEpistasis.n_latent_phenotypes` > 1),
        also indicates the phenotype number of the latent effect.

        """
        assert len(self.binarymap.all_subs) == self._latenteffects.shape[1] - 1
        d = {'mutation': self.binarymap.all_subs * self.n_latent_phenotypes,
             'latent_effect': self._latenteffects[:, : -1].ravel(),
             }
        if self.n_latent_phenotypes > 1:
            d['latent_phenotype_number'] = numpy.repeat(
                    range(1, self.n_latent_phenotypes + 1),
                    len(self.binarymap.all_subs))
        return pd.DataFrame(d)

    def add_phenotypes_to_df(self,
                             df,
                             *,
                             substitutions_col=None,
                             latent_phenotype_col='latent_phenotype',
                             observed_phenotype_col='observed_phenotype',
                             phenotype_col_overwrite=False,
                             unknown_as_nan=False,
                             ):
        """Add predicted phenotypes to data frame of variants.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing variants.
        substitutions_col : str or None
            Column in `df` giving variants as substitution strings in format
            that can be processed by :attr:`AbstractEpistasis.binarymap`.
            If `None`, defaults to the `substitutions_col` attribute of
            that binary map.
        latent_phenotype_col : str
            Column(s) added to `df` containing predicted latent phenotypes.
            If there are multiple latent phenotypes, this string is suffixed
            with the latent phenotype number (i.e., 'latent_phenotype_1').
        observed_phenotype_col : str
            Column added to `df` containing predicted observed phenotypes.
        phenotype_col_overwrite : bool
            If the specified latent or observed phenotype column already
            exist in `df`, overwrite it? If `False`, raise an error.
        unknown_as_nan : bool
            If some of the substitutions in a variant are not present in
            the model (not in :attr:`AbstractEpistasis.binarymap`) set the
            phenotypes to `nan` (not a number)? If `False`, raise an error.

        Returns
        -------
        pandas.DataFrame
            A copy of `df` with the phenotypes added. Phenotypes are predicted
            based on the current state of the model.

        """
        if substitutions_col is None:
            substitutions_col = self.binarymap.substitutions_col
        if substitutions_col not in df.columns:
            raise ValueError('`df` lacks `substitutions_col` '
                             f"{substitutions_col}")
        if self.n_latent_phenotypes == 1:
            latent_phenotype_cols = [latent_phenotype_col]
        else:
            latent_phenotype_cols = [f"{latent_phenotype_col}_{k}" for k in
                                     range(1, self.n_latent_phenotypes + 1)]
        if 2 + self.n_latent_phenotypes != len({substitutions_col,
                                                observed_phenotype_col,
                                                *latent_phenotype_cols}):
            raise ValueError('repeated name among `latent_phenotype_col`, '
                             '`observed_phenotype_col`, `substitutions_col`')
        for col in latent_phenotype_cols + [observed_phenotype_col]:
            if col in df.columns and not phenotype_col_overwrite:
                if not phenotype_col_overwrite:
                    raise ValueError(f"`df` already contains column {col}")

        # build binary variants as csr matrix
        row_ind = []  # row indices of elements that are one
        col_ind = []  # column indices of elements that are one
        nan_variant_indices = []  # indices of variants that are nan
        for ivariant, subs in enumerate(df[substitutions_col].values):
            try:
                for isub in self.binarymap.sub_str_to_indices(subs):
                    row_ind.append(ivariant)
                    col_ind.append(isub)
            except ValueError:
                if unknown_as_nan:
                    nan_variant_indices.append(ivariant)
                else:
                    raise ValueError('Variant has substitutions not in model:'
                                     f"\n{subs}\nMaybe use `unknown_as_nan`?")
        binary_variants = scipy.sparse.csr_matrix(
                            (numpy.ones(len(row_ind), dtype='int8'),
                             (row_ind, col_ind)),
                            shape=(len(df), self.binarymap.binarylength),
                            dtype='int8')

        df = df.copy()
        for col, k, phenotype in zip(
                latent_phenotype_cols + [observed_phenotype_col],
                list(range(1, self.n_latent_phenotypes + 1)) + [None],
                ['latent'] * self.n_latent_phenotypes + ['observed']
                ):
            vals = self.phenotypes_frombinary(binary_variants,
                                              phenotype,
                                              k=k)
            assert len(vals) == len(df)
            vals = vals.copy()  # needed because vals not might be writable
            vals[nan_variant_indices] = numpy.nan
            df[col] = vals
        return df

    @property
    def phenotypes_df(self):
        """pandas.DataFrame: Phenotypes of variants used to fit model.

        For each variant in :attr:`AbstractEpistasis.binarymap`, gives
        the current predicted latent and observed phenotypes as well
        as the functional score and its variance.

        """
        d = {self.binarymap.substitutions_col: (self.binarymap
                                                .substitution_variants),
             'func_score': self.binarymap.func_scores,
             'func_score_var': self.binarymap.func_scores_var
             }
        if self.n_latent_phenotypes == 1:
            d['latent_phenotype'] = self._latent_phenotypes(k=None)
        else:
            for k in range(1, self.n_latent_phenotypes + 1):
                d[f"latent_phenotype_{k}"] = self._latent_phenotypes(k=k)
        d['observed_phenotype'] = self._observed_phenotypes()
        return pd.DataFrame(d)

    def preferences(self, phenotype, base, *,
                    missing='average', exclude_chars=('*',),
                    returnformat='wide', stringency_param=1,
                    k=None):
        r"""Get preference of each site for each character.

        Use the latent or observed phenotype to estimate the preference
        :math:`\pi_{r,a}` of each site :math:`r` for each character (e.g.,
        amino acid) :math:`a`. These preferences can be displayed in logo plots
        or used as input to `phydms <https://jbloomlab.github.io/phydms/>`_
        in experimentally informed substitution models.

        The preferences are calculated from the phenotypes as follows. Let
        :math:`p_{r,a}` be the phenotype of the variant with the single
        mutation of site :math:`r` to :math:`a` (when :math:`a` is the wildtype
        character, then :math:`p_{r,a}` is the phenotype of the wildtype
        sequence). Then the preference :math:`\pi_{r,a}` is defined as

        .. math::

           \pi_{r,a} = \frac{b^{p_{r,a}}}{\sum_{a'} b^{p_{r,a'}}}

        where :math:`b` is the base for the exponent. This definition
        ensures that the preferences sum to one at each site.

        The alphabet from which the characters are drawn and the site
        numbers are extracted from :attr:`AbstractEpistasis.binarymap`.

        Note
        ----
        The "flatness" of the preferences is determined by the exponent base.
        A smaller `base` yields flatter preferences. There is no obvious "best"
        `base` as different values correspond to different linear scalings of
        of the phenotype. A recommended approach is simply to choose a value of
        `base` and then re-scale the preferences by using
        `phydms <https://jbloomlab.github.io/phydms/>`_ to optimize a
        stringency parameter (`see here <https://peerj.com/articles/3657>`_).
        The stringency parameter and the `base` chosen here both apply the
        same transformation to the data: linear scaling of the phenotypes.
        But note that `phydms <https://jbloomlab.github.io/phydms/>`_
        has an upper bound on the largest stringency parameter it can fit,
        so if you are hitting this upper bound then pre-scale the preferences
        to be less flat by using a larger value of `base`. In particular,
        the latent phenotypes from many of the epistasis models are scaled
        during fitting to have a relatively tight range, so you may need a
        large value of `base` such as 50.

        Parameters
        ----------
        phenotype : {'observed', 'latent'}
            Calculate the preferences from observed or latent phenotypes?
            Note that if there are multiple latent phenotypes, you must
            also set `k`.
        base : float
            Base to which the exponent is taken in computing the preferences.
        missing : {'average', 'site_average', 'error'}
            What to do when there is no estimate of the phenotype for one of
            the single mutants? Estimate the phenotype as the average of
            all single mutants, as the average of all single mutants at that
            site, or raise an error.
        exclude_chars : tuple or list
            Characters to exclude when calculating preferences (and when
            averaging values for missing mutants). For instance, you might
            want to exclude stop codons.
        returnformat : {'tidy', 'wide'}
            Return preferences in tidy or wide format data frame.
        stringency_param : float
            Re-scale preferences by this stringency parameter. This
            involves raising each preference to the power of
            `stringency_param`, and then re-normalizes. A similar
            effect can be achieved by changing `base`.
        k : int or None
            Which latent phenotype to use (1 <= k <=
            :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`. Has no
            meaning if `phenotype` is 'observed'.

        Returns
        -------
        pandas.DataFrame
            Data frame where first column is named 'site', other columns are
            named for each character, and rows give preferences for each site.

        """
        effects = self.single_mut_effects(phenotype,
                                          include_wildtype=False,
                                          standardize_range=False,
                                          k=k,
                                          )

        # get alphabet of non-excluded characters
        alphabet = [a for a in self.binarymap.alphabet
                    if a not in exclude_chars]

        return dms_variants.utils.scores_to_prefs(
                    df=effects[['mutation', 'effect']],
                    mutation_col='mutation',
                    score_col='effect',
                    base=base,
                    wt_score=0,
                    missing=missing,
                    alphabet=alphabet,
                    exclude_chars=exclude_chars,
                    returnformat=returnformat,
                    stringency_param=stringency_param,
                    )

    def single_mut_effects(self,
                           phenotype,
                           *,
                           include_wildtype=True,
                           standardize_range=True,
                           k=None,
                           ):
        """Effects of single mutations on latent or observed phenotype.

        For the effects on observed phenotype, this is how much the mutation
        changes the observed phenotype relative to wildtype. Effects are
        reported only for mutations present in `AbstractEpistasis.binarymap`.

        Parameters
        -----------
        phenotype : {'latent', 'observed'}
            Get effect on this phenotype. If there are multiple latent
            phenotypes, you must also set `k`.
        include_wildtype : bool
            Include the effect of "mutating" to wildtype identity at a site
            (always zero).
        standardize_range : bool
            Scale effects so that the mean absolute value effect is one
            (scaling is done before including wildtype).
        k : int or None
            Which latent phenotype to use (1 <= k <=
            :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`. Has no
            meaning if `phenotype` is 'observed'.

        Returns
        -------
        pandas.DataFrame
            The effects of all single mutations. Columns are:

              - 'mutation': mutation as str
              - 'wildtype': wildtype identity at site
              - 'site': site number
              - 'mutant': mutant identity at site
              - 'effect': effect of mutation on latent or observed phenotype

        """
        if phenotype == 'observed':
            phenotypecol = 'observed_phenotype'
        elif phenotype == 'latent':
            if self.n_latent_phenotypes == 1:
                phenotypecol = 'latent_phenotype'
            else:
                k = self._process_latent_phenotype_k(k) + 1
                phenotypecol = f"latent_phenotype_{k}"
        else:
            raise ValueError(f"invalid `phenotype` {phenotype}")

        # data frame with all observed single mutations and phenotypes
        df = self.add_phenotypes_to_df(
                    pd.DataFrame({'mutation': self.binarymap.all_subs}),
                    substitutions_col='mutation')

        # get wildtype phenotype
        wt_phenotype = (self.add_phenotypes_to_df(
                            pd.DataFrame({'mutation': ['']}),
                            substitutions_col='mutation')
                        [phenotypecol]
                        .values
                        [0]
                        )

        # subtract wildtype phenotype to get effects
        df['effect'] = df[phenotypecol] - wt_phenotype

        if standardize_range:
            df['effect'] = df['effect'] / df['effect'].abs().mean()

        # extract wildtype, site, mutant from mutation
        chars_regex = ''.join(map(re.escape, self.binarymap.alphabet))
        df = (df.join(df
                      ['mutation']
                      .str
                      .extract(rf"^(?P<wildtype>[{chars_regex}])" +
                               r'(?P<site>\-?\d+)' +
                               rf"(?P<mutant>[{chars_regex}])$")
                      )
              [['mutation', 'wildtype', 'site', 'mutant', 'effect']]
              )

        if include_wildtype:
            df = pd.concat([df,
                            (df
                             [['wildtype', 'site']]
                             .drop_duplicates()
                             .assign(mutant=lambda x: x['wildtype'],
                                     mutation=lambda x: (x['wildtype'] +
                                                         x['site'] +
                                                         x['mutant']),
                                     effect=0)
                             )
                            ],
                           sort=False)

        return (df
                .assign(site=lambda x: x['site'].astype(int))
                .sort_values(['effect', 'site', 'mutant'])
                .reset_index(drop=True)
                )

    def enrichments(self, observed_phenotypes, base=2):
        r"""Calculated enrichment ratios from observed phenotypes.

        Note
        ----
        In many cases, the functional scores used to fit the model are the
        logarithm (most commonly base 2) of experimentally observed enrichments
        For example, this is how functional scores are calculated by
        :meth:`dms_variants.codonvarianttable.CodonVariantTable.func_scores`.
        In that case, the predicted enrichment value :math:`E\left(v\right)`
        for each variant :math:`v` can be computed from the observed phenotype
        :math:`p\left(v\right)` as:

        .. math::

           E\left(v\right) = B^{p\left(v\right) - p\left(\rm{wt}\right)}

        where :math:`p\left(\rm{wt}\right)` is the observed phenotype
        of wildtype, and :math:`B` is the base for the exponent (by default
        :math:`B = 2`).

        Parameters
        ----------
        observed_phenotypes : float or numpy.ndarray
            The observed phenotypes.
        base : float
            The base for the exponent used to convert observed phenotypes
            to enrichments.

        Returns
        -------
        float or numpy.ndarray
            The enrichments.

        """
        observed_phenotype_wt = self.phenotypes_frombinary(
                        numpy.zeros((1, self._n_latent_effects)),
                        'observed')[0]
        return base**(observed_phenotypes - observed_phenotype_wt)

    # ------------------------------------------------------------------------
    # Methods / properties used for model fitting. Many of these are properties
    # that store the current state for the variants we are fitting, using the
    # cache so that they don't have to be re-computed needlessly.
    # ------------------------------------------------------------------------
    def fit(self, *, use_grad=True, optimize_method='L-BFGS-B', ftol=1e-7,
            clearcache=True):
        """Fit all model params to maximum likelihood values.

        Parameters
        ----------
        use_grad : bool
            Use analytical gradients to help with fitting.
        optimize_method : {'L-BFGS-B', 'TNC'}
            Optimization method used by `scipy.optimize.minimize`.
        ftol : float
            Function convergence tolerance for optimization, used by
            `scipy.optimize.minimize`.
        clearcache : bool
            Clear the cache after model fitting? This slightly increases
            the time needed to compute properties after fitting, but
            greatly saves memory usage.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The results of optimizing the full model.

        """
        # Least squares fit of latent effects for reasonable initial values
        # for the first latent phenotype:
        if self.n_latent_phenotypes == 1:
            _ = self._fit_latent_leastsquares(self.n_latent_phenotypes,
                                              self.binarymap.func_scores)
            self._prescale_params(k=1,
                                  g_k_range=(min(self.binarymap.func_scores),
                                             max(self.binarymap.func_scores)),
                                  )
        else:
            assert self.n_latent_phenotypes > 1
            phenos_Kminus1 = list(range(1, self.n_latent_phenotypes))
            residuals = (self.binarymap.func_scores -
                         self._observed_phenotypes(phenos_Kminus1))
            max_abs_residual = numpy.abs(residuals).max()
            self._prescale_params(
                        k=self.n_latent_phenotypes,
                        g_k_range=(-max_abs_residual, max_abs_residual))

        # optimize full model by maximum likelihood
        optres = scipy.optimize.minimize(
                        fun=self._loglik_by_allparams,
                        jac=self._dloglik_by_allparams if use_grad else None,
                        x0=self._allparams,
                        method=optimize_method,
                        bounds=self._allparams_bounds,
                        options={'ftol': ftol},
                        )
        if not optres.success:
            raise EpistasisFittingError(
                    f"Fitting of {self.__class__.__name__} failed after "
                    f"{optres.nit} iterations. Message:\n{optres.message}\n"
                    f"{optres}")
        self._allparams = optres.x

        # postscale parameters to desired range
        self._postscale_params()

        if clearcache:
            self._cache = {}

        return optres

    def _loglik_by_allparams(self, allparams, negative=True):
        """(Negative) log likelihood after setting all parameters.

        Note
        ----
        Calling this method alters the internal model parameters, so only
        use if you understand what you are doing.

        Parameters
        ----------
        allparams : numpy.ndarray
            Parameters used to set :meth:`AbstractEpistasis._allparams`.
        negative : bool
            Return negative log likelihood. Useful if using a minimizer to
            optimize.

        Returns
        -------
        float
            (Negative) log likelihood after setting parameters to `allparams`.

        """
        self._allparams = allparams
        if negative:
            return -self.loglik
        else:
            return self.loglik

    def _dloglik_by_allparams(self, allparams, negative=True):
        """(Negative) derivative of log likelihood with respect to all params.

        Note
        ----
        Calling this method alters the internal model parameters, so only
        use if you understand what you are doing.

        Parameters
        ----------
        allparams: numpy.ndarray
            Parameters used to set :meth:`AbstractEpistasis._allparams`.
        negative : bool
            Return negative log likelihood. Useful if using a minimizer to
            optimize.

        Returns
        --------
        numpy.ndarray
            (Negative) derivative of log likelihood with respect to
            :meth:`AbstractEpistasis._allparams`.

        """
        self._allparams = allparams
        val = numpy.concatenate((*[self._dloglik_dlatent(k) for k in
                                   range(1, self.n_latent_phenotypes + 1)],
                                 self._dloglik_dlikelihood_calc_params,
                                 *[self._dloglik_depistasis_func_params(k) for
                                   k in range(1, self.n_latent_phenotypes + 1)]
                                 )
                                )
        assert val.shape == (self.nparams,)
        if negative:
            return -val
        else:
            return val

    @property
    def _allparams(self):
        """numpy.ndarray: All model parameters in a single array.

        Note
        ----
        This property should only be used for purposes in which it is
        necessary to get or set all params in a single vector (typically
        for model optimiziation), **not** to access the values of specific
        parameters, since the order of parameters in the array may change
        in future implementations.

        """
        val = numpy.concatenate((self._latenteffects.ravel(),
                                 self._likelihood_calc_params,
                                 self._epistasis_func_params.ravel(),
                                 )
                                )
        return val

    @_allparams.setter
    def _allparams(self, val):
        if val.shape != (self.nparams,):
            raise ValueError(f"invalid `_allparams`: {val}")

        istart = 0
        ncol = self._n_latent_effects + 1  # add 1 for wt latent phenotype
        assert (self.n_latent_phenotypes, ncol) == self._latenteffects.shape
        n = ncol * self.n_latent_phenotypes
        assert self._latenteffects.size == n
        self._latenteffects = val[istart: n].reshape(self.n_latent_phenotypes,
                                                     ncol)
        istart += n

        assert self._likelihood_calc_params.ndim == 1
        n = len(self._likelihood_calc_params)
        self._likelihood_calc_params = val[istart: istart + n]
        istart += n

        ncol = len(self._epistasis_func_param_names)
        assert ((self.n_latent_phenotypes, ncol) ==
                self._epistasis_func_params.shape)
        n = ncol * self.n_latent_phenotypes
        assert self._epistasis_func_params.size == n
        self._epistasis_func_params = (val[istart: istart + n]
                                       .reshape(self.n_latent_phenotypes,
                                                ncol)
                                       )

    @property
    def _allparams_bounds(self):
        """list: Bounds for :meth:`AbstractEpistasis._allparams`.

        Can be passed to `scipy.optimize.minimize`.

        """
        bounds = ([(None, None)] * self._latenteffects.size +
                  self._likelihood_calc_param_bounds +
                  self._epistasis_func_param_bounds * self.n_latent_phenotypes)
        assert len(bounds) == len(self._allparams), (
                f"len(bounds) = {len(bounds)}\n"
                f"_allparams.shape = {self._allparams.shape}")
        return bounds

    def _latent_phenotypes(self, k=None):
        """Latent phenotypes.

        Parameters
        -----------
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`),
            or can be `None` if just one latent phenotype.

        Returns
        -------
        numpy.ndarray
            Latent phenotypes of all variants being used to fit model.

        """
        k = self._process_latent_phenotype_k(k) + 1
        key = f"_latent_phenotypes_{k}"
        if key not in self._cache:
            self._cache[key] = self.phenotypes_frombinary(
                                binary_variants=self._binary_variants,
                                phenotype='latent',
                                wt_col=True,
                                k=k,
                                )
            self._cache[key].flags.writeable = False
        return self._cache[key]

    def _observed_phenotypes(self, latent_phenos='all'):
        r"""Observed phenotypes of variants being fit.

        Parameters
        ----------
        latent_phenos : 'all' or list
            The numbers (:math:`k = 1, 2, \ldots...) of the latent phenotypes
            used to calculate the observed phenotype. If 'all' use all
            latent phenotypes. Otherwise only include the terms in
            Eq. :eq:`observed_phenotype_multi` corresponding to the
            :math:`k` values listed here.

        Returns
        --------
        numpy.ndarray
            Observed phenotypes.

        """
        if latent_phenos == 'all':
            latent_phenos = list(range(1, self.n_latent_phenotypes + 1))
        if isinstance(latent_phenos, list):
            if len(latent_phenos) != len(set(latent_phenos)):
                raise ValueError('duplicate entries in `latent_phenos`')
            if not set(latent_phenos).issubset(
                    set(range(1, self.n_latent_phenotypes + 1))):
                raise ValueError('invalid entries in `latent_phenos`')
            if not latent_phenos:
                raise ValueError('empty `latent_phenos`')
            latent_phenos = sorted(latent_phenos)
        else:
            raise ValueError('`latent_phenos` not a list')
        key = f"_observed_phenotypes_{'_'.join(map(str, latent_phenos))}"
        if key not in self._cache:
            observed_phenos = self.epistasis_func(
                    self._latent_phenotypes(latent_phenos[0]),
                    k=latent_phenos[0]
                    ).copy()
            for k in latent_phenos[1:]:
                observed_phenos += self.epistasis_func(
                                        self._latent_phenotypes(k),
                                        k=k)
            self._cache[key] = observed_phenos
            self._cache[key].flags.writeable = False
        return self._cache[key]

    def _dobserved_phenotypes_dlatent(self, k=None):
        """Derivative observed phenotype by latent effects.

        Parameters
        ----------
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`),
            or can be `None` if just one latent phenotype.

        Returns
        -------
        scipy.parse.csr_matrix
            Derivative observed pheno by latent effects for phenotype
            :math:`k`. See Eq. :eq:`dobserved_phenotype_dlatent_effect`.
            This is a :math:`M + 1` by :math:`V` matrix.

        """
        k = self._process_latent_phenotype_k(k) + 1
        key = f"_dobserved_phenotypes_dlatent_{k}"
        if key not in self._cache:
            self._cache[key] = (
                    self._binary_variants
                    .transpose()  # convert from V by M to M by V
                    .multiply(self._depistasis_func_dlatent(
                                self._latent_phenotypes(k),
                                k=k))
                    )
            assert self._cache[key].shape == (self._n_latent_effects + 1,
                                              self.binarymap.nvariants)
        return self._cache[key]

    def _dloglik_dlatent(self, k=None):
        """Derivative log likelihood by latent effects.

        Parameters
        ----------
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`),
            or can be `None` if just one latent phenotype.

        Returns
        -------
        numpy.ndarray:
            Derivative log likelihood by latent effects for phenotype
            :math:`k`. See Eq. :eq:`dloglik_dlatent_effect`.

        """
        k = self._process_latent_phenotype_k(k) + 1
        key = f"_dloglik_dlatent_{k}"
        if key not in self._cache:
            self._cache[key] = self._dobserved_phenotypes_dlatent(k).dot(
                    self._dloglik_dobserved_phenotype)
            self._cache[key].flags.writeable = False
            assert self._cache[key].shape == (self._n_latent_effects + 1,)
        return self._cache[key]

    def _fit_latent_leastsquares(self, k=None, fit_to=None):
        """Least-squares fit latent effects for quick "reasonable" values.

        Parameters
        ----------
        k : int or None
            Fit effects for this latent phenotype (1 <= `k` <=
            `n_latent_phenotypes`); can be `None` if just one latent phenotype.
        fit_to : numpy.ndarray or None
            Fit latent effects to these values. If `None`, fits to
            :attr:`AbstractEpistasis.binarymap.func_scores`.

        Returns
        -------
        tuple
            Results of fitting described here:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html

        In addition to returning the fitting results, sets the latent
        effects to the new fit value.

        """
        ki = self._process_latent_phenotype_k(k)
        if fit_to is None:
            fit_to = self.binarymap.func_scores
        else:
            if fit_to.shape != self.binarymap.func_scores.shape:
                raise ValueError('`invalid shape for `fit_to`')

        # fit by least squares
        fitres = scipy.sparse.linalg.lsqr(
                    A=self._binary_variants,
                    b=fit_to,
                    x0=self._latenteffects[ki],
                    )

        # use fit result to update latenteffects
        new_latenteffects = self._latenteffects.copy()
        new_latenteffects[ki] = fitres[0]
        self._latenteffects = new_latenteffects

        return fitres

    # ------------------------------------------------------------------------
    # Abstract methods for global epistasis func, must implement in subclasses
    # specific for that epistasis model.
    # ------------------------------------------------------------------------
    @abc.abstractmethod
    def epistasis_func(self, latent_phenotype, k=None):
        """The :ref:`global_epistasis_function` :math:`g`.

        Parameters
        -----------
        latent_phenotype : numpy.ndarray
            Latent phenotype(s) of one or more variants.
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`),
            or can be `None` if just one latent phenotype. See
            Eq. :ref:`multi_latent`.

        Returns
        -------
        numpy.ndarray
            Result of applying global epistasis function :math:`g_k` to latent
            phenotypes.

        """
        return NotImplementedError

    @abc.abstractmethod
    def _depistasis_func_dlatent(self, latent_phenotype, k=None):
        """Derivative of epistasis function by latent phenotype.

        Parameters
        -----------
        latent_phenotype : numpy.ndarray
            Latent phenotype(s) of one or more variants.
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`),
            or can be `None` if just one latent phenotype. See
            Eq. :ref:`multi_latent`.

        Returns
        -------
        numpy.ndarray
            Derivative of :meth:`NoEpistasis.epistasis_func` for
            latent phenotype `k` evaluated at `latent_phenotype`.

        """
        return NotImplementedError

    @abc.abstractmethod
    def _dloglik_depistasis_func_params(self, k=None):
        """Deriv log likelihood by `_epistasis_func_params` for :math:`g_k`.

        Parameters
        ----------
        k : int or None
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`),
            or can be `None` if just one latent phenotype. See
            Eq. :ref:`multi_latent`.

        Returns
        -------
        numpy.ndarray
            The derivative with respect to the parameters for epistasis
            function :math:`g_k`.

        """
        return NotImplementedError

    @property
    @abc.abstractmethod
    def _epistasis_func_param_names(self):
        """list: Names of :meth:`AbstractEpistasis._epistasis_func_params`.

        When there are multiple latent phenotypes and global epistasis
        functions (:ref:`multi_latent`) still just provide a list with
        one copy of each parameter name and they will be suffixed with
        :math:`k` by :meth:`AbstractEpistasis.epistasis_func_params_dict`.

        """
        return NotImplementedError

    @property
    @abc.abstractmethod
    def _epistasis_func_param_bounds(self):
        """list: Bounds for the epistasis function parameters.

        For each entry in :meth:`AbstractEpistasis._epistasis_func_param_names`
        a 2-tuple gives the lower and upper bound for optimization by
        `scipy.optimize.minimize`.

        """
        return NotImplementedError

    @abc.abstractmethod
    def _prescale_params(self, k, g_k_range):
        """Set / scale parameters prior to the global fitting.

        This method is designed to set / re-scale parameters relevant
        to the latent phenotype :math:`k` and its associated global
        epistasis function prior to fitting. The re-scaling differs
        for different model classes, and is implemented in concrete
        subclasses. See :ref:`fitting_workflow`.

        Importantly, this is the method that sets initial values
        for `_epistasis_func_params`.

        Parameters
        -----------
        k : int
            Latent phenotype number (1 <= `k` <= `n_latent_phenotypes`).
        g_k_range : tuple
            Gives desired min and max of :math:`g_k`.

        """
        return NotImplementedError

    @abc.abstractmethod
    def _postscale_params(self):
        """Rescale parameters after the global fitting.

        This is an abstract method, any actual post-scaling is done in concrete
        subclasses.

        """
        return NotImplementedError

    # ------------------------------------------------------------------------
    # Abstract methods for likelihood calculations, implement in subclasses
    # ------------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def loglik(self):
        """float: Current log likelihood of model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _likelihood_calc_param_names(self):
        """list: Names of :meth:`AbstractEpistasis._likelihood_calc_params`."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _init_likelihood_calc_params(self):
        """numpy.ndarray: Initial `_likelihood_calc_params` values."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def _likelihood_calc_param_bounds(self):
        """list: Bounds for the likelihood calculation parameters.

        For entries in :meth:`AbstractEpistasis._likelihood_calc_param_names`,
        a 2-tuple gives the lower and upper bound for optimization by
        `scipy.optimize.minimize`.

        """
        return NotImplementedError

    @property
    @abc.abstractmethod
    def _dloglik_dobserved_phenotype(self):
        """numpy.ndarray: Derivative log likelihood by observed phenotype."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _dloglik_dlikelihood_calc_params(self):
        """numpy.ndarray: Derivative log lik by `_likelihood_calc_params`."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _zero_wt_observed_pheno(self):
        """bool: Re-scale wildtype observed phenotype to 0 after fitting?

        Should be set to `False` for likelihood calculation methods that
        fit observed phenotypes directly to functional scores directly,
        and `True` for those that only fit observed phenotypes up to an
        arbitrary additive constant.

        """
        raise NotImplementedError


class CauchyLikelihood(AbstractEpistasis):
    """Cauchy likelihood calculation.

    Note
    ----
    Subclass of :class:`AbstractEpistasis` that implements the
    :ref:`cauchy_likelihood`.

    """

    @property
    def loglik(self):
        """float: Current log likelihood from Eq. :eq:`loglik_cauchy`."""
        key = 'loglik'
        if key not in self._cache:
            scales = numpy.sqrt(self._pseudo_variances)
            if not (scales > 0).all():
                raise ValueError('scales not all > 0')
            self._cache[key] = (scipy.stats.cauchy.logpdf(
                                    self.binarymap.func_scores,
                                    loc=self._observed_phenotypes(),
                                    scale=scales)
                                ).sum()
        return self._cache[key]

    def _fit_latent_leastsquares(self, k=None, fit_to=None):
        r"""Also get initial value for scale parameter.

        Overrides :meth:`AbstractEpistasis._fit_latent_leastsquares`
        to make initial estimate of :math:`\gamma^2` as residual not
        from functional score variance. This is based on the supposition
        that the scale parameter can be treated like the variance for
        a Gaussian distribution (not sure how good this supposition is...).

        """
        fitres = super()._fit_latent_leastsquares(k=k, fit_to=fit_to)
        residuals2 = fitres[3]**2
        if self.binarymap.func_scores_var is None:
            scale_param2 = max(residuals2 / self.binarymap.nvariants,
                               self._NEARLY_ZERO)
        else:
            scale_param2 = max((residuals2 -
                                self.binarymap.func_scores_var.sum()
                                ) / self.binarymap.nvariants,
                               self._NEARLY_ZERO)
        self._likelihood_calc_params = numpy.sqrt([scale_param2])

    @property
    def _likelihood_calc_param_names(self):
        r"""list: Likelihood calculation parameter names.

        For :class:`CauchyLikelihood`, this is the scale parameter
        :math:`\gamma`.

        """
        return ['scale_parameter']

    @property
    def _init_likelihood_calc_params(self):
        r"""numpy.ndarray: Initial `_likelihood_calc_params`.

        The initial scale parameter :math:`\gamma` is 1.

        """
        init_d = {'scale_parameter': 1.0}
        return numpy.array([init_d[name] for name in
                            self._likelihood_calc_param_names],
                           dtype='float')

    @property
    def _likelihood_calc_param_bounds(self):
        r"""list: Bounds for likelihood calculation parameters.

        For :class:`CauchyLikelihood`, :math:`\gamma` must be > 0.

        """
        bounds_d = {'scale_parameter': (self._NEARLY_ZERO, None)}
        return [bounds_d[name] for name in self._likelihood_calc_param_names]

    @property
    def _dloglik_dobserved_phenotype(self):
        r"""numpy.ndarray: Derivative of log likelihood by observed phenotype.

        Calculated using Eq. :eq:`dloglik_cauchy_dobserved_phenotype`.

        """
        key = '_dloglik_dobserved_phenotype'
        if key not in self._cache:
            diff = self.binarymap.func_scores - self._observed_phenotypes()
            self._cache[key] = 2 * diff / (self._pseudo_variances + diff**2)
            self._cache[key].flags.writeable = False
        return self._cache[key]

    @property
    def _dloglik_dlikelihood_calc_params(self):
        """numpy.ndarray: Derivative log lik by `_likelihood_calc_params`.

        See Eq. :eq:`dloglik_cauchy_dscale_parameter`.

        """
        key = '_dloglik_dlikelihood_calc_params'
        if key not in self._cache:
            scale_param = self.likelihood_calc_params_dict['scale_parameter']
            diff2 = (self.binarymap.func_scores -
                     self._observed_phenotypes())**2
            self._cache[key] = numpy.array([
                                (scale_param * (diff2 - self._pseudo_variances)
                                 / (self._pseudo_variances *
                                    (self._pseudo_variances + diff2)
                                    )
                                 ).sum()
                                ])
            self._cache[key].flags.writeable = False
            assert self._cache[key].shape == self._likelihood_calc_params.shape
        return self._cache[key]

    @property
    def _pseudo_variances(self):
        r"""numpy.ndarray: Functional score variance plus scale param squared.

        :math:`\sigma_{y_v}^2 + \gamma^2` in Eq. :eq:`loglik_cauchy`.

        """
        key = '_pseudo_variances'
        if key not in self._cache:
            scale_param = self.likelihood_calc_params_dict['scale_parameter']
            if self.binarymap.func_scores_var is not None:
                var = self.binarymap.func_scores_var + scale_param**2
            else:
                var = numpy.full(self.binarymap.nvariants, scale_param**2)
            if (var <= 0).any():
                raise ValueError('variance <= 0')
            self._cache[key] = var
            self._cache[key].flags.writeable = False
        return self._cache[key]

    @property
    def _zero_wt_observed_pheno(self):
        """False: Do not re-scale wildtype observed phenotype to 0."""
        return False


class BottleneckLikelihood(AbstractEpistasis):
    r"""Bottleneck likelihood calculation.

    Note
    ----
    Subclass of :class:`AbstractEpistasis` that implements the
    :ref:`bottleneck_likelihood`.

    Note
    ----
    The :attr:`AbstractEpistasis.binarymap` must have non-`None`
    counts in :attr:`dms_variants.binarymap.BinaryMap.n_pre` and
    :attr:`dms_variants.binarymap.BinaryMap.n_post`, since
    as described in :ref:`bottleneck_likelihood`, the model is
    actually fit to the :attr:`BottleneckLikelihood.f_pre` and
    :attr:`BottleneckLikelihood.f_post` values calculated from
    these counts.

    Parameters
    ----------
    bottleneck : float
        The estimated size of the experimental bottleneck between the
        pre- and post-selection conditions. This is the :math:`N_{\rm{bottle}}`
        parameter described in :ref:`bottleneck_likelihood`.
    pseudocount : float
        The pseudocount used when converting the counts to frequencies
        vi Eq. :eq:`f_v_pre_post`.
    base : float
        The exponent base in Eq. :eq:`n_v_bottle` used when exponentiating
        the observed phenotypes :math:`p\left(v\right)`. It is written
        as 2 in Eq. :eq:`n_v_bottle`.

    """

    def __init__(self,
                 binarymap,
                 bottleneck,
                 *,
                 n_latent_phenotypes=1,
                 model_one_less_latent=None,
                 pseudocount=0.5,
                 base=2.0,
                 ):
        """See main class docstring."""
        if pseudocount <= 0:
            raise ValueError('`pseudocount` must be > 0')
        for cond in ['pre', 'post']:
            n = getattr(binarymap, f"n_{cond}")
            if n is None:
                raise ValueError(f"`binarymap.n_{cond}` is `None`")
            elif n.shape != (binarymap.nvariants,):
                raise ValueError(f"invalid `binarymap.n_{cond}` shape")
            if (n < 0).any():
                raise ValueError(f"negative values in `binarymap.n_{cond}`")
            f = n + pseudocount
            f = f / f.sum()
            if (f <= 0).any():
                raise ValueError(f"non-positive values in `f_{cond}`")
            if (f < self._NEARLY_ZERO).any():
                warnings.warn(f"`f_{cond}` has values that are nearly zero "
                              'which *might* cause numerical issues. Consider '
                              'increasing `pseudocount` if you have fitting '
                              'problems',
                              EpistasisFittingWarning)
            f.flags.writeable = False
            setattr(self, f"_f_{cond}", f)

        if base <= 0:
            raise ValueError(f"invalid `base` of {base}")
        self._base = base

        if bottleneck <= 0:
            raise ValueError('`bottleneck` must be > 0')
        self._bottleneck = bottleneck

        super().__init__(binarymap,
                         n_latent_phenotypes=n_latent_phenotypes,
                         model_one_less_latent=model_one_less_latent,
                         )

    @property
    def bottleneck(self):
        r"""int: Bottleneck pre- to post-selection, :math:`N_{\rm{bottle}]`."""
        return self._bottleneck

    @property
    def f_pre(self):
        r"""numpy.ndarray: Pre-selection frequency of each variant.

        The :math:`f_v^{\rm{pre}}` values in Eq. :eq:`f_v_pre_post`.

        """
        return self._f_pre

    @property
    def f_post(self):
        r"""numpy.ndarray: Post-selection frequency of each variant.

        The :math:`f_v^{\rm{post}}` values in Eq. :eq:`f_v_pre_post`.

        """
        return self._f_post

    @property
    def loglik(self):
        """float: Current log likelihood from Eq. :eq:`loglik_bottleneck`."""
        key = 'loglik'
        if key not in self._cache:
            self._cache[key] = (self._n_v_bottle * self._log_N_bottle_f_pre -
                                scipy.special.loggamma(self._n_v_bottle + 1)
                                ).sum() - self.bottleneck
        return self._cache[key]

    @property
    def _likelihood_calc_param_names(self):
        r"""list: Likelihood calculation parameter names.

        For :class:`BottleneckLikelihood`, there are no such parameters as
        :math:`N_{\rm{bottle}}` must be determined experimentally.

        """
        return []

    @property
    def _init_likelihood_calc_params(self):
        """numpy.ndarray: Initial `_likelihood_calc_params`."""
        return numpy.array([], dtype='float')

    @property
    def _likelihood_calc_param_bounds(self):
        """list: Bounds for likelihood calculation parameters."""
        return []

    @property
    def _dloglik_dobserved_phenotype(self):
        """numpy.ndarray: Derivative of log likelihood by observed phenotype.

        Calculating using Eq. :eq:`dloglik_bottleneck_dobserved_phenotype`.

        """
        key = '_dloglik_dobserved_phenotype'
        if key not in self._cache:
            self._cache[key] = (
                    numpy.log(2) *
                    self.f_pre *
                    self._base_to_observed_pheno *
                    self.bottleneck *
                    (self.f_post /
                     self._base_to_observed_pheno *
                     (self._log_N_bottle_f_pre -
                      self._digamma_n_v_bottle_1
                      )
                     ).sum() -
                    numpy.log(2) *
                    self._n_v_bottle *
                    (self._log_N_bottle_f_pre -
                     self._digamma_n_v_bottle_1
                     )
                    )
            self._cache[key].flags.writeable = False
        assert self._cache[key].shape == (self.binarymap.nvariants,)
        assert numpy.isfinite(self._cache[key]).all()
        return self._cache[key]

    @property
    def _dloglik_dlikelihood_calc_params(self):
        """numpy.ndarray: Derivative of log lik by `_likelihood_calc_params`"""
        return numpy.array([], dtype='float')

    @property
    def _base_to_observed_pheno(self):
        r"""numpy.ndarray: `base` raised to observed phenotype.

        This is :math:`2^{p\left(v\right)` in Eq. :eq:`n_v_bottle`.

        """
        key = '_base_to_observed_pheno'
        if key not in self._cache:
            self._cache[key] = self._base**self._observed_phenotypes()
            self._cache[key].flags.writeable = False
        assert self._cache[key].shape == (self.binarymap.nvariants,)
        assert numpy.isfinite(self._cache[key]).all()
        return self._cache[key]

    @property
    def _log_N_bottle_f_pre(self):
        r"""numpy.ndarray: Log of N_bottle * f_v_pre.

        :math:`\ln\left(N_{\rm{bottle}} f_v^{\rm{pre}}\right)` in
        Eq. :eq:`loglik_bottleneck`.

        """
        key = '_log_N_bottle_f_pre'
        if key not in self._cache:
            self._cache[key] = numpy.log(self.bottleneck * self.f_pre)
            self._cache[key].flags.writeable = False
        assert self._cache[key].shape == (self.binarymap.nvariants,)
        assert numpy.isfinite(self._cache[key]).all()
        return self._cache[key]

    @property
    def _n_v_bottle(self):
        r"""numpy.ndarray: :math:`n_v^{\rm{bottle}}` (Eq. :eq:`n_v_bottle`)."""
        key = '_n_v_bottle'
        if key not in self._cache:
            sumterm = (self.f_pre * self._base_to_observed_pheno).sum()
            self._cache[key] = (self.f_post *
                                self.bottleneck *
                                sumterm /
                                self._base_to_observed_pheno)
            self._cache[key].flags.writeable = False
        assert self._cache[key].shape == (self.binarymap.nvariants,)
        assert numpy.isfinite(self._cache[key]).all()
        assert (self._cache[key] >= 0).all()
        return self._cache[key]

    @property
    def _digamma_n_v_bottle_1(self):
        r"""numpy.ndarray: :math:`\psi_0\left(n_v^{\rm{bottle}} + 1\right)`."""
        key = '_digamma_n_v_bottle_1'
        if key not in self._cache:
            self._cache[key] = scipy.special.digamma(self._n_v_bottle + 1)
            self._cache[key].flags.writeable = False
        assert self._cache[key].shape == (self.binarymap.nvariants,)
        assert numpy.isfinite(self._cache[key]).all()
        return self._cache[key]

    @property
    def _zero_wt_observed_pheno(self):
        """True: Re-scale wildtype observed phenotype to 0 after fitting."""
        return True


class GaussianLikelihood(AbstractEpistasis):
    """Gaussian likelihood calculation.

    Note
    ----
    Subclass of :class:`AbstractEpistasis` that implements the
    :ref:`gaussian_likelihood`.

    """

    @property
    def loglik(self):
        """float: Current log likelihood from Eq. :eq:`loglik_gaussian`."""
        key = 'loglik'
        if key not in self._cache:
            standard_devs = numpy.sqrt(self._variances)
            if not (standard_devs > 0).all():
                raise ValueError('standard deviations not all > 0')
            self._cache[key] = (scipy.stats.norm.logpdf(
                                    self.binarymap.func_scores,
                                    loc=self._observed_phenotypes(),
                                    scale=standard_devs)
                                ).sum()
        return self._cache[key]

    def _fit_latent_leastsquares(self, k=None, fit_to=None):
        r"""Also get initial value for HOC epistasis.

        Overrides :meth:`AbstractEpistasis._fit_latent_leastsquares`
        to make initial estimate of :math:`\sigma^2_{\rm{HOC}}` as
        residual not from functional score variance.

        """
        fitres = super()._fit_latent_leastsquares(k=k, fit_to=fit_to)
        residuals2 = fitres[3]**2
        if self.binarymap.func_scores_var is None:
            epistasis_HOC = max(residuals2 / self.binarymap.nvariants,
                                self._NEARLY_ZERO)
        else:
            epistasis_HOC = max((residuals2 -
                                 self.binarymap.func_scores_var.sum()
                                 ) / self.binarymap.nvariants,
                                self._NEARLY_ZERO)
        self._likelihood_calc_params = numpy.array([epistasis_HOC])

    @property
    def _likelihood_calc_param_names(self):
        r"""list: Likelihood calculation parameter names.

        For :class:`GaussianLikelihood`, this :math:`\sigma^2_{\rm{HOC}}`.

        """
        return ['epistasis_HOC']

    @property
    def _init_likelihood_calc_params(self):
        r"""numpy.ndarray: Initial `_likelihood_calc_params`.

        The initial HOC epistasis :math:`\sigma^2_{\rm{HOC}}` is 1.

        """
        init_d = {'epistasis_HOC': 1.0}
        return numpy.array([init_d[name] for name in
                            self._likelihood_calc_param_names],
                           dtype='float')

    @property
    def _likelihood_calc_param_bounds(self):
        r"""list: Bounds for likelihood calculation parameters.

        For :class:`GaussianLikelihood`, :math:`\sigma^2_{\rm{HOC}}` must
        be > 0.

        """
        bounds_d = {'epistasis_HOC': (self._NEARLY_ZERO, None)}
        return [bounds_d[name] for name in self._likelihood_calc_param_names]

    @property
    def _dloglik_dobserved_phenotype(self):
        r"""numpy.ndarray: Derivative of log likelihood by observed phenotype.

        Calculated using Eq. :eq:`dloglik_gaussian_dobserved_phenotype`.

        """
        key = '_dloglik_dobserved_phenotype'
        if key not in self._cache:
            self._cache[key] = (self.binarymap.func_scores -
                                self._observed_phenotypes()) / self._variances
            self._cache[key].flags.writeable = False
        return self._cache[key]

    @property
    def _dloglik_dlikelihood_calc_params(self):
        """numpy.ndarray: Derivative log lik by `_likelihood_calc_params`.

        See Eq. :eq:`dloglik_gaussian_depistasis_HOC`.

        """
        key = '_dloglik_dlikelihood_calc_params'
        if key not in self._cache:
            self._cache[key] = numpy.array([
                0.5 *
                (self._dloglik_dobserved_phenotype**2 -
                 1 / self._variances).sum()
                ])
            self._cache[key].flags.writeable = False
            assert self._cache[key].shape == self._likelihood_calc_params.shape
        return self._cache[key]

    @property
    def _variances(self):
        r"""numpy.ndarray: Functional score variance plus HOC epistasis.

        :math:`\sigma_{y_v}^2 + \sigma_{\rm{HOC}}^2` in
        Eq. :eq:`loglik_gaussian`.

        """
        key = '_variances'
        if key not in self._cache:
            epistasis_HOC = self.likelihood_calc_params_dict['epistasis_HOC']
            if self.binarymap.func_scores_var is not None:
                var = self.binarymap.func_scores_var + epistasis_HOC
            else:
                var = numpy.full(self.binarymap.nvariants, epistasis_HOC)
            if (var <= 0).any():
                raise ValueError('variance <= 0')
            self._cache[key] = var
            self._cache[key].flags.writeable = False
        return self._cache[key]

    @property
    def _zero_wt_observed_pheno(self):
        """False: Do not re-scale wildtype observed phenotype to 0."""
        return False


class NoEpistasis(AbstractEpistasis):
    """Non-epistatic model.

    Note
    ----
    Subclass of :class:`AbstractEpistasis` that implements the
    :ref:`no_epistasis_function`.

    """

    def epistasis_func(self, latent_phenotype, k=None):
        """Global epistasis function :math:`g` in Eq. :eq:`noepistasis`.

        Concrete implementation of :meth:`AbstractEpistasis.epistasis_func`.

        """
        return latent_phenotype

    def _depistasis_func_dlatent(self, latent_phenotype, k=None):
        """Derivative of `epistasis_func` by latent phenotype.

        Concrete implementation of
        :meth:`AbstractEpistasis._depistasis_func_dlatent`.

        """
        return numpy.ones(latent_phenotype.shape, dtype='float')

    def _dloglik_depistasis_func_params(self, k=None):
        """Implements :meth:`AbstractEpistasis._dloglik_depistasis_func_params`

        For :class:`NoEpistasis` models, this is just an empty array as there
        are no epistasis function parameters.

        """
        assert len(self.epistasis_func_params_dict) == 0
        return numpy.array([], dtype='float')

    @property
    def _epistasis_func_param_names(self):
        """list: Epistasis function parameter names.

        For :class:`NoEpistasis`, this is just an empty list as there are
        no epistasis function parameters.

        """
        return []

    @property
    def _epistasis_func_param_bounds(self):
        """list: Bounds for the epistasis function parameters.

        For :class:`NoEpistasis` models, this is just an empty list as
        there are no epistasis function parameters.

        """
        bounds_d = {}
        return [bounds_d[name] for name in self._epistasis_func_param_names]

    def _prescale_params(self, k, g_k_range):
        """Do nothing, as no need to prescale for :class:`NoEpistasis`."""
        pass

    def _postscale_params(self):
        """If `_zero_wt_observed_pheno`, all wildtype latent -> 0."""
        if self._zero_wt_observed_pheno:
            rescaled_latenteffects = self._latenteffects.copy()
            oldloglik = self.loglik
            for ki in range(self.n_latent_phenotypes):
                rescaled_latenteffects[ki] = numpy.append(
                                    rescaled_latenteffects[ki][: -1],
                                    0.0)
            self._latenteffects = rescaled_latenteffects
            # make sure log likelihood hasn't changed too much
            if not numpy.allclose(self.loglik, oldloglik):
                raise EpistasisFittingError('post-scaling changed loglik '
                                            f"{oldloglik} to {self.loglik}")
            assert numpy.allclose(0, self.phenotypes_frombinary(
                                      numpy.zeros((1, self._n_latent_effects)),
                                      'observed')
                                  )


class MonotonicSplineEpistasis(AbstractEpistasis):
    """Monotonic spline global epistasis model.

    Note
    ----
    Subclass of :class:`AbstractEpistasis` that implements the
    :ref:`monotonic_spline_epistasis_function`.

    Parameters
    ----------
    spline_order : int
        Order of the I-splines defining the global epistasis function.
    meshpoints : int
        Number of evenly spaced mesh points for the I-spline defining the
        global epistasis function.

    """

    def __init__(self,
                 binarymap,
                 *,
                 n_latent_phenotypes=1,
                 model_one_less_latent=None,
                 spline_order=3,
                 meshpoints=4,
                 **kwargs,
                 ):
        """See main class docstring."""
        if not (isinstance(meshpoints, int) and meshpoints > 1):
            raise ValueError('`meshpoints` must be int > 1')
        self._mesh = (numpy.tile(numpy.linspace(0, 1, meshpoints),
                                 n_latent_phenotypes)
                      .reshape(n_latent_phenotypes, meshpoints)
                      )
        self._spline_order = spline_order
        super().__init__(binarymap, n_latent_phenotypes=n_latent_phenotypes,
                         model_one_less_latent=model_one_less_latent,
                         **kwargs)

    def _set_lower_latent_phenotype_params(self, model_one_less_latent):
        """Overrides :meth:`AbstractEpistasis._set_lower_latent_phenotypes`.

        Augments that base method to also set mesh.

        """
        super()._set_lower_latent_phenotype_params(model_one_less_latent)
        for k in range(1, self.n_latent_phenotypes):
            ki = k - 1
            self._mesh[ki] = model_one_less_latent._mesh[ki]

    def _isplines_total(self, k=None):
        """I-splines for global epistasis function.

        Parameters
        -----------
        k : int or None
            Which global epistasis function to get I-splines for (1 <= k <=
            :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`.

        Returns
        --------
        :class:`dms_variants.ispline.Isplines_total`
            The I-spline family defined with the current values of
            the latent phenotypes as `x`.

        """
        k = self._process_latent_phenotype_k(k) + 1
        key = f"_isplines_total_{k}"
        if key not in self._cache:
            self._cache[key] = dms_variants.ispline.Isplines_total(
                                        order=self._spline_order,
                                        mesh=self._mesh[k - 1],
                                        x=self._latent_phenotypes(k))
        return self._cache[key]

    def epistasis_func(self, latent_phenotype, k=None):
        """Global epistasis function :math:`g` in Eq. :eq:`monotonicspline`.

        Concrete implementation of :meth:`AbstractEpistasis.epistasis_func`.

        """
        if not isinstance(latent_phenotype, numpy.ndarray):
            raise ValueError('`latent_phenotype` not numpy array')
        if ((latent_phenotype.shape == self._latent_phenotypes(k).shape) and
                (latent_phenotype == self._latent_phenotypes(k)).all()):
            return self._isplines_total(k).Itotal(weights=self.alpha_ms(k),
                                                  w_lower=self.c_alpha(k))
        else:
            return dms_variants.ispline.Isplines_total(
                        order=self._spline_order,
                        mesh=self._mesh[self._process_latent_phenotype_k(k)],
                        x=latent_phenotype).Itotal(weights=self.alpha_ms(k),
                                                   w_lower=self.c_alpha(k))

    def _depistasis_func_dlatent(self, latent_phenotype, k=None):
        """Derivative of `epistasis_func` by latent phenotype.

        Concrete implementation of
        :meth:`AbstractEpistasis._depistasis_func_dlatent`.

        """
        return self._isplines_total(k).dItotal_dx(weights=self.alpha_ms(k))

    def _dloglik_depistasis_func_params(self, k=None):
        """Implements :meth:`AbstractEpistasis._dloglik_depistasis_func_params`

        See Eqs. :eq:`dspline_epistasis_dcalpha` and
        :eq:`dspline_epistasis_dalpham`.

        """
        ki = self._process_latent_phenotype_k(k)
        assert self._epistasis_func_params[ki, 0] == self.c_alpha(k)
        assert (self._epistasis_func_params[ki, 1:] == self.alpha_ms(k)).all()
        dcalpha = self._dloglik_dobserved_phenotype.dot(
                self._isplines_total(k).dItotal_dw_lower())
        dalpham = self._dloglik_dobserved_phenotype.dot(
                self._isplines_total(k).dItotal_dweights(self.alpha_ms(k),
                                                         self.c_alpha(k)))
        deriv = numpy.append(dcalpha, dalpham)
        assert deriv.shape == (len(self._epistasis_func_param_names),)
        return deriv

    @property
    def _epistasis_func_param_names(self):
        r"""list: Epistasis function parameter names.

        These are the :math:`c_{\alpha}` and :math:`\alpha_m` parameters
        in Eq. :eq:`monotonicspline`.

        """
        return ['c_alpha'] + [f"alpha_{m}" for m in
                              range(1, self._isplines_total(1).n + 1)]

    @property
    def _epistasis_func_param_bounds(self):
        r"""list: Bounds for the epistasis function parameters.

        There is no bound on :math:`c_{\alpha}`, and the :math:`\alpha_m`
        parameters must be > 0.

        """
        bounds_d = {'c_alpha': (None, None)}
        for m in range(1, self._isplines_total(1).n + 1):
            bounds_d[f"alpha_{m}"] = (self._NEARLY_ZERO, None)
        return [bounds_d[name] for name in self._epistasis_func_param_names]

    def c_alpha(self, k=None):
        r""":math:`c_{\alpha}` in Eq. :eq:`monotonicspline`.

        Parameters
        ----------
        k : int or None
            Which global epistasis function to get I-splines for (1 <= k <=
            :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`.

        Returns
        -------
        float
            :math:`c_{\alpha}` for global epistasis function `k`.

        """
        if self.n_latent_phenotypes == 1:
            return self.epistasis_func_params_dict['c_alpha']
        else:
            k = self._process_latent_phenotype_k(k) + 1
            return self.epistasis_func_params_dict[f"c_alpha_{k}"]

    def alpha_ms(self, k=None):
        r""":math:`\alpha_m` in Eq. :eq:`monotonicspline`.

        Parameters
        ----------
        k : int or None
            Which global epistasis function to get I-splines for (1 <= k <=
            :attr:`AbstractEpistasis.n_latent_phenotypes`). If there
            is just one latent phenotype, can also be `None`.

        Returns
        -------
        numpy.ndarray
            :math:`\alpha_m` values for global epistasis function `k`.

        """
        if self.n_latent_phenotypes == 1:
            return numpy.array(
                    [self.epistasis_func_params_dict[f"alpha_{m}"]
                     for m in range(1, self._isplines_total(k).n + 1)],
                    dtype='float')
        else:
            k = self._process_latent_phenotype_k(k) + 1
            return numpy.array(
                    [self.epistasis_func_params_dict[f"alpha_{m}_{k}"]
                     for m in range(1, self._isplines_total(k).n + 1)],
                    dtype='float')

    def _prescale_params(self, k, g_k_range):
        r"""Get latent phenotypes in mesh and :math:`g_k` with desired limits.

        See :meth:`AbstractEpistasis._prescale_params` for description of
        parameters.

        Specifically, if `k == 1` then the latent phenotypes are re-scaled
        to span the mesh. The parameters of the global epistasis function
        :math:`g_k` are set so that :math:`c_alpha` is :math:`g_k_range[0]`,
        and all :math:`\alpha_m` values are set to
        :math:`\left[\max\left(y_v\right) - \min\left(y_v\right)\right] / M`
        so that the range of :math:`g_k` over its mesh spans `g_k_range`.

        If `k > 1`, then the latent effects are all zero, the latent phenotype
        of wildtype is chosen so that :math:`g_k = 0` for wildtype, and
        the parameters of :math:`g_k` are chosen so that the limits on the
        mesh are `g_k_range` and all :math:`\alpha_m` values are equal.
        In addition, we require that `g_k_range[0] = -g_k_range[1]`.
        """
        if not (isinstance(k, int) and self.n_latent_phenotypes >= k >= 1):
            raise ValueError(f"invalid `k` of {k}")
        ki = k - 1

        # check g_k_range, and make sure > 0
        if g_k_range[1] < g_k_range[0]:
            raise ValueError('invalid `g_k_range`')
        if g_k_range[1] - g_k_range[0] < 2 * self._NEARLY_ZERO:
            g_k_range = (g_k_range[0] - self._NEARLY_ZERO,
                         g_k_range[1] + self._NEARLY_ZERO)

        # set initial epistasis func params
        g_k_params = self._epistasis_func_params.copy()
        assert g_k_params.shape == (self.n_latent_phenotypes,
                                    len(self._epistasis_func_param_names))
        init_d = {'c_alpha': g_k_range[0]}
        for m in range(1, self._isplines_total(k).n + 1):
            init_d[f"alpha_{m}"] = ((g_k_range[1] - g_k_range[0]) /
                                    self._isplines_total(k).n)
        for iparam, param in enumerate(self._epistasis_func_param_names):
            g_k_params[ki, iparam] = init_d[param]
        self._epistasis_func_params = g_k_params

        if k == 1:
            rescale_min, rescale_max = min(self._mesh[ki]), max(self._mesh[ki])
            rescalerange = rescale_max - rescale_min
            assert rescalerange > self._NEARLY_ZERO

            rescaled_latenteffects = self._latenteffects.copy()
            currentrange = (self._latent_phenotypes(k).max() -
                            self._latent_phenotypes(k).min())

            if currentrange <= self._NEARLY_ZERO:
                warnings.warn(f"range of latent phenotype {k} is nearly zero "
                              f"({currentrange}); so cannot pre-scale. Just "
                              'setting all latent effects to zero',
                              EpistasisFittingWarning)
                rescaled_latenteffects[ki] = 0
                rescaled_latenteffects[ki] = numpy.append(
                     rescaled_latenteffects[ki][: -1],
                     (rescaled_latenteffects[ki][-1] + rescale_min -
                      self._latent_phenotypes(k).min()))
                self._latenteffects = rescaled_latenteffects

            else:
                # rescale so latent phenotypes span desired range
                rescaled_latenteffects[ki] = (rescaled_latenteffects[ki] *
                                              rescalerange / currentrange)
                self._latenteffects = rescaled_latenteffects
                # change wt latent phenotype so latent phenos have right min
                rescaled_latenteffects[ki] = numpy.append(
                     rescaled_latenteffects[ki][: -1],
                     (rescaled_latenteffects[ki][-1] + rescale_min -
                      self._latent_phenotypes(k).min()))
                self._latenteffects = rescaled_latenteffects

                assert numpy.allclose(rescale_min,
                                      self._latent_phenotypes(k).min())
                assert numpy.allclose(rescale_max,
                                      self._latent_phenotypes(k).max())
                assert numpy.allclose(rescalerange,
                                      (self._latent_phenotypes(k).max() -
                                       self._latent_phenotypes(k).min()))

        else:
            if g_k_range[0] != -g_k_range[1]:
                raise ValueError(f"`g_k_range` not symmetric: {g_k_range}")

            assert k > 1

            # midpoint of mesh, g_k should be 0 here, set to wildtype latent k
            mid_mesh = (self._mesh[ki].max() - self._mesh[ki].min()) / 2
            assert numpy.allclose(0,
                                  self.epistasis_func(numpy.array([mid_mesh]),
                                                      k=k))
            latenteffects = self._latenteffects.copy()
            latenteffects[ki] = 0.0
            latenteffects[ki, self._n_latent_effects] = mid_mesh
            self._latenteffects = latenteffects
            assert numpy.allclose(0, self._observed_phenotypes([k]))

    def _postscale_params(self):
        """Rescale parameters after global epistasis fitting.

        The parameters are re-scaled so that:
          - The mean absolute value latent effect is 1.
          - The latent phenotype of wildtype is 0.

        """
        rescaled_latenteffects = self._latenteffects.copy()
        oldloglik = self.loglik
        for ki in range(self.n_latent_phenotypes):
            # make mean absolute latent effect equal to one
            mean_abs_latent_effect = (numpy.abs(self._latenteffects[ki][: -1])
                                      .mean()
                                      )
            if mean_abs_latent_effect < self._NEARLY_ZERO:
                warnings.warn(f"mean latent effect for phenotype {ki + 1} "
                              f"is nearly zero ({mean_abs_latent_effect}); "
                              'so cannot rescale',
                              EpistasisFittingWarning)
            else:
                rescaled_latenteffects[ki] = (rescaled_latenteffects[ki] /
                                              mean_abs_latent_effect)
                self._mesh[ki] = self._mesh[ki] / mean_abs_latent_effect

            # make latent phenotype of wildtype equal to 0
            self._mesh[ki] = self._mesh[ki] - rescaled_latenteffects[ki][-1]
            rescaled_latenteffects[ki] = numpy.append(
                                            rescaled_latenteffects[ki][: -1],
                                            0.0)

        self._latenteffects = rescaled_latenteffects
        assert all(0 == self.latent_phenotype_wt(k)
                   for k in range(1, self.n_latent_phenotypes + 1))

        # make sure log likelihood hasn't changed too much
        if not numpy.allclose(self.loglik, oldloglik):
            raise EpistasisFittingError('post-scaling latent effects changed '
                                        f"loglik {oldloglik} to {self.loglik}")

        if self._zero_wt_observed_pheno:
            rescaled_func_params = self._epistasis_func_params.copy()
            c_alpha_index = self._epistasis_func_param_names.index('c_alpha')
            for ki in range(self.n_latent_phenotypes):
                k = ki + 1
                wt_obs_pheno_k = self.epistasis_func(
                                    numpy.array([self.latent_phenotype_wt(k)]),
                                    k=k)[0]
                rescaled_func_params[ki, c_alpha_index] -= wt_obs_pheno_k
            self._epistasis_func_params = rescaled_func_params
            if not numpy.allclose(self.loglik, oldloglik):
                raise EpistasisFittingError(
                            'post-scaling wt observed pheno changed likelihood'
                            f" from loglik {oldloglik} to {self.loglik}")
            assert numpy.allclose(0, self.phenotypes_frombinary(
                                      numpy.zeros((1, self._n_latent_effects)),
                                      'observed')
                                  )


class MonotonicSplineEpistasisGaussianLikelihood(MonotonicSplineEpistasis,
                                                 GaussianLikelihood):
    """Monotonic spline global epistasis model with Gaussian likelihood.

    Note
    ----
    This class implements the :ref:`monotonic_spline_epistasis_function`
    with a :ref:`gaussian_likelihood`. See documentation for the base
    classes :class:`MonotonicSplineEpistasis`, :class:`GaussianLikelihood`,
    and :class:`AbstractEpistasis` for details.

    """

    pass


class MonotonicSplineEpistasisCauchyLikelihood(MonotonicSplineEpistasis,
                                               CauchyLikelihood):
    """Monotonic spline global epistasis model with Cauchy likelihood.

    Note
    ----
    This class implements the :ref:`monotonic_spline_epistasis_function`
    with a :ref:`cauchy_likelihood`. See documentation for the base
    classes :class:`MonotonicSplineEpistasis`, :class:`CauchyLikelihood`,
    and :class:`AbstractEpistasis` for details.

    """

    pass


class MonotonicSplineEpistasisBottleneckLikelihood(MonotonicSplineEpistasis,
                                                   BottleneckLikelihood):
    """Monotonic spline global epistasis model with bottleneck likelihood.

    Note
    ----
    This class implements the :ref:`monotonic_spline_epistasis_function`
    with a :ref:`bottleneck_likelihood`. See documentation for the base
    classes :class:`MonotonicSplineEpistasis`, :class:`BottleneckLikelihood`,
    and :class:`AbstractEpistasis` for details.

    """

    def __init__(self,
                 binarymap,
                 bottleneck,
                 *,
                 n_latent_phenotypes=1,
                 model_one_less_latent=None,
                 spline_order=3,
                 meshpoints=4,
                 pseudocount=0.5,
                 base=2,
                 ):
        """See main class docstring."""
        super().__init__(binarymap,
                         bottleneck=bottleneck,
                         n_latent_phenotypes=n_latent_phenotypes,
                         model_one_less_latent=model_one_less_latent,
                         spline_order=spline_order,
                         meshpoints=meshpoints,
                         pseudocount=pseudocount,
                         base=base)


class NoEpistasisGaussianLikelihood(NoEpistasis,
                                    GaussianLikelihood):
    """No-epistasis model with Gaussian likelihood.

    Note
    ----
    This class implements the :ref:`no_epistasis_function` with a
    :ref:`gaussian_likelihood`. See documentation for the base classes
    :class:`NoEpistasis`, :class:`GaussianLikelihood`, and
    :class:`AbstractEpistasis` for details.

    """

    pass


class NoEpistasisCauchyLikelihood(NoEpistasis,
                                  CauchyLikelihood):
    """No-epistasis model with Cauchy likelihood.

    Note
    ----
    This class implements the :ref:`no_epistasis_function` with a
    :ref:`cauchy_likelihood`. See documentation for the base classes
    :class:`NoEpistasis`, :class:`CauchyLikelihood`, and
    :class:`AbstractEpistasis` for details.

    """

    pass


class NoEpistasisBottleneckLikelihood(NoEpistasis,
                                      BottleneckLikelihood):
    """No-epistasis model with bottleneck likelihood.

    Note
    ----
    This class implements the :ref:`no_epistasis_function` with a
    :ref:`bottleneck_likelihood`. See documentation for the base classes
    :class:`NoEpistasis`, :class:`BottleneckLikelihood`, and
    :class:`AbstractEpistasis` for details.

    """

    pass


def fit_models(binarymap,
               likelihood,
               *,
               bottleneck=None,
               max_latent_phenotypes=1,
               ):
    r"""Fit and compare global epistasis models.

    This function is useful when you want to examine the fit of several
    different models to the same data. It does the following:

     1. Fits a non-epistatic model to the data.

     2. Fits a global epistasis model with :math:`K = 1` latent phenotypes
        to the data. If the global epistasis model outperforms the no-
        epistasis model by AIC_, proceed to next step. Otherwise stop.

     3. Fit a global epistasis model with :math:`K = 2` latent phenotypes.
        If this model outperforms (by AIC_) the model with :math:`K - 1`
        latent phenotypes, repeat for :math:`K = 3` etc until adding more
        latent phenotypes no longer improves fit. Note that it only does
        continues this process while :math:`K \le` `max_latent_phenotypes`,
        so set `max_latent_phenotypes` > 1 if you want to fit multiple
        latent phenotypes.

    .. _AIC: https://en.wikipedia.org/wiki/Akaike_information_criterion

    Note
    ----
    All of the fitting is done with the same likelihood-calculation method
    because you can **not** compare models fit with different likelihood-
    calculation methods.

    Parameters
    ----------
    binarymap : :class:`dms_variants.binarymap.BinaryMap`
        Contains the variants, their functional scores, and score variances.
        The models are fit to these data.
    likelihood : {'Gaussian', 'Cauchy', 'Bottleneck'}
        Likelihood calculation method to use when fitting models. See
        :ref:`likelihood_calculation`.
    bottleneck : float or None
        Required if using 'Bottleneck' `likelihood`. In that case, is
        the experimentally estimated bottleneck between the pre-
        and post-selection conditions.
    max_latent_phenotypes : int
        Maximum number of latent phenotypes that are potentially be fit.
        See the :math:`K` parameter in :ref:`multi_latent`.

    Returns
    -------
    pandas.DataFrame
        Summarizes the results of the model fitting and contains the
        fit models. Columns are:

          - 'description': description of model
          - 'n_latent_phenotypes': number of latent phenotypes in model
          - 'AIC': AIC_
          - 'nparams': number of parameters
          - 'log_likelihood': log likelihood
          - 'model': the actual model (subclass of :class:`AbstractEpistasis`)
          - 'fitting_time': time in seconds that it took to fit model

        The data frame is sorted from best to worst model by AIC_.

    """
    if not (isinstance(max_latent_phenotypes, int) and
            max_latent_phenotypes >= 1):
        raise ValueError('`max_latent_phenotypes` must be int >= 1')

    if likelihood == 'Gaussian':
        NoEpistasisClass = NoEpistasisGaussianLikelihood
        EpistasisClass = MonotonicSplineEpistasisGaussianLikelihood
        bottleneck_args = {}
    elif likelihood == 'Cauchy':
        NoEpistasisClass = NoEpistasisCauchyLikelihood
        EpistasisClass = MonotonicSplineEpistasisCauchyLikelihood
        bottleneck_args = {}
    elif likelihood == 'Bottleneck':
        NoEpistasisClass = NoEpistasisBottleneckLikelihood
        EpistasisClass = MonotonicSplineEpistasisBottleneckLikelihood
        if not bottleneck:
            raise ValueError('specify `bottleneck` for Bottleneck likelihood')
        bottleneck_args = {'bottleneck': bottleneck}
    else:
        raise ValueError(f"invalid `likelihood` {likelihood}")

    FitData = collections.namedtuple('FitData',
                                     ['description', 'n_latent_phenotypes',
                                      'AIC', 'nparams', 'log_likelihood',
                                      'model', 'fitting_time']
                                     )

    def fit(modelclass, description, k=1, model_one_less_latent=None):
        model = modelclass(binarymap,
                           n_latent_phenotypes=k,
                           model_one_less_latent=model_one_less_latent,
                           **bottleneck_args)
        start = time.time()
        _ = model.fit()
        return FitData(description=description,
                       n_latent_phenotypes=model.n_latent_phenotypes,
                       AIC=model.aic,
                       nparams=model.nparams,
                       log_likelihood=model.loglik,
                       model=model,
                       fitting_time=time.time() - start
                       )

    fitlist = [fit(NoEpistasisClass, 'no epistasis')]

    for k in range(1, max_latent_phenotypes + 1):
        if max_latent_phenotypes == 1:
            description = 'global epistasis'
        else:
            description = f"global epistasis with {k} latent phenotypes"
        fitlist.append(
                fit(EpistasisClass, description, k,
                    model_one_less_latent=None if k == 1 else fitlist[-1].model
                    )
                )
        if fitlist[-1].AIC > fitlist[-2].AIC:
            break

    return (pd.DataFrame.from_records(fitlist, columns=FitData._fields)
            .sort_values('AIC')
            .reset_index(drop=True)
            )


if __name__ == '__main__':
    import doctest
    doctest.testmod()
