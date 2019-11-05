Bottleneck-based likelihood
==============================
This form of the likelihood is appropriate when most noise in the experiment comes from a bottleneck when passaging the library from the pre-selection to post-selection condition.
This will be the case when the total pre- and post-selection sequencing depths greatly exceed the number of variants that were physically passaged from the pre-selection library to the post-selection one.
In this case, noise in the statistical estimation of the pre- and post-selection variant frequencies is overwhelmed by the experimental noise associated with the bottlenecking.

Note that this form of the likelihood also assumes that an appreciable fraction of the variants are wildtype, such that there is not substantial bottleneck-induced noise in the fraction of the library that is wildtype.

Let :math:`n_v^{\text{pre}}` and :math:`n_v^{\text{post}}` be the pre-selection and post-selection counts for variant :math:`v`.
We estimate the frequencies of the variant pre- and post-selection as

.. math::

   f_v^{\text{pre}}
   &=&
   \frac{n_v^{\text{pre}} + C}
        {\sum_{v'} \left(n_{v'}^{\text{pre}} + C\right)} \\

   f_v^{\text{post}}
   &=&
   \frac{n_v^{\text{post}} + C}
        {\sum_{v'} \left(n_{v'}^{\text{post}} + C\right)}

where :math:`C` is a pseudocount which by default is 0.5.
Because the sequencing depth greatly exceeds the experiment bottleneck, we disregard statistical error in the estimation of :math:`f_v^{\text{pre}}` and :math:`f_v^{\text{post}}` and instead take these as exact measurements.

Let :math:`N_{\text{bottle}}` be the bottleneck when passaging the pre-selection library to the post-selection condition.
As mentioned above, we assume this bottleneck is much smaller than the sequencing depth (meaning that :math:`N_{\text{bottle}} \ll \sum_{v} n_{v}^{\text{pre}}, \sum_{v} n_{v}^{\text{pre}}`).
Let :math:`n_v^{\text{bottle}}` be the number of variants :math:`v` that survive the bottleneck, with :math:`N_{\text{bottle}} = \sum_v n_v^{\text{bottle}}`.
Note that :math:`n_v^{\text{bottle}}` is **not** an experimental observable, although it may be possible to experimentally estimate :math:`N_{\text{bottle}}`.

Furthemore, let :math:`F_{\text{wt}}^{\text{pre}} = \sum_{v = \text{wt}} f_v^{\text{pre}}` be the total fraction of the library that is composed of wildtype variants pre-selection, and let :math:`F_{\text{wt}}^{\text{post}} = \sum_{v = \text{wt}} f_v^{\text{post}}` be the fraction post-selection.
We assume that :math:`F_{\text{wt}}^{\text{post}}` is sufficiently large that it is not much affected by the bottleneck (in other words, we assume that :math:`\frac{\sum_{v = \text{wt}} n_v^{\text{bottle}}}{N_{\text{bottle}}} \simeq F_{\text{wt}}^{\text{post}}`.

After the bottleneck, selection will change the frequency of variant :math:`v` relative to wildtype by an amount proportional to :math:`2^{p\left(v\right)` where :math:`p\left(v\right)` is the observed phenotype of the variant.
So

.. math::

   \frac{f_v^{\text{post}}}{F_{\text{wt}}^{\text{post}}}
   =
   2^{p\left(v\right) \frac{n_v^{\text{bottle}}}
                           {F_{\text{wt}}^{\text{pre}} \times N_{\text{bottle}}}

We can rearrange this equation to yield:

.. math::

   n_v^{\text{bottle}}
   = \frac{f_v^{\text{post}} \times F_{\text{wt}}^{\text{pre}}
           \times N_{\text{bottle}}}
          {F_{\text{wt}}^{\text{post}} \times 2^{p\left(v\right)}}.

We can then calculate the likelihood of observing :math:`n_v^{\text{bottle}}` as simply the Poisson probability given the initial frequency and the bottleneck size:

.. math::

   \mathcal{L}_v
   &=&
   \exp\left(-N_{\text{bottle}} f_v^{\text{pre}}\right)
   \frac{\left(N_{\text{bottle}} f_v^{\text{pre}}\right)^{n_v^{\text{bottle}}}}
        {\Gamma\left(n_v^{\text{bottle}} + 1\right)}

where we have used the fact that :math:`N_{\text{bottle}} f_v^{\text{pre}}` is the expectation value for :math:`n_v^{\text{bottle}}`, and we have used the "continuous Poisson distribution" function defined by `Ilenko (2013) <https://arxiv.org/abs/1303.5990>`_ and `Abid and Mohammed (2016) <http://pubs.sciepub.com/ijdeaor/2/1/2/>`_, but dropped the normalizing factor :math:`c_{\lambda}` from their equations as the likelihoods do not have to integrate to one.


