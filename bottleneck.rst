Bottleneck-based likelihood
==============================
This form of the likelihood is most appropriate when most of the noise in the experiment comes from a bottleneck when passaging the library from the pre-selection to post-selection condition.
This will typically be the case when the total pre- and post-selection sequencing depths exceed the number of actual variants that were physically passaged from the pre-selection library to the post-selection one.
In this case, most of the noise isn't from statistical sampling in the variant-barcode sequencing counts, but rather from the bottleneck during the experimental passaging.

Let :math:`n_v^{\text{pre}}` and :math:`n_v^{\text{post}}` be the pre-selection and post-selection counts for variant :math:`v`, and let :math:`N_{\text{pre}} = \sum_v n_v^{\text{pre}}` and :math:`N_{\text{post}} = \sum_v n_v^{\text{post}}` be the total pre- and post-selection counts.
Let :math:`N_{\text{bottle}}` be the bottleneck when passaging the pre-selection library to the post-selection condition.
As mentioned above, this bottleneck-based likelihood is most appropriate when :math:`N_{\text{bottle}} \ll N_{\text{post}}, N_{\text{pre}}`.

Let :math:`n_v^{\text{bottle}}` be the number of variants :math:`v` that survive the bottleneck, with :math:`N_{\text{bottle}} = \sum_v n_v^{\text{bottle}}`.
Note that :math:`n_v^{\text{bottle}}` is **not** an experimental observable, although it may be possible to experimentally estimate :math:`N_{\text{bottle}}`.
We can, however, calculate the probability distribution over :math:`n_v^{\text{bottle}}` as

.. math::

   \text{DCM}\left(\left\{ {n_v^{\text{bottle}} \right\} \mid
                   \left\{n_v^{\text{pre}} \right\})
   = \frac{N_{\text{bottle}} B\left(n_v^{\text{pre}},
                                              N_{\text{bottle}}\right)}
          {\prod_{v: n_v^{\text{bottle}} > 0}
           n_v^{\text{bottle}} B\left(n_v^{\text{pre}}, n_v^{\text{bottle}}\right)
           }

where :math:`\text{DCM}` is the `Dirichlet compound multinomial distribution <https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution>`_, which is defined in terms of the `beta function <https://en.wikipedia.org/wiki/Beta_function>`_ :math:`B`.


