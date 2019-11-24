=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

0.4.5
------

Added
+++++++
- The new ``AbstractEpistasis.single_mut_effects`` method.

- Options ``returnformat`` and ``stringency_param`` to ``AbstractEpistasis.preferences`` and ``utils.scores_to_prefs``.

Changed
+++++++
- ``AbstractEpistasis.preferences`` and ``utils.scores_to_prefs`` return site as integer.

0.4.4
------

Fixed
++++++
- Errors related to using ``pandas.query`` for ``nan`` values. Not sure of the cause, but the errors are fixed now.

0.4.3
------

Changed
++++++++
- Eliminated the default log base for conversion of scores / phenotypes. This is because base 2 gave excessively flat preferences, and the choice of a base is something that the user should need to think about. Added explanation about the consequences of this choice to docs and examples.

- The preferenes returned by ``scores_to_prefs`` and ``AbstractEpistasis.preferences`` are now naturally sorted by site.

0.4.2
------

Added
++++++
- The new ``AbstractEpistasis.preferences`` method gets amino-acid preferences from phenotypes.

- Added ``utils.scores_to_prefs``.

0.4.1
------

Fixed
++++++
- The ``isplines`` module now uses a simple dict-implemented cache rather than ``methodtools.lru_cache``. This fixes excess memory usage and allows objects to be pickled.

- ``AbstractEpistasis`` internally clears the cache via ``__getstate__`` to reduce size of pickled objects. This avoids pickled models being huge. Also added the ``clearcache`` option to ``AbstractEpistasis.fit`` to serve a similar purpose of memory savings.

0.4.0
--------

Added
++++++
- Added additional forms of likelihood function to the global epistasis models. This involves substantial re-factoring the epistasis models in ``globalepistasis``.
  In particular, the ``MonotonicSplineEpistasis`` and ``NoEpistasis`` classes no longer are fully concrete subclasses of ``AbstractEpistasis``.
  Instead, there are also likelihood calculation subclasses (``GaussianLikelihood`` and ``CauchyLikelihood``), and the concrete subclasses inherit from both an epistasis function and likelihood calculation subclass.
  So for instance, what was previously ``MonotonicSplineEpistasis`` (with Gaussian likelihood assumed) is now ``MonotonicSplineEpistasisGaussianLikelihood``.
  **Note that this an API-breaking change.**

- Added the ``narrow_bottleneck.ipynb`` notebook to demonstrate use of the Cauchy likelihood for analysis of experiments with a lot of noise.

- Added the ``predict_variants.ipynb`` to demonstrate prediction of variant phenotypes using global epistasis models.

- Added ``simulate.codon_muts``.

Fixed
++++++++
- Some minor fixes to ``codonvariat_sim_data.ipynb``.

0.3.0
-----

Added
++++++++
- Added ``utils.tidy_to_corr``.

- Added ``binarymap`` module.

- Added ``globalepistasis`` module.

- Added ``ispline`` module.

Changed
++++++++
- Order of rows in data frames from ``CodonVariantTable.func_scores``.

- Updated ``codonvariant_sim_data.ipynb`` to be smaller and fit global epistasis models, and move plot formatting examples to a new dedicated notebook.

- Changed ``SigmoidPhenotypeSimulator`` so that the **enrichment** is a sigmoidal function of the latent phenotype, and the observed phenotype is the log (base 2) of the latent phenotype. 
  This change harmonizes the simulator with the definitions in the new ``globalepistasis`` module.
  Also changed the input to the ``latentPhenotype`` and ``observedPhenotype`` methods.
  Note that these are backwards-compatibility breaking changes.

Fixed
++++++
- Removed use of deprecated ``Bio.Alphabet``

0.2.0
--------

Added
++++++
- Capabilities to parse barcodes from Illumina data: FASTQ readers and ``IlluminaBarcodeParser``.

- ``CodonVariantTable.numCodonMutsByType`` method to get numerical values for codon mutations per variant.

- Can specify names of columns when initializing a ``CodonVariantTable``.

- ``CodonVariantTable.func_scores`` now takes ``libraries`` rather than ``combine_libs`` argument.

- Added ``CodonVariantTable.add_sample_counts_df`` method.

- Added ``CodonVariantTable.plotVariantSupportHistogram`` method.

- Added ``CodonVariantTable.avgCountsPerVariant`` and ``CodonVariantTable.plotAvgCountsPerVariant`` methods.

- Add custom ``plotnine`` theme in ``plotnine_themes`` and improved formatting of plots from ``CodonVariantTable``.

- Added ``sample_rename`` parameter to ``CodonVariantTable`` plotting methods.

- Added ``syn_as_wt`` to ``CodonVariantTable.classifyVariants``.

- Added ``random_seq`` and ``mutate_seq`` to ``simulate`` module.

Changed
--------
- Changed how ``variant_call_support`` set in ``simulate_CodonVariantTable``.

- Better xlimits on ``CodonVariantTable.plotCumulMutCoverage``.

Fixed
-----
- Docs /formatting in Jupyter notebooks.

- Fixed bugs that arose when ``pandas`` updated to 0.25 (related to ``groupby`` no longer dropping empty categories).

- Bugs in ``CodonVariantTable`` histogram plots when ``samples`` set.

0.1.0
-----
Initial release. Ported code from ``dms_tools2`` and made some improvements.

