=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

0.8.9
------
Fixed
++++++
- Fixed bug in ``extra_cols`` to ``CodonVariantTable.from_variant_count_df`` implementation.

0.8.8
------
Added
+++++
- ``CodonVariantTable.from_variant_count_df`` accepts ``extra_cols`` parameter.

0.8.7
-----
Fixed
+++++
- Fixed datatype for ``maxpoints`` default in ``barcodes.rarefyBarcodes`` 

Changed
+++++++
- Discourage and stop testing multiple latent phenotypes.
- Remove from docs and stop testing ``predict_variants.ipynb`` as this doesn't seem to be a common use case.

0.8.6
-----
Fixed
+++++
- Updated compilation arguments for Windows.

- Pass Travis tests.

0.8.5
------

Added
+++++
- ``pdb_utils`` module with ``reassign_b_factor`` function.

0.8.4
-----

Added
+++++
- Classify amino-acid mutations as single-nucleotide accessible: added ``constants.SINGLE_NT_AA_MUTS`` and ``utils.single_nt_accessible``.

Fixed
+++++
- Made compatible with ``biopython`` 1.78 by fixing import of ``ambiguous_dna_values`` to be from ``Bio.Data.IUPACData``.

0.8.3
-----

Fixed
+++++
- Unpin ``plotnine`` now that `this bug <https://github.com/has2k1/plotnine/issues/403>`_ fixed.

Changed
+++++++
- Only test on Python 3.7.

0.8.2
------

Fixed
++++++

- Bug fix in ``filter_by_subs_observed``.

0.8.1
-----

Added
+++++
- ``CodonVariantTable.escape_scores`` now computes score type ``frac_escape``.

- Added ``filter_by_subs_observed``.

0.8.0
-----

Changed
++++++++
- ``CodonVariantTable.escape_scores`` now requires specification of score type, and implements a new score type of log fraction escape. The output of this method is also slightly changed.

Fixed
+++++
- Bug in calculation of variance in ``CodonVariantTable.escape_scores``.

0.7.1
------

Fixed
+++++
- Fixed bug in ``CodonVariantTable.escape_scores`` that sometimes gives null escape scores.

0.7.0
------

Added
+++++
- Added ``CodonVariantTable.escape_scores``

- Added ``CodonVariantTable.add_frac_counts``

- Added ``CodonVariantTable.plotCountsPerVariant``

Fixed
++++++
- ``CodonVariantTable.classifyVariants`` requires instructions on how to handle non-primary targets.

0.6.0
------

Added
+++++
- Added capability of having other "reference" targets in a ``CodonVariantTable``.

Fixed
+++++
- ``simulate.rand_seq`` generates unique sequences.

0.5.3
------

Fixed
++++++
- ``plotCumultMutCoverage`` now has y-axis that extends from 0 to 1.

0.5.2
------

Added
++++++
- In ``CodonVariantTable`` plotting, by default do not label facets for library when just one library, and add ``one_lib_facet`` parameter to plotting functions.

- Made compatible with ``pandas`` >= 1.0

0.5.1
-------

Fixed
++++++
- Show estimates data frame for ``bottlenecks.estimateBottleneck`` doctest.

- Remove use of deprecated ``scipy.array`` for ``numpy.array``.

0.5.0
--------

Added
++++++
- The ability to fit **multiple** latent phenotypes in the global epistasis models. This adds the ``n_latent_phenotypes`` flag to ``AbstractEpistasis`` models, and changes calls to certain methods / properties of that abstract model class and its concrete subclasses.

- The concept of "bottleneck" likelihoods in global epistasis models, implemented in ``BottleneckLikelihood``.

- The ``bottlenecks`` module to estimate bottlenecks.

- Added ``AbstractEpistasis.aic`` property.

- Added ``globalepistasis.fit_models``

- Added ``MultiLatentSigmoidPhenotypeSimulator``.

- An equals (``__eq__``) comparison operation to ``BinaryMap``.

- Added ``n_pre`` and ``n_post`` attributes to ``BinaryMap``. This changes the initialization to add new parameters, ``n_pre_col``, ``n_post_col``, and ``cols_optional``.

Fixed
++++++
- ``BinaryMap`` objects can now be deep copied (they don't have a compiled regex as attributed).

0.4.7
------

Added
+++++
- The ``expand`` option to ``BinaryMap`` to have maps encode all possible characters at each site.

0.4.6
-----

Fixed
+++++
- Fixed bug in ``AbstractEpistasis.preferences`` with ``returnformat`` of 'tidy'. Previously the wildtype was set incorrectly for missing values.

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

