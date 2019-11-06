=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

0.4.dev0
--------

Added
++++++
- Added forms of likelihood function beyond the normal distribution one to the global epistasis models. This involves substantial re-factoring the epistasis models in `globalepistasis`.

- Added the `narrow_bottleneck.ipynb` notebook.

Fixed
++++++++
- Some minor fixes to `codonvariat_sim_data.ipynb`.

0.3.0
-----

Added
++++++++
- Added `utils.tidy_to_corr`.

- Added `binarymap` module.

- Added `globalepistasis` module.

- Added `ispline` module.

Changed
++++++++
- Order of rows in data frames from `CodonVariantTable.func_scores`.

- Updated `codonvariant_sim_data.ipynb` to be smaller and fit global epistasis models, and move plot formatting examples to a new dedicated notebook.

- Changed `SigmoidPhenotypeSimulator` so that the **enrichment** is a sigmoidal function of the latent phenotype, and the observed phenotype is the log (base 2) of the latent phenotype. 
  This change harmonizes the simulator with the definitions in the new `globalepistasis` module.
  Also changed the input to the `latentPhenotype` and `observedPhenotype` methods.
  Note that these are backwards-compatibility breaking changes.

Fixed
++++++
- Removed use of deprecated `Bio.Alphabet`

0.2.0
--------

Added
++++++
- Capabilities to parse barcodes from Illumina data: FASTQ readers and `IlluminaBarcodeParser`.

- `CodonVariantTable.numCodonMutsByType` method to get numerical values for codon mutations per variant.

- Can specify names of columns when initializing a `CodonVariantTable`.

- `CodonVariantTable.func_scores` now takes `libraries` rather than `combine_libs` argument.

- Added `CodonVariantTable.add_sample_counts_df` method.

- Added `CodonVariantTable.plotVariantSupportHistogram` method.

- Added `CodonVariantTable.avgCountsPerVariant` and `CodonVariantTable.plotAvgCountsPerVariant` methods.

- Add custom `plotnine` theme in `plotnine_themes` and improved formatting of plots from `CodonVariantTable`.

- Added `sample_rename` parameter to `CodonVariantTable` plotting methods.

- Added `syn_as_wt` to `CodonVariantTable.classifyVariants`.

- Added `random_seq` and `mutate_seq` to `simulate` module.

Changed
--------
- Changed how `variant_call_support` set in `simulate_CodonVariantTable`.

- Better xlimits on `CodonVariantTable.plotCumulMutCoverage`.

Fixed
-----
- Docs /formatting in Jupyter notebooks.

- Fixed bugs that arose when `pandas` updated to 0.25 (related to `groupby` no longer dropping empty categories).

- Bugs in `CodonVariantTable` histogram plots when `samples` set.

0.1.0
-----
Initial release. Ported code from `dms_tools2` and made some improvements.

