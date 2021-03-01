=========
Examples
=========

The ``dms_variants`` package is designed to aid in the analysis of deep mutational scanning of barcoded variants of genes.
It contains many Python functions and classes that you can use in scripts or Jupyter notebooks.
Major functionality includes:

 - The :class:`dms_variants.codonvarianttable.CodonVariantTable` class for managing and analyze libraries of barcoded codon variants of genes.

 - The :class:`dms_variants.illuminabarcodeparser.IlluminaBarcodeParser` for easily parsing Illumina barcode sequencing that counts variants.

 - The :mod:`dms_variants.globalepistasis` module for fitting global epistasis models to experiments on variants with multiple mutations.

Below are a series of examples that illustrate usage of this functionality and other aspects of the ``dms_variants`` package.

.. toctree::
   :maxdepth: 1

   codonvariant_sim_data
   codonvariant_sim_data_multi_targets
   narrow_bottleneck
   codonvariant_plot_formatting
   parsebarcodes_sim_data

The above examples can be run as interactive Jupyter notebooks on `mybinder <https://mybinder.readthedocs.io>`_ by going to the `following link <https://mybinder.org/v2/gh/jbloomlab/dms_variants/master?filepath=notebooks>`_ (it may take a minute to load) and then opening the notebook you want to run.
