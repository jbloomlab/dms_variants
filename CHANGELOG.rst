=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

0.2.dev0
--------

Added
-----
- `CodonVariantTable.numCodonMutsByType` method to get numerical values for codon mutations per variant.

- Can specify names of columns when initializing a `CodonVariantTable`.

Fixed
-----
- Docs /formatting in Jupyter notebooks.

- Fixed bugs that arose when `pandas` updated to 0.25 (related to `groupby` no logner dropping empty categories).

0.1.0
-----
Initial release. Ported code from `dms_tools2` and made some improvements.

