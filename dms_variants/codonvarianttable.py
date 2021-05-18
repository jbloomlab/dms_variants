"""
=================
codonvarianttable
=================

Defines :class:`CodonVariantTable` objects for storing and
handling codon variants of a gene.

"""

import collections
import itertools
import os
import re
import tempfile

import Bio.SeqUtils.ProtParamData

import numpy

import pandas as pd

import plotnine as p9

import dms_variants.utils
from dms_variants.constants import (AAS_NOSTOP,
                                    AAS_WITHSTOP,
                                    AA_TO_CODONS,
                                    CBPALETTE,
                                    CODONS,
                                    CODON_TO_AA,
                                    NTS,
                                    )


class CodonVariantTable:
    """Store and handle counts of barcoded codon variants of a gene.

    Parameters
    ----------
    barcode_variant_file : str
        CSV file giving barcodes and variants. Must have columns "library",
        "barcode", "substitutions" (nucleotide mutations in 1, ..., numbering
        in a format like "G301A A302T G856C"), and "variant_call_support"
        (sequences supporting barcode-variant call). Additional columns are
        removed unless they are specified `extra_cols`.
    geneseq : str
        Sequence of wildtype protein-coding gene.
    substitutions_are_codon : bool
        If `True`, then "substitutions" column in `barcode_variant_file` gives
        substitutions as codon rather than nucleotide mutations (e.g.,
        "ATG1ATA GTA5CCC" for substitutions at codons 1 and 5.
    extra_cols : list
        Additional columns in `barcode_variant_file` to retain when creating
        `barcode_variant_df` and `variant_count_df` attributes.
    substitutions_col : str
        Name of substitutions column in `barcode_variant_file` (use if you
        want it to be something other than "substitutions").
    primary_target : str or None
        Use this option if you have additional targets beyond the main gene
        for which we are analyzing variants. This might be the case if you
        have spiked other genes into the library. If this option is set to
        something other than `None`, then there must be a column in
        `barcode_variant_file` named "target" and one of these targets must
        be equal to 'primary_target'. If there are other targets, they should
        **not** have any substitutions as we don't parse substitutions in
        non-primary targets. Instead, `substitutions_col` for secondary targets
        should be empty or just have the name of the secondary target.

    Attributes
    ----------
    geneseq : str
        Wild-type sequence passed at initialization.
    sites : list
        List of all codon sites in 1, 2, ... numbering.
    codons : collections.OrderedDict
        `codons[r]` is wildtype codon at site `r`, ordered by sites.
    aas : collections.OrderedDict
        `aas[r]` is wildtype amino acid at site `r`, ordered by sites.
    libraries : list
        List of libraries in `barcode_variant_file`.
    barcode_variant_df : pandas.DataFrame
        Info about codon mutations parsed from `barcode_variant_file`.
        For non-primary targets, the mutations column just hold the
        target name.
    variant_count_df : pandas.DataFrame or None
        Initially `None`, but after data added with
        :class:`CodonVariantTable.addSampleCounts`, holds counts of
        each variant for each sample. Differs from `barcode_variant_df`
        in that the former just holds barcode-variant definitions,
        whereas `variant_count_df` has counts for each sample.
    primary_target : str or None
        If multiple targets, name of the main target for which we are
        calling variants.

    """

    _CODON_SUB_RE = re.compile(f"^(?P<wt>{'|'.join(CODONS)})"
                               r'(?P<r>\d+)'
                               f"(?P<mut>{'|'.join(CODONS)})$")
    """re Pattern : Match codon substitution; groups 'wt', 'r', 'mut'."""

    _AA_SUB_RE = re.compile((f"^(?P<wt>{'|'.join(AAS_WITHSTOP)})"
                             r'(?P<r>\d+)'
                             f"(?P<mut>{'|'.join(AAS_WITHSTOP)})$"
                             ).replace('*', r'\*'))
    """re Pattern : Match amino-acid substitution; groups 'wt', 'r', 'mut'."""

    _NT_SUB_RE = re.compile(f"^(?P<wt>{'|'.join(NTS)})"
                            r'(?P<r>\d+)'
                            f"(?P<mut>{'|'.join(NTS)})$")
    """re Pattern : Match nucleotide substitution; groups 'wt', 'r', 'mut'."""

    def __eq__(self, other):
        """Test if equal to object `other`."""
        # following here: https://stackoverflow.com/a/390640
        if type(other) is not type(self):
            return False
        elif self.__dict__.keys() != other.__dict__.keys():
            return False
        else:
            for key, val in self.__dict__.items():
                val2 = getattr(other, key)
                if isinstance(val, pd.DataFrame):
                    if not val.equals(val2):
                        return False
                else:
                    if val != val2:
                        return False
            return True

    @classmethod
    def from_variant_count_df(cls, *, variant_count_df_file, geneseq,
                              drop_all_libs=True, primary_target=None,
                              extra_cols=None):
        """:class:`CodonVariantTable` from CSV of `variant_count_df`.

        Note
        ----
        Use this method when you have written a CSV file of `variant_count_df`
        attribute of a :class:`CodonVariantTable`, and wish to re-initialize.

        Parameters
        ----------
        variant_count_df_file : str
            Name of CSV file containing the `variant_count_df`. Must have
            following columns: "barcode", "library", "variant_call_support",
            "codon_substitutions", "sample", and "count".
        geneseq : str
            Sequence of wildtype protein-coding gene.
        drop_all_libs : bool
            If there is a library named "all libraries", drop it as it probably
            added by :meth:`CodonVariantTable.addMergedLibraries` and
            duplicates information for the individual libraries.
        primary_target : None or str
            Meaning described in main :class:`CodonVariantTable` doc string.
        extra_cols : list
            Meaning described in main :class:`CodonVariantTable` doc string.

        Returns
        -------
        :class:`CodonVariantTable`

        """
        df = pd.read_csv(variant_count_df_file)

        req_cols = ['barcode', 'library', 'variant_call_support',
                    'codon_substitutions', 'sample', 'count']

        if primary_target is not None:
            if 'target' not in set(df.columns):
                raise ValueError('primary_target not None but no "target" col')
            req_cols.append('target')
        else:
            if 'target' in set(df.columns):
                raise ValueError('primary_target is None but "target" col')

        if not set(req_cols).issubset((df.columns)):
            raise ValueError(f"{variant_count_df_file} lacks required "
                             f"columns {req_cols}. It has: {set(df.columns)}")
        if extra_cols and not set(extra_cols).issubset((df.columns)):
            raise ValueError(f"{variant_count_df_file} lacks `extra_cols` "
                             f"columns {extra_cols}. Has: {set(df.columns)}")
        else:
            if extra_cols:
                df = df[req_cols + extra_cols]
            else:
                df = df[req_cols]

        if drop_all_libs:
            dropcol = 'all libraries'
            if dropcol in df['library'].unique():
                df = df.query('library != @dropcol')

        with tempfile.NamedTemporaryFile(mode='w') as f:
            (df
             .drop(columns=['sample', 'count'])
             .drop_duplicates()
             .to_csv(f, index=False)
             )
            f.flush()
            cvt = cls(barcode_variant_file=f.name,
                      geneseq=geneseq,
                      substitutions_are_codon=True,
                      substitutions_col='codon_substitutions',
                      primary_target=primary_target,
                      extra_cols=extra_cols,
                      )

        cvt.add_sample_counts_df(df[['library', 'sample', 'barcode', 'count']])

        return cvt

    def __init__(self, *, barcode_variant_file, geneseq,
                 substitutions_are_codon=False, extra_cols=None,
                 substitutions_col='substitutions',
                 primary_target=None):
        """See main class doc string."""
        self.geneseq = geneseq.upper()
        if not re.match(f"^[{''.join(NTS)}]+$", self.geneseq):
            raise ValueError(f"invalid nucleotides in {self.geneseq}")
        if ((len(geneseq) % 3) != 0) or len(geneseq) == 0:
            raise ValueError(f"`geneseq` invalid length {len(self.geneseq)}")
        self.sites = list(range(1, len(self.geneseq) // 3 + 1))
        self.codons = collections.OrderedDict([
                (r, self.geneseq[3 * (r - 1): 3 * r]) for r in self.sites])
        self.aas = collections.OrderedDict([
                (r, CODON_TO_AA[codon]) for r, codon in self.codons.items()])

        df = (pd.read_csv(barcode_variant_file)
              .rename(columns={substitutions_col: 'substitutions'})
              .assign(substitutions=lambda x: x['substitutions'].fillna(''))
              )
        required_cols = ['library', 'barcode',
                         'substitutions', 'variant_call_support']
        sort_cols = ['library', 'barcode']
        self.primary_target = primary_target
        if self.primary_target is not None:
            required_cols.insert(0, 'target')
            sort_cols.insert(0, 'target')
            if 'target' not in set(df.columns):
                raise ValueError('cannot use `primary_target` as the variant '
                                 'file lacks column named "target"')
            if self.primary_target not in set(df['target']):
                raise ValueError(f"{self.primary_target} not in 'target' col")
            df = (df  # if substitutions col is secondary target name, make ''
                  .assign(substitutions=lambda x: x['substitutions'].where(
                            ((x['target'] == self.primary_target) |
                             (x['substitutions'] != x['target'])),
                            '')
                          )
                  )
            subs_non_primary = (
                    df
                    .query('target != @self.primary_target')
                    .assign(has_subs=lambda x: (x['substitutions']
                                                .str.strip()
                                                .str.len()
                                                .astype(bool)
                                                )
                            )
                    .query('has_subs == True')
                    )
            if len(subs_non_primary):
                raise ValueError('non-primary targets have substitutions:\n' +
                                 subs_non_primary.head().to_csv())
        elif 'target' in set(df.columns):
            raise ValueError('variant file has column "target" but '
                             'you did not specify `primary_target`')
        if not set(df.columns).issuperset(set(required_cols)):
            raise ValueError("`variantfile` does not have "
                             f"required columns {required_cols}")
        if extra_cols and not set(df.columns).issuperset(set(extra_cols)):
            raise ValueError(f"`variantfile` lacks `extra_cols` {extra_cols}")
        if extra_cols:
            df = df[required_cols + extra_cols]
        else:
            df = df[required_cols]

        self.libraries = sorted(df.library.unique().tolist())
        self._valid_barcodes = {}
        for lib in self.libraries:
            barcodes = df.query('library == @lib').barcode
            if len(set(barcodes)) != len(barcodes):
                raise ValueError(f"duplicated barcodes for {lib}")
            self._valid_barcodes[lib] = set(barcodes)

        self._samples = {lib: [] for lib in self.libraries}
        self.variant_count_df = None

        if substitutions_are_codon:
            codonSubsFunc = self._sortCodonMuts
        else:
            codonSubsFunc = self._ntToCodonMuts

        self.barcode_variant_df = (
                df
                # info about codon and amino-acid substitutions
                .assign(codon_substitutions=lambda x: (x['substitutions']
                                                       .apply(codonSubsFunc)),
                        aa_substitutions=lambda x: (x.codon_substitutions
                                                    .apply(self.codonToAAMuts)
                                                    ),
                        n_codon_substitutions=lambda x: (x.codon_substitutions
                                                         .str.split()
                                                         .apply(len)),
                        n_aa_substitutions=lambda x: (x['aa_substitutions']
                                                      .str.split().apply(len))
                        )
                # we no longer need initial `substitutions` column
                .drop('substitutions', axis='columns')
                # sort to ensure consistent order
                .assign(library=lambda x: pd.Categorical(x['library'],
                                                         self.libraries,
                                                         ordered=True)
                        )
                .sort_values(sort_cols)
                .reset_index(drop=True)
                )

        assert ((self.primary_target is None and
                 'target' not in set(self.barcode_variant_df.columns)
                 ) or
                (self.primary_target is not None and
                 'target' in set(self.barcode_variant_df.columns)
                 )
                )
        assert self.primary_target is None or not (
                            self.barcode_variant_df
                            .query('target != @self.primary_target')
                            ['n_codon_substitutions']
                            .any()
                            )

        # check validity of codon substitutions given `geneseq`
        for codonmut in itertools.chain.from_iterable(
                        self.barcode_variant_df
                        .codon_substitutions.str.split()):
            m = self._CODON_SUB_RE.match(codonmut)
            if m is None:
                raise ValueError(f"invalid mutation {codonmut}")
            wt = m.group('wt')
            r = int(m.group('r'))
            mut = m.group('mut')
            if r not in self.sites:
                raise ValueError(f"invalid site {r} in mutation {codonmut}")
            if self.codons[r] != wt:
                raise ValueError(f"Wrong wildtype codon in {codonmut}. "
                                 f"Expected wildtype of {self.codons[r]}.")
            if wt == mut:
                raise ValueError(f"invalid mutation {codonmut}")

        # define some colors for plotting
        self._mutation_type_colors = {
                'nonsynonymous': CBPALETTE[1],
                'synonymous': CBPALETTE[2],
                'stop': CBPALETTE[3]
                }

        # for "safety" make the substitutions column for non-primary targets
        # just the target name
        if self.primary_target is not None:
            targets = self.barcode_variant_df['target']
            primary = (targets == self.primary_target)
            for col in ['aa_substitutions', 'codon_substitutions']:
                self.barcode_variant_df[col] = (
                        self.barcode_variant_df[col].where(primary, targets)
                        )

    @classmethod
    def add_frac_counts(self, variant_count_df):
        """Add fraction of counts from each variant in library/sample.

        Parameters
        ----------
        variant_count_df : pandas.DataFrame
            Same format as :attr:`CodonVariantTable`.

        Returns
        -------
        pandas.DataFrame
            A copy of `variant_count_df` with added column 'frac_counts'
            that gives fraction of all counts in that library / sample for
            that variant.

        Example
        --------
        >>> variant_count_df = pd.DataFrame.from_records(
        ...        [('lib1', 's1', 'AA', 1),
        ...         ('lib1', 's1', 'AT', 3),
        ...         ('lib1', 's2', 'GG', 0),
        ...         ('lib1', 's2', 'GA', 10),
        ...         ('lib2', 's1', 'CC', 5),
        ...         ('lib2', 's1', 'GA', 5),
        ...         ],
        ...        columns=['library', 'sample', 'barcode', 'count'],
        ...        )
        >>> CodonVariantTable.add_frac_counts(variant_count_df)
          library sample barcode  count  frac_counts
        0    lib1     s1      AA      1         0.25
        1    lib1     s1      AT      3         0.75
        2    lib1     s2      GG      0         0.00
        3    lib1     s2      GA     10         1.00
        4    lib2     s1      CC      5         0.50
        5    lib2     s1      GA      5         0.50

        """
        if variant_count_df is None:
            return None
        return (variant_count_df
                .assign(frac_counts=lambda x: (x['count'] /
                                               x.groupby(['library', 'sample'],
                                                         observed=True,
                                                         sort=False)
                                               ['count']
                                               .transform('sum')
                                               )
                        )
                )

    def samples(self, library):
        """List of all samples for `library`.

        Parameters
        ----------
        library : str
            Valid `library` for the :class:`CodonVariantTable`.

        Returns
        -------
        list
            All samples for which barcode counts have been added.

        """
        try:
            return self._samples[library]
        except KeyError:
            raise ValueError(f"invalid `library` {library}")

    def addSampleCounts(self, library, sample, barcodecounts):
        """Add variant counts for a sample to `variant_count_df`.

        Note
        ----
        If you have many samples to add at once, it is faster to use
        :meth:`CodonVariantTable.add_sample_counts_df` rather than
        repeatedly calling this method.

        Parameters
        ----------
        library : str
            Valid `library` for the :class:`CodonVariantTable`.
        sample : str
            Sample name, must **not** already be in
            :class:`CodonVariantTable.samples` for `library`.
        barcodecounts : pandas.DataFrame
            Counts for each variant by barcode. Must have columns "barcode"
            and "count". The "barcode" column must have all barcodes in
            :class:`CodonVariantTable.valid_barcodes` for `library`. Such
            data frames are returned by
            :mod:`dms_variants.illuminabarcodeparser.IlluminaBarcodeParser`.

        """
        if library not in self.libraries:
            raise ValueError(f"invalid library {library}")

        if sample in self.samples(library):
            raise ValueError(f"`library` {library} already "
                             f"has `sample` {sample}")

        req_cols = ['barcode', 'count']
        if not set(barcodecounts.columns).issuperset(set(req_cols)):
            raise ValueError(f"`barcodecounts` lacks columns {req_cols}")
        if len(barcodecounts) != len(set(barcodecounts.barcode.unique())):
            raise ValueError("`barcodecounts` has non-unique barcodes")
        if set(barcodecounts.barcode.unique()) != self.valid_barcodes(library):
            raise ValueError("barcodes in `barcodecounts` do not match "
                             f"those expected for `library` {library}")

        self._samples[library].append(sample)

        df = (barcodecounts
              [req_cols]
              .assign(library=library, sample=sample)
              .merge(self.barcode_variant_df,
                     how='inner',
                     on=['library', 'barcode'],
                     sort=False,
                     validate='one_to_one')
              )

        if self.variant_count_df is None:
            self.variant_count_df = df
        else:
            self.variant_count_df = pd.concat(
                              [self.variant_count_df, df],
                              axis='index',
                              ignore_index=True,
                              sort=False
                              )

        # samples in order added after ordering by library, getting
        # unique ones as here: https://stackoverflow.com/a/39835527
        unique_samples = list(collections.OrderedDict.fromkeys(
                itertools.chain.from_iterable(
                    [self.samples(lib) for lib in self.libraries])
                ))

        # make library and sample categorical and sort
        sort_cols = ['library', 'sample', 'count']
        order_cols = self.variant_count_df.columns.tolist()
        if self.primary_target is not None:
            sort_cols.insert(0, 'target')
            assert 'target' in order_cols
            order_cols.remove('target')
            order_cols.insert(0, 'target')
        self.variant_count_df = (
                self.variant_count_df
                .assign(library=lambda x: pd.Categorical(x['library'],
                                                         self.libraries,
                                                         ordered=True),
                        sample=lambda x: pd.Categorical(x['sample'],
                                                        unique_samples,
                                                        ordered=True),
                        )
                .sort_values(sort_cols,
                             ascending=[True] * (len(sort_cols) - 1) + [False])
                .reset_index(drop=True)
                [order_cols]
                )

    def add_sample_counts_df(self, counts_df):
        """Add variant counts for several samples to `variant_count_df`.

        Parameters
        ----------
        counts_df : pandas.DataFrame
            Must have columns 'library', 'sample', 'barcode', and 'count'.
            The sample must **not** already be in `CodonVariantTable.samples`
            for that library. The barcode columns must have all barcodes for
            that library including zero-count ones.

        """
        req_cols = ['library', 'sample', 'barcode', 'count']
        if not (set(counts_df.columns) >= set(req_cols)):
            raise ValueError(f"`counts_df` lacks required columns {req_cols}")

        for lib in counts_df['library'].unique():
            if lib not in self.libraries:
                raise ValueError(f"`counts_df` has unknown library {lib}")
            for s in counts_df.query('library == @lib')['sample'].unique():
                if s in self.samples(lib):
                    raise ValueError(f"library {lib} already has counts for "
                                     f"sample {s}, so you cannot add them")
                else:
                    self._samples[lib].append(s)

        df = (counts_df
              [req_cols]
              .merge(self.barcode_variant_df,
                     on=['library', 'barcode'],
                     sort=False,
                     how='inner',
                     validate='many_to_one',
                     )
              )

        if self.variant_count_df is None:
            self.variant_count_df = df
        else:
            assert not (set(df.groupby(['library', 'sample']).groups)
                        .intersection(set(self.variant_count_df
                                          .groupby(['library', 'sample'])
                                          .groups))
                        )
            self.variant_count_df = pd.concat(
                    [self.variant_count_df, df],
                    axis='index',
                    ignore_index=True,
                    sort=False,
                    )

        # samples in order added after ordering by library, getting
        # unique ones as here: https://stackoverflow.com/a/39835527
        unique_samples = list(collections.OrderedDict.fromkeys(
                itertools.chain.from_iterable(
                    [self.samples(lib) for lib in self.libraries])
                ))

        # make library and sample categorical and sort
        sort_cols = ['library', 'sample', 'count']
        order_cols = self.variant_count_df.columns.tolist()
        if self.primary_target is not None:
            sort_cols.insert(0, 'target')
            assert 'target' in order_cols
            order_cols.remove('target')
            order_cols.insert(0, 'target')
        self.variant_count_df = (
                self.variant_count_df
                .assign(library=lambda x: pd.Categorical(x['library'],
                                                         self.libraries,
                                                         ordered=True),
                        sample=lambda x: pd.Categorical(x['sample'],
                                                        unique_samples,
                                                        ordered=True),
                        )
                .sort_values(sort_cols,
                             ascending=[True] * (len(sort_cols) - 1) + [False])
                .reset_index(drop=True)
                [order_cols]
                )

    def valid_barcodes(self, library):
        """Set of valid barcodes for `library`.

        Parameters
        ----------
        library : str
            Name of a valid library.

        Returns
        -------
        set

        """
        if library not in self.libraries:
            raise ValueError(f"invalid `library` {library}; "
                             f"valid libraries are {self.libraries}")
        else:
            return self._valid_barcodes[library]

    def escape_scores(self,
                      sample_df,
                      score_type,
                      *,
                      pseudocount=0.5,
                      by='barcode',
                      logbase=2,
                      floor_B=0.01,
                      floor_E=0.01,
                      ceil_B=1.0,
                      ceil_E=1.0,
                      ):
        r"""Compute a score that represents escape from binding.

        Note
        ----
        Here we couch the explanation in terms of variant escape from
        antibody binding.

        Let :math:`v` be a variant, let :math:`B_v` be the fraction of this
        variant that is bound by antibody, and let :math:`E_v = 1 - B_v` be
        the fraction that escapes binding. A variant that completely escapes
        binding has :math:`B_v = 0` and :math:`E_v = 1`; a variant that is
        completely bound has :math:`B_v = 1` and :math:`E_v = 0`.

        We define the escape score :math:`s_v` in one of three ways depending
        on the value of `score_type` parameter:

          1. As the escape fraction, so :math:`s_v = E_v`.

          2. As the log of the escape fraction, so
             :math:`s_v = \log_b E_v` where :math:`b` is the logarithm base.

          3. As minus the log of the binding fraction, so
             :math:`s_v = -\log_b B_v`.

        In all cases, larger values of :math:`s_v` indicate more escape.

        We calculate :math:`E_v` as follows. Let :math:`f_v^{\rm{pre}}` be
        the frequency of :math:`v` prior  to selection for binding (so
        :math:`\sum_v f_v^{\rm{pre}} = 1`). Then the fraction of the library
        that is :math:`v` **after** selecting for unbound variants is

        .. math::

           f_v^{\rm{post}} =
           \frac{f_v^{\rm{pre}} \times E_v}
                {\sum_{v'} f_{v'}^{\rm{pre}} \times E_{v'}}.

        Note that the denominator of the above equation,
        :math:`F = \sum_v f_v^{\rm{pre}} \times E_v`,
        represents the overall fraction of the library that escapes binding,
        **which we assume is directly measured experimentally**.

        We can easily solve for :math:`E_v` as

        .. math::

           E_v = \frac{F \times f_v^{\rm{post}}}{f_v^{\rm{pre}}},

        and can similarly obtain :math:`B_v` from :math:`B_v = 1 - E_v`.

        We calculate :math:`E_v` directly from the actual counts of the
        variants pre- and post-selection. Let :math:`n_v^{\rm{pre}}` and
        :math:`n_v^{\rm{post}}` be the counts of variant :math:`v` pre-
        and post-selection **after** adding a pseudocount of :math:`P \ge 0`,
        and let :math:`N^{\rm{pre}} = \sum_v n_v^{\rm{pre}}` and
        :math:`N^{\rm{post}} = \sum_v n_v^{\rm{post}}` be the total counts of
        all variants pre- and post-selection. Then:

        .. math::

           E_v
           =
           F \times \frac{n_v^{\rm{post}} N^{\rm{pre}}}
                         {n_v^{\rm{pre}} N^{\rm{post}}}

        A complication is that :math:`s_v` can be undefined for certain values
        of :math:`E_v` or :math:`B_v`. Specifically, when using the log escape
        fraction definition (:math:`s_v = \log_b E_v`) then :math:`s_v` is
        undefined if :math:`E_v = 0`, so we put a floor on :math:`E_v` by
        defining :math:`s_v = \log_b \max\left(E_v, E_{\rm{floor}}\right)`.
        When using the minus log binding fraction definition
        (:math:`s_v = -\log_b B_v`), then :math:`s_v` is undefined if
        :math:`B_v \le 0`, so we put a floor on :math:`B_v` by defining
        defining :math:`s_v = -\log_b \max\left(B_v, B_{\rm{floor}}\right)`.
        We similarly define ceilings on :math:`E_v` and :math:`B_v`, although
        these ceilings are optional.

        We also estimate the variance :math:`\sigma_{s_v}^2` on
        :math:`s_v` from the variances on the counts, which we assume
        are :math:`\sigma_{n_v^{\rm{pre}}}^2 = n_v^{\rm{pre}}` and
        :math:`\sigma_{n_v^{\rm{post}}}^2 = n_v^{\rm{post}}` from
        Poisson counting statistics. To do this, we propagate the errors:

        .. math::

           \sigma_{s_v}^2
           &=&
           \left(\frac{\partial s_v}{\partial n_v^{\rm{pre}}}\right)^2
           \sigma_{n_v^{\rm{pre}}}^2 +
           \left(\frac{\partial s_v}{\partial n_v^{\rm{post}}}\right)^2
           \sigma_{n_v^{\rm{post}}}^2 \\
           &=&
           \left(\frac{\partial s_v}{\partial n_v^{\rm{pre}}}\right)^2
           n_v^{\rm{pre}} +
           \left(\frac{\partial s_v}{\partial n_v^{\rm{post}}}\right)^2
           n_v^{\rm{post}}.

        We calculate the derivatives of :math:`s_v` with respect to the counts
        numerically with a step size of one rather than analytically, since
        analytical calculations are confounded by the floors on :math:`E_v`
        or :math:`B_v`.

        Parameters
        -----------
        sample_df : pandas.DataFrame
            Comparisons we use to compute the functional scores. Should have
            these columns: 'pre_sample' (pre-selection sample), 'post_sample'
            (post-selection sample), 'library', 'name' (name for output),
            'frac_escape' (the overall fraction escaping :math:`F`).
        score_type : {'frac_escape', 'log_escape', 'minus_log_bind'}
            How to define escape score: :math:`E_v` if 'frac_escape';
            :math:`\log_b E_v` if 'log_escape'; :math:`-\log_b B_v` if
            'minus_log_bind'.
        pseudocount : float
            Pseudocount added to each count.
        by : {'barcode', 'aa_substitutions', 'codon_substitutions'}
            Compute effects for each barcode", set of amino-acid substitutions,
            or set of codon substitutions. In the last two cases, all barcodes
            with each set of substitutions are combined.
        logbase : float
            Base for logarithm when calculating functional score.
        floor_B : float
            Floor assigned to :math:`B_v`, :math:`B_{\rm{floor}}`
            if `score_type` is 'minus_log_bind'.
        floor_E : float
            Floor assigned to :math:`E_v`, :math:`E_{\rm{floor}}`
            if `score_type` is 'frac_escape' or 'log_escape'.
        ceil_B : float or None
            Ceiling assigned to :math:`B_v`, or `None` if no ceiling.
        ceil_E : float or None
            Ceiling assigned to :math:`E_v`, or `None` if no ceiling.

        Returns
        -------
        pandas.DataFrame
            Has the following columns:
              - 'name': specified in `sample_df`
              - 'library': the library
              - 'pre_sample': specified in `sample_df`
              - 'post_sample': specified in `sample_df`
              - the grouping used to compute scores (the value of `by`)
              - 'score': :math:`s_v`
              - 'score_var': :math:`\sigma_{s_v}^2`
              - 'pre_count': :math:`n_v^{\rm{pre}}` (without pseudocount)
              - 'post_count': :math:`n_v^{\rm{post}}` (without pseudocount)
              - as many of 'aa_substitutions', 'n_aa_substitutions',
                'codon_substitutions', and 'n_codon_substitutions' as
                makes sense to retain given value of `by`.

        """
        req_cols = {'pre_sample', 'post_sample', 'library', 'name',
                    'frac_escape'}
        if not set(sample_df.columns).issuperset(req_cols):
            raise ValueError(f"`sample_df` lacks required columns: {req_cols}")
        if len(sample_df) != len(sample_df.groupby(['name', 'library'])):
            raise ValueError('names / libraries in `sample_df` not unique')
        if (0 >= sample_df['frac_escape']).any() or (sample_df['frac_escape']
                                                     >= 1).any():
            raise ValueError('in `sample_df`, `frac_escape` must be > 0, < 1')

        # get data frame with samples of interest
        df = []
        already_added = set()
        for tup in sample_df.itertuples():
            lib = tup.library
            for stype in ['pre_sample', 'post_sample']:
                sample = getattr(tup, stype)
                if (sample, lib) in already_added:
                    continue
                already_added.add((sample, lib))
                tup_df = (self.variant_count_df
                          .query('(sample == @sample) and (library == @lib)')
                          )
                if len(tup_df) < 1:
                    raise ValueError(f"no sample {sample} library {lib}")
                df.append(tup_df)
        df = pd.concat(df, ignore_index=True, sort=False)

        # sum counts in groups specified by `by`
        group_cols = ['codon_substitutions', 'n_codon_substitutions',
                      'aa_substitutions', 'n_aa_substitutions']
        if self.primary_target is not None:
            group_cols.append('target')
        if by in {'aa_substitutions', 'codon_substitutions'}:
            group_cols = group_cols[group_cols.index(by) + 1:]
            df = (df
                  .groupby(['library', 'sample', by, *group_cols],
                           observed=True, sort=False)
                  .aggregate({'count': 'sum'})
                  .reset_index()
                  )
        elif by != 'barcode':
            raise ValueError(f"invalid `by` of {by}")

        # get data frame with pre- and post-selection samples / counts
        df_scores = []
        for tup in sample_df.itertuples():
            name_dfs = []
            lib = tup.library  # noqa: F841
            for stype in ('pre_sample', 'post_sample'):
                s_name = getattr(tup, stype)  # noqa: F841
                name_dfs.append(
                    df
                    .query('(sample == @s_name) and (library == @lib)')
                    .rename(columns={'count': stype.split('_')[0] + '_count',
                                     'sample': stype})
                    .assign(name=tup.name, frac_escape=tup.frac_escape)
                    )
            df_scores.append(pd.merge(*name_dfs, how='inner', validate='1:1'))
        df_scores = pd.concat(df_scores, ignore_index=True, sort=False)

        # check pseudocount
        if pseudocount < 0:
            raise ValueError(f"`pseudocount` is < 0: {pseudocount}")
        elif (pseudocount == 0) and any((df_scores[c] <= 0).any() for c
                                        in ['pre_count', 'post_count']):
            raise ValueError('some counts are zero, you must use '
                             '`pseudocount` > 0')
        if floor_B <= 0:
            raise ValueError('`floor_B` must be > 0')
        if floor_E <= 0 and score_type != 'frac_escape':
            raise ValueError('`floor_E` must be > 0')
        if (ceil_B is not None) and (ceil_B <= floor_B):
            raise ValueError('`ceil_B` must be > `floor_B`')
        if (ceil_E is not None) and (ceil_E <= floor_E):
            raise ValueError('`ceil_E` must be > `floor_E`')

        # compute escape scores
        def _compute_escape_scores():
            _df_scores = (
                df_scores
                .assign(
                    n_v_pre=lambda x: x['pre_count'] + pseudocount,
                    n_v_post=lambda x: x['post_count'] + pseudocount,
                    N_pre=lambda x: (x
                                     .groupby(['name', 'library'],
                                              observed=True,
                                              sort=False)
                                     ['n_v_pre']
                                     .transform('sum')
                                     ),
                    N_post=lambda x: (x
                                      .groupby(['name', 'library'],
                                               observed=True,
                                               sort=False)
                                      ['n_v_post']
                                      .transform('sum')
                                      ),
                    E_v=lambda x: (x['frac_escape'] * x['n_v_post'] *
                                   x['N_pre'] / (x['n_v_pre'] * x['N_post'])),
                    B_v=lambda x: 1 - x['E_v'],
                    # for computing derivatives, increment counts by one
                    n_v_pre_d=lambda x: x['n_v_pre'] + 1,
                    n_v_post_d=lambda x: x['n_v_post'] + 1,
                    N_pre_d=lambda x: x['N_pre'] + 1,
                    N_post_d=lambda x: x['N_post'] + 1,
                    E_v_dpre=lambda x: (x['frac_escape'] * x['n_v_post'] *
                                        x['N_pre_d'] /
                                        (x['n_v_pre_d'] * x['N_post'])),
                    E_v_dpost=lambda x: (x['frac_escape'] * x['n_v_post_d'] *
                                         x['N_pre'] /
                                         (x['n_v_pre'] * x['N_post_d'])),
                    B_v_dpre=lambda x: 1 - x['E_v_dpre'],
                    B_v_dpost=lambda x: 1 - x['E_v_dpost'],
                    )
                )
            if score_type == 'minus_log_bind':
                floor = floor_B
                ceil = ceil_B
                cols = ['B_v', 'B_v_dpre', 'B_v_dpost']
                def func(x): return -numpy.log(x) / numpy.log(logbase)
            elif score_type == 'log_escape':
                floor = floor_E
                ceil = ceil_E
                cols = ['E_v', 'E_v_dpre', 'E_v_dpost']
                def func(x): return numpy.log(x) / numpy.log(logbase)
            elif score_type == 'frac_escape':
                floor = floor_E
                ceil = ceil_E
                cols = ['E_v', 'E_v_dpre', 'E_v_dpost']
                def func(x): return x
            else:
                raise ValueError(f"invalid `score_type` {score_type}")
            for col in cols:
                _df_scores[col] = numpy.clip(_df_scores[col], floor, ceil)
            _df_scores['score'] = func(_df_scores[cols[0]])
            _df_scores['score_dpre'] = func(_df_scores[cols[1]])
            _df_scores['score_dpost'] = func(_df_scores[cols[2]])
            _df_scores['score_var'] = (
                    (_df_scores['score_dpre'] - _df_scores['score'])**2 *
                    _df_scores['n_v_pre'] +
                    (_df_scores['score_dpost'] - _df_scores['score'])**2 *
                    _df_scores['n_v_post'])
            return _df_scores

        df_scores = _compute_escape_scores()
        assert df_scores['score'].notnull().all(), df_scores

        # get columns to keep
        col_order = ['name', 'library', 'pre_sample', 'post_sample', by,
                     'score', 'score_var',
                     'pre_count', 'post_count', *group_cols]
        if self.primary_target is not None:
            assert col_order.count('target') == 1
            col_order.remove('target')
            col_order.insert(1, 'target')
        else:
            assert 'target' not in col_order

        return df_scores[col_order]

    def func_scores(self, preselection, *,
                    pseudocount=0.5, by="barcode",
                    libraries='all', syn_as_wt=False, logbase=2,
                    permit_zero_wt=False, permit_self_comp=False):
        r"""Get data frame with functional scores for variants.

        Note
        ----
        The functional score is calculated from the change in counts
        for a variant pre- and post-selection using the formula in
        `Otwinoski et al (2018) <https://doi.org/10.1073/pnas.1804015115>`_.

        Specifically, let :math:`n^v_{pre}` and :math:`n^v_{post}` be
        the counts of variant :math:`v` pre- and post-selection, and
        let :math:`n^{wt}_{pre}` and :math:`n^{wt}_{post}` be
        the summed counts of **all** wildtype variants pre- and post-
        selection.

        Then the functional score of the variant is:

        .. math::

            f_v = \log_b\left(\frac{n^v_{post} /
            n^{wt}_{post}}{n^v_{pre} / n^{wt}_{pre}}\right).

        The variance due to Poisson sampling statistics is:

        .. math::

            \sigma^2_v = \frac{1}{\left(\ln b\right)^2}
            \left(\frac{1}{n^v_{post}} + \frac{1}{n^{wt}_{post}} +
            \frac{1}{n^v_{pre}} + \frac{1}{n^{wt}_{pre}}\right)

        where :math:`b` is logarithm base (see `logbase` parameter).

        For both calculations, a pseudocount (see `pseudocount` parameter)
        is added to each count first. The wildtype counts are computed
        across all **fully wildtype** variants (see `syn_as_wt` for
        how this is defined).

        If there are multiple targets, the functional scores for all
        targets are relative to the wildtype of the primary target.

        Parameters
        ----------
        preselection : str or dict
            Pre-selection sample. If the same for all post-selection then
            provide the name as str. If it differs among post-selection
            samples, then provide a dict keyed by each post-selection
            sample with the pre-selection sample being the value.
        pseudocount : float
            Pseudocount added to each count.
        by : {'barcode', 'aa_substitutions', 'codon_substitutions'}
            Compute effects for each barcode", set of amino-acid substitutions,
            or set of codon substitutions. In the last two cases, all barcodes
            with each set of substitutions are combined. If you use
            "aa_substitutions" then it may be more sensible to set
            `syn_as_wt` to `True`.
        syn_as_wt : bool
            In formula for functional scores, consider variants with only
            synonymous mutations when determining wildtype counts? If `False`,
            only variants with **no** mutations of any type contribute.
        libraries : {'all', 'all_only', list}
            Perform calculation for all libraries including a merge
            (named "all libraries"), only for the merge of all libraries,
            or for the libraries in the list. If `by` is 'barcode', then
            the barcodes for the merge have the library name pre-pended.
        logbase : float
            Base for logarithm when calculating functional score.
        permit_zero_wt : bool
            If the wildtype counts are zero for any sample, raise an error
            or permit the calculation to proceed just using pseudocount?
        permit_self_comp : bool
            Permit comparisons for sample pre- and post-selection?

        Returns
        -------
        pandas.DataFrame
            Has the following columns:
              - "library": the library
              - "pre_sample": the pre-selection sample
              - "post_sample": the post-selection sample
              - value corresponding to grouping used to compute effects (`by`)
              - "func_score": the functional score
              - "func_score_var": variance on the functional score
              - "pre_count": pre-selection counts
              - "post_count: post-selection counts
              - "pre_count_wt": pre-selection counts for all wildtype
              - "post_count_wt": post-selection counts for all wildtype
              - "pseudocount": the pseudocount value
              - as many of "aa_substitutions", "n_aa_substitutions",
                "codon_substitutions", and "n_codon_substitutions"
                as can be retained given value of `by`.

        """
        ordered_samples = self.variant_count_df['sample'].unique()
        if isinstance(preselection, str):
            # make `preselection` into dict
            preselection = {s: preselection for s in ordered_samples
                            if s != preselection or permit_self_comp}
        elif not isinstance(preselection, dict):
            raise ValueError('`preselection` not str or dict')
        if not permit_self_comp:
            if any(pre == post for pre, post in preselection.items()):
                raise ValueError('`permit_self_comp` is False but there'
                                 ' are identical pre and post samples')

        # all samples of interest
        samples = set(preselection.keys()).union(set(preselection.values()))
        if not samples.issubset(set(ordered_samples)):
            extra_samples = samples - set(ordered_samples)
            raise ValueError(f"invalid samples: {extra_samples}")

        # get data frame with samples of interest
        if self.variant_count_df is None:
            raise ValueError('no sample variant counts have been added')
        df = (self.variant_count_df
              .query('sample in @samples')
              )

        if libraries == 'all':
            df = self.addMergedLibraries(df)
        elif libraries == 'all_only':
            df = (self.addMergedLibraries(df)
                  .query('library == "all libraries"')
                  )
        else:
            if set(libraries) > set(self.libraries):
                raise ValueError(f"invalid `libraries` of {libraries}. Must "
                                 'be "all", "all_only", or a list containing '
                                 f"some subset of {self.libraries}")
            df = df.query('library in @libraries')

        # get wildtype counts for each sample and library
        if syn_as_wt:
            wt_col = 'n_aa_substitutions'
        else:
            wt_col = 'n_codon_substitutions'
        wt_counts = (
                df
                .assign(count=lambda x: (x['count'] *
                                         (0 == x[wt_col]).astype('int') *
                                         (self.primary_target is None or
                                          x['target'] == self.primary_target)
                                         )
                        )
                .groupby(['library', 'sample'], sort=False, observed=True)
                .aggregate({'count': 'sum'})
                .reset_index()
                )
        if (wt_counts['count'] <= 0).any() and not permit_zero_wt:
            raise ValueError(f"no wildtype counts:\n{wt_counts}")

        # sum counts in groups specified by `by`
        group_cols = ['codon_substitutions', 'n_codon_substitutions',
                      'aa_substitutions', 'n_aa_substitutions']
        if self.primary_target is not None:
            group_cols.append('target')
        if by in {'aa_substitutions', 'codon_substitutions'}:
            group_cols = group_cols[group_cols.index(by) + 1:]
            df = (df
                  .groupby(['library', 'sample', by, *group_cols],
                           observed=True, sort=False)
                  .aggregate({'count': 'sum'})
                  .reset_index()
                  )
        elif by != 'barcode':
            raise ValueError(f"invalid `by` of {by}")

        # get data frame with pre- and post-selection samples / counts
        df_func_scores = []
        for post_sample in ordered_samples:
            if post_sample not in preselection:
                continue
            pre_sample = preselection[post_sample]
            sample_dfs = []
            for stype, s in [('pre', pre_sample),  # noqa: B007
                             ('post', post_sample)]:
                sample_dfs.append(
                        df
                        .query('sample == @s')
                        .rename(columns={'count': f"{stype}_count"})
                        .merge(wt_counts
                               .rename(columns={'count': f"{stype}_count_wt"}),
                               how='inner', validate='many_to_one'
                               )
                        .rename(columns={'sample': f"{stype}_sample"})
                        )
            df_func_scores.append(
                    pd.merge(sample_dfs[0], sample_dfs[1],
                             how='inner', validate='1:1')
                    )
        df_func_scores = pd.concat(df_func_scores,
                                   ignore_index=True, sort=False)

        # check pseudocount
        if pseudocount < 0:
            raise ValueError(f"`pseudocount` is < 0: {pseudocount}")
        elif (pseudocount == 0) and any((df_func_scores[c] <= 0).any() for c
                                        in ['pre_count', 'post_count',
                                            'pre_count_wt', 'post_count_wt']):
            raise ValueError('some counts are zero, you must use '
                             '`pseudocount` > 0')

        # calculate functional score and variance
        df_func_scores = (
                df_func_scores
                .assign(
                    pseudocount=pseudocount,
                    func_score=lambda x: numpy.log(
                                         ((x.post_count + x.pseudocount) /
                                          (x.post_count_wt + x.pseudocount)) /
                                         ((x.pre_count + x.pseudocount) /
                                          (x.pre_count_wt + x.pseudocount))
                                         ) / numpy.log(logbase),
                    func_score_var=lambda x: (
                                1 / (x.post_count + x.pseudocount) +
                                1 / (x.post_count_wt + x.pseudocount) +
                                1 / (x.pre_count + x.pseudocount) +
                                1 / (x.pre_count_wt + x.pseudocount)
                                ) / (numpy.log(logbase)**2)
                    )
                )

        col_order = ['library', 'pre_sample', 'post_sample', by,
                     'func_score', 'func_score_var', 'pre_count',
                     'post_count', 'pre_count_wt', 'post_count_wt',
                     'pseudocount', *group_cols]
        if self.primary_target is not None:
            assert col_order.count('target') == 1
            col_order.remove('target')
            col_order.insert(0, 'target')
        else:
            assert 'target' not in col_order

        return df_func_scores[col_order]

    def n_variants_df(self, *, libraries='all', samples='all',
                      min_support=1, variant_type='all',
                      mut_type=None, sample_rename=None,
                      primary_target_only=False):
        """Get number variants per library / sample (and target if specified).

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <= 1 `mut_type` mutation.
        mut_type : {'aa', 'codon', None}
            If `variant_type` is 'single', indicate what type of single
            mutants we are filtering for.
        primary_target_only : bool
            Only return counts for the primary target.
        All other args
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        pandas.DataFrame

        """
        df, nlibraries, nsamples = self._getPlotData(
                                    libraries,
                                    samples,
                                    min_support,
                                    primary_target_only=primary_target_only,
                                    sample_rename=sample_rename)

        if variant_type == 'single':
            if mut_type in {'aa', 'codon'}:
                df = df.query(f"n_{mut_type}_substitutions <= 1")
            else:
                raise ValueError('`mut_type` must be "aa" or "single"')
        elif variant_type != 'all':
            raise ValueError(f"invalid `variant_type` {variant_type}")

        group_cols = ['library', 'sample']
        if (self.primary_target is not None) and (not primary_target_only):
            group_cols.insert(0, 'target')
            assert 'target' in set(df.columns)
        else:
            assert 'target' not in set(df.columns)

        return (df
                .groupby(group_cols, observed=True)
                .aggregate({'count': 'sum'})
                .reset_index()
                )

    def mutCounts(self, variant_type, mut_type, *,
                  libraries='all', samples='all', min_support=1,
                  sample_rename=None):
        """Get counts of each individual mutations (only in primary target).

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include just single mutants, or all mutants?
        other_parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        pandas.DataFrame
            Tidy data frame with columns named "library", "sample",
            "mutation", "count", "mutation_type", and "site". If there
            are multiple targets, only returns counts for the primary
            target.

        """
        df, nlibraries, nsamples = self._getPlotData(libraries,
                                                     samples,
                                                     min_support,
                                                     primary_target_only=True)
        assert 'target' not in set(df.columns)

        samplelist = df['sample'].unique().tolist()
        librarylist = df['library'].unique().tolist()

        if mut_type == 'codon':
            wts = self.codons
            chars = CODONS
            mutation_types = ['nonsynonymous', 'synonymous', 'stop']
        elif mut_type == 'aa':
            wts = self.aas
            chars = AAS_WITHSTOP
            mutation_types = ['nonsynonymous', 'stop']
        else:
            raise ValueError(f"invalid mut_type {mut_type}")

        # data frame listing all mutations with count 0
        mut_list = []
        for r, wt in wts.items():
            for mut in chars:
                if mut != wt:
                    mut_list.append(f'{wt}{r}{mut}')
        all_muts = pd.concat([
                    pd.DataFrame({'mutation': mut_list,
                                  'library': library,
                                  'sample': sample,
                                  'count': 0})
                    for library, sample in
                    itertools.product(librarylist, samplelist)])

        if variant_type == 'single':
            df = df.query(f'n_{mut_type}_substitutions == 1')
        elif variant_type == 'all':
            df = df.query(f'n_{mut_type}_substitutions >= 1')
        else:
            raise ValueError(f"invalid variant_type {variant_type}")

        def _classify_mutation(mut_str):
            if mut_type == 'aa':
                m = self._AA_SUB_RE.match(mut_str)
                assert m is not None, f"cannot match aa mut: {mut_str}"
                wt_aa = m.group('wt')
                mut_aa = m.group('mut')
            else:
                m = self._CODON_SUB_RE.match(mut_str)
                assert m is not None, f"cannot match codon mut: {mut_str}"
                wt_aa = CODON_TO_AA[m.group('wt')]
                mut_aa = CODON_TO_AA[m.group('mut')]
            if wt_aa == mut_aa:
                return 'synonymous'
            elif mut_aa == '*':
                return 'stop'
            else:
                return 'nonsynonymous'

        def _get_site(mut_str):
            if mut_type == 'aa':
                m = self._AA_SUB_RE.match(mut_str)
            else:
                m = self._CODON_SUB_RE.match(mut_str)
            site = int(m.group('r'))
            assert site in self.sites
            return site

        if sample_rename is None:
            sample_rename_dict = _dict_missing_is_key()
        else:
            if len(sample_rename) != len(set(sample_rename.values())):
                raise ValueError('duplicates in `sample_rename`')
            sample_rename_dict = _dict_missing_is_key(sample_rename)

        df = (df
              .rename(columns={f"{mut_type}_substitutions": 'mutation'})
              [['library', 'sample', 'mutation', 'count']]
              .pipe(dms_variants.utils.tidy_split, column='mutation')
              .merge(all_muts, how='outer')
              .groupby(['library', 'sample', 'mutation'])
              .aggregate({'count': 'sum'})
              .reset_index()
              .assign(library=lambda x: pd.Categorical(x['library'],
                                                       librarylist,
                                                       ordered=True),
                      sample=lambda x: pd.Categorical(
                                x['sample'].map(sample_rename_dict),
                                [sample_rename_dict[s] for s in samplelist],
                                ordered=True),
                      mutation_type=lambda x:
                      pd.Categorical(x['mutation'].apply(_classify_mutation),
                                     mutation_types,
                                     ordered=True),
                      site=lambda x: x['mutation'].apply(_get_site),
                      )
              .sort_values(
                ['library', 'sample', 'count', 'mutation'],
                ascending=[True, True, False, True])
              .reset_index(drop=True)
              )

        return df

    def plotMutHeatmap(self, variant_type, mut_type, *,
                       count_or_frequency='frequency',
                       libraries='all', samples='all', plotfile=None,
                       orientation='h', widthscale=1, heightscale=1,
                       min_support=1, sample_rename=None,
                       one_lib_facet=False):
        """Heatmap of mutation counts or frequencies (for primary target only).

        Parameters
        ----------
        count_or_frequency : {'count', 'frequency'}
            Plot mutation counts or frequencies?
        other_parameters
            Same as for :meth:`CodonVariantTable.plotCumulMutCoverage`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.mutCounts(variant_type, mut_type, samples=samples,
                            libraries=libraries, min_support=min_support,
                            sample_rename=sample_rename)
        assert 'target' not in set(df.columns)

        n_variants = (self.n_variants_df(
                                    libraries=libraries,
                                    samples=samples,
                                    min_support=min_support,
                                    variant_type=variant_type,
                                    mut_type=mut_type,
                                    sample_rename=sample_rename,
                                    primary_target_only=True)
                      .rename(columns={'count': 'nseqs'})
                      )

        # order amino acids by Kyte-Doolittle hydrophobicity,
        aa_order = [tup[0] for tup in sorted(
                    Bio.SeqUtils.ProtParamData.kd.items(),
                    key=lambda tup: tup[1])] + ['*']
        if mut_type == 'codon':
            height_per = 5.5
            mut_desc = 'codon'
            # order codons by the amino acid they encode
            order = list(itertools.chain.from_iterable(
                         [AA_TO_CODONS[aa] for aa in aa_order]))
            pattern = self._CODON_SUB_RE.pattern
        elif mut_type == 'aa':
            height_per = 1.7
            mut_desc = 'amino acid'
            order = aa_order
            pattern = self._AA_SUB_RE.pattern
        else:
            raise ValueError(f"invalid `mut_type` {mut_type}")

        df = (df
              [['library', 'sample', 'mutation', 'site', 'count']]
              .merge(n_variants, on=['library', 'sample'])
              .assign(frequency=lambda x: x['count'] / x['nseqs'],
                      mut_char=lambda x: pd.Categorical(x['mutation'].str
                                                        .extract(pattern).mut,
                                                        order,
                                                        ordered=True)
                      )
              )
        assert 'target' not in set(df.columns)

        if count_or_frequency not in {'count', 'frequency'}:
            raise ValueError(f"invalid count_or_frequency "
                             f"{count_or_frequency}")

        nlibraries = len(df['library'].unique())
        nsamples = len(df['sample'].unique())

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1.6 + 3.5 * nlibraries)
            height = heightscale * (0.8 + height_per * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1.6 + 3.5 * nsamples)
            height = heightscale * (0.8 + height_per * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        p = (p9.ggplot(df, p9.aes('site', 'mut_char',
                                  fill=count_or_frequency)) +
             p9.geom_tile() +
             p9.theme(figure_size=(width, height),
                      legend_key=p9.element_blank(),
                      axis_text_y=p9.element_text(size=6)
                      ) +
             p9.scale_x_continuous(
                name=f'{mut_desc} site',
                limits=(min(self.sites) - 1, max(self.sites) + 1),
                expand=(0, 0)
                ) +
             p9.ylab(mut_desc) +
             p9.scale_fill_cmap('gnuplot')
             )

        if samples is None:
            if nlibraries > 1 or one_lib_facet:
                p = (p +
                     p9.facet_wrap('~ library',
                                   nrow={'h': 1, 'v': nlibraries}[orientation])
                     )
        else:
            p = p + p9.facet_grid(facet_str)

        if plotfile:
            p.save(plotfile, height=height, width=width,
                   verbose=False, limitsize=False)

        return p

    def plotMutFreqs(self, variant_type, mut_type, *,
                     libraries='all', samples='all', plotfile=None,
                     orientation='h', widthscale=1, heightscale=1,
                     min_support=1, sample_rename=None,
                     one_lib_facet=False):
        """Mutation frequency along length of gene (primary target only).

        Parameters
        ----------
        All parameters
            Same as for :meth:`CodonVariantTable.plotCumulMutCoverage`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.mutCounts(variant_type, mut_type, samples=samples,
                            libraries=libraries, min_support=min_support,
                            sample_rename=sample_rename)

        n_variants = (self.n_variants_df(libraries=libraries,
                                         samples=samples,
                                         min_support=min_support,
                                         variant_type=variant_type,
                                         mut_type=mut_type,
                                         sample_rename=sample_rename,
                                         primary_target_only=True)
                      .rename(columns={'count': 'nseqs'})
                      )

        assert 'target' not in set(df.columns).union(set(n_variants.columns))

        df = (df
              .groupby(['library', 'sample', 'mutation_type', 'site'])
              .aggregate({'count': 'sum'})
              .reset_index()
              .merge(n_variants, on=['library', 'sample'])
              .assign(freq=lambda x: x['count'] / x['nseqs'])
              )

        nlibraries = len(df['library'].unique())
        nsamples = len(df['sample'].unique())

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1.6 + 1.8 * nlibraries)
            height = heightscale * (0.8 + 1 * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1.6 + 1.8 * nsamples)
            height = heightscale * (0.8 + 1 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        if mut_type == 'aa':
            mut_desc = 'amino-acid'
        else:
            mut_desc = mut_type

        if height < 3:
            ylabel = (f'{mut_desc} mutation\nfrequency '
                      f'({variant_type} mutants)')
        else:
            ylabel = (f'{mut_desc} mutation frequency '
                      f'({variant_type} mutants)')

        p = (p9.ggplot(df, p9.aes('site', 'freq', color='mutation_type')) +
             p9.geom_step() +
             p9.scale_color_manual(
                [self._mutation_type_colors[m] for m in
                 df.mutation_type.unique().sort_values().tolist()],
                name='mutation type'
                ) +
             p9.scale_x_continuous(
                name=f'{mut_desc} site',
                limits=(min(self.sites), max(self.sites))
                ) +
             p9.ylab(ylabel) +
             p9.theme(figure_size=(width, height),
                      legend_key=p9.element_blank(),
                      )
             )

        if samples is None:
            if nlibraries > 1 or one_lib_facet:
                p = (p +
                     p9.facet_wrap('~ library',
                                   nrow={'h': 1, 'v': nlibraries}[orientation])
                     )
        else:
            p = p + p9.facet_grid(facet_str)

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotCountsPerVariant(self,
                             *,
                             ystat='frac_counts',
                             logy=True,
                             by_variant_class=False,
                             classifyVariants_kwargs=None,
                             variant_type='all',
                             libraries='all', samples='all', plotfile=None,
                             orientation='h', widthscale=1, heightscale=1,
                             min_support=1, mut_type='aa',
                             sample_rename=None, one_lib_facet=False,
                             primary_target_only=False):
        """Plot variant index versus counts (or frac counts).

        Parameters
        -----------
        ystat : {'frac_counts', 'count'}
            Is y-axis counts from variant, or fraction of counts in
            library / sample from variant?
        logy : bool
            Show the y-axis on a log scale. If so, all values of 0 are
            set to half the minimum observed value > 0, and dashed line
            is drawn to indicate that points below it are not observed.
        other_parameters
            Same as for :meth:`CodonVariantTable.plotCumulVariantCounts`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        if samples is None:
            raise ValueError('plot nonsensical with `samples` of `None`')

        df, nlibraries, nsamples = self._getPlotData(
                                                libraries,
                                                samples,
                                                min_support,
                                                primary_target_only,
                                                sample_rename=sample_rename)

        if variant_type == 'single':
            if mut_type == 'aa':
                mutstr = 'amino acid'
            elif mut_type == 'codon':
                mutstr = mut_type
            else:
                raise ValueError(f"invalid `mut_type` {mut_type}")
            ylabel = f"single {mutstr} variants with >= this many counts"
            df = df.query(f"n_{mut_type}_substitutions <= 1")
        elif variant_type == 'all':
            ylabel = 'variants with >= this many counts'
        else:
            raise ValueError(f"invalid `variant_type` {variant_type}")

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1 + 1.8 * nlibraries)
            height = heightscale * (0.6 + 1.5 * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1 + 1.5 * nsamples)
            height = heightscale * (0.6 + 1.5 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        if ystat == 'frac_counts':
            df = self.add_frac_counts(df).drop(columns='count')
            ylabel = 'fraction of counts'
        elif ystat == 'count':
            ylabel = 'number of counts'
        else:
            raise ValueError(f"invalid `ystat` of {ystat}")

        ivariant_group_cols = ['library', 'sample']
        if by_variant_class:
            ivariant_group_cols.append('variant class')
            if not classifyVariants_kwargs:
                kw_args = {}
            else:
                kw_args = {k: v for k, v in classifyVariants_kwargs.items()}
            if 'primary_target' not in kw_args:
                kw_args['primary_target'] = self.primary_target
            if 'class_as_categorical' not in kw_args:
                kw_args['class_as_categorical'] = True
            df = (self.classifyVariants(df, **kw_args)
                  .rename(columns={'variant_class': 'variant class'})
                  )
            aes = p9.aes('ivariant', ystat, color='variant class')
        else:
            aes = p9.aes('ivariant', ystat)

        df = (df
              .sort_values(ystat, ascending=False)
              .assign(ivariant=lambda x: (x
                                          .groupby(ivariant_group_cols)
                                          .cumcount()
                                          + 1
                                          )
                      )
              )

        if logy:
            min_gt_0 = df.query(f"{ystat} > 0")[ystat].min()
            min_y = min_gt_0 / 2
            df[ystat] = numpy.clip(df[ystat], min_y, None)
            yscale = p9.scale_y_log10(
                        labels=dms_variants.utils.latex_sci_not)
            hline = p9.geom_hline(yintercept=(min_y + min_gt_0) / 2,
                                  linetype='dotted',
                                  color=CBPALETTE[0],
                                  size=1,
                                  )
        else:
            yscale = p9.scale_y_continuous(
                        labels=dms_variants.utils.latex_sci_not)
            hline = None

        p = (p9.ggplot(df) +
             aes +
             p9.geom_step() +
             p9.xlab('variant number') +
             p9.ylab(ylabel) +
             yscale +
             p9.theme(figure_size=(width, height),
                      axis_text_x=p9.element_text(angle=90),
                      ) +
             p9.facet_grid(facet_str) +
             p9.scale_color_manual(values=CBPALETTE[1:])
             )
        if hline is not None:
            p = p + hline

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotCumulVariantCounts(self, *, variant_type='all',
                               libraries='all', samples='all', plotfile=None,
                               orientation='h', widthscale=1, heightscale=1,
                               min_support=1, mut_type='aa',
                               tot_variants_hline=True,
                               sample_rename=None, one_lib_facet=False,
                               primary_target_only=True):
        """Plot number variants with >= that each number of counts.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <=1 `mut_type` mutation.
        tot_variants_hline : bool
            Include dotted horizontal line indicating total number of variants.
        primary_target_only : bool
            Only show counts for the primary target.
        other_parameters
            Same as for :meth:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df, nlibraries, nsamples = self._getPlotData(
                                                libraries,
                                                samples,
                                                min_support,
                                                primary_target_only,
                                                sample_rename=sample_rename)

        if variant_type == 'single':
            if mut_type == 'aa':
                mutstr = 'amino acid'
            elif mut_type == 'codon':
                mutstr = mut_type
            else:
                raise ValueError(f"invalid `mut_type` {mut_type}")
            ylabel = f"single {mutstr} variants with >= this many counts"
            df = df.query(f"n_{mut_type}_substitutions <= 1")
        elif variant_type == 'all':
            ylabel = 'variants with >= this many counts'
        else:
            raise ValueError(f"invalid `variant_type` {variant_type}")

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1 + 1.5 * nlibraries)
            height = heightscale * (0.6 + 1.5 * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1 + 1.5 * nsamples)
            height = heightscale * (0.6 + 1.5 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        df = (dms_variants.utils.cumul_rows_by_count(
                    df,
                    n_col='nvariants',
                    tot_col='total_variants',
                    group_cols=['library', 'sample'])
              .query('count > 0')
              )

        p = (p9.ggplot(df, p9.aes('count', 'nvariants')) +
             p9.geom_step() +
             p9.xlab('number of counts') +
             p9.ylab(ylabel) +
             p9.scale_x_log10(labels=dms_variants.utils.latex_sci_not) +
             p9.scale_y_continuous(labels=dms_variants.utils.latex_sci_not) +
             p9.theme(figure_size=(width, height))
             )

        if samples is None:
            p = p + p9.ylab('number of variants')
            if nlibraries > 1 or one_lib_facet:
                p = (p +
                     p9.facet_wrap('~ library',
                                   nrow={'h': 1, 'v': nlibraries}[orientation])
                     )
        else:
            p = p + p9.facet_grid(facet_str)

        if tot_variants_hline:
            p = p + p9.geom_hline(p9.aes(yintercept='total_variants'),
                                  linetype='dashed', color=CBPALETTE[1])

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotCumulMutCoverage(self, variant_type, mut_type, *,
                             libraries='all', samples='all', plotfile=None,
                             orientation='h', widthscale=1, heightscale=1,
                             min_support=1, max_count=None,
                             sample_rename=None, one_lib_facet=False):
        """Frac mutations seen <= some number of times (primary target only).

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <=1 `mut_type` mutation.
        max_count : None or int
            Plot cumulative fraction plot out to this number of observations.
            If `None`, a reasonable value is automatically determined.
        other_parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.mutCounts(variant_type, mut_type, samples=samples,
                            libraries=libraries, min_support=min_support,
                            sample_rename=sample_rename)
        assert 'target' not in set(df.columns)

        # add one to counts to plot fraction found < this many
        # as stat_ecdf by default does <=
        df = df.assign(count=lambda x: x['count'] + 1)

        if max_count is None:
            max_count = (df
                         .groupby(['library', 'sample'], observed=True)
                         ['count']
                         .quantile(0.6)
                         .median()
                         )

        nlibraries = len(df['library'].unique())
        nsamples = len(df['sample'].unique())

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1.6 + 1.3 * nlibraries)
            height = heightscale * (1 + 1.2 * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1.6 + 1.3 * nsamples)
            height = heightscale * (1 + 1.2 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        if width > 4:
            xlabel = f'counts among {variant_type} mutants'
        else:
            xlabel = f'counts among\n{variant_type} mutants'

        mut_desc = {'aa': 'amino-acid', 'codon': 'codon'}[mut_type]
        if height > 3:
            ylabel = f'frac {mut_desc} mutations found < this many times'
        else:
            ylabel = f'frac {mut_desc} mutations\nfound < this many times'

        p = (p9.ggplot(df, p9.aes('count', color='mutation_type')) +
             p9.stat_ecdf(geom='step', size=0.75) +
             p9.coord_cartesian(xlim=(0, max_count),
                                ylim=(0, 1)) +
             p9.scale_color_manual(
                [self._mutation_type_colors[m] for m in
                 df.mutation_type.unique().sort_values().tolist()],
                name='mutation type'
                ) +
             p9.xlab(xlabel) +
             p9.ylab(ylabel) +
             p9.theme(figure_size=(width, height),
                      legend_key=p9.element_blank(),
                      axis_text_x=p9.element_text(angle=90),
                      )
             )

        if samples is None:
            if nlibraries > 1 or one_lib_facet:
                p = (p +
                     p9.facet_wrap('~ library',
                                   nrow={'h': 1, 'v': nlibraries}[orientation])
                     )
        else:
            p = p + p9.facet_grid(facet_str)

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def numCodonMutsByType(self, variant_type, *,
                           libraries='all', samples='all', min_support=1,
                           sample_rename=None):
        """Get average nonsynonymous, synonymous, stop mutations per variant.

        These statistics are only for the primary target.

        Parameters
        ----------
        all_parameters
            Same as for :meth:`CodonVariantTable.plotNumCodonMutsByType`.

        Returns
        -------
        pandas.DataFrame
            Data frame with average mutations of each type per variant.

        """
        df, _, _ = self._getPlotData(libraries, samples, min_support,
                                     primary_target_only=True,
                                     sample_rename=sample_rename)
        assert 'target' not in set(df.columns)

        if variant_type == 'single':
            df = df.query('n_codon_substitutions <= 1')
        elif variant_type != 'all':
            raise ValueError(f"invalid variant_type {variant_type}")

        codon_mut_types = ['nonsynonymous', 'synonymous', 'stop']

        # mutations from stop to another amino-acid counted as nonsyn
        df = (df
              .assign(
                synonymous=lambda x: (x.n_codon_substitutions -
                                      x.n_aa_substitutions),
                stop=lambda x: (x.aa_substitutions.str
                                .findall(rf"[{AAS_NOSTOP}]\d+\*").apply(len)),
                nonsynonymous=lambda x: (x.n_codon_substitutions -
                                         x.synonymous - x.stop),
                )
              .melt(id_vars=['library', 'sample', 'count'],
                    value_vars=codon_mut_types,
                    var_name='mutation_type',
                    value_name='num_muts')
              .assign(
                  mutation_type=lambda x: pd.Categorical(x['mutation_type'],
                                                         codon_mut_types,
                                                         ordered=True),
                  num_muts_count=lambda x: x.num_muts * x['count']
                  )
              .groupby(['library', 'sample', 'mutation_type'],
                       observed=True)
              .aggregate({'num_muts_count': 'sum', 'count': 'sum'})
              .reset_index()
              .assign(number=lambda x: x.num_muts_count / x['count'])
              )

        return df

    def plotNumCodonMutsByType(self, variant_type, *,
                               libraries='all', samples='all', plotfile=None,
                               orientation='h', widthscale=1, heightscale=1,
                               min_support=1, ylabel=None, sample_rename=None,
                               one_lib_facet=False):
        """Plot average nonsynonymous, synonymous, stop mutations per variant.

        These statistics are only for the primary target.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <=1 codon mutation.
        ylabel : None or str
            If not `None`, specify y-axis label (otherwise it is autoset).
        other_parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        _, nlibraries, nsamples = self._getPlotData(
                                            libraries,
                                            samples,
                                            min_support,
                                            primary_target_only=True,
                                            sample_rename=sample_rename)

        df = self.numCodonMutsByType(variant_type=variant_type,
                                     libraries=libraries,
                                     samples=samples,
                                     min_support=min_support,
                                     sample_rename=sample_rename)
        assert 'target' not in set(df.columns)

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1 + 1.4 * nlibraries)
            height = heightscale * (1 + 1.3 * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1 + 1.4 * nsamples)
            height = heightscale * (1 + 1.3 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        if ylabel is None:
            if height > 3:
                ylabel = f'mutations per variant ({variant_type} mutants)'
            else:
                ylabel = f'mutations per variant\n({variant_type} mutants)'

        p = (p9.ggplot(df, p9.aes('mutation_type', 'number',
                                  fill='mutation_type', label='number')) +
             p9.geom_bar(stat='identity') +
             p9.geom_text(size=9, va='bottom', format_string='{0:.2f}') +
             p9.scale_y_continuous(name=ylabel,
                                   expand=(0.03, 0, 0.15, 0)) +
             p9.scale_fill_manual(
                [self._mutation_type_colors[m] for m in
                 df.mutation_type.unique().sort_values().tolist()]
                ) +
             p9.theme(figure_size=(width, height),
                      axis_title_x=p9.element_blank(),
                      axis_text_x=p9.element_text(angle=90),
                      legend_position='none')
             )

        if samples is None:
            if nlibraries > 1 or one_lib_facet:
                p = (p +
                     p9.facet_wrap('~ library',
                                   nrow={'h': 1, 'v': nlibraries}[orientation])
                     )
        else:
            p = p + p9.facet_grid(facet_str)

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotVariantSupportHistogram(self, *,
                                    libraries='all', plotfile=None,
                                    orientation='h', widthscale=1,
                                    heightscale=1, max_support=None,
                                    sample_rename=None, one_lib_facet=False,
                                    primary_target_only=False):
        """Plot histogram of variant call support for variants.

        Parameters
        ----------
        max_support : int or None
            Group together all variants with >= this support.
        other_parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.
        primary_target_only : bool
            Only include variants that are of the primary target.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df, nlibraries, nsamples = self._getPlotData(
                                    libraries,
                                    None,
                                    min_support=1,
                                    primary_target_only=primary_target_only,
                                    sample_rename=sample_rename)

        if orientation == 'h':
            width = widthscale * (1 + 1.4 * nlibraries)
            height = 2.3 * heightscale
            nrow = 1
        elif orientation == 'v':
            width = 2.4 * widthscale
            height = heightscale * (1 + 1.3 * nlibraries)
            nrow = nlibraries
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        df = (df
              .assign(variant_call_support=lambda x:
                      numpy.clip(x['variant_call_support'], None, max_support)
                      )
              .groupby(['library', 'variant_call_support'],
                       observed=True)
              .aggregate({'count': 'sum'})
              .reset_index()
              )

        p = (p9.ggplot(df, p9.aes('variant_call_support', 'count')) +
             p9.geom_bar(stat='identity') +
             p9.scale_x_continuous(name='supporting sequences',
                                   breaks=dms_variants.utils.integer_breaks) +
             p9.scale_y_continuous(name='number of variants',
                                   labels=dms_variants.utils.latex_sci_not) +
             p9.theme(figure_size=(width, height))
             )

        if nlibraries > 1 or one_lib_facet:
            p = p + p9.facet_wrap('~ library', nrow=nrow)

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def avgCountsPerVariant(self, *,
                            libraries='all', samples='all', min_support=1,
                            sample_rename=None, by_target=True):
        """Get average counts per variant.

        Parameters
        ----------
        libraries :  {'all', 'all_only', list}
            Include all libraries including a merge, only a merge of all
            libraries, or a list of libraries.
        samples : {'all', list}
            Include all samples or just samples in list.
        min_support : int
            Only include variants with variant call support >= this.
        sample_rename : dict or None
            Rename samples by specifying original name as key and new name
            as value.
        by_target : bool
            If `True`, also group counts by target if multiple targets.

        Returns
        -------
        pandas.DataFrame
            Average counts per variant for each library and sample, and
            possibly target.

        """
        if samples is None:
            raise ValueError('`samples` cannot be `None`')

        df, nlibraries, nsamples = self._getPlotData(
                                                libraries,
                                                samples,
                                                min_support,
                                                primary_target_only=False,
                                                sample_rename=sample_rename)

        group_cols = ['library', 'sample']
        if self.primary_target is None:
            assert 'target' not in set(df.columns)
        else:
            assert 'target' in set(df.columns)
            if by_target:
                group_cols.insert(0, 'target')

        return (df
                .groupby(group_cols, observed=True)
                .aggregate({'count': 'mean'})
                .rename(columns={'count': 'avg_counts_per_variant'})
                .reset_index()
                )

    def plotAvgCountsPerVariant(self, *,
                                libraries='all', samples='all', plotfile=None,
                                orientation='h', widthscale=1, heightscale=1,
                                min_support=1, sample_rename=None,
                                one_lib_facet=False, by_target=True):
        """Plot average counts per variant.

        Parameters
        ----------
        libraries :  {'all', 'all_only', list}
            Include all libraries including a merge, only a merge of all
            libraries, or a list of libraries.
        samples : {'all', list}
            Include all samples or just samples in list.
        by_target : bool
            If `True`, also group counts by target if multiple targets.
        other_parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.avgCountsPerVariant(libraries=libraries, samples=samples,
                                      min_support=min_support,
                                      sample_rename=sample_rename,
                                      by_target=by_target)

        nsamples = df['sample'].nunique()
        nlibraries = df['library'].nunique()
        if 'target' in set(df.columns):
            assert (self.primary_target is not None) and by_target
            ntargets = df['library'].nunique()
        else:
            ntargets = 1
            assert (self.primary_target is None) or not by_target

        if orientation == 'h':
            nrow = ntargets
            width = widthscale * nlibraries * (0.9 + 0.2 * nsamples)
            height = 2.1 * heightscale * ntargets
            facet_grid_str = 'target ~ library'
        elif orientation == 'v':
            nrow = nlibraries
            width = widthscale * (0.9 + 0.2 * nsamples) * ntargets
            height = 2.1 * nlibraries
            facet_grid_str = 'library ~ target'
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        p = (p9.ggplot(df, p9.aes('sample', 'avg_counts_per_variant')) +
             p9.geom_bar(stat='identity') +
             p9.xlab('') +
             p9.ylab('average counts per variant') +
             p9.theme(figure_size=(width, height),
                      axis_text_x=p9.element_text(angle=90)
                      )
             )

        if nlibraries > 1 or one_lib_facet:
            if ntargets > 1:
                p = p + p9.facet_grid(facet_grid_str)
            else:
                p = p + p9.facet_wrap('~ library', nrow=nrow)
        elif ntargets > 1:
            p = p + p9.facet_wrap('~ target', nrow=nrow)

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotNumMutsHistogram(self, mut_type, *,
                             libraries='all', samples='all', plotfile=None,
                             orientation='h', widthscale=1, heightscale=1,
                             min_support=1, max_muts=None, sample_rename=None,
                             one_lib_facet=False):
        """Plot histogram of num mutations per variant (primary target only).

        Parameters
        ----------
        mut_type : {'codon' or 'aa'}
            Mutation type.
        libraries :  {'all', 'all_only', list}
            Include all libraries including a merge, only a merge of all
            libraries, or a list of libraries.
        samples : {'all', None, list}
            Include all samples, a list of samples, or `None` to just count
            each barcoded variant once.
        plotfile : None or str
            Name of file to which to save plot.
        orientation : {'h', 'v'}
            Facet libraries horizontally or vertically?
        widthscale : float
            Expand width of plot by this factor.
        heightscale : float
            Expand height of plot by this factor.
        min_support : int
            Only include variants with variant call support >= this.
        max_muts : int or None
            Group together all variants with >= this many mutations.
        sample_rename : dict or None
            Rename samples by specifying original name as key and new name
            as value.
        one_lib_facet : bool
            Plot library facet title even if just one library.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df, nlibraries, nsamples = self._getPlotData(
                                            libraries,
                                            samples,
                                            min_support,
                                            primary_target_only=True,
                                            sample_rename=sample_rename)
        assert 'target' not in set(df.columns)

        if mut_type == 'aa':
            mut_col = 'n_aa_substitutions'
            xlabel = 'amino-acid mutations'
        elif mut_type == 'codon':
            mut_col = 'n_codon_substitutions'
            xlabel = 'codon mutations'
        else:
            raise ValueError(f"invalid mut_type {mut_type}")

        if orientation == 'h':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'sample ~ library'
            else:
                facet_str = 'sample ~'
            width = widthscale * (1 + 1.5 * nlibraries)
            height = heightscale * (0.6 + 1.5 * nsamples)
        elif orientation == 'v':
            if nlibraries > 1 or one_lib_facet:
                facet_str = 'library ~ sample'
            else:
                facet_str = '~ sample'
            width = widthscale * (1 + 1.5 * nsamples)
            height = heightscale * (0.6 + 1.5 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        df[mut_col] = numpy.clip(df[mut_col], None, max_muts)

        df = (df
              .groupby(['library', 'sample', mut_col],
                       observed=True)
              .aggregate({'count': 'sum'})
              .reset_index()
              )

        p = (p9.ggplot(df, p9.aes(mut_col, 'count')) +
             p9.geom_bar(stat='identity') +
             p9.scale_x_continuous(name=xlabel,
                                   breaks=dms_variants.utils.integer_breaks) +
             p9.scale_y_continuous(labels=dms_variants.utils.latex_sci_not) +
             p9.theme(figure_size=(width, height))
             )

        if samples is None:
            p = p + p9.ylab('number of variants')
            if nlibraries > 1 or one_lib_facet:
                p = (p +
                     p9.facet_wrap('~ library',
                                   nrow={'h': 1, 'v': nlibraries}[orientation])
                     )
        else:
            p = p + p9.facet_grid(facet_str)

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def writeCodonCounts(self, single_or_all, *,
                         outdir=None, include_all_libs=False):
        """Write codon counts files for all libraries and samples.

        Only writes the counts for the primary target.

        Note
        ----
        Useful if you want to analyze individual mutations using
        `dms_tools2 <https://jbloomlab.github.io/dms_tools2/>`_. File format:
        https://jbloomlab.github.io/dms_tools2/dms2_bcsubamp.html#counts-file

        Parameters
        ----------
        single_or_all : {'single', 'all'}
            If 'single', then counts just from single-codon mutants and
            wildtype, and count all wildtype codons and just mutated codons
            for single-codon mutants. If 'all', count all codons for all
            variants at all sites. Appropriate if enrichment of each mutation
            is supposed to represent its effect for 'single', and if enrichment
            of mutation is supposed to represent its average effect across
            genetic backgrounds in the library for 'all' provided mutations are
            Poisson distributed.
        outdir : None or str
            Name of directory to write counts, created if it does not exist.
            Use `None` to write to current directory.
        include_all_libs : bool
            Include data for a library (named "all-libraries") that has
            summmed data for all individual libraries if multiple libraries.

        Returns
        -------
        pandas.DataFrame
            Gives names of created files. Has columns "library", "sample",
            and "countfile". The "countfile" columns gives name of the
            created CSV file, ``<library>_<sample>_codoncounts.csv``.

        """
        def _parseCodonMut(mutstr):
            m = self._CODON_SUB_RE.match(mutstr)
            return (m.group('wt'), int(m.group('r')), m.group('mut'))

        if self.variant_count_df is None:
            raise ValueError("no samples with counts")

        if single_or_all not in {'single', 'all'}:
            raise ValueError(f"invalid `single_or_all` {single_or_all}")

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
        else:
            outdir = ''

        if include_all_libs:
            df = self.addMergedLibraries(self.variant_count_df,
                                         all_lib='all-libraries')
        else:
            df = self.variant_count_df

        if self.primary_target is not None:
            assert 'target' in set(df.columns)
            df = (df
                  .query('target == @self.primary_target')
                  .drop(columns='target')
                  )
        else:
            assert 'target' not in set(df.columns)

        countfiles = []
        liblist = []
        samplelist = []

        for lib, sample in itertools.product(
                            df['library'].unique().tolist(),
                            df['sample'].unique().tolist()
                            ):

            i_df = df.query('library == @lib & sample == @sample')
            if len(i_df) == 0:
                continue  # no data for this library and sample

            countfile = os.path.join(outdir, f"{lib}_{sample}_codoncounts.csv")
            countfiles.append(countfile)
            liblist.append(lib)
            samplelist.append(sample)

            codoncounts = {codon: [0] * len(self.sites) for codon in CODONS}

            if single_or_all == 'single':
                n_wt = (i_df
                        .query('n_codon_substitutions == 0')
                        ['count']
                        .sum()
                        )
                for isite, site in enumerate(self.sites):
                    codoncounts[self.codons[site]][isite] += n_wt
                for mut, count in (i_df
                                   .query('n_codon_substitutions == 1')
                                   [['codon_substitutions', 'count']]
                                   .itertuples(index=False, name=None)
                                   ):
                    wtcodon, r, mutcodon = _parseCodonMut(mut)
                    codoncounts[mutcodon][r - 1] += count

            elif single_or_all == 'all':
                n_wt = i_df['count'].sum()
                for isite, site in enumerate(self.sites):
                    codoncounts[self.codons[site]][isite] += n_wt
                for muts, count in (i_df
                                    .query('n_codon_substitutions > 0')
                                    [['codon_substitutions', 'count']]
                                    .itertuples(index=False, name=None)
                                    ):
                    for mut in muts.split():
                        wtcodon, r, mutcodon = _parseCodonMut(mut)
                        codoncounts[mutcodon][r - 1] += count
                        codoncounts[wtcodon][r - 1] -= count

            else:
                raise ValueError(f"invalid `single_or_all` {single_or_all}")

            counts_df = pd.DataFrame(collections.OrderedDict(
                         [('site', self.sites),
                          ('wildtype', [self.codons[r] for r in self.sites])] +
                         [(codon, codoncounts[codon]) for codon in CODONS]
                         ))
            counts_df.to_csv(countfile, index=False)

        assert all(map(os.path.isfile, countfiles))

        return pd.DataFrame({'library': liblist,
                             'sample': samplelist,
                             'countfile': countfiles})

    @staticmethod
    def classifyVariants(df,
                         *,
                         variant_class_col='variant_class',
                         max_aa=2,
                         syn_as_wt=False,
                         primary_target=None,
                         non_primary_target_class='secondary target',
                         class_as_categorical=False,
                         ):
        """Classifies codon variants in `df`.

        Parameters
        ----------
        df : pandas.DataFrame
            Must have columns named 'aa_substitutions', 'n_aa_substitutions',
            and 'n_codon_substitutions'. Such a data frame can be obtained via
            `variant_count_df` or `barcode_variant_df` attributes of a
            :class:`CodonVariantTable` or via
            :meth:`CodonVariantTable.func_scores`.
        variant_class_col : str
            Name of column added to `df` that contains variant classification.
            Overwritten if already exists.
        max_aa : int
            When classifying, group all with >= this many amino-acid mutations.
        syn_as_wt : bool
            Do not have a category of 'synonymous' and instead classify
            synonymous variants as 'wildtype'. If using this option, `df`
            does not need column 'n_codon_substitutions'.
        primary_target : None or str
            If `df` has a column named 'target', then this must specify
            primary target (e.g., :attr:`CodonVariantTable.primary_target`).
            Variants not from primary target are classified as
            `non_primary_target_class`.
        non_primary_target_class : str
            Classification used for non-primary targets.
        class_as_categorical : bool
            Return `variant_class` as a categorical variable with a
            reasonable ordering.

        Returns
        -------
        pandas.DataFrame
            Copy of `df` with column specified by `variant_class_col` as:
              - 'wildtype': no codon mutations
              - 'synonymous': only synonymous codon mutations
              - 'stop': at least one stop-codon mutation
              - '{n_aa} nonsynonymous' where `n_aa` is number of amino-acid
                mutations, or is '>{max_aa}' if more than `max_aa`.
              - potentially `non_primary_target_class`.

        Example
        -------
        >>> df = pd.DataFrame.from_records(
        ...         [('AAA', '', 0, 0),
        ...          ('AAG', '', 0, 1),
        ...          ('ATA', 'M1* G5K', 2, 3),
        ...          ('GAA', 'G5H', 1, 2),
        ...          ('CTT', 'M1C G5C', 2, 3),
        ...          ('CTT', 'M1A L3T G5C', 3, 3),
        ...          ('AAG', '', 0, 1),
        ...          ],
        ...         columns=['barcode', 'aa_substitutions',
        ...                  'n_aa_substitutions', 'n_codon_substitutions']
        ...         )
        >>> df_classify = CodonVariantTable.classifyVariants(df)
        >>> all(df_classify.columns == ['barcode', 'aa_substitutions',
        ...                             'n_aa_substitutions',
        ...                             'n_codon_substitutions',
        ...                             'variant_class'])
        True
        >>> df_classify[['barcode', 'variant_class']]
          barcode     variant_class
        0     AAA          wildtype
        1     AAG        synonymous
        2     ATA              stop
        3     GAA   1 nonsynonymous
        4     CTT  >1 nonsynonymous
        5     CTT  >1 nonsynonymous
        6     AAG        synonymous
        >>> df_syn_as_wt = CodonVariantTable.classifyVariants(df,
        ...                                                   syn_as_wt=True)
        >>> df_syn_as_wt[['barcode', 'variant_class']]
          barcode     variant_class
        0     AAA          wildtype
        1     AAG          wildtype
        2     ATA              stop
        3     GAA   1 nonsynonymous
        4     CTT  >1 nonsynonymous
        5     CTT  >1 nonsynonymous
        6     AAG          wildtype

        Now show how we need to specify how to handle when multiple targets:

        >>> df['target'] = ['secondary'] + ['primary'] * 6
        >>> (CodonVariantTable.classifyVariants(df)
        ...  [['target', 'barcode', 'variant_class']])
        Traceback (most recent call last):
            ...
        ValueError: `df` has "target" so give `primary_target`

        We need to specify how to handle targets:

        >>> (CodonVariantTable.classifyVariants(
        ...                         df,
        ...                         primary_target='primary',
        ...                         non_primary_target_class='homolog')
        ...  [['target', 'barcode', 'variant_class']])
              target barcode     variant_class
        0  secondary     AAA           homolog
        1    primary     AAG        synonymous
        2    primary     ATA              stop
        3    primary     GAA   1 nonsynonymous
        4    primary     CTT  >1 nonsynonymous
        5    primary     CTT  >1 nonsynonymous
        6    primary     AAG        synonymous

        """
        req_cols = ['aa_substitutions', 'n_aa_substitutions']
        if not syn_as_wt:
            req_cols.append('n_codon_substitutions')
        if not (set(req_cols) <= set(df.columns)):
            raise ValueError(f"`df` does not have columns {req_cols}")

        cats = ['wildtype', 'synonymous',
                *[f"{n} nonsynonymous" for n in range(1, max_aa)],
                f">{max_aa - 1} nonsynonymous", 'stop']
        if syn_as_wt:
            cats.remove('synonymous')
        if 'target' in set(df.columns):
            req_cols.append('target')
            if primary_target is None:
                raise ValueError('`df` has "target" so give `primary_target`')
            if primary_target not in set(df['target']):
                raise ValueError(f"`primary_target` {primary_target} not in "
                                 f"`df` targets:\n{set(df['target'])}")
            cats.append(non_primary_target_class)
        else:
            primary_target = None

        def _classify_func(row):
            if (primary_target is not None) and (row['target'] !=
                                                 primary_target):
                return non_primary_target_class
            elif row['n_aa_substitutions'] == 0:
                if syn_as_wt:
                    return 'wildtype'
                elif row['n_codon_substitutions'] == 0:
                    return 'wildtype'
                else:
                    return 'synonymous'
            elif '*' in row['aa_substitutions']:
                return 'stop'
            elif row['n_aa_substitutions'] < max_aa:
                return f"{row['n_aa_substitutions']} nonsynonymous"
            else:
                return f">{max_aa - 1} nonsynonymous"

        # to speed up, if variants present multiple times just classify
        # once and then merge into overall data frame.
        class_df = df[req_cols].drop_duplicates()
        class_df[variant_class_col] = class_df.apply(_classify_func, axis=1)
        if class_as_categorical:
            assert set(class_df[variant_class_col]).issubset(set(cats))
            class_df[variant_class_col] = pd.Categorical(
                                            class_df[variant_class_col],
                                            cats,
                                            ordered=True)
        return (df
                .drop(columns=variant_class_col, errors='ignore')
                .merge(class_df,
                       on=req_cols,
                       validate='many_to_one',
                       how='left',
                       )
                )

    @staticmethod
    def addMergedLibraries(df, *, all_lib='all libraries'):
        """Add data to `df` for all libraries merged.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame that includes columns named 'library' and 'barcode'.
        all_lib : str
            Name given to library that is merge of all other libraries.

        Returns
        -------
        pandas.DataFrame
            If `df` only has data for one library, just returns `df`. Otherwise
            returns copy of `df` with new library with name given by `all_lib`
            that contains data for all individual libraries with the 'barcode'
            column giving library name followed by a hyphen and barcode.

        """
        libs = df.library.unique().tolist()

        if len(libs) <= 1:
            return df

        if all_lib in libs:
            raise ValueError(f"library {all_lib} already exists")

        df = (pd.concat([df, df.assign(barcode=(lambda x: x.library.str
                                                .cat(x.barcode, sep='-')),
                                       library=all_lib
                                       )],
                        axis='index',
                        ignore_index=True,
                        sort=False)
              .assign(library=lambda x: pd.Categorical(x['library'],
                                                       libs + [all_lib],
                                                       ordered=True)
                      )
              )

        return df

    def _getPlotData(self, libraries, samples, min_support,
                     primary_target_only, *, sample_rename=None):
        """Get data to plot from library and sample filters.

        Parameters
        ----------
        primary_target_only : bool
            Only return data for the primary target.
        All other parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        tuple
            `(df, nlibraries, nsamples)` where:
                - `df`: DataFrame with data to plot.
                - `nlibraries`: number of libraries being plotted.
                - `nsamples`: number of samples being plotted.

        """
        if samples is None:
            df = (self.barcode_variant_df
                  .assign(sample='barcoded variants')
                  .assign(count=1)
                  )
        elif samples == 'all':
            if self.variant_count_df is None:
                raise ValueError('no samples have been added')
            df = self.variant_count_df
        elif isinstance(samples, list):
            all_samples = set(itertools.chain.from_iterable(
                    self.samples(lib) for lib in self.libraries))
            if not all_samples.issuperset(set(samples)):
                raise ValueError(f"invalid sample(s) in {samples}")
            if len(samples) != len(set(samples)):
                raise ValueError(f"duplicate samples in {samples}")
            df = self.variant_count_df.query('sample in @samples')
        else:
            raise ValueError(f"invalid `samples` {samples}")

        df = df.query('variant_call_support >= @min_support')

        if not len(df):
            raise ValueError(f"no samples {samples}")
        else:
            nsamples = len(df['sample'].unique())

        if libraries == 'all':
            df = self.addMergedLibraries(df)
        elif libraries == 'all_only':
            df = (self.addMergedLibraries(df)
                  .query('library == "all libraries"')
                  )
        elif isinstance(libraries, list):
            if not set(self.libraries).issuperset(set(libraries)):
                raise ValueError(f"invalid library in {libraries}")
            if len(libraries) != len(set(libraries)):
                raise ValueError(f"duplicate library in {libraries}")
            df = df.query('library in @libraries')
        else:
            raise ValueError(f"invalid `libraries` {libraries}")
        if not len(df):
            raise ValueError(f"no libraries {libraries}")
        else:
            nlibraries = len(df['library'].unique())

        if sample_rename is None:
            sample_rename_dict = _dict_missing_is_key()
        else:
            if len(sample_rename) != len(set(sample_rename.values())):
                raise ValueError('duplicates in `sample_rename`')
            sample_rename_dict = _dict_missing_is_key(sample_rename)
        df = (df
              .assign(
                library=lambda x: pd.Categorical(x['library'],
                                                 x['library'].unique(),
                                                 ordered=True),
                sample=lambda x: pd.Categorical(
                                x['sample'].map(sample_rename_dict),
                                x['sample'].map(sample_rename_dict).unique(),
                                ordered=True),
                )
              )

        if self.primary_target is not None:
            assert 'target' in set(df.columns)
            if primary_target_only:
                df = (df
                      .query('target == @self.primary_target')
                      .drop(columns='target')
                      )
        else:
            assert 'target' not in set(df.columns)

        return (df, nlibraries, nsamples)

    @classmethod
    def codonToAAMuts(self, codon_mut_str):
        """Convert string of codon mutations to amino-acid mutations.

        Parameters
        ----------
        codon_mut_str : str
            Codon mutations, delimited by a space and in 1, ... numbering.

        Returns
        -------
        str
            Amino acid mutations in 1, ... numbering.

        Example
        -------
        >>> CodonVariantTable.codonToAAMuts('ATG1GTG GGA2GGC TGA3AGA')
        'M1V *3R'

        """
        aa_muts = {}
        for mut in codon_mut_str.upper().split():
            m = self._CODON_SUB_RE.match(mut)
            if not m:
                raise ValueError(f"invalid mutation {mut} in {codon_mut_str}")
            r = int(m.group('r'))
            if r in aa_muts:
                raise ValueError(f"duplicate codon mutation for {r}")
            wt_codon = m.group('wt')
            mut_codon = m.group('mut')
            if wt_codon == mut_codon:
                raise ValueError(f"invalid mutation {mut}")
            wt_aa = CODON_TO_AA[wt_codon]
            mut_aa = CODON_TO_AA[mut_codon]
            if wt_aa != mut_aa:
                aa_muts[r] = f"{wt_aa}{r}{mut_aa}"

        return ' '.join([mut_str for r, mut_str in sorted(aa_muts.items())])

    def _sortCodonMuts(self, mut_str):
        """Sort space-delimited codon mutations and make uppercase.

        Example
        -------
        >>> geneseq = 'ATGGGATGA'
        >>> with tempfile.NamedTemporaryFile(mode='w') as f:
        ...     _ = f.write('library,barcode,substitutions,'
        ...                 'variant_call_support')
        ...     f.flush()
        ...     variants = CodonVariantTable(
        ...                 barcode_variant_file=f.name,
        ...                 geneseq=geneseq
        ...                 )
        >>> variants._sortCodonMuts('GGA2CGT ATG1GTG')
        'ATG1GTG GGA2CGT'

        """
        muts = {}
        for mut in mut_str.upper().split():
            m = self._CODON_SUB_RE.match(mut)
            if not m:
                raise ValueError(f"invalid codon mutation {mut}")
            wt_codon = m.group('wt')
            r = int(m.group('r'))
            mut_codon = m.group('mut')
            if wt_codon == mut_codon:
                raise ValueError(f"invalid codon mutation {mut}")
            if r not in self.sites:
                raise ValueError(f"invalid site in codon mutation {mut}")
            if wt_codon != self.codons[r]:
                raise ValueError(f"invalid wt in codon mutation {mut}")
            if r in muts:
                raise ValueError(f"duplicate mutation at codon {mut}")
            muts[r] = mut
        return ' '.join(mut for r, mut in sorted(muts.items()))

    def _ntToCodonMuts(self, nt_mut_str):
        """Convert string of nucleotide mutations to codon mutations.

        Parameters
        ----------
        nt_mut_str : str
            Nucleotide mutations, delimited by space in 1, ... numbering.

        Returns
        -------
        str
            Codon mutations in 1, 2, ... numbering of codon sites.

        Example
        -------
        >>> geneseq = 'ATGGGATGA'
        >>> with tempfile.NamedTemporaryFile(mode='w') as f:
        ...     _ = f.write('library,barcode,substitutions,'
        ...                 'variant_call_support')
        ...     f.flush()
        ...     variants = CodonVariantTable(
        ...                 barcode_variant_file=f.name,
        ...                 geneseq=geneseq
        ...                 )
        >>> variants._ntToCodonMuts('A1G G4C A6T')
        'ATG1GTG GGA2CGT'
        >>> variants._ntToCodonMuts('G4C A6T A1G')
        'ATG1GTG GGA2CGT'
        >>> variants._ntToCodonMuts('A1G G4C G6T')
        Traceback (most recent call last):
        ...
        ValueError: nucleotide 6 should be A not G in A1G G4C G6T

        """
        mut_codons = collections.defaultdict(set)
        for mut in nt_mut_str.upper().split():
            m = self._NT_SUB_RE.match(mut)
            if not m:
                raise ValueError(f"invalid mutation {mut}")
            wt_nt = m.group('wt')
            i = int(m.group('r'))
            mut_nt = m.group('mut')
            if wt_nt == mut_nt:
                raise ValueError(f"invalid mutation {mut}")
            if i > len(self.geneseq) or i < 1:
                raise ValueError(f"invalid nucleotide site {i}")
            if self.geneseq[i - 1] != wt_nt:
                raise ValueError(f"nucleotide {i} should be "
                                 f"{self.geneseq[i - 1]} not {wt_nt} in "
                                 f"{nt_mut_str}")
            icodon = (i - 1) // 3 + 1
            i_nt = (i - 1) % 3
            assert self.codons[icodon][i_nt] == wt_nt
            if i_nt in mut_codons[icodon]:
                raise ValueError(f"duplicate mutations {i_nt} in {icodon}")
            mut_codons[icodon].add((i_nt, mut_nt))

        codon_mut_list = []
        for r, r_muts in sorted(mut_codons.items()):
            wt_codon = self.codons[r]
            mut_codon = list(wt_codon)
            for i, mut_nt in r_muts:
                mut_codon[i] = mut_nt
            codon_mut_list.append(f"{wt_codon}{r}{''.join(mut_codon)}")

        return ' '.join(codon_mut_list)

    def subs_to_seq(self, subs, subs_type='codon'):
        """Convert substitutions to full sequence.

        Parameters
        ----------
        subs : str
            Space delimited substitutions in 1, ... numbering.
        subs_type : {'codon', 'aa'}
            Are substitutions codon or amino acid?

        Returns
        -------
        str
            Sequence created by these mutations, either codon or
            amino acid depending on value of `subs_type`.

        Example
        -------
        >>> geneseq = 'ATGGGATGA'
        >>> with tempfile.NamedTemporaryFile(mode='w') as f:
        ...     _ = f.write('library,barcode,substitutions,'
        ...                 'variant_call_support')
        ...     f.flush()
        ...     variants = CodonVariantTable(
        ...                 barcode_variant_file=f.name,
        ...                 geneseq=geneseq
        ...                 )
        >>> variants.subs_to_seq('GGA2CGT ATG1GTG')
        'GTGCGTTGA'
        >>> variants.subs_to_seq('*3W M1C', subs_type='aa')
        'CGW'

        """
        if subs_type == 'codon':
            submatcher = self._CODON_SUB_RE
            seqdict = self.codons.copy()
        elif subs_type == 'aa':
            submatcher = self._AA_SUB_RE
            seqdict = self.aas.copy()
        else:
            raise ValueError(f"invalid `subs_type` of {subs_type}")

        mutated_sites = set()
        for sub in subs.split():
            m = submatcher.match(sub)
            if not m:
                raise ValueError(f"Invalid substitution {sub}")
            r = int(m.group('r'))
            if r in mutated_sites:
                raise ValueError(f"Multiple substitutions at {r}")
            mutated_sites.add(r)
            if seqdict[r] != m.group('wt'):
                raise ValueError(f"Invalid wildtype in {sub}")
            seqdict[r] = m.group('mut')

        return ''.join(seqdict.values())

    def add_full_seqs(self,
                      df,
                      *,
                      aa_seq_col='aa_sequence',
                      codon_seq_col='codon_sequence',
                      ):
        """Add full sequences to data frame, for primary target only.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame that specifies substitutions. Might be the
            `barcode_variant_df` or `variant_count_df` attributes,
            or the result of calling :meth:`CodonVariantTable.func_scores`.
        aa_seq_col : str or None
            Name of added column with amino-acid sequences. In order to add,
            `df` must have column named 'aa_substitutions'.
        codon_seq_col : str or None
            Name of added column with codon sequences. In order to add,
            `df` must have column named 'codon_substitutions'.

        Returns
        -------
        pandas.DataFrame
            Copy of `df` with columns `aa_seq_col` and/or `codon_seq_col`,
            and only the primary target retained if there is a 'target'
            column in `df`.

        """
        if (not aa_seq_col) and (not codon_seq_col):
            raise ValueError('specify either `aa_seq_col` or `codon_seq_col`')

        df = df.copy(deep=True)

        if 'target' in set(df.columns):
            if self.primary_target is None:
                raise ValueError('`df` has "target" col but no primary target')
            else:
                df = df.query('target == @self.primary_target')

        for coltype, col in [('aa', aa_seq_col), ('codon', codon_seq_col)]:
            if col:
                if col in df.columns:
                    raise ValueError(f"`df` already has column {col} "
                                     f"specified by `{coltype}_seq_col`")
                subs_col = f"{coltype}_substitutions"
                if subs_col not in df.columns:
                    raise ValueError(f"cannot specify `{coltype}_seq_col "
                                     f"because `df` lacks {subs_col} column")
                df[col] = df[subs_col].apply(self.subs_to_seq, args=(coltype,))

        if aa_seq_col and codon_seq_col:
            if not all(df[codon_seq_col].apply(dms_variants.utils.translate) ==
                       df[aa_seq_col]):
                raise ValueError('codon seqs != translated amino-acid seqs')

        return df


def filter_by_subs_observed(df,
                            min_single_counts,
                            min_any_counts,
                            *,
                            and_vs_or='and',
                            subs_col='aa_substitutions',
                            n_subs_col='n_aa_substitutions',
                            ):
    """Filter for variants by observations substitutions in entire data frame.

    Filter data frames of the type returned by :class:`CodonVariantTable` to
    get just variants that have substitutions that are observed in some other
    number of variants.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame containing variants as rows.
    min_single_counts : int
        Only keep variants with substitutions that are observed as single-
        substitution variants in at least this many variants.
    min_any_counts : int
        Only keep variants with substitutions that are observed in at
        least this many variants with any number of substitutions.
    and_vs_or : {'and', 'or'}
        Require variants to pass the `min_single_counts` **and** the
        'min_any_counts' filters, or just require to pass one **or** the
        other.
    subs_col : 'aa_substitutions'
        Column in `df` with the substitutions as space-delimited strings.
    n_subs_col : 'n_aa_substitutions'
        Column in `df` with the number of substitutions per variant.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` that only contains the variants that pass the filters.

    Examples
    --------
    Create a data frame of variants to filter:

    >>> df = pd.DataFrame.from_records([
    ...          ('var1', '', 0),
    ...          ('var2', 'M1A', 1),
    ...          ('var3', 'M1A G2A', 2),
    ...          ('var4', 'M1A G2C', 2),
    ...          ('var5', 'G2A', 1),
    ...          ('var6', 'M1A', 1),
    ...          ('var7', 'M1C', 1),
    ...          ],
    ...          columns=['barcode', 'aa_substitutions', 'n_aa_substitutions'])

    Calling with `min_single_counts=1` and `min_any_counts=1` gets only
    variants with substitutions that are observed in single-substitution
    variants:

    >>> filter_by_subs_observed(df, 1, 1)
      barcode aa_substitutions  n_aa_substitutions
    0    var1                                    0
    1    var2              M1A                   1
    2    var3          M1A G2A                   2
    3    var5              G2A                   1
    4    var6              M1A                   1
    5    var7              M1C                   1

    We can also require substitutions to be seen multiple times as
    single variants:

    >>> filter_by_subs_observed(df, 2, 1)
      barcode aa_substitutions  n_aa_substitutions
    0    var1                                    0
    1    var2              M1A                   1
    2    var6              M1A                   1

    Or that substitutions be seen a specified number of times
    in both single- and multi-substitution contexts:

    >>> filter_by_subs_observed(df, 1, 2)
      barcode aa_substitutions  n_aa_substitutions
    0    var1                                    0
    1    var2              M1A                   1
    2    var3          M1A G2A                   2
    3    var5              G2A                   1
    4    var6              M1A                   1

    We can also make the requirement **or** rather than **and**:

    >>> filter_by_subs_observed(df, 1, 2, and_vs_or='or')
      barcode aa_substitutions  n_aa_substitutions
    0    var1                                    0
    1    var2              M1A                   1
    2    var3          M1A G2A                   2
    3    var5              G2A                   1
    4    var6              M1A                   1
    5    var7              M1C                   1

    Do not filter any variants:

    >>> filter_by_subs_observed(df, 0, 0)
      barcode aa_substitutions  n_aa_substitutions
    0    var1                                    0
    1    var2              M1A                   1
    2    var3          M1A G2A                   2
    3    var4          M1A G2C                   2
    4    var5              G2A                   1
    5    var6              M1A                   1
    6    var7              M1C                   1

    """
    df = df.copy()

    for col in [subs_col, n_subs_col]:
        if col not in df.columns:
            raise ValueError(f"`df` lacks column {col}")

    if and_vs_or not in {'and', 'or'}:
        raise ValueError(f"invalid `and_vs_or` of {and_vs_or}")

    for var_type, min_counts, df_to_count in [
            ('any', min_any_counts, df),
            ('single', min_single_counts, df.query(f"{n_subs_col} == 1")),
            ]:
        filter_col = f"_pass_{var_type}_filter"
        if filter_col in df.columns:
            raise ValueError(f"`df` cannot have column {filter_col}")
        if min_counts == 0:
            df[filter_col] = True
            continue
        subs_counts = collections.Counter(
                    itertools.chain.from_iterable(df_to_count
                                                  [subs_col]
                                                  .str
                                                  .split()
                                                  ))
        subs_valid = {s for s, n in subs_counts.items() if n >= min_counts}
        df[filter_col] = df[subs_col].map(lambda s: set(s.split()).issubset(
                                                                subs_valid))

    return (df
            .query('(_pass_any_filter == True) ' + and_vs_or +
                   '(_pass_single_filter == True)')
            .drop(columns=['_pass_any_filter', '_pass_single_filter'])
            .reset_index(drop=True)
            )


class _dict_missing_is_key(dict):
    """dict that returns key as value for missing keys.

    Note
    ----
    See here: https://stackoverflow.com/a/6229253

    """

    def __missing__(self, key):
        return key


if __name__ == '__main__':
    import doctest
    doctest.testmod()
