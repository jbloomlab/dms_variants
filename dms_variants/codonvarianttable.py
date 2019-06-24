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

import pandas as pd

import plotnine as p9

import scipy

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
    variant_count_df : pandas.DataFrame or None
        Initially `None`, but after data added with
        :class:`CodonVariantTable.addSampleCounts`, holds counts of
        each variant for each sample. Differs from `barcode_variant_df`
        in that the former just holds barcode-variant definitions,
        whereas `variant_count_df` has counts for each sample.

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
                              drop_all_libs=True):
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

        Returns
        -------
        :class:`CodonVariantTable`

        """
        df = pd.read_csv(variant_count_df_file)

        req_cols = ['barcode', 'library', 'variant_call_support',
                    'codon_substitutions', 'sample', 'count']
        if not (set(req_cols) < set(df.columns)):
            raise ValueError(f"{variant_count_df_file} lacks required "
                             f"columns {req_cols}")
        else:
            df = df[req_cols]

        if drop_all_libs:
            dropcol = "all libraries"
            if dropcol in df['library'].unique():
                df = df.query('library != @dropcol')

        with tempfile.NamedTemporaryFile(mode='w') as f:
            (df
             .drop(columns=['sample', 'count'])
             .rename(columns={'codon_substitutions': 'substitutions'})
             .drop_duplicates()
             .to_csv(f, index=False)
             )
            f.flush()
            cvt = cls(barcode_variant_file=f.name,
                      geneseq=geneseq,
                      substitutions_are_codon=True)

        for sample in df['sample'].unique():
            for lib in cvt.libraries:
                idf = df.query('sample == @sample & library == @lib')
                if len(idf):
                    cvt.addSampleCounts(lib,
                                        sample,
                                        idf[['barcode', 'count']]
                                        )

        return cvt

    def __init__(self, *, barcode_variant_file, geneseq,
                 substitutions_are_codon=False, extra_cols=None):
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

        df = pd.read_csv(barcode_variant_file)
        required_cols = ['library', 'barcode',
                         'substitutions', 'variant_call_support']
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
                                                       .fillna('')
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
                .sort_values(['library', 'barcode'])
                .reset_index(drop=True)
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

        Parameters
        ----------
        library : str
            Valid `library` for the :class:`CodonVariantTable`.
        sample : str
            Sample name, must **not** already be in
            :class:`CodonVariantTable.samples` for `library`.
        barcodecounts : pandas.DataFrame
            Counts for each variant by barcode. Must have columns "barcode"
            and "count". The "barcode" column must contain all barcodes in
            :class:`CodonVariantTable.valid_barcodes` for `library`.

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
        self.variant_count_df = (
                self.variant_count_df
                .assign(library=lambda x: pd.Categorical(x['library'],
                                                         self.libraries,
                                                         ordered=True),
                        sample=lambda x: pd.Categorical(x['sample'],
                                                        unique_samples,
                                                        ordered=True),
                        )
                .sort_values(['library', 'sample', 'count'],
                             ascending=[True, True, False])
                .reset_index(drop=True)
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

    def func_scores(self, preselection, *,
                    pseudocount=0.5, by="barcode",
                    combine_libs=False, syn_as_wt=False, logbase=2,
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

        Parameters
        ----------
        preselection : str or dict
            Pre-selection sample. If the same for all post-selection then
            provide the name as str. If it differs among post-selection
            samples, then provide a dict keyed by each post-selection
            sample with the pre-selection sample being the value.
        pseudocount : float
            Pseudocount added to each count.
        by : str
            Compute effects for each "barcode", set of "aa_substitutions", or
            set of "codon_substitutions". In the latter two cases, all barcodes
            with each set of substitutions are combined (see `combine_libs`).
            If you use "aa_substitutions" then it may be more sensible to set
            `syn_as_wt` to `True`.
        syn_as_wt : bool
            In formula for functional scores, consider variants with only
            synonymous mutations when determining wildtype counts? If `False`,
            only variants with **no** mutations of any type contribute.
        combine_libs : bool
            If `by` is "aa_substitutions" or "codon_substitutions", do we
            combine across libraries as well as barcodes?
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
              - "library": the library ("all libraries" if `combine_libs`)
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

        if combine_libs and (len(self.libraries) > 1):
            if any(s not in self.samples(lib) for lib in self.libraries
                   for s in samples):
                raise ValueError('cannot use `combine_libs`, not every '
                                 f"library has every sample: {samples}")
            df = (self.addMergedLibraries(df)
                  .query('library == "all libraries"')
                  )

        # get wildtype counts for each sample and library
        if syn_as_wt:
            wt_col = 'n_aa_substitutions'
        else:
            wt_col = 'n_codon_substitutions'
        wt_counts = (
                df
                .assign(count=lambda x: (x['count'] *
                                         (0 == x[wt_col]).astype('int')))
                .groupby(['library', 'sample'])
                .aggregate({'count': 'sum'})
                .reset_index()
                )
        if (wt_counts['count'] <= 0).any() and not permit_zero_wt:
            raise ValueError(f"no wildtype counts:\n{wt_counts}")

        # sum counts in groups specified by `by`
        group_cols = ['codon_substitutions', 'n_codon_substitutions',
                      'aa_substitutions', 'n_aa_substitutions']
        if by in {'aa_substitutions', 'codon_substitutions'}:
            group_cols = group_cols[group_cols.index(by) + 1:]
        elif by != 'barcode':
            raise ValueError(f"invalid `by` of {by}")
        df = (df
              .groupby(['library', 'sample', by] + group_cols)
              .aggregate({'count': 'sum'})
              .reset_index()
              )

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
                    func_score=lambda x: scipy.log(
                                         ((x.post_count + x.pseudocount) /
                                          (x.post_count_wt + x.pseudocount)) /
                                         ((x.pre_count + x.pseudocount) /
                                          (x.pre_count_wt + x.pseudocount))
                                         ) / scipy.log(logbase),
                    func_score_var=lambda x: (
                                1 / (x.post_count + x.pseudocount) +
                                1 / (x.post_count_wt + x.pseudocount) +
                                1 / (x.pre_count + x.pseudocount) +
                                1 / (x.pre_count_wt + x.pseudocount)
                                ) / (scipy.log(logbase)**2)
                    )
                # set column order in data frame
                [['library', 'pre_sample', 'post_sample', by,
                  'func_score', 'func_score_var', 'pre_count',
                  'post_count', 'pre_count_wt', 'post_count_wt',
                  'pseudocount'] + group_cols]
                )

        return df_func_scores

    def n_variants_df(self, *, libraries='all', samples='all',
                      min_support=1, variant_type='all',
                      mut_type=None):
        """Get number of variants per library / sample.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <= 1 `mut_type` mutation.
        mut_type : {'aa', 'codon', None}
            If `variant_type` is 'single', indicate what type of single
            mutants we are filtering for.
        All other args
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        pandas.DataFrame

        """
        df, nlibraries, nsamples = self._getPlotData(libraries,
                                                     samples,
                                                     min_support)

        if variant_type == 'single':
            if mut_type in {'aa', 'codon'}:
                df = df.query(f"n_{mut_type}_substitutions <= 1")
            else:
                raise ValueError('`mut_type` must be "aa" or "single"')
        elif variant_type != 'all':
            raise ValueError(f"invalid `variant_type` {variant_type}")

        return (df
                .groupby(['library', 'sample'])
                .aggregate({'count': 'sum'})
                .reset_index()
                )

    def mutCounts(self, variant_type, mut_type, *,
                  libraries='all', samples='all', min_support=1):
        """Get counts of each individual mutation.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include just single mutants, or all mutants?
        Other args
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        pandas.DataFrame
            Tidy data frame with columns named "library", "sample",
            "mutation", "count", "mutation_type", and "site".

        """
        df, nlibraries, nsamples = self._getPlotData(libraries,
                                                     samples,
                                                     min_support)

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
                wt_aa = m.group('wt')
                mut_aa = m.group('mut')
            else:
                m = self._CODON_SUB_RE.match(mut_str)
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
                      sample=lambda x: pd.Categorical(x['sample'],
                                                      samplelist,
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
                       min_support=1):
        """Heatmap of mutation counts or frequencies.

        Parameters
        ----------
        count_or_frequency : {'count', 'frequency'}
            Plot mutation counts or frequencies?
        All other args
            Same as for :meth:`CodonVariantTable.plotCumulMutCoverage`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.mutCounts(variant_type, mut_type, samples=samples,
                            libraries=libraries, min_support=min_support)

        n_variants = (self.n_variants_df(libraries=libraries,
                                         samples=samples,
                                         min_support=min_support,
                                         variant_type=variant_type,
                                         mut_type=mut_type)
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

        if count_or_frequency not in {'count', 'frequency'}:
            raise ValueError(f"invalid count_or_frequency "
                             f"{count_or_frequency}")

        nlibraries = len(df['library'].unique())
        nsamples = len(df['sample'].unique())

        if orientation == 'h':
            facet_str = 'sample ~ library'
            width = widthscale * (1.6 + 3.5 * nlibraries)
            height = heightscale * (0.8 + height_per * nsamples)
        elif orientation == 'v':
            facet_str = 'library ~ sample'
            width = widthscale * (1.6 + 3.5 * nsamples)
            height = heightscale * (0.8 + height_per * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        p = (p9.ggplot(df, p9.aes('site', 'mut_char',
                                  fill=count_or_frequency)) +
             p9.geom_tile() +
             p9.facet_grid(facet_str) +
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

        if plotfile:
            p.save(plotfile, height=height, width=width,
                   verbose=False, limitsize=False)

        return p

    def plotMutFreqs(self, variant_type, mut_type, *,
                     libraries='all', samples='all', plotfile=None,
                     orientation='h', widthscale=1, heightscale=1,
                     min_support=1):
        """Mutation frequency along length of gene.

        Parameters
        ----------
        All parameters
            Same as for :meth:`CodonVariantTable.plotCumulMutCoverage`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.mutCounts(variant_type, mut_type, samples=samples,
                            libraries=libraries, min_support=min_support)

        n_variants = (self.n_variants_df(libraries=libraries,
                                         samples=samples,
                                         min_support=min_support,
                                         variant_type=variant_type,
                                         mut_type=mut_type)
                      .rename(columns={'count': 'nseqs'})
                      )

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
            facet_str = 'sample ~ library'
            width = widthscale * (1.6 + 1.8 * nlibraries)
            height = heightscale * (0.8 + 1 * nsamples)
        elif orientation == 'v':
            facet_str = 'library ~ sample'
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
             p9.facet_grid(facet_str) +
             p9.theme(figure_size=(width, height),
                      legend_key=p9.element_blank(),
                      legend_text=p9.element_text(size=11)
                      )
             )

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotCumulVariantCounts(self, *, variant_type='all',
                               libraries='all', samples='all', plotfile=None,
                               orientation='h', widthscale=1, heightscale=1,
                               min_support=1, mut_type='aa',
                               tot_variants_hline=True):
        """Plot number variants with >= that each number of counts.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <=1 `mut_type` mutation.
        tot_variants_hline : bool
            Include dotted horizontal line indicating total number of variants.
        Other parameters
            Same as for :meth:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df, nlibraries, nsamples = self._getPlotData(libraries,
                                                     samples,
                                                     min_support)

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
            facet_str = 'sample ~ library'
            width = widthscale * (1 + 1.5 * nlibraries)
            height = heightscale * (0.6 + 1.5 * nsamples)
        elif orientation == 'v':
            facet_str = 'library ~ sample'
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
             p9.facet_grid(facet_str) +
             p9.xlab('number of counts') +
             p9.ylab(ylabel) +
             p9.scale_x_log10(labels=dms_variants.utils.latex_sci_not) +
             p9.scale_y_continuous(labels=dms_variants.utils.latex_sci_not) +
             p9.theme(figure_size=(width, height))
             )

        if tot_variants_hline:
            p = p + p9.geom_hline(p9.aes(yintercept='total_variants'),
                                  linetype='dashed', color=CBPALETTE[1])

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotCumulMutCoverage(self, variant_type, mut_type, *,
                             libraries='all', samples='all', plotfile=None,
                             orientation='h', widthscale=1, heightscale=1,
                             min_support=1, max_count=None):
        """Fraction of mutations seen <= some number of times.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <=1 `mut_type` mutation.
        max_count : None or int
            Plot cumulative fraction plot out to this number of observations.
            If `None`, a reasonable value is automatically determined.
        Other parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df = self.mutCounts(variant_type, mut_type, samples=samples,
                            libraries=libraries, min_support=min_support)

        # add one to counts to plot fraction found < this many
        # as stat_ecdf by default does <=
        df = df.assign(count=lambda x: x['count'] + 1)

        if max_count is None:
            max_count = df['count'].quantile(0.75)

        nlibraries = len(df['library'].unique())
        nsamples = len(df['sample'].unique())

        if orientation == 'h':
            facet_str = 'sample ~ library'
            width = widthscale * (1.6 + 1.3 * nlibraries)
            height = heightscale * (1 + 1.2 * nsamples)
        elif orientation == 'v':
            facet_str = 'library ~ sample'
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
             p9.coord_cartesian(xlim=(0, max_count)) +
             p9.scale_color_manual(
                [self._mutation_type_colors[m] for m in
                 df.mutation_type.unique().sort_values().tolist()],
                name='mutation type'
                ) +
             p9.xlab(xlabel) +
             p9.ylab(ylabel) +
             p9.facet_grid(facet_str) +
             p9.theme(figure_size=(width, height),
                      legend_key=p9.element_blank(),
                      legend_text=p9.element_text(size=11),
                      axis_text_x=p9.element_text(angle=90),
                      )
             )

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotNumCodonMutsByType(self, variant_type, *,
                               libraries='all', samples='all', plotfile=None,
                               orientation='h', widthscale=1, heightscale=1,
                               min_support=1):
        """Nonsynonymous, synonymous, stop mutations per variant.

        Parameters
        ----------
        variant_type : {'single', 'all'}
            Include all variants or just those with <=1 `mut_type` mutation.
        Other parameters
            Same as for :class:`CodonVariantTable.plotNumMutsHistogram`.

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df, nlibraries, nsamples = self._getPlotData(libraries,
                                                     samples,
                                                     min_support)

        if variant_type == 'single':
            df = df.query('n_codon_substitutions <= 1')
        elif variant_type != 'all':
            raise ValueError(f"invalid variant_type {variant_type}")

        if orientation == 'h':
            facet_str = 'sample ~ library'
            width = widthscale * (1 + 1.4 * nlibraries)
            height = heightscale * (1 + 1.3 * nsamples)
        elif orientation == 'v':
            facet_str = 'library ~ sample'
            width = widthscale * (1 + 1.4 * nsamples)
            height = heightscale * (1 + 1.3 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        if height > 3:
            ylabel = f'mutations per variant ({variant_type} mutants)'
        else:
            ylabel = f'mutations per variant\n({variant_type} mutants)'

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
              .groupby(['library', 'sample', 'mutation_type'])
              .aggregate({'num_muts_count': 'sum', 'count': 'sum'})
              .reset_index()
              .assign(number=lambda x: x.num_muts_count / x['count'])
              )

        p = (p9.ggplot(df, p9.aes('mutation_type', 'number',
                                  fill='mutation_type', label='number')) +
             p9.geom_bar(stat='identity') +
             p9.geom_text(size=8, va='bottom', format_string='{0:.3f}') +
             p9.facet_grid(facet_str) +
             p9.scale_y_continuous(name=ylabel,
                                   expand=(0.03, 0, 0.12, 0)) +
             p9.scale_fill_manual(
                [self._mutation_type_colors[m] for m in
                 df.mutation_type.unique().sort_values().tolist()]
                ) +
             p9.theme(figure_size=(width, height),
                      axis_title_x=p9.element_blank(),
                      axis_text_x=p9.element_text(angle=90, size=11),
                      legend_position='none')
             )

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def plotNumMutsHistogram(self, mut_type, *,
                             libraries='all', samples='all', plotfile=None,
                             orientation='h', widthscale=1, heightscale=1,
                             min_support=1, max_muts=None):
        """Plot histograms of number of mutations per variant.

        Parameters
        ----------
        mut_type : {'codon' or 'aa'}
            Mutation type.
        libraries :  {'all', 'all_only', list}
            Include all libraries including a marge, only a merge of all
            libraries, or a list of libraries.
        samples : {'all', None, list}
            Include all samples, a list of simple, or `None` to just count
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

        Returns
        -------
        plotnine.ggplot.ggplot

        """
        df, nlibraries, nsamples = self._getPlotData(libraries,
                                                     samples,
                                                     min_support)

        if mut_type == 'aa':
            mut_col = 'n_aa_substitutions'
            xlabel = 'amino-acid mutations'
        elif mut_type == 'codon':
            mut_col = 'n_codon_substitutions'
            xlabel = 'codon mutations'
        else:
            raise ValueError(f"invalid mut_type {mut_type}")

        if orientation == 'h':
            facet_str = 'sample ~ library'
            width = widthscale * (1 + 1.5 * nlibraries)
            height = heightscale * (0.6 + 1.5 * nsamples)
        elif orientation == 'v':
            facet_str = 'library ~ sample'
            width = widthscale * (1 + 1.5 * nsamples)
            height = heightscale * (0.6 + 1.5 * nlibraries)
        else:
            raise ValueError(f"invalid `orientation` {orientation}")

        df[mut_col] = scipy.clip(df[mut_col], None, max_muts)

        df = (df
              .groupby(['library', 'sample', mut_col])
              .aggregate({'count': 'sum'})
              .reset_index()
              )

        p = (p9.ggplot(df, p9.aes(mut_col, 'count')) +
             p9.geom_bar(stat='identity') +
             p9.facet_grid(facet_str) +
             p9.xlab(xlabel) +
             p9.scale_y_continuous(labels=dms_variants.utils.latex_sci_not) +
             p9.theme(figure_size=(width, height))
             )

        if plotfile:
            p.save(plotfile, height=height, width=width, verbose=False)

        return p

    def writeCodonCounts(self, single_or_all, *,
                         outdir=None, include_all_libs=False):
        """Write codon counts files for all libraries and samples.

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
                         max_aa=2):
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

        Returns
        -------
        pandas.DataFrame
            Copy of `df` with column specified by `variant_class_col` as:
              - 'wildtype': no codon mutations
              - 'synonymous': only synonymous codon mutations
              - 'stop': at least one stop-codon mutation
              - '{n_aa} nonsynonymous' where `n_aa` is number of amino-acid
                mutations, or is '>{max_aa}' if more than `max_aa`.

        Example
        -------
        >>> df = pd.DataFrame.from_records(
        ...         [('AAA', '', 0, 0),
        ...          ('AAG', '', 0, 1),
        ...          ('ATA', 'M1* G5K', 2, 3),
        ...          ('GAA', 'G5H', 1, 2),
        ...          ('CTT', 'M1C G5C', 2, 3),
        ...          ('CTT', 'M1A L3T G5C', 3, 3),
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
          barcode      variant_class
        0     AAA           wildtype
        1     AAG         synonymous
        2     ATA               stop
        3     GAA    1 nonsynonymous
        4     CTT  >=2 nonsynonymous
        5     CTT  >=2 nonsynonymous

        """
        req_cols = ['aa_substitutions', 'n_aa_substitutions',
                    'n_codon_substitutions']
        if not (set(req_cols) <= set(df.columns)):
            raise ValueError(f"`df` does not have columns {req_cols}")

        def _classify_func(row):
            if row['n_codon_substitutions'] == 0:
                return 'wildtype'
            elif row['n_aa_substitutions'] == 0:
                return 'synonymous'
            elif '*' in row['aa_substitutions']:
                return 'stop'
            elif row['n_aa_substitutions'] < max_aa:
                return f"{row['n_aa_substitutions']} nonsynonymous"
            else:
                return f">={max_aa} nonsynonymous"

        df = df.copy(deep=True)
        df[variant_class_col] = df.apply(_classify_func, axis=1)
        return df

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

    def _getPlotData(self, libraries, samples, min_support):
        """Get data to plot from library and sample filters.

        Parameters
        ----------
        All parameters
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
        ValueError: nucleotide 6 should be A not G

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
                                 f"{self.geneseq[i - 1]} not {wt_nt}")
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
        """Add full sequences to data frame.

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
            Copy of `df` with columns `aa_seq_col` and/or `codon_seq_col`.

        """
        if (not aa_seq_col) and (not codon_seq_col):
            raise ValueError('specify either `aa_seq_col` or `codon_seq_col`')

        df = df.copy(deep=True)
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
