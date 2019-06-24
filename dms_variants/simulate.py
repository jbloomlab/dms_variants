"""
========
simulate
========

Simulate data.

"""

import collections
import itertools
import math
import random
import re
import tempfile

import pandas as pd

import plotnine as p9

import scipy

import dms_variants.codonvarianttable
from dms_variants.constants import (AAS_WITHSTOP,
                                    CBPALETTE,
                                    CODONS,
                                    CODON_TO_AA,
                                    NTS,
                                    )


def simulate_CodonVariantTable(*, geneseq, bclen, library_specs,
                               seed=1, variant_call_support=1):
    """Simulate :class:`dms_variants.codonvarianttable.CodonVariantTable`.

    Note
    ----
    Only simulates the variants, not counts for samples. To simulate counts,
    use :func:`simulateSampleCounts`.

    Parameters
    -----------
    geneseq : str
        Sequence of wildtype protein-coding gene.
    bclen : int
        Length of the barcodes; must enable complexity at least
        10-fold greater than max number of variants.
    library_specs : dict
        Specifications for each simulated library. Keys are 'avgmuts'
        and 'nvariants'. Mutations per variant are Poisson distributed.
    seed : int or None
        Random number seed or `None` to set no seed.
    variant_call_support : int or 2-tuple
        If integer, all supports are set to this. If 2-tuple, support is
        drawn as random integer uniformly from this range (inclusive).

    Returns
    -------
    :class:`dms_variants.codonvarianttable.CodonVariantTable`

    """
    if seed is not None:
        scipy.random.seed(seed)
        random.seed(seed)

    if len(library_specs) < 1:
        raise ValueError('empty `library_specs`')

    if isinstance(variant_call_support, int):
        variant_call_support = tuple([variant_call_support] * 2)

    if len(geneseq) % 3 != 0:
        raise ValueError('length of `geneseq` not multiple of 3')
    genelength = len(geneseq) // 3

    barcode_variant_dict = collections.defaultdict(list)
    for lib, specs_dict in library_specs.items():

        nvariants = specs_dict['nvariants']
        avgmuts = specs_dict['avgmuts']
        if 10 * nvariants > (len(NTS))**bclen:  # safety factor 10
            raise ValueError('barcode too short for nvariants')
        existing_barcodes = set()

        for _ivariant in range(nvariants):

            barcode = ''.join(random.choices(NTS, k=bclen))
            while barcode in existing_barcodes:
                barcode = ''.join(random.choices(NTS, k=bclen))
            existing_barcodes.add(barcode)

            support = random.randint(*variant_call_support)

            # get mutations
            substitutions = []
            nmuts = scipy.random.poisson(avgmuts)
            for icodon in random.sample(range(1, genelength + 1), nmuts):
                wtcodon = geneseq[3 * (icodon - 1): 3 * icodon]
                mutcodon = random.choice([c for c in CODONS if c != wtcodon])
                for i_nt, (wt_nt, mut_nt) in enumerate(zip(wtcodon, mutcodon)):
                    if wt_nt != mut_nt:
                        igene = 3 * (icodon - 1) + i_nt + 1
                        substitutions.append(f"{wt_nt}{igene}{mut_nt}")
            substitutions = ' '.join(substitutions)

            barcode_variant_dict['barcode'].append(barcode)
            barcode_variant_dict['substitutions'].append(substitutions)
            barcode_variant_dict['library'].append(lib)
            barcode_variant_dict['variant_call_support'].append(support)

    barcode_variants = pd.DataFrame(barcode_variant_dict)

    with tempfile.NamedTemporaryFile(mode='w') as f:
        barcode_variants.to_csv(f, index=False)
        f.flush()
        cvt = dms_variants.codonvarianttable.CodonVariantTable(
                        barcode_variant_file=f.name,
                        geneseq=geneseq)

    return cvt


def simulateSampleCounts(*,
                         variants,
                         phenotype_func,
                         variant_error_rate,
                         pre_sample,
                         post_samples,
                         pre_sample_name='pre-selection',
                         seed=1):
    """Simulate pre- and post-selection variant counts.

    Note
    ----
    Add to :class:`dms_variants.codonvarianttable.CodonVariantTable` using
    :meth:`dms_variants.codonvarianttable.CodonVariantTable.addSampleCounts`

    Parameters
    ----------
    variants : class:`dms_variants.codonvarianttable.CodonVariantTable`
        Holds variants used in simulation.
    phenotype_func : function
        Takes row from `variants.barcode_variant_df` and returns phenotype
        as number giving enrichment of variant relative to wildtype. For
        instance, :meth:`SigmoidPhenotypeSimulator.observedPhenotype`.
    variant_error_rate : float
        Rate at which variants mis-called. Provide the probability that a
        variant has a spuriously called (or missing) codon mutation; each
        variant then has a random codon mutation added or removed with this
        probability before being passed to `phenotype_func`.
    pre_sample : pandas.DataFrame or dict
        Counts of each variant pre-selection. To specify counts, provide
        data frame with columns "library", "barcode", and "count". To
        simulate, provide dict with keys "total_count" and "uniformity",
        and for each library, we simulate pre-selection counts as a draw
        of "total_count" counts from a multinomial parameterized by pre-
        selection frequencies drawn from a Dirichlet distribution with
        concentration parameter of "uniformity" (5 is reasonable value).
    post_samples : dict
        Keyed by name of each sample with value another dict keyed by
        'total_count', 'noise', and 'bottleneck'. Counts drawn from
        multinomial parameterized by pre-selection frerquencies after
        passing through bottleneck of indicated size, and adding noise
        by mutiplying phenotype by a random variable with mean 1 and
        standard deviation specified by 'noise' (0 is no noise).
    pre_sample_name : str
        Name used for the pre-selection sample.
    seed : None or int
        If not `None`, random number seed.

    Returns
    -------
    pandas.DataFrame
        Data frame with the following columns:
            - "library"
            - "barcode"
            - "sample"
            - "count"

    """
    if seed is not None:
        scipy.random.seed(seed)

    if pre_sample_name in post_samples:
        raise ValueError('`pre_sample_name` is in `post_samples`')

    # -----------------------------------------
    # internal function
    def _add_variant_errors(codon_substitutions):
        """Add errors to variant according to `variant_error_rate`."""
        if scipy.random.random() < variant_error_rate:
            muts = codon_substitutions.split()
            if len(muts) == 0 or scipy.random.random() < 0.5:
                # add mutation
                mutatedsites = set(map(int,
                                       [re.match(f"^({'|'.join(CODONS)})"
                                                 r'(?P<r>\d+)'
                                                 f"({'|'.join(CODONS)})$",
                                                 mut).group('r')
                                        for mut in muts]))
                unmutatedsites = [r for r in variants.sites
                                  if r not in mutatedsites]
                if not unmutatedsites:
                    raise RuntimeError("variant already has all mutations")
                errorsite = scipy.random.choice(unmutatedsites)
                wtcodon = variants.codons[errorsite]
                mutcodon = scipy.random.choice([c for c in CODONS
                                                if c != wtcodon])
                muts.append(f'{wtcodon}{errorsite}{mutcodon}')
                return ' '.join(muts)
            else:
                # remove mutation
                muts = muts.pop(scipy.random.randint(0, len(muts)))
                return muts
        else:
            return codon_substitutions
    # -----------------------------------------

    barcode_variant_df = (
        variants.barcode_variant_df
        [['library', 'barcode', 'codon_substitutions']]
        .assign(
            codon_substitutions=(lambda x: x.codon_substitutions
                                 .apply(_add_variant_errors)),
            aa_substitutions=(lambda x: x.codon_substitutions
                              .apply(dms_variants.codonvarianttable
                                     .CodonVariantTable.codonToAAMuts)),
            phenotype=lambda x: x.apply(phenotype_func, axis=1)
            )
        [['library', 'barcode', 'phenotype']]
        )

    libraries = variants.libraries

    if isinstance(pre_sample, pd.DataFrame):
        # pre-sample counts specified
        req_cols = ['library', 'barcode', 'count']
        if not set(req_cols).issubset(set(pre_sample.columns)):
            raise ValueError(f"pre_sample lacks cols {req_cols}:"
                             f"\n{pre_sample}")
        cols = ['library', 'barcode']
        if any(pre_sample[cols].sort_values(cols).reset_index() !=
               barcode_variant_df[cols].sort_values(cols).reset_index()):
            raise ValueError("pre_sample DataFrame lacks required "
                             "library and barcode columns")
        barcode_variant_df = (
                barcode_variant_df
                .merge(pre_sample[req_cols], on=['library', 'barcode'])
                .rename(columns={'count': pre_sample_name})
                )
        # "true" pre-selection freqs are just input counts
        nperlib = (barcode_variant_df
                   .groupby('library')
                   .rename('total_count')
                   .reset_index()
                   )
        barcode_variant_df = (
                barcode_variant_df
                .merge(nperlib, on='library')
                .assign(pre_freqs=lambda x: x[pre_sample_name] / x.total_count)
                .drop(columns='total_count')
                )

    elif isinstance(pre_sample, dict):
        pre_req_keys = {'uniformity', 'total_count'}
        if set(pre_sample.keys()) != pre_req_keys:
            raise ValueError(f"pre_sample lacks required keys {pre_req_keys}")

        pre_df_list = []
        for lib in libraries:  # noqa: B007
            df = (
                barcode_variant_df
                .query('library == @lib')
                .assign(
                    pre_freq=lambda x: scipy.random.dirichlet(
                             pre_sample['uniformity'] *
                             scipy.ones(len(x))),
                    count=lambda x: scipy.random.multinomial(
                            pre_sample['total_count'], x.pre_freq),
                    sample=pre_sample_name
                    )
                )
            pre_df_list.append(df)
        barcode_variant_df = pd.concat(pre_df_list)

    else:
        raise ValueError("pre_sample not DataFrame / dict: "
                         f"{pre_sample}")

    cols = ['library', 'barcode', 'sample', 'count',
            'pre_freq', 'phenotype']
    assert set(barcode_variant_df.columns) == set(cols), (
            f"cols = {set(cols)}\nbarcode_variant_df.columns = "
            f"{set(barcode_variant_df.columns)}")
    for col in cols:
        if col in post_samples:
            raise ValueError(f"post_samples can't have key {col}; "
                             "choose another sample name")

    df_list = [barcode_variant_df[cols[: 4]]]

    def _bottleneck_freqs(pre_freq, bottleneck):
        if bottleneck is None:
            return pre_freq
        else:
            return scipy.random.multinomial(bottleneck, pre_freq) / bottleneck

    post_req_keys = {'bottleneck', 'noise', 'total_count'}
    for lib, (sample, sample_dict) in itertools.product(  # noqa: B007
            libraries, sorted(post_samples.items())):

        if set(sample_dict.keys()) != post_req_keys:
            raise ValueError(f"post_samples {sample} lacks {post_req_keys}")

        lib_df = (
            barcode_variant_df.query('library == @lib')
            .assign(
                sample=sample,
                # simulated pre-selection freqs after bottleneck
                bottleneck_freq=(lambda x:
                                 _bottleneck_freqs(x.pre_freq,
                                                   sample_dict['bottleneck'])),
                # post-selection freqs with noise
                noise=scipy.clip(scipy.random.normal(1, sample_dict['noise']),
                                 0, None),
                post_freq_nonorm=lambda x: (x.bottleneck_freq *
                                            x.phenotype * x.noise),
                post_freq=lambda x: (x.post_freq_nonorm /
                                     x.post_freq_nonorm.sum()),
                # post-selection counts simulated from frequencies
                count=(lambda x:
                       scipy.random.multinomial(sample_dict['total_count'],
                                                x.post_freq))
                )
            .rename(columns={'post_counts': sample})
            [['library', 'barcode', 'sample', 'count']]
            )

        df_list.append(lib_df)

    return pd.concat(df_list)


class SigmoidPhenotypeSimulator:
    """Simulate phenotypes under sigmoid global epistasis model.

    Note
    ----
    Mutational effects on latent phenotype are simulated to follow
    compound normal distribution; latent phenotype maps to observed
    phenotype via sigmoid. This distinction between latent and
    observed phenotype parallel the "global epistasis" models of
    `Otwinoski et al <https://doi.org/10.1073/pnas.1804015115>`_ and
    `Sailer and Harms <http://www.genetics.org/content/205/3/1079>`_.

    The exact sigmoid used is defined in
    :meth:`SigmoidPhenotypeSimulator.latentToObserved`.

    Parameters
    ----------
    geneseq : str
        Codon sequence of wild-type gene.
    seed : int or None
        Random number seed.
    wt_latent : float
        Latent phenotype of wildtype.
    norm_weights : list or tuple of tuples
        Specify compound normal distribution of mutational effects on
        latent phenotype as `(weight, mean, sd)` for each Gaussian.
    stop_effect : float
        Effect of stop codon at any position.

    Attributes
    ----------
    wt_latent : float
        Wildtype latent phenotype.
    muteffects : dict
        Effect on latent phenotype of each amino-acid mutation.

    """

    def __init__(self, geneseq, *, seed=1, wt_latent=4,
                 norm_weights=((0.4, -0.5, 1), (0.6, -5, 2.5)),
                 stop_effect=-10):
        """See main class docstring for how to initialize."""
        self.wt_latent = wt_latent

        # simulate muteffects from compound normal distribution
        self.muteffects = {}
        if seed is not None:
            scipy.random.seed(seed)
        weights, means, sds = zip(*norm_weights)
        cumweights = scipy.cumsum(weights)
        for icodon in range(len(geneseq) // 3):
            wt_aa = CODON_TO_AA[geneseq[3 * icodon: 3 * icodon + 3]]
            for mut_aa in AAS_WITHSTOP:
                if mut_aa != wt_aa:
                    if mut_aa == '*':
                        muteffect = stop_effect
                    else:
                        # choose Gaussian from compound normal
                        i = scipy.argmin(cumweights < scipy.random.rand())
                        # draw mutational effect from chosen Gaussian
                        muteffect = scipy.random.normal(means[i], sds[i])
                    self.muteffects[f"{wt_aa}{icodon + 1}{mut_aa}"] = muteffect

    def latentPhenotype(self, v):
        """Latent phenotype of a variant.

        Parameters
        ----------
        v : dict or row of pandas.DataFrame
            Must have key 'aa_substitutions' that gives space-delimited
            amino-acid mutations.

        Returns
        -------
        float
            Latent phenotype of variant.

        """
        return self.wt_latent + sum(self.muteffects[m] for m in
                                    v['aa_substitutions'].split())

    def observedPhenotype(self, v):
        """Observed phenotype of a variant.

        Parameters
        ----------
        v : dict or row of pandas.DataFrame
            Must have key 'aa_substitutions' that gives space-delimited
            amino-acid mutations.

        Returns
        -------
        float
            Observed phenotype of variant.

        """
        return self.latentToObserved(self.latentPhenotype(v))

    @staticmethod
    def latentToObserved(latent):
        r"""Observed phenotype from latent phenotype.

        Note
        ----
        The observed phenotype :math:`p_{obs}` is calculated from the
        latent phenotype :math:`p_{latent}` as:

        .. math::

            p_{obs} = \frac{1}{1 + e^{-p_{latent}}}

        Parameters
        ----------
        latent : float
            Latent phenotype.

        Returns
        -------
        float
            Observed phenotype.

        """
        return 1 / (1 + math.exp(-latent - 3))

    def plotLatentVsObservedPhenotype(self, *, latent_min=-15,
                                      latent_max=5, npoints=200,
                                      wt_vline=True):
        """Plot observed phenotype as function of latent phenotype.

        Parameters
        ----------
        latent_min : float
            Smallest value of latent phenotype on plot.
        latent_max : float
            Largest value of latent phenotype on plot.
        npoints : int
            Plot a line fit to this many points.
        wt_vline : bool
            Draw a vertical line at the wildtype latent phenotype.

        Returns
        -------
        plotnine.ggplot.ggplot
            Plot of observed phenotype as function of latent phenotype.

        """
        latent = scipy.linspace(latent_min, latent_max, npoints)

        p = (p9.ggplot(pd.DataFrame(
                            {'latent': latent,
                             'observed': map(latent, self.latentToObserved)}),
                       p9.aes('latent', 'observed')
                       ) +
             p9.geom_line() +
             p9.theme(figure_size=(3.5, 2.5)) +
             p9.xlab('latent phenotype') +
             p9.ylab('observed phenotype')
             )

        if wt_vline:
            p = p + p9.geom_vline(xintercept=self.wt_latent,
                                  color=CBPALETTE[1],
                                  linetype='dashed')

        return p

    def plotMutsHistogram(self, latent_or_observed, *,
                          mutant_order=1, bins=30, wt_vline=True):
        """Plot distribution of phenotype for all mutants.

        Parameters
        ----------
        latent_or_observed : {'latent', 'observed'}
            Which type of phenotype to plot.
        mutant_order : int
            Plot mutations of this order. Currently only works for 1
            (single mutants).
        bins : int
            Number of bins in histogram.
        wt_vline : bool
            Draw a vertical line at the wildtype latent phenotype.

        Returns
        -------
        plotnine.ggplot.ggplot
            Histogram of phenotype for all mutants.

        """
        if mutant_order != 1:
            raise ValueError('only implemented for `mutant_order` of 1')

        if latent_or_observed == 'latent':
            phenoFunc = self.latentPhenotype
        elif latent_or_observed == 'observed':
            phenoFunc = self.observedPhenotype
        else:
            raise ValueError('invalid value of `latent_or_observed`')

        phenotypes = [phenoFunc({'aa_substitutions': m}) for m in
                      self.muteffects.keys()]

        p = (p9.ggplot(pd.DataFrame({'phenotype': phenotypes}),
                       p9.aes('phenotype')) +
             p9.geom_histogram(bins=bins) +
             p9.theme(figure_size=(3.5, 2.5)) +
             p9.ylab(f"number of {mutant_order}-mutants") +
             p9.xlab(f"{latent_or_observed} phenotype")
             )

        if wt_vline:
            p = p + p9.geom_vline(
                        xintercept=phenoFunc({'aa_substitutions': ''}),
                        color=CBPALETTE[1],
                        linetype='dashed')

        return p


if __name__ == '__main__':
    import doctest
    doctest.testmod()
