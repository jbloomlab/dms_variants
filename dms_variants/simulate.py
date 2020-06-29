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


def rand_seq(seqlen, *, exclude=None, nseqs=1):
    """Random nucleotide sequence(s).

    Parameters
    ----------
    seqlen : int
        Length of sequence(s).
    exclude : None or list
        Do not generate any of the sequences in this list.
    nseqs : int
        If >1, return a list of this many sequences.

    Returns
    -------
    str or list
        A random sequence or list of them depending on value of `nseqs`.
        If multiple sequences, they are all unique.

    Seed `random.seed` for reproducible output.

    Example
    -------
    >>> random.seed(1)
    >>> rand_seq(4)
    'CAGA'

    >>> random.seed(1)
    >>> rand_seq(5, nseqs=3)
    ['CAGAT', 'TTTCA', 'TATTA']

    >>> random.seed(1)
    >>> rand_seq(5, nseqs=3, exclude=['TTTCA'])
    ['CAGAT', 'TATTA', 'TGCAG']

    """
    if exclude:
        exclude = {s.upper() for s in exclude}
        if len(exclude) >= len(NTS)**seqlen:
            raise ValueError('excluding too many sequences, none valid.')
    else:
        exclude = set()
    seqs = []
    while len(seqs) < nseqs:
        seq = ''.join(random.choice(NTS) for _ in range(seqlen))
        if seq not in exclude:
            seqs.append(seq)
            exclude.add(seq)
    if len(seqs) == 1:
        return seqs[0]
    else:
        return seqs


def mutate_seq(seq, nmuts):
    """Mutate a nucleotide sequence.

    Parameters
    ----------
    seq : str
        Sequence to mutate.
    nmuts: int
        Number of mutations to make to sequence.

    Returns
    -------
    str
        Mutagenized sequence, all upper-case.

    Seed `random.seed` for reproducible output.

    Example
    -------
    >>> random.seed(1)
    >>> mutate_seq('ATGCAA', 2)
    'AAGCGA'

    """
    if nmuts > len(seq):
        raise ValueError(f"Can not make {nmuts} mutations to sequence of "
                         f"length {len(seq)}.")
    seq = list(seq.upper())
    sites = random.sample(range(len(seq)), nmuts)
    for r in sites:
        seq[r] = random.choice([nt for nt in NTS if nt != seq[r]])
    return ''.join(seq)


def codon_muts(codonseq, nmuts, nvariants):
    """Get random codon mutations to sequence.

    Parameters
    ----------
    codonseq : str
        A valid codon sequence
    nmuts : int
        Number of mutations to make per variant.
    nvariants : int
        Number of variants to generate. There is no guarantee that the
        variants are unique.

    Returns
    -------
    list
        List of mutations in each of the `nvariants` mutated variants
        in the form 'ATG1TAC GGC3GGG' where the number refers to the
        **codon** position.

    Seed `random.seed` for reproducible output.

    Example
    -------
    >>> codonseq = 'ATGGAAGGGCATGGA'
    >>> random.seed(1)
    >>> codon_muts(codonseq, nmuts=2, nvariants=3)
    ['ATG1CAC GAA2ACT', 'CAT4CTT GGA5GGG', 'GAA2ACG CAT4GAA']

    """
    if not re.fullmatch(f"(?:{'|'.join(CODONS)})+", codonseq):
        raise ValueError(f"`codonseq` not valid codon sequence:\n{codonseq}")
    assert len(codonseq) % 3 == 0
    codons = {i + 1: codonseq[3 * i: 3 * i + 3]
              for i in range(len(codonseq) // 3)}
    if nmuts > len(codons):
        raise ValueError('`nmuts` greater than number of codons in `codonseq`')
    mutlist = []
    for _ in range(nvariants):
        sitemuts = []
        for site, wt in sorted(random.sample(codons.items(), nmuts)):
            mut = random.choice([c for c in CODONS if c != wt])
            sitemuts.append(f"{wt}{site}{mut}")
        mutlist.append(' '.join(sitemuts))
    return mutlist


def simulate_CodonVariantTable(*, geneseq, bclen, library_specs,
                               seed=1, variant_call_support=0.5):
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
    variant_call_support : int > 0
        Support is floor of 1 plus draw from exponential distribution
        with this mean.

    Returns
    -------
    :class:`dms_variants.codonvarianttable.CodonVariantTable`

    """
    if seed is not None:
        scipy.random.seed(seed)
        random.seed(seed)

    if len(library_specs) < 1:
        raise ValueError('empty `library_specs`')

    if variant_call_support <= 0:
        raise ValueError('`variant_call_support` must be int > 0')

    if len(geneseq) % 3 != 0:
        raise ValueError('length of `geneseq` not multiple of 3')
    genelength = len(geneseq) // 3

    barcode_variant_dict = collections.defaultdict(list)
    for lib, specs_dict in sorted(library_specs.items()):

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

            support = math.floor(1 + random.expovariate(variant_call_support))

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
                         secondary_target_phenotypes=None,
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
        Takes a space-delimited string of amino-acid substitutions and returns
        a number giving enrichment of variant relative to wildtype. For
        instance, :meth:`SigmoidPhenotypeSimulator.observedEnrichment`.
        If there are multiple targets in `variants`, this function is only
        applied to the primary target. See `secondary_target_phenotypes` for
        the secondary targets.
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
    secondary_target_phenotypes : None or dict
        If there are secondary targets in `variants`, this should be keyed
        by each secondary target name and give the associated phenotypes
        as values.
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
        random.seed(seed)

    if pre_sample_name in post_samples:
        raise ValueError('`pre_sample_name` is in `post_samples`')

    # -----------------------------------------
    # internal function
    def _add_variant_errors(codon_substitutions):
        """Add errors to variant according to `variant_error_rate`."""
        if random.random() < variant_error_rate:
            muts = codon_substitutions.split()
            if len(muts) == 0 or random.random() < 0.5:
                # add mutation
                mutatedsites = {int(re.match(f"^({'|'.join(CODONS)})"
                                             r'(?P<r>\d+)'
                                             f"({'|'.join(CODONS)})$",
                                             mut).group('r'))
                                for mut in muts}
                unmutatedsites = [r for r in variants.sites
                                  if r not in mutatedsites]
                if not unmutatedsites:
                    raise RuntimeError("variant already has all mutations")
                errorsite = random.choice(unmutatedsites)
                wtcodon = variants.codons[errorsite]
                mutcodon = random.choice([c for c in CODONS if c != wtcodon])
                muts.append(f'{wtcodon}{errorsite}{mutcodon}')
                return ' '.join(muts)
            else:
                # remove mutation
                _ = muts.pop(random.randint(0, len(muts) - 1))
                return ' '.join(muts)
        else:
            return codon_substitutions
    # -----------------------------------------

    if variants.primary_target is None:
        primary_df = variants.barcode_variant_df.assign(target=None)
        secondary_df = None
    else:
        primary_df = (variants.barcode_variant_df
                      .query('target == @variants.primary_target'))
        secondary_df = (variants.barcode_variant_df
                        .query('target != @variants.primary_target'))

    primary_df = (
        primary_df
        [['target', 'library', 'barcode', 'codon_substitutions']]
        .assign(
            codon_substitutions=(lambda x: x['codon_substitutions']
                                 .map(_add_variant_errors)),
            aa_substitutions=(lambda x: x['codon_substitutions']
                              .map(dms_variants.codonvarianttable
                                   .CodonVariantTable.codonToAAMuts)),
            phenotype=lambda x: x['aa_substitutions'].map(phenotype_func)
            )
        [['library', 'barcode', 'phenotype']]
        )

    if secondary_df is not None:
        if secondary_target_phenotypes is None:
            secondary_target_phenotypes = {}
        if not set(secondary_target_phenotypes).issuperset(
                    secondary_df['target']):
            raise ValueError('`secondary_target_phenotypes` lacks some '
                             'secondary targets')
        secondary_df = (
                secondary_df
                .assign(phenotype=lambda x: x['target'].map(
                                        secondary_target_phenotypes))
                [['library', 'barcode', 'phenotype']]
                )
        barcode_variant_df = primary_df.append(secondary_df)
    else:
        barcode_variant_df = primary_df

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
                noise=lambda x: scipy.clip(
                                    [random.gauss(1, sample_dict['noise'])
                                     for _ in range(len(x))],
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
    r"""Simulate phenotypes under sigmoid global epistasis model.

    Note
    ----
    Mutational effects on latent phenotype are simulated to follow
    compound normal distribution; latent phenotype maps to observed
    enrichment via sigmoid, and the observed phenotype is the
    :math:`\log_2` of the enrichment. This distinction between latent and
    observed phenotype parallels the "global epistasis" models of
    `Otwinoski et al <https://doi.org/10.1073/pnas.1804015115>`_ and
    `Sailer and Harms <http://www.genetics.org/content/205/3/1079>`_.

    Specifically, let :math:`p_{\rm{latent}}` be the latent phenotype
    of a variant, let :math:`p_{\rm{observed}}` be the observed phenotype,
    and let :math:`E_{\rm{observed}}` be the observed enrichment. Also
    let :math:`p_{\rm{latent, wt}}` be the latent phenotype of the
    wildtype. Then:

    .. math::

       E_{\rm{observed}}
       &=& \frac{\left(1 + \exp\left(-p_{\rm{latent, wt}}\right)\right)
                 \left(1 - E_{\rm{min}}\right)}
                {1 + \exp\left(-p_{\rm{latent}}\right)} +
                E_{\rm{min}} \\
       p_{\rm{observed}} &=& \log_2 E_{\rm{observed}}

    where :math:`0 \le E_{\rm{min}} < 1` is the minimum possible observed
    enrichment. Typically :math:`E_{\rm{min}} > 0` in real experiments
    as pseudocounts are used so estimates of enrichments can't be zero.

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
    min_observed_enrichment : float
        Minimum possible observed enrichment.

    Attributes
    ----------
    wt_latent : float
        Wildtype latent phenotype.
    muteffects : dict
        Effect on latent phenotype of each amino-acid mutation.

    """

    def __init__(self, geneseq, *, seed=1, wt_latent=5,
                 norm_weights=((0.4, -0.7, 1.5), (0.6, -7, 3.5)),
                 stop_effect=-15, min_observed_enrichment=0.001):
        """See main class docstring for how to initialize."""
        self.wt_latent = wt_latent

        if not (0 <= min_observed_enrichment < 1):
            raise ValueError('not 0 <= `min_observed_enrichment` < 1')
        self.min_observed_enrichment = min_observed_enrichment

        # simulate muteffects from compound normal distribution
        self.muteffects = {}
        if seed is not None:
            random.seed(seed)
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
                        i = scipy.argmin(cumweights < random.random())
                        # draw mutational effect from chosen Gaussian
                        muteffect = random.gauss(means[i], sds[i])
                    self.muteffects[f"{wt_aa}{icodon + 1}{mut_aa}"] = muteffect

    def latentPhenotype(self, subs):
        """Latent phenotype of a variant.

        Parameters
        ----------
        subs : str
            Space-delimited list of amino-acid substitutions.

        Returns
        -------
        float
            Latent phenotype of variant.

        """
        return self.wt_latent + sum(self.muteffects[m] for m in subs.split())

    def observedPhenotype(self, subs):
        """Observed phenotype of a variant.

        Parameters
        ----------
        subs : str
            Space-delimited list of amino-acid substitutions.

        Returns
        -------
        float
            Observed phenotype of variant.

        """
        return self.latentToObserved(self.latentPhenotype(subs), 'phenotype')

    def observedEnrichment(self, subs):
        """Observed enrichment of a variant.

        Parameters
        ----------
        subs : str
            Space-delimited list of amino-acid substitutions.

        Returns
        -------
        float
            Observed enrichment relative to wildtype.

        """
        return self.latentToObserved(self.latentPhenotype(subs), 'enrichment')

    def latentToObserved(self, latent, value):
        r"""Observed enrichment or observed phenotype from latent phenotype.

        Parameters
        ----------
        latent : float or numpy.ndarray
            The latent phenotype.
        value : {'enrichment', 'phenotype'}
            Get the observed enrichment or the observed phenotype?

        Returns
        -------
        float or numpy.ndarray
            The observed enrichment or phenotype.

        """
        enrichment = ((1 + scipy.exp(-self.wt_latent)) *
                      (1 - self.min_observed_enrichment) /
                      (1 + scipy.exp(-latent)) +
                      self.min_observed_enrichment)
        if value == 'enrichment':
            return enrichment
        elif value == 'phenotype':
            return scipy.log(enrichment) / scipy.log(2)
        else:
            raise ValueError(f"invalid `value` of {value}")

    def plotLatentVsObserved(self, value, *, latent_min=-15,
                             latent_max=10, npoints=200,
                             wt_vline=True):
        """Plot observed enrichment/phenotype as function of latent phenotype.

        Parameters
        ----------
        value : {'enrichment', 'phenotype'}
            Do we plot observed enrichment or observed phenotype?
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
            Plot of observed enrichment or phenotype as function of latent
            phenotype.

        """
        latent = scipy.linspace(latent_min, latent_max, npoints)
        observed = self.latentToObserved(latent, value)

        p = (p9.ggplot(pd.DataFrame({'latent': latent, 'observed': observed}),
                       p9.aes('latent', 'observed')
                       ) +
             p9.geom_line() +
             p9.theme(figure_size=(3.5, 2.5)) +
             p9.xlab('latent phenotype') +
             p9.ylab(f"observed {value}")
             )

        if wt_vline:
            p = p + p9.geom_vline(xintercept=self.wt_latent,
                                  color=CBPALETTE[1],
                                  linetype='dashed')

        return p

    def plotMutsHistogram(self, value, *,
                          mutant_order=1, bins=30, wt_vline=True):
        """Plot distribution of phenotype for all mutants of a given order.

        Parameters
        ----------
        value : {'latentPhenotype', 'observedPhenotype', 'observedEnrichment'}
            What value to plot.
        mutant_order : int
            Plot mutations of this order. Currently only works for 1
            (single mutants).
        bins : int
            Number of bins in histogram.
        wt_vline : bool
            Draw a vertical line at the wildtype value.

        Returns
        -------
        plotnine.ggplot.ggplot
            Histogram of phenotype for all mutants.

        """
        if mutant_order != 1:
            raise ValueError('only implemented for `mutant_order` of 1')

        if value not in {'latentPhenotype', 'observedPhenotype',
                         'observedEnrichment'}:
            raise ValueError(f"invalid `value` of {value}")
        func = getattr(self, value)

        xlist = [func(m) for m in self.muteffects.keys()]

        p = (p9.ggplot(pd.DataFrame({value: xlist}),
                       p9.aes(value)) +
             p9.geom_histogram(bins=bins) +
             p9.theme(figure_size=(3.5, 2.5)) +
             p9.ylab(f"number of {mutant_order}-mutants")
             )

        if wt_vline:
            p = p + p9.geom_vline(
                        xintercept=func(''),
                        color=CBPALETTE[1],
                        linetype='dashed')

        return p


class MultiLatentSigmoidPhenotypeSimulator:
    r"""Simulate phenotype that derives from several latent phenotypes.

    Note
    -----
    The observed phenotype is the sum of several sigmoid-transformed latent
    phenotypes. Specifically, let there be :math:`k = 1, \ldots, K` latent
    phenotypes, each of which is transformed into an observed phenotype
    :math:`p_{\rm{observed}}^k as described in the docs for
    :class:`SigmoidPhenotypeSimulator`. The overall observed phenotype is:

    .. math::

       p_{\rm{observed}} = \sum_k p_{\rm{observed}}^k

    and the observed enrichment is:

    .. math::

       E_{\rm{observed}} = 2^{p_{\rm{observed}}}.

    Parameters
    -----------
    sigmoid_phenotype_simulators : list
        A list of :class:`SigmoidPhenotypeSimulator` objects, each of which
        transforms one of the latent phenotypes to a component of the
        observed phenotype.

    Attributes
    ----------
    n_latent_phenotypes : int
        The number of latent phenotypes.

    """

    def __init__(self, sigmoid_phenotype_simulators):
        """See main class docstring."""
        self.n_latent_phenotypes = len(sigmoid_phenotype_simulators)
        if self.n_latent_phenotypes < 1:
            raise ValueError('`sigmoid_phenotype_simulators` cannot be empty')
        if not all(isinstance(m, SigmoidPhenotypeSimulator) for m in
                   sigmoid_phenotype_simulators):
            raise ValueError('entries in `sigmoid_phenotype_simulators` not '
                             'all of SigmoidPhenotypeSimulator type')
        self._sigmoid_phenotype_simulators = sigmoid_phenotype_simulators
        self._all_subs = list(self._sigmoid_phenotype_simulators[0]
                              .muteffects.keys())
        for m in self._sigmoid_phenotype_simulators[1:]:
            if set(self._all_subs) != set(m.muteffects.keys()):
                raise ValueError('entries in `sigmoid_phenotype_simulators` '
                                 'do not all represent same gene sequence')

    def observedEnrichment(self, subs):
        """Observed enrichment of a variant.

        Parameters
        ----------
        subs : str
            Space-delimited list of amino-acid substitutions.

        Returns
        -------
        float
            Observed enrichment relative to wildtype.

        """
        return 2**self.observedPhenotype(subs)

    def observedPhenotype(self, subs):
        """Observed phenotype of a variant.

        Parameters
        ----------
        subs : str
            Space-delimited list of amino-acid substitutions.

        Returns
        -------
        float
            Observed phenotype of a variant.

        """
        return sum(m.observedPhenotype(subs) for m in
                   self._sigmoid_phenotype_simulators)

    def latentPhenotype(self, subs, k):
        """Latent phenotype :math:`k` of a variant.

        Parameters
        ----------
        subs : str
            Space-delimited list of amino-acid substitutions.
        k : int
            Latent phenotype to get (1 <= `k` <=
            :attr:`MultiLatentSigmoidPhenotypeSimulator.n_latent_phenotypes`).

        Returns
        -------
        float
            Latent phenotype `k`.

        """
        if isinstance(k, int) and 1 <= k <= self.n_latent_phenotypes:
            return (self._sigmoid_phenotype_simulators[k - 1]
                    .observedPhenotype(subs))
        else:
            raise ValueError(f"invalid `k` of {k}")

    def plotMutsHistogram(self,
                          value, *,
                          k=None,
                          mutant_order=1,
                          bins=30,
                          wt_vline=True,
                          ):
        """Plot distribution of phenotype for all mutants of a given order.

        Parameters
        ----------
        value : {'latentPhenotype', 'observedPhenotype', 'observedEnrichment'}
            What value to plot.
        k : int or None
            If value is `latentPhenotype, which phenotype (1 <= `k` <=
            :attr:`MultiLatentSigmoidPhenotypeSimulator.n_latent_phenotypes`)
            to plot.
        mutant_order : int
            Plot mutations of this order. Currently only works for 1
            (single mutants).
        bins : int
            Number of bins in histogram.
        wt_vline : bool
            Draw a vertical line at the wildtype value.

        Returns
        -------
        plotnine.ggplot.ggplot
            Histogram of phenotype for all mutants.

        """
        if mutant_order != 1:
            raise ValueError('only implemented for `mutant_order` of 1')

        if value == 'latentPhenotype':
            if isinstance(k, int) and 1 <= k <= self.n_latent_phenotypes:
                kwargs = {'k': k}
                xlabel = f"latentPhenotype {k}"
            else:
                raise ValueError(f"invalid `k` of {k}")
        else:
            kwargs = {}
            xlabel = value

        if value not in {'latentPhenotype', 'observedPhenotype',
                         'observedEnrichment'}:
            raise ValueError(f"invalid `value` of {value}")
        func = getattr(self, value)

        xlist = [func(m, **kwargs) for m in self._all_subs]

        p = (p9.ggplot(pd.DataFrame({value: xlist}),
                       p9.aes(value)) +
             p9.geom_histogram(bins=bins) +
             p9.theme(figure_size=(3.5, 2.5)) +
             p9.ylab(f"number of {mutant_order}-mutants") +
             p9.xlab(xlabel)
             )

        if wt_vline:
            p = p + p9.geom_vline(
                        xintercept=func('', **kwargs),
                        color=CBPALETTE[1],
                        linetype='dashed')

        return p


if __name__ == '__main__':
    import doctest
    doctest.testmod()
