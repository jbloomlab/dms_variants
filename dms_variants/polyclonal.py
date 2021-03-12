"""
==========
polyclonal
==========

Defines :class:`Polyclonal` objects for handling antibody mixtures.

"""


import collections
import re

import numpy

import pandas as pd

import dms_variants.binarymap
import dms_variants.constants


class Polyclonal:
    r"""Represent polyclonal antibody mixtures targeting multiple epitopes.

    Note
    ----
    At each of several concentrations :math:`c` of an antibody mixture, we
    measure :math:`p_v\left(c\right)`, the probability that variant :math:`v`
    that is **not** bound (or neutralized) that concentration. We assume
    antibodies bind independently to one of :math:`E` epitopes, such that the
    probability :math:`U_e\left(v, c\right)` that variant :math:`v` is unbound
    at concentration :math:`c` is related to the probability that epitope
    :math:`e` is unbound by

    .. math::
       :label: p_v

       p_v\left(c\right) = \prod_{e=1}^E U_e\left(v, c\right).

    We furthermore assume that :math:`U_e\left(v, c\right)` is related to the
    total binding activity :math:`\phi_e\left(v\right)` of antibodies targeting
    epitope :math:`e` on variant :math:`v` by

    .. math::
       :label: U_e

       U_e\left(v,c\right)=\frac{1}{1+c\exp\left(-\phi_e\left(v\right)\right)}

    where smaller (more negative) values of :math:`\phi_e\left(v\right)`
    correspond to higher overall binding activity against epitope :math:`e`
    variant :math:`v`.

    We define :math:`\phi_e\left(v\right)` in terms of the underlying
    quantities of biological interest as

    .. math::
       :label: phi_

       \phi_e\left(v\right) = -a_{\rm{wt}, e} +
                              \sum_{m=1}^M \beta_{m,e} b\left(v\right)_m,

    where :math:`a_{\rm{wt}, e}` is the activity of the serum against
    epitope :math:`e` for the "wildtype" (unmutated) protein (larger values
    indicate higher activity against this epitope), :math:`\beta_{m,e}`
    is the extent to which mutation :math:`m` (where :math:`1 \le m \le M`)
    escapes binding from antibodies targeting epitope :math:`e` (larger
    values indicate more escape by this mutation), and
    :math:`b\left(v\right)_m` is 1 if variant :math:`v` has mutation :math:`m`
    and 0 otherwise.

    Parameters
    ----------
    activity_wt_df : pandas.DataFrame
        Should have columns named 'epitope' and 'activity', giving the names
        of the epitopes and the activity against epitope in the wildtype
        protein, :math:`a_{\rm{wt}, e}`.
    mut_escape_df : pandas.DataFrame
        Should have columns named 'mutation', 'epitope', and 'escape' that
        give the :math:`\beta_{m,e}` values (in the 'escape' column).
    alphabet : array-like
        Allowed characters in mutation strings.

    Attributes
    ----------
    epitopes : tuple
        Names of all epitopes, in order provided in `activity_wt_df`.
    mutations : tuple
        All mutations, sorted by site and then in the order of the alphabet
        provided in `alphabet`.
    alphabet : tuple
        Allowed characters in mutation strings.
    sites : tuple
        List of all sites.
    wts : dict
        Keyed by site, value is wildtype at that site.

    Example
    --------
    A simple example with two epitopes (`e1` and `e2`) and a small
    number of mutations:

    >>> activity_wt_df = pd.DataFrame({'epitope':  ['e1', 'e2'],
    ...                                'activity': [ 2.0,  1.0]})
    >>> mut_escape_df = pd.DataFrame({
    ...      'mutation': ['M1C', 'M1A', 'M1A', 'M1C', 'A2K', 'A2K'],
    ...      'epitope':  [ 'e1',  'e2',  'e1',  'e2',  'e1',  'e2'],
    ...      'escape':   [  2.0,   0.0,   3.0,  0.0,   0.0,   2.5]})
    >>> polyclonal = Polyclonal(activity_wt_df=activity_wt_df,
    ...                         mut_escape_df=mut_escape_df)
    >>> polyclonal.epitopes
    ('e1', 'e2')
    >>> polyclonal.mutations
    ('M1A', 'M1C', 'A2K')
    >>> polyclonal.sites
    (1, 2)
    >>> polyclonal.wts
    {1: 'M', 2: 'A'}
    >>> polyclonal.activity_wt_df
      epitope  activity
    0      e1       2.0
    1      e2       1.0
    >>> polyclonal.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      A      M1A     3.0
    1      e1     1        M      C      M1C     2.0
    2      e1     2        A      K      A2K     0.0
    3      e2     1        M      A      M1A     0.0
    4      e2     1        M      C      M1C     0.0
    5      e2     2        A      K      A2K     2.5

    Note that we can **not** initialize a :class:`Polyclonal` object if we are
    missing escape estimates for any mutations for any epitopes:

    >>> Polyclonal(activity_wt_df=activity_wt_df,
    ...            mut_escape_df=mut_escape_df.head(n=5))
    Traceback (most recent call last):
      ...
    ValueError: not all expected mutations for e2

    Now make a data frame with some variants:

    >>> variants_df = pd.DataFrame.from_records(
    ...         [('AA', 'A2K'),
    ...          ('AC', 'M1A A2K'),
    ...          ('AG', 'M1A'),
    ...          ('AT', ''),
    ...          ('CA', 'A2K')],
    ...         columns=['barcode', 'aa_substitutions'])

    Get the escape probabilities:

    >>> polyclonal.prob_escape(variants_df=variants_df,
    ...                        concentrations=[1, 2, 4]).round(3)
       barcode aa_substitutions  concentration  prob_escape
    0       AA              A2K            1.0        0.097
    1       AC          M1A A2K            1.0        0.598
    2       AG              M1A            1.0        0.197
    3       AT                             1.0        0.032
    4       CA              A2K            1.0        0.097
    5       AA              A2K            2.0        0.044
    6       AC          M1A A2K            2.0        0.398
    7       AG              M1A            2.0        0.090
    8       AT                             2.0        0.010
    9       CA              A2K            2.0        0.044
    10      AA              A2K            4.0        0.017
    11      AC          M1A A2K            4.0        0.214
    12      AG              M1A            4.0        0.034
    13      AT                             4.0        0.003
    14      CA              A2K            4.0        0.017

    """

    def __init__(self,
                 *,
                 activity_wt_df,
                 mut_escape_df,
                 alphabet=dms_variants.constants.AAS_NOSTOP,
                 ):
        """See main class docstring."""
        if len(set(alphabet)) != len(alphabet):
            raise ValueError('duplicate letters in `alphabet`')
        self.alphabet = tuple(alphabet)
        chars = []
        for char in self.alphabet:
            if char.isalpha():
                chars.append(char)
            elif char == '*':
                chars.append(r'\*')
            else:
                raise ValueError(f"invalid alphabet character: {char}")
        chars = '|'.join(chars)
        self._mutation_regex = re.compile(rf"(?P<wt>{chars})"
                                          rf"(?P<site>\d+)"
                                          rf"(?P<mut>{chars})")

        if pd.isnull(activity_wt_df['epitope']).any():
            raise ValueError('epitope name cannot be null')
        self.epitopes = tuple(activity_wt_df['epitope'].unique())
        if len(self.epitopes) != len(activity_wt_df):
            raise ValueError('duplicate epitopes in `activity_wt_df`:\n' +
                             str(activity_wt_df))
        self._activity_wt = (activity_wt_df
                             .set_index('epitope')
                             ['activity']
                             .astype(float)
                             .to_dict()
                             )

        # get sites, wts, mutations
        self.wts = {}
        mutations = collections.defaultdict(list)
        for mutation in mut_escape_df['mutation'].unique():
            wt, site, mut = self._parse_mutation(mutation)
            if site not in self.wts:
                self.wts[site] = wt
            elif self.wts[site] != wt:
                raise ValueError(f"inconsistent wildtype for site {site}")
            mutations[site].append(mutation)
        self.sites = tuple(sorted(self.wts.keys()))
        self.wts = dict(sorted(self.wts.items()))
        assert set(mutations.keys()) == set(self.sites) == set(self.wts)
        char_order = {c: i for i, c in enumerate(self.alphabet)}
        self.mutations = tuple(mut for site in self.sites for mut in
                               sorted(mutations[site],
                                      key=lambda m: char_order[m[-1]]))

        # get mutation escape values
        if set(mut_escape_df['epitope']) != set(self.epitopes):
            raise ValueError('`mut_escape_df` does not have same epitopes as '
                             '`activity_wt_df`')
        self._mut_escape = {}
        for epitope, df in mut_escape_df.groupby('epitope'):
            if set(df['mutation']) != set(self.mutations):
                raise ValueError(f"not all expected mutations for {epitope}")
            self._mut_escape[epitope] = (df
                                         .set_index('mutation')
                                         ['escape']
                                         .astype(float)
                                         .to_dict()
                                         )
        assert set(self.epitopes) == set(self._activity_wt)
        assert set(self.epitopes) == set(self._mut_escape)

        # below are set to non-null values in `_set_binarymap` when
        # specific variants provided
        self._binarymap = None
        self._beta = None  # M by E matrix of betas
        self._a = None  # length E vector of activities

    @property
    def activity_wt_df(self):
        r"""pandas.DataFrame: activities :math:`a_{\rm{wt,e}}` for epitopes."""
        return pd.DataFrame({'epitope': self.epitopes,
                             'activity': [self._activity_wt[e]
                                          for e in self.epitopes],
                             })

    @property
    def mut_escape_df(self):
        r"""pandas.DataFrame: escape :math:`\beta_{m,e}` for each mutation."""
        return (pd.concat([pd.DataFrame({'mutation': self.mutations,
                                         'escape': [self._mut_escape[e][m]
                                                    for m in self.mutations],
                                         })
                           .assign(epitope=e)
                           for e in self.epitopes],
                          ignore_index=True)
                .assign(
                    site=lambda x: x['mutation'].map(
                                        lambda m: self._parse_mutation(m)[1]),
                    mutant=lambda x: x['mutation'].map(
                                        lambda m: self._parse_mutation(m)[2]),
                    wildtype=lambda x: x['site'].map(self.wts),
                    )
                [['epitope', 'site', 'wildtype', 'mutant',
                  'mutation', 'escape']]
                )

    def prob_escape(self,
                    *,
                    variants_df,
                    concentrations,
                    substitutions_col='aa_substitutions',
                    concentration_col='concentration',
                    prob_escape_col='prob_escape',
                    ):
        r"""Compute probability of escape :math:`p_v\left(c\right)`.

        Arguments
        ---------
        variants_df : pandas.DataFrame
            Input data frame defining variants and concentrations.
        concentrations : array-like
            Concentrations at which we compute probability of escape.
        substitutions_col : str
            Column in `variants_df` defining variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T').
        concentration_col : str
            Column in returned data frame with concentrations.
        prob_escape_col : str
            Column in returned data frame with :math:`p_v\left(c\right)`.

        Returns
        -------
        pandas.DataFrame
            A copy of `variants_df` with `concentration_col` and
            `prob_escape_col` added, giving the probability of escape
            (:math:`_v\left(c\right)`) values.

        """
        for col in [concentration_col, prob_escape_col]:
            if col in variants_df.columns:
                raise ValueError(f"`variants_df` already has column {col}")
        self._set_binarymap(variants_df, substitutions_col)
        cs = numpy.array(concentrations, dtype='float')
        if not (cs > 0).all():
            raise ValueError('concentrations must be > 0')
        if cs.ndim != 1:
            raise ValueError('concentrations must be 1-dimensional')
        p_v_c = self._compute_pv(cs)
        assert p_v_c.shape == (self._binarymap.nvariants, len(cs))
        return (pd.concat([variants_df.assign(**{concentration_col: c})
                           for c in cs],
                          ignore_index=True)
                .assign(**{prob_escape_col: p_v_c.ravel(order='F')})
                )

    def _compute_pv(self, cs):
        r"""Compute :math:`p_v\left(c\right)`. Call `_set_binarymap` first."""
        if self._binarymap is None or self._a is None or self._beta is None:
            raise ValueError('call `_set_binarymap` first')
        assert (cs > 0).all()
        assert cs.ndim == 1
        phi_e_v = self._binarymap.binary_variants.dot(self._beta) - self._a
        assert phi_e_v.shape == (self._binarymap.nvariants, len(self.epitopes))
        exp_minus_phi_e_v = numpy.exp(-phi_e_v)
        U_e_v_c = 1.0 / (1.0 + numpy.multiply.outer(exp_minus_phi_e_v, cs))
        assert U_e_v_c.shape == (self._binarymap.nvariants,
                                 len(self.epitopes),
                                 len(cs))
        p_v_c = U_e_v_c.prod(axis=1)
        assert p_v_c.shape == (self._binarymap.nvariants, len(cs))
        return p_v_c

    def _set_binarymap(self,
                       variants_df,
                       substitutions_col,
                       ):
        """Set `_binarymap`, `_beta`, `_a` attributes."""
        self._binarymap = dms_variants.binarymap.BinaryMap(
                variants_df,
                substitutions_col=substitutions_col,
                )
        extra_muts = set(self._binarymap.all_subs) - set(self.mutations)
        if extra_muts:
            raise ValueError('variants contain mutations for which no '
                             'escape value initialized:\n'
                             '\n'.join(extra_muts))

        self._a = numpy.array([self._activity_wt[e] for e in self.epitopes],
                              dtype='float')
        assert self._a.shape == (len(self.epitopes),)

        self._beta = numpy.array(
                        [[self._mut_escape[e][m] for e in self.epitopes]
                         for m in self._binarymap.all_subs],
                        dtype='float')
        assert self._beta.shape == (self._binarymap.binarylength,
                                    len(self.epitopes))
        assert self._beta.shape[0] == self._binarymap.binary_variants.shape[1]

    def _parse_mutation(self, mutation):
        """Returns `(wt, site, mut)`."""
        m = self._mutation_regex.fullmatch(mutation)
        if not m:
            raise ValueError(f"invalid mutation {mutation}")
        else:
            return (m.group('wt'), int(m.group('site')), m.group('mut'))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
