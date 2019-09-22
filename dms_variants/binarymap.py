"""
=================
binarymap
=================

Defines :class:`BinaryMap` objects for handling binary representations
of variants and their functional scores.

"""


import collections
import itertools
import re

import numpy

import pandas as pd  # noqa: F401

import dms_variants.constants


class BinaryMap:
    r"""Binary representations of variants and their functional scores.

    Note
    ----
    These maps represent variants as arrays of 0 and 1 integers indicating
    whether a particular variant has a substitution. The wildtype is all 0.
    Such representations are useful for fitting estimates of the effect of
    each substitution.

    The binary maps only cover substitutions that are present in at least one
    of the variants used to create the map.

    Parameters
    ----------
    func_scores_df : pandas.DataFrame
        Data frame of variants and their functional scores. Each row is
        a different variant, defined by space-delimited list of substitutions.
        Data frames of this type are returned by
        :meth:`dms_variants.codonvarianttable.CodonVariantTable.func_scores`.
    substitutions_col : str
        Column in `func_scores_df` giving substitutions for each variant.
    func_score_col : str
        Column in `func_scores_df` giving functional score for each variant.
    func_score_var_col : str or None
        Column in `func_scores_df` giving variance on functional score
        estimate, or `None` if no variance available.
    alphabet : list or tuple
        Allowed characters (e.g., amino acids or codons).

    Attributes
    ----------
    binarylength : int
        Length of the binary representation of each variant.
    nvariants : int
        Number of variants.
    binary_variants : numpy.ndarray of dtype int8
        A 2D array of shape `nvariants` by `binarylength`. So
        `binary_variants[ivariant]` gives the binary representation of
        variant `ivariant`, and `binary_variants[ivariant, i]` is 1
        if the variant has the substitution :meth:`BinaryMap.i_to_sub`
        and 0 otherwise.
    func_scores : numpy.ndarray of floats
        A 1D array of length `nvariants` giving score for each variant.
    func_scores_var : numpy.ndarray of floats, or None
        A 1D array of length `nvariants` giving variance on score for each
        variant, or `None` if no variance estimates provided.

    Example
    -------
    Create a binary map:

    >>> func_scores_df = pd.DataFrame.from_records(
    ...         [('', 0.0, 0.2),
    ...          ('M1A', -0.2, 0.1),
    ...          ('M1C K3A', -0.4, 0.3),
    ...          ('', 0.01, 0.15),
    ...          ('A2C K3A', -0.05, 0.1),
    ...          ('A2*', -1.2, 0.4),
    ...          ],
    ...         columns=['aa_substitutions', 'func_score', 'func_score_var'])
    >>> binmap = BinaryMap(func_scores_df)

    The length of the binary representation equals the number of unique
    substitutions:

    >>> binmap.binarylength
    5
    >>> binmap.all_subs
    ['M1A', 'M1C', 'A2*', 'A2C', 'K3A']

    Here are the scores, score variances, and binary representations:

    >>> binmap.nvariants
    6
    >>> binmap.func_scores
    array([ 0.  , -0.2 , -0.4 ,  0.01, -0.05, -1.2 ])
    >>> binmap.func_scores_var
    array([0.2 , 0.1 , 0.3 , 0.15, 0.1 , 0.4 ])
    >>> binmap.binary_variants
    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 1, 0, 0]], dtype=int8)

    Validate binary map interconverts binary representations and substitutions:

    >>> for ivar in range(binmap.nvariants):
    ...     binvar = binmap.binary_variants[ivar]
    ...     subs_from_df = func_scores_df.at[ivar, 'aa_substitutions']
    ...     subs = ' '.join(map(binmap.i_to_sub, numpy.flatnonzero(binvar)))
    ...     assert subs == subs_from_df, f"{subs}\n{subs_from_df}"
    ...     bin_from_subs = numpy.zeros(binmap.binarylength)
    ...     for sub in subs_from_df.split():
    ...         bin_from_subs[binmap.sub_to_i(sub)] = 1
    ...     assert all(binvar == bin_from_subs), f"{binvar}\n{bin_from_subs}"

    """

    def __init__(self,
                 func_scores_df,
                 *,
                 substitutions_col='aa_substitutions',
                 func_score_col='func_score',
                 func_score_var_col='func_score_var',
                 alphabet=dms_variants.constants.AAS_WITHSTOP,
                 ):
        """Initialize object; see main class docstring."""
        self.nvariants = len(func_scores_df)

        if func_score_col not in func_scores_df.columns:
            raise ValueError('`func_scores_df` lacks `func_score_col` ' +
                             func_score_col)
        self.func_scores = func_scores_df[func_score_col].values.astype(float)
        assert self.func_scores.shape == (self.nvariants,)
        if any(numpy.isnan(self.func_scores)):
            raise ValueError('some functional scores are NaN')

        if func_score_var_col is None:
            self.func_scores_var = None
        else:
            if func_score_var_col not in func_scores_df.columns:
                raise ValueError('`func_scores_df` lacks `func_score_var_col` '
                                 + func_score_var_col)
            self.func_scores_var = (func_scores_df[func_score_var_col]
                                    .values.astype(float))
            assert self.func_scores_var.shape == (self.nvariants,)
            if any(numpy.isnan(self.func_scores_var)):
                raise ValueError('some functional score variances are NaN')
            if any(self.func_scores_var < 0):
                raise ValueError('some functional score variances are < 0')

        # get lists of lists of substitutions for each variant
        if substitutions_col not in func_scores_df.columns:
            raise ValueError('`func_scores_df` lacks `substitutions_col` ' +
                             substitutions_col)
        substitutions = func_scores_df[substitutions_col].tolist()
        if not all(isinstance(s, str) for s in substitutions):
            raise ValueError('values in `substitutions_col` not all str')
        substitutions = [s.split() for s in substitutions]
        assert len(substitutions) == self.nvariants

        # regex that matches substitution
        chars = []
        for char in alphabet:
            if char.isalpha():
                chars.append(char)
            elif char == '*':
                chars.append(r'\*')
            else:
                raise ValueError(f"invalid alphabet character: {char}")
        chars = '|'.join(chars)
        sub_regex = re.compile(rf"(?P<wt>{chars})"
                               rf"(?P<site>\d+)"
                               rf"(?P<mut>{chars})")

        # build mapping from substitution to binary map index
        wts = {}
        muts = collections.defaultdict(set)
        for sub in itertools.chain.from_iterable(substitutions):
            m = sub_regex.fullmatch(sub)
            if not m:
                raise ValueError(f"could not match substitution: {sub}")
            site = int(m.group('site'))
            if site not in wts:
                wts[site] = m.group('wt')
            elif m.group('wt') != wts[site]:
                raise ValueError(f"different wildtype identities at {site}:\n"
                                 f"{m.group('wt')} versus {wts[site]}")
            if m.group('mut') == wts[site]:
                raise ValueError(f"wildtype and mutant the same in {sub}")
            muts[site].add(m.group('mut'))
        self._i_to_sub = {}
        i = 0
        for site, wt in sorted(wts.items()):
            for mut in sorted(muts[site]):
                self._i_to_sub[i] = f"{wt}{site}{mut}"
                i += 1
        self.binarylength = i
        assert (set(itertools.chain.from_iterable(substitutions)) ==
                set(self._i_to_sub.values()))
        self._sub_to_i = {sub: i for i, sub in self._i_to_sub.items()}
        assert len(self._sub_to_i) == len(self._i_to_sub) == self.binarylength

        # build binary_variants
        self.binary_variants = numpy.zeros(
                                shape=(self.nvariants, self.binarylength),
                                dtype='int8')
        for ivariant, subs in enumerate(substitutions):
            # check that variant doesn't have multiple subs at same site
            sites = [int(sub_regex.fullmatch(s).group('site')) for s in subs]
            if len(sites) != len(set(sites)):
                raise ValueError(f"variant {ivariant} has multiple "
                                 f"substitutions at the same site:\n{subs}")
            for sub in subs:
                self.binary_variants[ivariant, self.sub_to_i(sub)] = 1
            assert len(subs) == self.binary_variants[ivariant].sum()

    def i_to_sub(self, i):
        """Mutation corresponding to index in binary representation.

        Parameters
        ----------
        i : int
            Index in binary representation, 0 <= `i` < `binarylength`.

        Returns
        -------
        str
            The substitution corresponding to that index.

        """
        try:
            return self._i_to_sub[i]
        except KeyError:
            if i < 0 or i >= self.binarylength:
                raise ValueError(f"invalid i of {i}. Must be >= 0 and "
                                 f"< {self.binarylength}")
            else:
                raise ValueError(f"unexpected error, i = {i} should be in map")

    def sub_to_i(self, sub):
        """Index in binary representation corresponding to substitution.

        Parameters
        ----------
        sub : str
            The substitution.

        Returns
        -------
        int
            Index in binary representation, will be >= 0 and < `binarylength`.

        """
        try:
            return self._sub_to_i[sub]
        except KeyError:
            raise ValueError(f"sub of {sub} is not in the binary map. The map "
                             'only contains substitutions in the variants.')

    @property
    def all_subs(self):
        """list: all substitutions in order encoded in binary map."""
        if not hasattr(self, '_all_subs'):
            self._all_subs = [self.i_to_sub(i) for i in
                              range(self.binarylength)]
            assert len(self._all_subs) == len(set(self._all_subs))
        return self._all_subs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
