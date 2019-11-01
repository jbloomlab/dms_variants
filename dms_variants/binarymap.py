"""
=================
binarymap
=================

Defines :class:`BinaryMap` objects for handling binary representations
of variants and their functional scores.

"""


import collections
import re

import numpy

import pandas as pd  # noqa: F401

import scipy.sparse

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
    binary_variants : scipy.sparse.csr.csr_matrix of dtype int8
        Sparse matrix of shape `nvariants` by `binarylength`. Row
        `binary_variants[ivariant]` gives the binary representation of
        variant `ivariant`, and `binary_variants[ivariant, i]` is 1
        if the variant has the substitution :meth:`BinaryMap.i_to_sub`
        and 0 otherwise. To convert to dense `numpy.ndarray`, use
        `toarray` method of the sparse matrix.
    substitution_variants : list
        All variants as substitution strings as provided in `substitutions_col`
        of `func_scores_df`.
    func_scores : numpy.ndarray of floats
        A 1D array of length `nvariants` giving score for each variant.
    func_scores_var : numpy.ndarray of floats, or None
        A 1D array of length `nvariants` giving variance on score for each
        variant, or `None` if no variance estimates provided.
    alphabet : tuple
        Allowed characters (e.g., amino acids or codons).
    substitutions_col : str
        Value set when initializing object.

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

    Scores, score variances, binary and string representations:

    >>> binmap.nvariants
    6
    >>> binmap.func_scores
    array([ 0.  , -0.2 , -0.4 ,  0.01, -0.05, -1.2 ])
    >>> binmap.func_scores_var
    array([0.2 , 0.1 , 0.3 , 0.15, 0.1 , 0.4 ])
    >>> type(binmap.binary_variants)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> binmap.binary_variants.toarray()
    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 1],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 1, 0, 0]], dtype=int8)
    >>> binmap.substitution_variants
    ['', 'M1A', 'M1C K3A', '', 'A2C K3A', 'A2*']
    >>> binmap.substitutions_col
    'aa_substitutions'

    Validate binary map interconverts binary representations and substitutions:

    >>> for ivar in range(binmap.nvariants):
    ...     binvar = binmap.binary_variants.toarray()[ivar]
    ...     subs_from_df = func_scores_df.at[ivar, 'aa_substitutions']
    ...     assert subs_from_df == binmap.binary_to_sub_str(binvar)
    ...     assert all(binvar == binmap.sub_str_to_binary(subs_from_df))

    Demonstrate :meth:`BinaryMap.sub_str_to_indices`:

    >>> for sub in binmap.substitution_variants:
    ...     print(binmap.sub_str_to_indices(sub))
    []
    [0]
    [1, 4]
    []
    [3, 4]
    [2]

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
        self.alphabet = tuple(alphabet)

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

        # get list of substitution strings for each variant
        if substitutions_col not in func_scores_df.columns:
            raise ValueError('`func_scores_df` lacks `substitutions_col` ' +
                             substitutions_col)
        substitutions = func_scores_df[substitutions_col].tolist()
        if not all(isinstance(s, str) for s in substitutions):
            raise ValueError('values in `substitutions_col` not all str')
        self.substitution_variants = substitutions
        self.substitutions_col = substitutions_col

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
        self._sub_regex = re.compile(rf"(?P<wt>{chars})"
                                     rf"(?P<site>\d+)"
                                     rf"(?P<mut>{chars})")

        # build mapping from substitution to binary map index
        wts = {}
        muts = collections.defaultdict(set)
        for subs in substitutions:
            for sub in subs.split():
                m = self._sub_regex.fullmatch(sub)
                if not m:
                    raise ValueError(f"could not match substitution: {sub}")
                site = int(m.group('site'))
                if site not in wts:
                    wts[site] = m.group('wt')
                elif m.group('wt') != wts[site]:
                    raise ValueError(f"different wildtypes at {site}:\n"
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
        self._sub_to_i = {sub: i for i, sub in self._i_to_sub.items()}
        assert len(self._sub_to_i) == len(self._i_to_sub) == self.binarylength

        # build binary_variants
        row_ind = []  # row indices of elements that are one
        col_ind = []  # column indices of elements that are one
        for ivariant, subs in enumerate(substitutions):
            for isub in self.sub_str_to_indices(subs):
                row_ind.append(ivariant)
                col_ind.append(isub)
        self.binary_variants = scipy.sparse.csr_matrix(
                        (numpy.ones(len(row_ind), dtype='int8'),
                         (row_ind, col_ind)),
                        shape=(self.nvariants, self.binarylength),
                        dtype='int8')

    def sub_str_to_binary(self, sub_str):
        """Convert space-delimited substitutions to binary representation.

        Parameters
        ----------
        sub_str : str
            Space-delimited substitutions.

        Returns
        -------
        numpy.ndarray of dtype `int8`
            Binary representation.

        """
        binrep = numpy.zeros(self.binarylength, dtype='int8')
        binrep[self.sub_str_to_indices(sub_str)] = 1
        return binrep

    def sub_str_to_indices(self, sub_str):
        """Convert space-delimited substitutions to list of non-zero indices.

        Parameters
        -----------
        sub_str : str
            Space-delimited substitutions.

        Returns
        -------
        list
            Contains binary representation index for each mutation, so wildtype
            is an empty list.

        """
        sites = set()
        indices = []
        for sub in sub_str.split():
            m = self._sub_regex.fullmatch(sub)
            if not m:
                raise ValueError(f"substitution {sub} in {sub_str} invalid "
                                 f"for alphabet {self.alphabet}")
            if m.group('site') in sites:
                raise ValueError("multiple subs at same site in {sub_str}")
            sites.add(m.group('site'))
            indices.append(self.sub_to_i(sub))
        assert len(sites) == len(sub_str.split()) == len(indices)
        return sorted(indices)

    def binary_to_sub_str(self, binary):
        """Convert binary representation to space-delimited substitutions.

        Note
        ----
        This method is the inverse of :meth:`BinaryMap.sub_str_to_binary`.

        Parameters
        ----------
        binary : numpy.ndarray
            Binary representation.

        Returns
        -------
        str
            Space-delimited substitutions.

        """
        if binary.shape != (self.binarylength,):
            raise ValueError(f"`binary` not length {self.binarylength}:\n" +
                             str(binary))
        if not set(binary).issubset({0, 1}):
            raise ValueError(f"`binary` not all 0 or 1:\n{binary}")
        subs = list(map(self.i_to_sub, numpy.flatnonzero(binary)))
        sites = [self._sub_regex.fullmatch(sub) for sub in subs]
        if len(sites) != len(set(sites)):
            raise ValueError('`binary` specifies multiple substitutions '
                             f"at same site:\n{binary}\n{' '.join(subs)}")
        return ' '.join(subs)

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
        """list: All substitutions in order encoded in binary map."""
        if not hasattr(self, '_all_subs'):
            self._all_subs = [self.i_to_sub(i) for i in
                              range(self.binarylength)]
            assert len(self._all_subs) == len(set(self._all_subs))
        return self._all_subs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
