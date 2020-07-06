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

    Unless you are using the `expand` option, the binary maps only cover
    substitutions relative to wildtype that are present in at least one
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
    func_score_col : str or None
        Column in `func_scores_df` giving functional score for each variant,
        or `None` if no functional scores available.
    func_score_var_col : str or None
        Column in `func_scores_df` giving variance on functional score
        estimate, or `None` if no variance available.
    n_pre_col : str or None
        Column in `func_scores_df` giving pre-selection counts for each
        variant, or `None` if counts not available.
    n_post_col : str or None
        Column in `func_scores_df` giving post-selection counts for each
        variant, or `None` if counts not available.
    cols_optional : True
        All of the `*_col` parameters are optional except `substitutions_col`.
        If `cols_optional` is `True`, the absence of any of these columns
        is taken the same as setting that column's parameter to zero: the
        corresponding attribute is set to `None`.
    alphabet : list or tuple
        Allowed characters (e.g., amino acids or codons).
    expand : bool
        If `False` (the default) the encoding only covers substitutions
        relative to wildtype that are observed in the set of variants. If
        `True` then the encoding covers all allowed characters at each
        site regardless of whether they are wildtype or observed. In this
        latter case, each binary representation is of length (alphabet size)
        :math:`\times` (sequence length), and sums to the sequence length.
    wtseq : None or str
        Only set this option if `expand` is `True`. In that case, it
        should be the wildtype sequence.

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
    n_pre : numpy.dnarray of integers, or None
        A 1D array of length `nvariants` giving pre-selection counts for each
        variant, or `None` if counts not provided.
    n_post : numpy.dnarray of integers, or None
        A 1D array of length `nvariants` giving post-selection counts for each
        variant, or `None` if counts not provided.
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

    Now do similar operation but using `expand` to include full alphabet
    (although to keep size manageable, we use an alphabet smaller than
    all amino acids):

    >>> wtseq = 'MAKG'
    >>> alphabet = ['A', 'C', 'G', 'K', 'M', '*']
    >>> binmap_expand = BinaryMap(func_scores_df,
    ...                           alphabet=alphabet,
    ...                           expand=True,
    ...                           wtseq=wtseq)
    >>> binmap_expand.binarylength == len(wtseq) * len(alphabet)
    True

    >>> binmap_expand.all_subs
    ... # doctest: +NORMALIZE_WHITESPACE
    ['M1A', 'M1C', 'M1G', 'M1K', 'M1*',
     'A2C', 'A2G', 'A2K', 'A2M', 'A2*',
     'K3A', 'K3C', 'K3G', 'K3M', 'K3*',
     'G4A', 'G4C', 'G4K', 'G4M', 'G4*']

    >>> binmap_expand.binary_variants.toarray()
    ... # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0],
           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0],
           [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0],
           [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0],
           [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            0, 0]], dtype=int8)

    >>> all(numpy.sum(binmap_expand.binary_variants.toarray(), axis=1) ==
    ...     numpy.full(binmap_expand.nvariants, len(wtseq)))
    True

    >>> binmap_expand.substitution_variants
    ['', 'M1A', 'M1C K3A', '', 'A2C K3A', 'A2*']

    >>> for ivar in range(binmap_expand.nvariants):
    ...     binvar = binmap_expand.binary_variants.toarray()[ivar]
    ...     subs_from_df = func_scores_df.at[ivar, 'aa_substitutions']
    ...     assert subs_from_df == binmap_expand.binary_to_sub_str(binvar)
    ...     assert all(binvar == binmap_expand.sub_str_to_binary(subs_from_df))

    Note that `binamp` does not have `n_pre` and `n_post` attributes set:

    >>> binmap.n_pre == binmap.n_post == None
    True

    We would not have been able to initialize `binmap` if we weren't using
    the `cols_optional` flag:

    >>> BinaryMap(func_scores_df, alphabet=alphabet, cols_optional=False)
    Traceback (most recent call last):
      ...
    ValueError: `func_scores_df` lacks column pre_count

    Now assign values to `n_pre` and `n_post` attributes:

    >>> func_scores_df_counts = (
    ...         func_scores_df.assign(pre_count=[10, 20, 15, 5, 6, 8],
    ...                               post_count=[0, 3, 12, 11, 9, 8])
    ...         )
    >>> binmap_counts = BinaryMap(func_scores_df_counts, alphabet=alphabet)
    >>> binmap_counts.n_pre
    array([10, 20, 15,  5,  6,  8])
    >>> binmap_counts.n_post
    array([ 0,  3, 12, 11,  9,  8])

    """

    def __eq__(self, other):
        """Test if equal to object `other`.

        >>> df = pd.DataFrame({'aa_substitutions': ['', 'M1A'],
        ...                    'func_score': [0.0, -1.2],
        ...                    'func_score_var': [0.1, 0.15]})
        >>> df2 = df.copy()
        >>> df3 = df.assign(func_score=lambda x: x['func_score'] + 0.1)
        >>> bmap1 = BinaryMap(df)
        >>> bmap2 = BinaryMap(df2)
        >>> bmap3 = BinaryMap(df3)
        >>> bmap1 == bmap2
        True
        >>> bmap1 == bmap3
        False

        """
        # following here: https://stackoverflow.com/a/390640
        if type(other) != type(self):
            return False
        elif self.__dict__.keys() != other.__dict__.keys():
            return False
        else:
            for key, val in self.__dict__.items():
                val2 = getattr(other, key)
                if type(val) != type(val2):
                    return False
                elif isinstance(val, numpy.ndarray):
                    if not numpy.array_equal(val, val2):
                        return False
                elif isinstance(val, scipy.sparse.csr.csr_matrix):
                    if (val - val2).nnz:
                        return False
                elif isinstance(val, (pd.DataFrame, pd.Series)):
                    if not val.equals(val2):
                        return False
                else:
                    if val != val2:
                        return False
            return True

    def __init__(self,
                 func_scores_df,
                 *,
                 substitutions_col='aa_substitutions',
                 func_score_col='func_score',
                 func_score_var_col='func_score_var',
                 n_pre_col='pre_count',
                 n_post_col='post_count',
                 cols_optional=True,
                 alphabet=dms_variants.constants.AAS_WITHSTOP,
                 expand=False,
                 wtseq=None,
                 ):
        """Initialize object; see main class docstring."""
        self.nvariants = len(func_scores_df)
        self.alphabet = tuple(alphabet)

        for col, attr, dtype, lim_min, lim_max in [
                (func_score_col, 'func_scores', float, None, None),
                (func_score_var_col, 'func_scores_var', float, 0, None),
                (n_pre_col, 'n_pre', int, 0, None),
                (n_post_col, 'n_post', int, 0, None),
                ]:
            if col not in func_scores_df.columns:
                if cols_optional:
                    setattr(self, attr, None)
                else:
                    raise ValueError(f"`func_scores_df` lacks column {col}")
            else:
                vals = func_scores_df[col].values.astype(dtype)
                if not all(vals == func_scores_df[col].values):
                    raise ValueError(f"{col} not of type {dtype}")
                assert vals.shape == (self.nvariants,)
                if any(numpy.isnan(vals)):
                    raise ValueError(f"some entries in {col} are NaN")
                if (lim_min is not None) and any(vals < lim_min):
                    raise ValueError(f"some entries in {col} < {lim_min}")
                if (lim_max is not None) and any(vals > lim_max):
                    raise ValueError(f"some entries in {col} < {lim_min}")
                setattr(self, attr, vals)

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
        self._sub_regex = (rf"(?P<wt>{chars})"
                           rf"(?P<site>\d+)"
                           rf"(?P<mut>{chars})")

        # build mapping from substitution to binary map index
        wts = {}
        muts = collections.defaultdict(set)
        for subs in substitutions:
            for sub in subs.split():
                m = re.fullmatch(self._sub_regex, sub)
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
        self._wt_indices = {}  # keyed by site, values wildtype indices
        if expand:
            if not isinstance(wtseq, str):
                raise ValueError('`wtseq` must be str if `expand` is True')
            if not set(wtseq).issubset(set(alphabet)):
                raise ValueError('`wtseq` has characters not in alphabet')
            if min(wts.keys()) < 1:
                raise ValueError('if `expand`, site numbers must start at 1')
            if max(wts.keys()) > len(wtseq):
                raise ValueError('`wtseq` not long enough given site numbers')
            for site, wt in wts.items():
                if wtseq[site - 1] != wt:
                    raise ValueError('`wtseq` and `func_scores_df` differ on '
                                     f"identity at site {site}")
            i = 0
            for site, wt in enumerate(wtseq, start=1):
                assert (site not in wts) or (wts[site] == wt)
                for char in self.alphabet:
                    self._i_to_sub[i] = f"{wt}{site}{char}"
                    if char == wt:
                        assert site not in self._wt_indices
                        self._wt_indices[site] = i
                    i += 1
        else:
            if wtseq is not None:
                raise ValueError('`wtseq` should be None if `expand` is False')
            i = 0
            for site, wt in sorted(wts.items()):
                for mut in sorted(muts[site]):
                    self._i_to_sub[i] = f"{wt}{site}{mut}"
                    i += 1
        self.binarylength = len(self._i_to_sub)
        self._sub_to_i = {sub: i for i, sub in self._i_to_sub.items()}
        self._wt_index_set = set(self._wt_indices.values())
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
            m = re.fullmatch(self._sub_regex, sub)
            if not m:
                raise ValueError(f"substitution {sub} in {sub_str} invalid "
                                 f"for alphabet {self.alphabet}")
            if m.group('site') in sites:
                raise ValueError("multiple subs at same site in {sub_str}")
            sites.add(m.group('site'))
            indices.append(self.sub_to_i(sub))
        sites = {int(site) for site in sites}
        for site, i in self._wt_indices.items():
            if site not in sites:
                indices.append(i)
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
        subs = [s for s in map(self.i_to_sub, numpy.flatnonzero(binary)) if s]
        sites = [re.fullmatch(self._sub_regex, sub) for sub in subs]
        if len(sites) != len(set(sites)):
            raise ValueError('`binary` specifies multiple substitutions '
                             f"at same site:\n{binary}\n{' '.join(subs)}")
        return ' '.join(subs)

    def i_to_sub(self, i):
        """Substitution corresponding to index in binary representation.

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
            if i in self._wt_index_set:
                return ''
            else:
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
        """list: Substitutions in order encoded in binary map."""
        if not hasattr(self, '_all_subs'):
            self._all_subs = [self.i_to_sub(i) for i in
                              range(self.binarylength) if self.i_to_sub(i)]
        return self._all_subs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
