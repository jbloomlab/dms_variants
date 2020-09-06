"""
===========
utils
===========

Miscellaneous utility functions.

"""


import re

import matplotlib.ticker

import pandas as pd  # noqa: F401

import dms_variants._cutils
from dms_variants.constants import (AAS_NOSTOP,
                                    CODON_TO_AA,
                                    NT_COMPLEMENT,
                                    SINGLE_NT_AA_MUTS,
                                    )


def reverse_complement(s, *, use_cutils=True):
    """Get reverse complement of DNA sequence.

    Parameters
    ----------
    s : str
        DNA sequence.
    use_cutils : bool
        Use faster C-extension implementation.

    Returns
    -------
    str
        Reverse complement of `s`.

    Example
    -------
    >>> s = 'ATGCAAN'
    >>> reverse_complement(s)
    'NTTGCAT'
    >>> reverse_complement(s, use_cutils=False) == reverse_complement(s)
    True

    """
    if use_cutils:
        return dms_variants._cutils.reverse_complement(s)
    else:
        return ''.join(reversed([NT_COMPLEMENT[nt] for nt in s]))


def translate(codonseq):
    """Translate codon sequence.

    Parameters
    ----------
    codonseq : str
        Codon sequence. Gaps currently not allowed.

    Returns
    -------
    str
        Amino-acid sequence.

    Example
    -------
    >>> translate('ATGGGATAA')
    'MG*'

    """
    if len(codonseq) % 3 != 0:
        raise ValueError('length of `codonseq` not multiple of 3')

    aaseq = []
    for icodon in range(len(codonseq) // 3):
        codon = codonseq[3 * icodon: 3 * icodon + 3]
        aaseq.append(CODON_TO_AA[codon])

    return ''.join(aaseq)


def latex_sci_not(xs):
    r"""Convert list of numbers to LaTex scientific notation.

    Parameters
    ----------
    xs : list
        Numbers to format.

    Returns
    -------
    list
        Formatted strings for numbers.

    Examples
    ----------
    >>> latex_sci_not([0, 3, 3120, -0.0000927])
    ['$0$', '$3$', '$3.1 \\times 10^{3}$', '$-9.3 \\times 10^{-5}$']

    >>> latex_sci_not([0.001, 1, 1000, 1e6])
    ['$0.001$', '$1$', '$10^{3}$', '$10^{6}$']

    >>> latex_sci_not([-0.002, 0.003, 0.000011])
    ['$-0.002$', '$0.003$', '$1.1 \\times 10^{-5}$']

    >>> latex_sci_not([-0.1, 0.0, 0.1, 0.2])
    ['$-0.1$', '$0$', '$0.1$', '$0.2$']

    >>> latex_sci_not([0, 1, 2])
    ['$0$', '$1$', '$2$']

    """
    formatlist = []
    for x in xs:
        xf = f"{x:.2g}"
        if xf[: 2] == '1e':
            xf = f"$10^{{{int(xf[2 : ])}}}$"
        elif xf[: 3] == '-1e':
            xf = f"$-10^{{{int(xf[3 : ])}}}$"
        elif 'e' in xf:
            d, exp = xf.split('e')
            xf = f"${d} \\times 10^{{{int(exp)}}}$"
        else:
            xf = f"${xf}$"
        formatlist.append(xf)
    return formatlist


def cumul_rows_by_count(df, *, count_col='count', n_col='n_rows',
                        tot_col='total_rows', group_cols=None,
                        group_cols_as_str=False):
    """Cumulative number of rows with >= each count.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with rows to analyze.
    count_col : str
        Column in `df` with count for row.
    n_col : str
        Name of column in result giving cumulative count threshold.
    tot_col : str
        Name of column in result giving total number of rows.
    group_cols : None or list
        Group by these columns and analyze each group separately.
    group_cols_as_str : bool
        Convert any `group_cols` columns to str. This is needed if calling
        in ``R`` using `reticulate <https://rstudio.github.io/reticulate/>`_.

    Returns
    -------
    pandas.DataFrame
        Give cumulative counts. For each count in `count_col`, column
        `n_col` gives number of rows with >= that many counts, and
        `tot_col` gives total number of counts.

    Examples
    --------
    >>> df = pd.DataFrame({'sample': ['a', 'a', 'b', 'b', 'a', 'a'],
    ...                    'count': [9, 0, 1, 4, 3, 3]})
    >>> cumul_rows_by_count(df)
       count  n_rows  total_rows
    0      9       1           6
    1      4       2           6
    2      3       4           6
    3      1       5           6
    4      0       6           6
    >>> cumul_rows_by_count(df, group_cols=['sample'])
      sample  count  n_rows  total_rows
    0      a      9       1           4
    1      a      3       3           4
    2      a      0       4           4
    3      b      4       1           2
    4      b      1       2           2

    """
    if count_col not in df.columns:
        raise ValueError(f"df does not have column {count_col}")

    if not group_cols:
        drop_group_cols = True
        group_cols = ['dummy_col']
        df[group_cols[0]] = '_'
    else:
        drop_group_cols = False
        if isinstance(group_cols, str):
            group_cols = [group_cols]

    if not set(group_cols).issubset(set(df.columns)):
        raise ValueError(f"invalid `group_cols` {group_cols}")

    for col in [count_col, n_col, tot_col]:
        if col in group_cols:
            raise ValueError(f"`group_cols` cannot contain {col}")

    df = (
         df

         # get number of rows with each count
         .assign(**{n_col: 1})
         .groupby(group_cols + [count_col])
         .aggregate({n_col: 'count'})
         .reset_index()
         .sort_values(group_cols + [count_col],
                      ascending=[True] * len(group_cols) + [False])
         .reset_index(drop=True)

         # get cumulative number with <= number of counts
         .assign(**{n_col: lambda x: x.groupby(group_cols)[n_col].cumsum()})

         # add new column that is total number of rows
         .assign(**{tot_col: lambda x: (x
                                        .groupby(group_cols, sort=False)
                                        [n_col]
                                        .transform('max'))})
         )
    assert df[tot_col].notnull().all()

    if drop_group_cols:
        df = df.drop(group_cols, axis='columns')
    elif group_cols_as_str:
        for col in group_cols:
            df[col] = df[col].astype('str')

    return df


def tidy_to_corr(df, sample_col, label_col, value_col, *,
                 group_cols=None, return_type='tidy_pairs',
                 method='pearson'):
    """Pairwise correlations between samples in tidy data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy data frame.
    sample_col : str
        Column in `df` with name of sample.
    label_col : str
        Column in `df` with labels for variable to correlate.
    value_col : str
        Column in `df` with values to correlate.
    group_cols : None, str, or list
        Additional columns used to group results.
    return_type : {'tidy_pairs', 'matrix'}
        Return results as tidy dataframe of pairwise correlations
        or correlation matrix.
    method : str
        A correlation metho passable to `pandas.DataFrame.corr`.

    Returns
    -------
    pandas.DataFrame
        Holds pairwise correlations in format specified by `return_type`.
        Correlations only calculated among values with shared label
        among samples.

    Example
    -------
    Define data frame with data to correlate:

    >>> df = pd.DataFrame({
    ...        'sample': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c'],
    ...        'barcode': ['A', 'C', 'G', 'G', 'A', 'C', 'T', 'G', 'C', 'A'],
    ...        'score': [1, 2, 3, 3, 1.5, 2, 4, 1, 2, 3],
    ...        'group': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'y', 'y', 'y'],
    ...        })

    Pairwise correlations between all samples ignoring group:

    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score')
      sample_1 sample_2  correlation
    0        a        a     1.000000
    1        b        a     0.981981
    2        c        a    -1.000000
    3        a        b     0.981981
    4        b        b     1.000000
    5        c        b    -0.981981
    6        a        c    -1.000000
    7        b        c    -0.981981
    8        c        c     1.000000

    The same but as a matrix rather than in tidy format:

    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score', return_type='matrix')
      sample         a         b         c
    0      a  1.000000  0.981981 -1.000000
    1      b  0.981981  1.000000 -0.981981
    2      c -1.000000 -0.981981  1.000000

    Now group before computing correlations:

    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score', group_cols='group')
      group sample_1 sample_2  correlation
    0     x        a        a     1.000000
    1     x        b        a     0.981981
    2     x        a        b     0.981981
    3     x        b        b     1.000000
    4     y        c        c     1.000000
    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score', group_cols='group',
    ...              return_type='matrix')
      group sample         a         b    c
    0     x      a  1.000000  0.981981  NaN
    1     x      b  0.981981  1.000000  NaN
    2     y      c       NaN       NaN  1.0

    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    elif group_cols is None:
        group_cols = []
    cols = [sample_col, value_col, label_col] + group_cols
    if set(cols) > set(df.columns):
        raise ValueError(f"`df` missing some of these columns: {cols}")
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names: {cols}")
    if 'correlation' in cols:
        raise ValueError('cannot have column named `correlation`')
    if sample_col + '_2' in group_cols:
        raise ValueError(f"cannot have column named `{sample_col}_2`")

    for _, g in df.groupby([sample_col] + group_cols):
        if len(g[label_col]) != g[label_col].nunique():
            raise ValueError(f"Entries in `df` column {label_col} not unique "
                             'after grouping by: ' +
                             ', '.join(c for c in [sample_col] + group_cols))

    df = (
        df
        .pivot_table(values=value_col,
                     columns=sample_col,
                     index=[label_col] + group_cols,
                     )
        .reset_index()
        )

    if group_cols:
        df = df.groupby(group_cols)

    corr = (
        df
        .corr(method=method)
        .dropna(how='all', axis='index')
        .reset_index()
        )

    corr.columns.name = None  # remove name of columns index

    if return_type == 'tidy_pairs':
        corr = (
            corr
            .melt(id_vars=group_cols + [sample_col],
                  var_name=sample_col + '_2',
                  value_name='correlation'
                  )
            .rename(columns={sample_col: sample_col + '_1'})
            .dropna()
            .reset_index(drop=True)
            )

    elif return_type != 'matrix':
        raise ValueError(f"invalid `return_type` of {return_type}")

    return corr


def tidy_split(df, column, sep=' ', keep=False):
    """Split values of column and expand into new rows.

    Note
    ----
    Taken from https://stackoverflow.com/a/39946744

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame with the column to split and expand.
    column : str
        Name of column to split and expand.
    sep : str
        The string used to split the column's values.
    keep : bool
        Retain the presplit value as it's own row.

    Returns
    -------
    pandas.DataFrame
        Data frame with the same columns as `df`. Rows lacking
        `column` are filtered.

    Example
    -------
    >>> df = pd.DataFrame({'col1': ['A', 'B', 'C'],
    ...                    'col2': ['d e', float('nan'), 'f']})
    >>> tidy_split(df, 'col2')
      col1 col2
    0    A    d
    0    A    e
    2    C    f

    """
    indexes = []
    new_values = []
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


def integer_breaks(x):
    """Integer breaks for axes labels.

    Note
    ----
    The breaks can be passed to `plotnine <http://plotnine.readthedocs.io>`_
    as in::

        scale_x_continuous(breaks=integer_breaks)

    Parameters
    ----------
    x : array-like
        Numerical data values.

    Returns
    -------
    numpy.ndarray
        Integer tick locations.

    Example
    -------
    >>> integer_breaks([0.5, 0.7, 1.2, 3.7, 7, 17])
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])

    """
    return (matplotlib.ticker.MaxNLocator(integer=True)
            .tick_values(min(x), max(x))
            )


def scores_to_prefs(df, mutation_col, score_col, base,
                    wt_score=0, missing='average',
                    alphabet=AAS_NOSTOP, exclude_chars=('*',),
                    returnformat='wide', stringency_param=1):
    r"""Convert functional scores to amino-acid preferences.

    Preferences are calculated from functional scores as follows. Let
    :math:`y_{r,a}` be the score of the variant with the single mutation
    of site :math:`r` to :math:`a` (when :math:`a` is the wildtype character,
    then :math:`p_{r,a}` is the score of the wildtype sequence). Then the
    preference :math:`\pi_{r,a}` is

    .. math::

        \pi_{r,a} = \frac{b^{y_{r,a}}}{\sum_{a'} b^{y_{r,a'}}}

    where :math:`b` is the base for the exponent. This definition ensures
    that the preferences sum to one at each site. These preferences can be
    displayed in logo pltos or used as input to
    `phydms <https://jbloomlab.github.io/phydms/>`_

    Note
    ----
    The "flatness" of the preferences is determined by the exponent base.
    A smaller `base` yields flatter preferences. There is no obvious "best"
    `base` as different values correspond to different linear scalings
    of the scores. A recommended approach is simply to choose a value of `base`
    (such as 10) and then re-scale the preferences by using
    `phydms <https://jbloomlab.github.io/phydms/>`_ to optimize a stringency
    parameter as `described here <https://peerj.com/articles/3657>`_. One
    thing to note is that `phydms <https://jbloomlab.github.io/phydms/>`_
    has an upper bound on the largest stringency parameter it can fit,
    so if you are hitting this upper bound then pre-scale the preferences
    to be less flat by using a larger value of `base`.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame holding the functional scores.
    mutation_col : str
        Column in `df` with mutations, in this format: 'M1A'.
    score_col : str
        Column in `df` with functional scores.
    base : float
        Base to which the exponent is taken in computing the preferences.
        Make sure not to choose an excessively small value if using
        in `phydms <https://jbloomlab.github.io/phydms/>`_ or the
        preferences will be too flat. In the examples below we use 2,
        but you may want a larger value.
    wt_score : float
        Functional score for wildtype sequence.
    missing : {'average', 'site_average', 'error'}
        What to do when there is no estimate of the score for a mutant?
        Estimate the phenotype as the average of all single mutants, the
        average of all single mutants at that site, or raise an error.
    alphabet : list or tuple
        Characters (e.g., amino acids) for which we compute preferences.
    exclude_chars : tuple or list
        Characters to exclude when calculating preferences (and when
        averaging values for missing mutants). For instance, you might
        want to exclude stop codons even if they are in `df`.
    returnformat : {'tidy', 'wide'}
        Return preferences in tidy or wide format data frame.
    stringency_param : float
        Re-scale preferences by this stringency parameter. This
        involves raising each preference to the power of
        `stringency_param`, and then re-normalizes. A similar
        effect can be achieved by changing `base`.

    Returns
    -------
    pandas.DataFrame
        Data frame where first column is named 'site', other columns are
        named for each character, and rows give preferences for each site.

    Example
    -------
    >>> func_scores_df = pd.DataFrame(
    ...         {'aa_substitutions': ['M1A', 'M1C', 'A2M', 'A2C', 'M1*'],
    ...          'func_score':       [-0.1,  -2.3,   0.8,  -1.2,  -3.0,]})

    >>> (scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                  alphabet=['M', 'A', 'C'], exclude_chars=['*'])
    ...  ).round(2)
       site     M     A     C
    0     1  0.47  0.44  0.10
    1     2  0.55  0.31  0.14

    >>> (scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                  alphabet=['M', 'A', 'C', '*'], exclude_chars=[])
    ...  ).round(2)
       site     M     A     C     *
    0     1  0.44  0.41  0.09  0.06
    1     2  0.48  0.28  0.12  0.12

    >>> (scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                  alphabet=['M', 'A', 'C', '*'], exclude_chars=[],
    ...                  missing='site_average')
    ...  ).round(2)
       site     M     A     C     *
    0     1  0.44  0.41  0.09  0.06
    1     2  0.43  0.25  0.11  0.22

    >>> scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                 alphabet=['M', 'A', 'C', '*'], exclude_chars=[],
    ...                 missing='error')
    Traceback (most recent call last):
        ...
    ValueError: missing functional scores for some mutations

    >>> (scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                  alphabet=['M', 'A', 'C'], exclude_chars=['*'],
    ...                  returnformat='tidy')
    ...  ).round(2)
      wildtype  site mutant  preference
    0        M     1      C        0.10
    1        A     2      C        0.14
    2        A     2      A        0.31
    3        M     1      A        0.44
    4        M     1      M        0.47
    5        A     2      M        0.55

    >>> (scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                  alphabet=['M', 'A', 'C'], exclude_chars=['*'],
    ...                  stringency_param=3)
    ...  ).round(2)
       site     M     A     C
    0     1  0.55  0.45  0.00
    1     2  0.83  0.16  0.01

    >>> (scores_to_prefs(func_scores_df, 'aa_substitutions', 'func_score', 2,
    ...                  alphabet=['M', 'A', 'C', '*'], exclude_chars=[],
    ...                  returnformat='tidy')
    ...  ).round(2)
      wildtype  site mutant  preference
    0        M     1      *        0.06
    1        M     1      C        0.09
    2        A     2      C        0.12
    3        A     2      *        0.12
    4        A     2      A        0.28
    5        M     1      A        0.41
    6        M     1      M        0.44
    7        A     2      M        0.48

    """
    if not isinstance(exclude_chars, (list, tuple)):
        raise ValueError('`exclude_chars` must be list, tuple (can be empty)')
    exclude_chars = list(exclude_chars)

    alphabet = list(alphabet)
    if [a for a in alphabet if a in exclude_chars]:
        raise ValueError(f"character in `exclude_chars` of {exclude_chars} "
                         f" in `alphabet` of {alphabet}. These lists must be "
                         'mutually exclusive')

    if score_col == mutation_col:
        raise ValueError('`score_col` and `mutation_col` must be different')
    for colname, col in [('mutation', mutation_col), ('score', score_col)]:
        if col not in df.columns:
            raise ValueError(f"`df` lacks `{colname}_col` of {col}")
        if col in {'wildtype', 'site', 'mutant'}:
            raise ValueError(f"`{colname}_col` cannot be named {score_col}")

    if len(df[mutation_col]) != df[mutation_col].nunique():
        raise ValueError('duplicated entries in `mutation_col` of `df`')

    # extract wildtype, site, mutant from mutation
    chars_regex = ''.join(re.escape(a) for a in alphabet + exclude_chars)
    df = (df.join(df
                  [mutation_col]
                  .str
                  .extract(rf"^(?P<wildtype>[{chars_regex}])" +
                           r'(?P<site>\-?\d+)' +
                           rf"(?P<mutant>[{chars_regex}])$")
                  .assign(site=lambda x: x['site'].astype(int))
                  )
          [['site', 'wildtype', 'mutant', score_col, mutation_col]]
          )
    if df.isnull().any().any():
        raise ValueError('unparseable mutations given specified alphabet:\n' +
                         ', '.join(df[mutation_col].tolist()))
    if len(df.query('wildtype == mutant')):
        raise ValueError('`df` contains mutations with same wildtype & mutant')

    # remove any excluded characters
    df = (df
          [['site', 'wildtype', 'mutant', score_col]]
          .query('wildtype not in @exclude_chars')
          .query('mutant not in @exclude_chars')
          )

    # get missing values for later use
    if missing == 'average':
        missing_val = df[score_col].mean()
    elif missing == 'site_average':
        missing_val = df.groupby('site')[score_col].mean().to_dict()
    elif missing != 'error':
        raise ValueError(f"invalid `missing` of {missing}")

    # add wildtype to data frame
    wt_df = (df
             [['wildtype', 'site']]
             .drop_duplicates()
             .assign(mutant=lambda x: x['wildtype'])
             .assign(**{score_col: wt_score})
             )
    df = pd.concat([df, wt_df], sort=False)

    # add missing characters from alphabet for each site as here:
    # https://stackoverflow.com/a/47118819
    mux = pd.MultiIndex.from_product([df['site'].unique(), alphabet],
                                     names=['site', 'mutant'])
    df = (mux
          .to_frame(index=False)
          .assign(wildtype=lambda x: x['site'].map(df
                                                   .set_index('site')
                                                   ['wildtype']
                                                   .to_dict()))
          .merge(df, on=['site', 'wildtype', 'mutant'], how='outer')
          )

    # fill missing values
    if missing == 'average':
        df = df.fillna(missing_val)
    elif missing == 'site_average':
        df_notmissing = df[df.notnull().all(axis=1)]
        df_missing = df[df.isnull().any(axis=1)]
        assert len(df) == len(df_notmissing) + len(df_missing)
        df_missing = (df_missing
                      .assign(**{score_col: lambda x: (x['site']
                                                       .map(missing_val))}
                              )
                      )
        df = pd.concat([df_notmissing, df_missing], sort=False)
    elif df.isnull().any().any():
        raise ValueError('missing functional scores for some mutations')

    # convert to prefs
    df = (df
          .assign(unscaled_prefs=lambda x: (base**x[score_col]
                                            )**stringency_param,
                  preference=lambda x: (x['unscaled_prefs'] /
                                        (x.groupby('site', sort=False)
                                         ['unscaled_prefs']
                                         .transform('sum')
                                         )
                                        )
                  )
          )

    # pivot to wide form
    if returnformat == 'wide':
        df = df.pivot_table(index='site',
                            columns='mutant',
                            values='preference')
        df = df[alphabet]
        df.columns.name = None
        df = df.reset_index()
    elif returnformat == 'tidy':
        df = (df
              [['wildtype', 'site', 'mutant', 'preference']]
              .sort_values('preference')
              .reset_index(drop=True)
              )
    else:
        raise ValueError(f"invalid `returnformat` {returnformat}")
    assert not df.isnull().any().any(), df

    return df


def single_nt_accessible(codon, aa, codon_encode_aa='raise'):
    """Is amino acid accessible from codon by single-nucleotide change?

    Parameters
    ----------
    codon : str
        The codon.
    aa : str
        The amino acid.
    codon_encode_aa : {'raise', 'true', 'false'}
        If `codon` encodes `aa`, raise an error, return `True`,
        or return `False`.

    Returns
    -------
    bool

    Example
    -------
    >>> single_nt_accessible('GGG', 'E')
    True
    >>> single_nt_accessible('GGC', 'E')
    False
    >>> single_nt_accessible('GGG', 'G')
    Traceback (most recent call last):
      ...
    ValueError: `codon` GGG already encodes `aa` G (see `codon_encode_aa`)
    >>> single_nt_accessible('GGG', 'G', codon_encode_aa='true')
    True
    >>> single_nt_accessible('TTT', 'L')
    True

    """
    if CODON_TO_AA[codon] == aa:
        if codon_encode_aa == 'raise':
            raise ValueError(f"`codon` {codon} already encodes `aa` {aa} "
                             '(see `codon_encode_aa`)')
        elif codon_encode_aa == 'false':
            return False
        elif codon_encode_aa == 'true':
            return True
        else:
            raise ValueError(f"invalid `codon_encode_aa` {codon_encode_aa}")
    elif aa in SINGLE_NT_AA_MUTS[codon]:
        return True
    else:
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()
