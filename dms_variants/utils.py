"""
===========
utils
===========

Miscellaneous utility functions.

"""


import matplotlib.ticker

import pandas as pd  # noqa: F401

import dms_variants._cutils
from dms_variants.constants import (CODON_TO_AA,
                                    NT_COMPLEMENT,
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
         .assign(**{tot_col: lambda x: (x.groupby(group_cols)[n_col]
                                        .transform('max'))})
         )

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

    del corr.columns.name  # remove name of columns index

    if return_type == 'tidy_pairs':
        corr = (
            corr
            .melt(id_vars=group_cols + [sample_col],
                  var_name=sample_col + '_2',
                  value_name='correlation'
                  )
            .rename(columns={sample_col: sample_col + '_1'})
            .query('not correlation.isna()')
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
