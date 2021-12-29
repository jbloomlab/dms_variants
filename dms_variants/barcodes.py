"""
========
barcodes
========

Utility functions to process and analyze barcodes.

"""

import collections
import math
import random  # noqa: F401

import numpy

import pandas as pd

import scipy.special


def inverse_simpson_index(
    barcodecounts,
    *,
    barcodecol="barcode",
    countcol="count",
    groupcols="library",
):
    """Inverse Simpson index (reciprocal probability two barcodes are same).

    Parameters
    ----------
    barcodecounts: pandas.DataFrame
        Data frame with barcode counts
    barcodecol : str
        Column in ``barcodecounts`` listing all unique barcodes.
    countcol : str
        Column in ``barcodecounts`` with counts of each barcode.
    groupcols : str, list, or None
        Columns in ``barcodecounts`` by which we group for calculations.

    Returns
    -------
    pandas.DataFrame

    Example
    -------
    >>> barcodecounts = pd.DataFrame.from_records(
    ...        [('lib1', 'AA', 10),
    ...         ('lib1', 'AT', 20),
    ...         ('lib1', 'AC', 30),
    ...         ('lib2', 'AA', 5)],
    ...        columns=['library', 'barcode', 'count'])
    >>> inverse_simpson_index(barcodecounts)
      library  inverse_simpson_index
    0    lib1               2.571429
    1    lib2               1.000000

    """
    # based on here: https://gist.github.com/martinjc/f227b447791df8c90568
    reserved_cols = ["dummy", "p2", "simpson_index", "inverse_simpson_index"]
    for col in reserved_cols:
        if col in barcodecounts.columns:
            raise ValueError(f"`barcodecounts` cannot have column {col}")

    if groupcols:
        if isinstance(groupcols, str):
            groupcols = [groupcols]
    else:
        groupcols = ["dummy"]
        barcodecounts["dummy"] = "dummy"
    req_cols = [barcodecol, countcol, *groupcols]
    if not set(barcodecounts.columns).issuperset(req_cols):
        raise ValueError(f"`barcodecounts` lacks columns {req_cols}")
    if len(barcodecounts) != len(barcodecounts.groupby(req_cols)):
        raise ValueError("`barcodecol` and `groupcols` not unique rows")

    df = (
        barcodecounts.assign(
            p2=lambda x: (
                x[countcol] / (x.groupby(groupcols)[countcol].transform("sum"))
            )
            ** 2
        )
        .groupby(groupcols, as_index=False)
        .aggregate(simpson_index=pd.NamedAgg("p2", "sum"))
        .assign(inverse_simpson_index=lambda x: 1 / x["simpson_index"])
    )
    if groupcols == ["dummy"]:
        groupcols = []
    return df[[*groupcols, "inverse_simpson_index"]]


def rarefyBarcodes(
    barcodecounts,
    *,
    barcodecol="barcode",
    countcol="count",
    maxpoints=100000,
    logspace=True,
):
    """Rarefaction curve of barcode observations.

    Note
    ----
    Uses analytical formula for rarefaction defined here:
    https://en.wikipedia.org/wiki/Rarefaction_(ecology)#Derivation

    Parameters
    ----------
    barcodecounts : pandas.DataFrame
        Data frame with counts to rarefy.
    barcodecol : str
        Column in `barcodecounts` listing all unique barcodes.
    countcol : str
        Column in `barcodecounts` with observed counts of each barcode.
    maxpoints : int
        Only calculate rarefaction curve at this many points. Benefit
        is that it is costly to calculate the curve for many points.
    logspace : bool
        Logarithmically space the points. If `False`, space linearly.

    Returns
    -------
    pandas.DataFrame
        A data frame with columns 'ncounts' and 'nbarcodes' giving number
        of unique barcodes observed for each total number of observed counts.

    Example
    -------
    >>> barcodecounts = pd.DataFrame({'barcode': ['A', 'G', 'C', 'T'],
    ...                               'count': [4, 2, 1, 0]})
    >>> rarefaction_curve = rarefyBarcodes(barcodecounts)
    >>> rarefaction_curve
       ncounts  nbarcodes
    0        1   1.000000
    1        2   1.666667
    2        3   2.114286
    3        4   2.428571
    4        5   2.666667
    5        6   2.857143
    6        7   3.000000

    Verify this result matches what is obtained by random sampling:

    >>> random.seed(1)
    >>> barcodelist = []
    >>> for tup in barcodecounts.itertuples(index=False):
    ...     barcodelist += [tup.barcode] * tup.count
    >>> nrand = 10000
    >>> ncounts = list(range(1, barcodecounts['count'].sum() + 1))
    >>> nbarcodes = []
    >>> for ncount in ncounts:
    ...     nbarcodes.append(sum(len(set(random.sample(barcodelist, ncount)))
    ...                      for _ in range(nrand)) / nrand)
    >>> sim_rarefaction_curve = pd.DataFrame({'ncounts': ncounts,
    ...                                       'nbarcodes': nbarcodes})
    >>> numpy.allclose(rarefaction_curve, sim_rarefaction_curve, atol=1e-2)
    True

    """
    if len(barcodecounts) != len(barcodecounts[barcodecol].unique()):
        raise ValueError("non-unique barcodes in `barcodecounts`")

    # follow nomenclature at
    # https://en.wikipedia.org/wiki/Rarefaction_(ecology)#Derivation
    Ni = barcodecounts.set_index(barcodecol)[countcol].to_dict()
    N = sum(Ni.values())
    K = len(barcodecounts)
    Mj = collections.Counter(Ni.values())

    Nk, num = map(numpy.array, zip(*Mj.items()))

    # use simplification that (N - Ni)Cr(n) / (N)Cr(n) =
    # [(N - Ni)! * (N - n)!] / [N! * (N - Ni - n)!]
    #
    # Also use fact that gamma(x + 1) = x!
    nbarcodes = []
    lnFactorial_N = scipy.special.gammaln(N + 1)
    if logspace and N > maxpoints:
        ncounts = list(
            numpy.unique(
                numpy.logspace(
                    math.log10(1), math.log10(N), num=min(N, maxpoints)
                ).astype("int")
            )
        )
    else:
        ncounts = list(
            numpy.unique(numpy.linspace(1, N, num=min(N, maxpoints)).astype("int"))
        )
    for n in ncounts:
        lnFactorial_N_minus_n = scipy.special.gammaln(N - n + 1)
        i = numpy.nonzero(N - Nk - n >= 0)  # indices where this is true
        nbarcodes.append(
            K
            - (
                num[i]
                * numpy.exp(
                    scipy.special.gammaln(N - Nk[i] + 1)
                    + lnFactorial_N_minus_n
                    - lnFactorial_N
                    - scipy.special.gammaln(N - Nk[i] - n + 1)
                )
            ).sum()
        )
    return pd.DataFrame({"ncounts": ncounts, "nbarcodes": nbarcodes})


if __name__ == "__main__":
    import doctest

    doctest.testmod()
