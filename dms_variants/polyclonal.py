"""
==========
polyclonal
==========

Defines :class:`Polyclonal` objects for handling antibody mixtures.

"""


import re

import pandas as pd


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

    Attributes
    ----------
    epitopes : tuple
        Names of all epitopes.
    mutations : tuple
        All mutations.

    Example
    --------
    A simple example with two epitopes (`e1` and `e2`) and a small
    number of mutations:

    >>> activity_wt_df = pd.DataFrame({'epitope':  ['e1', 'e2'],
    ...                                'activity': [ 2.0,  1.0]})
    >>> activity_wt_df
      epitope  activity
    0      e1       2.0
    1      e2       1.0

    >>> mut_escape_df = pd.DataFrame({
    ...      'mutation': ['M1A', 'M1A', 'M1C', 'M1C', 'A2K', 'A2K'],
    ...      'epitope':  [ 'e1',  'e2',  'e1',  'e2',  'e1',  'e2'],
    ...      'escape':   [  3.0,   0.0,   2.0,  0.0,   0.0,   2.5]})
    >>> mut_escape_df
      mutation epitope  escape
    0      M1A      e1     3.0
    1      M1A      e2     0.0
    2      M1C      e1     2.0
    3      M1C      e2     0.0
    4      A2K      e1     0.0
    5      A2K      e2     2.5

    >>> polyclonal = Polyclonal(activity_wt_df=activity_wt_df,
    ...                         mut_escape_df=mut_escape_df)
    >>> polyclonal.epitopes
    ('e1', 'e2')
    >>> polyclonal.mutations
    ('M1A', 'M1C', 'A2K')

    Note that we can **not** initialize a :class:`Polyclonal` object if we are
    missing escape estimates for any mutations for any epitopes:

    >>> Polyclonal(activity_wt_df=activity_wt_df,
    ...            mut_escape_df=mut_escape_df.head(n=5))
    Traceback (most recent call last):
      ...
    ValueError: not all expected mutations for e2

    """

    def __init__(self,
                 *,
                 activity_wt_df,
                 mut_escape_df,
                 ):
        """See main class docstring."""
        if pd.isnull(activity_wt_df['epitope']).any():
            raise ValueError('epitope name cannot be null')
        self.epitopes = tuple(activity_wt_df['epitope'].unique())
        if len(self.epitopes) != len(activity_wt_df):
            raise ValueError('duplicate epitopes in `activity_wt_df`:\n' +
                             str(activity_wt_df))
        self._activity_wt = (activity_wt_df
                             .set_index('epitope')
                             ['activity']
                             .to_dict()
                             )

        if set(mut_escape_df['epitope']) != set(self.epitopes):
            raise ValueError('`mut_escape_df` does not have same epitopes as '
                             '`activity_wt_df`')
        self.mutations = tuple(mut_escape_df['mutation'].unique())
        for mut in self.mutations:
            if not (isinstance(mut, str) and not re.search(r'\s', mut)):
                raise ValueError(f"invalid mutation: {mut}")
        self._mut_escape = {}
        for epitope, df in mut_escape_df.groupby('epitope'):
            if set(df['mutation']) != set(self.mutations):
                raise ValueError(f"not all expected mutations for {epitope}")
            self._mut_escape[epitope] = (df
                                         .set_index('mutation')
                                         ['escape']
                                         .to_dict()
                                         )
        assert set(self.epitopes) == set(self._activity_wt)
        assert set(self.epitopes) == set(self._mut_escape)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
