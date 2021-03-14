"""
==========
polyclonal
==========

Defines :class:`Polyclonal` objects for handling antibody mixtures.

"""


import collections
import itertools
import re

import altair as alt

import matplotlib.colors

import numpy

import pandas as pd

import dms_variants.binarymap
import dms_variants.constants
import dms_variants.utils

alt.data_transformers.disable_max_rows()


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
    epitope_colors : array-like or dict
        Maps each epitope to the color used for plotting. Either a dict keyed
        by each epitope, or an array of colors that are sequentially assigned
        to the epitopes.

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
    epitope_colors : dict
        Maps each epitope to its color.

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
                 epitope_colors=tuple(c for c in
                                      matplotlib.colors.TABLEAU_COLORS.values()
                                      if c != '#7f7f7f'),
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
        if isinstance(epitope_colors, dict):
            self.epitope_colors = {epitope_colors[e] for e in self.epitopes}
        elif len(epitope_colors) < len(self.epitopes):
            raise ValueError('not enough `epitope_colors`')
        else:
            self.epitope_colors = dict(zip(self.epitopes, epitope_colors))

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

    def activity_wt_barplot(self,
                            *,
                            epitopes=None,
                            width=110,
                            height_per_bar=25,
                            ):
        r"""Bar plot of activity against each epitope, :math:`a_{\rm{wt},e}`.

        Parameters
        ----------
        epitopes : array-like or None
            Include these epitopes in this order. If `None`, use all epitopes.
        width : float
            Width of plot.
        height_per_bar : float
            Height of plot for each bar (epitope).

        Returns
        -------
        altair.Chart
            Interactive plot.

        """
        if epitopes is None:
            epitopes = self.epitopes
        elif not set(epitopes).issubset(set(self.epitopes)):
            raise ValueError('invalid entries in `epitopes`')
        df = (self.activity_wt_df
              .query('epitope in @epitopes')
              .assign(epitope=lambda x: pd.Categorical(x['epitope'],
                                                       epitopes,
                                                       ordered=True)
                      )
              .sort_values('epitope')
              )

        barplot = (
            alt.Chart(df)
            .encode(x='activity:Q',
                    y='epitope:N',
                    color=alt.Color(
                       'epitope:N',
                       scale=alt.Scale(domain=epitopes,
                                       range=[self.epitope_colors[e]
                                              for e in epitopes]),
                       legend=None,
                       ),
                    tooltip=[alt.Tooltip('epitope:N'),
                             alt.Tooltip('activity:Q', format='.3g')],
                    )
            .mark_bar(size=0.75 * height_per_bar)
            .properties(width=width,
                        height={'step': height_per_bar})
            .configure_axis(grid=False)
            )

        return barplot

    def mut_escape_lineplot(self,
                            *,
                            epitopes=None,
                            all_sites=True,
                            share_ylims=True,
                            height=100,
                            width=900,
                            ):
        r"""Line plots of mutation escape :math:`\beta_{m,e}` at each site.

        Parameters
        -----------
        epitopes : array-like or None
            Make plots for these epitopes. If `None`, use all epitopes.
        all_sites : bool
            Plot all sites in range from first to last site even if some
            have no data.
        share_ylims : bool
            Should plots for all epitopes share same y-limits?
        height : float
            Height per facet.
        width : float
            Width of plot.

        Returns
        -------
        altair.Chart
            Interactive plot.

        """
        if epitopes is None:
            epitopes = self.epitopes
        elif not set(epitopes).issubset(set(self.epitopes)):
            raise ValueError('invalid entries in `epitopes`')
        df = self.mut_escape_df.query('epitope in @epitopes')

        if all_sites:
            sites = list(range(min(self.sites), max(self.sites) + 1))
        else:
            sites = self.sites
            assert set(sites) == set(df['site'])

        escape_metrics = {
                'mean': pd.NamedAgg('escape', 'mean'),
                'total positive': pd.NamedAgg('escape_gt_0', 'sum'),
                'max': pd.NamedAgg('escape', 'max'),
                'min': pd.NamedAgg('escape', 'min'),
                'total negative': pd.NamedAgg('escape_lt_0', 'sum'),
                }
        df = (df
              [['epitope', 'site', 'escape']]
              .assign(escape_gt_0=lambda x: x['escape'].clip(lower=0),
                      escape_lt_0=lambda x: x['escape'].clip(upper=0),
                      )
              .groupby(['epitope', 'site'], as_index=False)
              .aggregate(**escape_metrics)
              .merge(pd.DataFrame(itertools.product(sites, epitopes),
                                  columns=['site', 'epitope']),
                     on=['site', 'epitope'], how='right')
              .sort_values(['epitope', 'site'])
              .melt(id_vars=['epitope', 'site'],
                    var_name='metric',
                    value_name='escape'
                    )
              .pivot_table(index=['site', 'metric'],
                           values='escape',
                           columns='epitope',
                           dropna=False)
              .reset_index()
              .assign(wildtype=lambda x: x['site'].map(self.wts))
              )

        y_axis_dropdown = alt.binding_select(options=list(escape_metrics))
        y_axis_selection = alt.selection_single(fields=['metric'],
                                                bind=y_axis_dropdown,
                                                name='escape',
                                                init={'metric': 'mean'})

        zoom_brush = alt.selection_interval(encodings=['x'],
                                            mark=alt.BrushConfig(
                                                stroke='black',
                                                strokeWidth=2),
                                            )
        zoom_bar = (alt.Chart(df)
                    .mark_rect(color='gray')
                    .encode(x='site:O')
                    .add_selection(zoom_brush)
                    .properties(width=width, height=15, title='site zoom bar')
                    )

        site_selector = alt.selection(type='single',
                                      on='mouseover',
                                      fields=['site'],
                                      empty='none')

        charts = []
        for epitope in epitopes:
            base = (
                alt.Chart(df)
                .encode(x=alt.X('site:O',
                                title=('site' if epitope == epitopes[-1]
                                       else None),
                                axis=(alt.Axis() if epitope == epitopes[-1]
                                      else None),
                                ),
                        y=alt.Y(epitope,
                                type='quantitative',
                                title='escape',
                                scale=alt.Scale(),
                                ),
                        tooltip=[alt.Tooltip('site:O'),
                                 alt.Tooltip('wildtype:N'),
                                 *[alt.Tooltip(f"{epitope}:Q", format='.3g')
                                   for epitope in epitopes]
                                 ]
                        )
                )
            # in case some sites missing values, background thin transparent
            # over which we put darker foreground for measured points
            background = (
                base
                .transform_filter(f"isValid(datum['{epitope}'])")
                .mark_line(opacity=0.5, size=1,
                           color=self.epitope_colors[epitope])
                )
            foreground = (
                base
                .mark_line(opacity=1, size=1.5,
                           color=self.epitope_colors[epitope])
                )
            foreground_circles = (
                base
                .mark_circle(opacity=1,
                             color=self.epitope_colors[epitope])
                .encode(size=alt.condition(site_selector,
                                           alt.value(75),
                                           alt.value(25)),
                        stroke=alt.condition(site_selector,
                                             alt.value('black'),
                                             alt.value(None)),
                        )
                .add_selection(site_selector)
                )
            charts.append((background + foreground + foreground_circles)
                          .add_selection(y_axis_selection)
                          .transform_filter(y_axis_selection)
                          .transform_filter(zoom_brush)
                          .properties(
                                title=alt.TitleParams(
                                          f"{epitope} epitope",
                                          color=self.epitope_colors[epitope]),
                                width=width,
                                height=height)
                          )

        return (alt.vconcat(zoom_bar,
                            (alt.vconcat(*charts, spacing=10)
                             .resolve_scale(y='shared' if share_ylims
                                            else 'independent')
                             ),
                            spacing=10)
                .configure_axis(grid=False)
                .configure_title(anchor='start', fontSize=14)
                )

    def mut_escape_heatmap(self,
                           *,
                           epitopes=None,
                           alphabet=None,
                           all_sites=True,
                           all_alphabet=True,
                           floor_color_at_zero=True,
                           share_heatmap_lims=True,
                           cell_size=13,
                           ):
        r"""Heatmaps of the mutation escape values, :math:`\beta_{m,e}`.

        Parameters
        ----------
        epitopes : array-like or None
            Make plots for these epitopes. If `None`, use all epitopes.
        alphabet : array-like or None
            Order to plot alphabet letters (e.g., amino acids). If `None`, same
            order as `alphabet` used to initialize this `Polyclonal` object.
        all_sites : bool
            Plot all sites in range from first to last site even if some
            have no data.
        all_alphabet : bool
            Plot all letters in the alphabet (e.g., amino acids) even if some
            have no data.
        floor_color_at_zero : bool
            Set lower limit to color scale as zero, even if there are negative
            values or if minimum is >0.
        share_heatmap_lims : bool
            If `True`, let all epitopes share the same limits in color scale.
            If `False`, scale each epitopes colors to the min and max escape
            values for that epitope.
        cell_size : float
            Size of cells in heatmap.

        Returns
        -------
        altair.Chart
            Interactive heat maps.

        """
        if epitopes is None:
            epitopes = self.epitopes
        elif not set(epitopes).issubset(set(self.epitopes)):
            raise ValueError('invalid entries in `epitopes`')
        df = self.mut_escape_df.query('epitope in @epitopes')

        # get alphabet and sites, expanding to all if needed
        if alphabet is None:
            alphabet = self.alphabet
        elif set(alphabet) != set(self.alphabet):
            raise ValueError('`alphabet` and `Polyclonal.alphabet` do not '
                             'have same characters')
        if not all_alphabet:
            alphabet = [c for c in alphabet if c in set(df['mutant']) +
                        set(df['wildtype'])]
        if all_sites:
            sites = list(range(min(self.sites), max(self.sites) + 1))
        else:
            sites = self.sites
            assert set(sites) == set(df['site'])
        df = (df
              [['epitope', 'site', 'mutant', 'escape']]
              .pivot_table(index=['site', 'mutant'],
                           values='escape',
                           columns='epitope')
              .reset_index()
              .merge(pd.DataFrame(itertools.product(sites, alphabet),
                                  columns=['site', 'mutant']),
                     how='right')
              .assign(wildtype=lambda x: x['site'].map(self.wts),
                      mutation=lambda x: (x['wildtype'].fillna('') +
                                          x['site'].astype(str) + x['mutant']),
                      mutant=lambda x: pd.Categorical(x['mutant'], alphabet,
                                                      ordered=True),
                      # mark wildtype cells with a `x`
                      wildtype_char=lambda x: (x['mutant'] == x['wildtype']
                                               ).map({True: 'x', False: ''}),
                      )
              .sort_values(['site', 'mutant'])
              )
        # wildtype has escape of 0 by definition
        for epitope in epitopes:
            df[epitope] = df[epitope].where(df['mutant'] != df['wildtype'], 0)

        # zoom bar to put at top
        zoom_brush = alt.selection_interval(encodings=['x'],
                                            mark=alt.BrushConfig(
                                                        stroke='black',
                                                        strokeWidth=2)
                                            )
        zoom_bar = (alt.Chart(df)
                    .mark_rect(color='gray')
                    .encode(x='site:O')
                    .add_selection(zoom_brush)
                    .properties(width=800, height=15, title='site zoom bar')
                    )

        # select cells
        cell_selector = alt.selection_single(on='mouseover',
                                             empty='none')

        # make list of heatmaps for each epitope
        charts = [zoom_bar]
        for epitope in epitopes:
            # base chart
            base = (alt.Chart(df)
                    .encode(x=alt.X('site:O'),
                            y=alt.Y('mutant:O',
                                    sort=alt.EncodingSortField(
                                                'y',
                                                order='ascending')
                                    ),
                            )
                    )
            # heatmap for cells with data
            if share_heatmap_lims:
                vals = df[list(epitopes)].values
            else:
                vals = df[epitope].values
            escape_max = numpy.nanmax(vals)
            if floor_color_at_zero:
                escape_min = 0
            else:
                escape_min = numpy.nanmin(vals)
            if not (escape_min < escape_max):
                raise ValueError('escape min / max do not span a valid range')
            heatmap = (base
                       .mark_rect()
                       .encode(
                           color=alt.Color(
                                epitope,
                                type='quantitative',
                                scale=alt.Scale(
                                   range=dms_variants.utils.color_gradient_hex(
                                    'white', self.epitope_colors[epitope], 10),
                                   type='linear',
                                   domain=(escape_min, escape_max),
                                   clamp=True,
                                   ),
                                legend=alt.Legend(orient='left',
                                                  title='gray is n.d.',
                                                  titleFontWeight='normal',
                                                  gradientLength=100,
                                                  gradientStrokeColor='black',
                                                  gradientStrokeWidth=0.5)
                                ),
                           stroke=alt.value('black'),
                           strokeWidth=alt.condition(cell_selector,
                                                     alt.value(2.5),
                                                     alt.value(0.2)),
                           tooltip=[alt.Tooltip('mutation:N')] +
                                   [alt.Tooltip(f"{epitope}:Q", format='.3g')
                                    for epitope in epitopes],
                           )
                       )
            # nulls for cells with missing data
            nulls = (base
                     .mark_rect()
                     .transform_filter(f"!isValid(datum['{epitope}'])")
                     .mark_rect(opacity=0.25)
                     .encode(alt.Color('escape:N',
                                       scale=alt.Scale(scheme='greys'),
                                       legend=None),
                             )
                     )
            # mark wildtype cells
            wildtype = (base
                        .mark_text(color='black')
                        .encode(text=alt.Text('wildtype_char:N'))
                        )
            # combine the elements
            charts.append((heatmap + nulls + wildtype)
                          .interactive()
                          .add_selection(cell_selector)
                          .transform_filter(zoom_brush)
                          .properties(
                                title=alt.TitleParams(
                                        f"{epitope} epitope",
                                        color=self.epitope_colors[epitope]),
                                width={'step': cell_size},
                                height={'step': cell_size})
                          )

        return (alt.vconcat(*charts,
                            spacing=0,
                            )
                .configure_title(anchor='start', fontSize=14)
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
