"""
===============
plotnine_themes
===============

Defines `plotnine <https://plotnine.readthedocs.io>`_ themes.

See here for details on how to use the themes for plotting:
https://plotnine.readthedocs.io/en/stable/generated/plotnine.themes.theme.html

"""


import plotnine as p9


class theme_graygrid(p9.themes.theme_matplotlib):
    """Plot theme with a light gray grid and axes.

    Example
    -------
    .. plot::

       You can set this theme using the `plotnine.theme_set` command:

       >>> import pandas as pd
       >>> from plotnine import *
       >>> from dms_variants.plotnine_themes import theme_graygrid
       >>> df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 4, 9, 16]})
       >>> theme_set(theme_graygrid())
       >>> p = (ggplot(df, aes('x', 'y')) +
       ...      geom_point(size=3, color='orange') +
       ...      theme(figure_size=(2, 2))
       ...      )
       >>> _ = p.draw()

       Get rid of the vertical grid lines:

       >>> p_novertgrid = p + theme(panel_grid_major_x=element_blank())
       >>> _ = p_novertgrid.draw()

       Get rid of the axes border:

       >>> p_noborder = p + theme(panel_border=element_blank())
       >>> _ = p_noborder.draw()

    """

    def __init__(self, *args, **kwargs):
        """See main class docstring."""
        p9.theme_matplotlib.__init__(self, *args, **kwargs)

        gray = '#D9D9D9'  # gray used in themes.theme_matplotlib

        self.add_theme(
            p9.theme(
                panel_border=p9.element_rect(color=gray, size=0.7),
                axis_line=p9.element_blank(),
                axis_ticks_length=0,
                axis_ticks=p9.element_blank(),
                panel_grid_major=p9.element_line(color=gray, size=0.7),
                panel_grid_minor=p9.element_blank(),
                panel_ontop=True,  # plot panel on top of grid
                ),
            inplace=True)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
