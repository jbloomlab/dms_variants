"""
======================
bottleneck_likelihood
======================

Module for computing likelihoods of observing pre- and post-selection
counts when there are bottlenecks in the experiment.

"""


import mpmath

import scipy.special


def continuous_poisson(x, lamb):
    """Continuous Poisson distribution.

    Note
    ----
    Calculated according to formula provided by `Abid and Mohammed (2016)`_


    .. _`Abid and Mohammed (2016)`: http://pubs.sciepub.com/ijdeaor/2/1/2/ 

    """
    if x <= 0:
       raise ValueError('`x` must be > 0')
    if lamb <= 0:
        raise ValueError('`lamb` must be > 0')

    Fx = scipy.special.gammaincc(x, lamb)  # Eq 2 of Abid and Mohammed (2016)

    phi_x = scipy.special.digamma(x)  # digamma function

    T_3_x_lamb = mpmath.meijerg(?, ?, x)  # Meijer G function

    return Fx * (scipy.log(lamb) - phi_x) + lamb * T_3_x_lamb / gamma_x   

   


if __name__ == '__main__':
    import doctest
    doctest.testmod()
