import itertools
import numpy
import pandas as pd
import scipy.optimize
from dms_variants.ispline import Isplines_total

order = 3
mesh = [0.0, 0.3, 0.5, 0.6, 1.0]
x = numpy.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
isplines_total = Isplines_total(order, mesh, x)
weights = numpy.array([1.2, 2, 1.2, 1.2, 3, 0]) / 6
numpy.round(isplines_total.Itotal(weights, w_lower=0), 2)
# array([0.  , 0.38, 0.54, 0.66, 1.21, 1.43])

# Now calculate using some points that require linear extrapolation
# outside the mesh and also have a nonzero `w_lower`:

x2 = numpy.array([-0.5, -0.25, 0, 0.01, 1.0, 1.5])
isplines_total2 = Isplines_total(order, mesh, x2)
numpy.round(isplines_total2.Itotal(weights, w_lower=1), 3)
# array([0.   , 0.5  , 1.   , 1.02 , 2.433, 2.433])

# Test :meth:`Isplines_total.dItotal_dx`:

x_deriv = numpy.array([-0.5, -0.25, 0, 0.01, 0.5, 0.7, 1.0, 1.5])
for xval in x_deriv:
    xval = numpy.array([xval])
    def func(xval):
        return Isplines_total(order, mesh, xval).Itotal(weights, 0)
    def dfunc(xval):
        return Isplines_total(order, mesh, xval).dItotal_dx(weights)
    err = scipy.optimize.check_grad(func, dfunc, xval)
    if err > 1e-5:
        raise ValueError(f"excess err {err} for {xval}")

(isplines_total.dItotal_dw_lower() == numpy.ones(x.shape)).all()
# True

# Test :meth:`Isplines_total.dItotal_dweights`:

isplines_total3 = Isplines_total(order, mesh, x_deriv)
wl = 1.5
(isplines_total3.dItotal_dweights(weights, wl).shape ==
 (len(x_deriv), len(weights)))
# True
weightslist = list(weights)
for ix, iw in itertools.product(range(len(x_deriv)),
                                range(len(weights))):
    w = numpy.array([weightslist[iw]])
    def func(w):
        iweights = numpy.array(weightslist[: iw] +
                               list(w) +
                               weightslist[iw + 1:])
        return isplines_total3.Itotal(iweights, wl)[ix]
    def dfunc(w):
        iweights = numpy.array(weightslist[: iw] +
                               list(w) +
                               weightslist[iw + 1:])
        return isplines_total3.dItotal_dweights(iweights, wl)[ix,
                                                              iw]
    err = scipy.optimize.check_grad(func, dfunc, w)
    if err > 1e-6:
        raise ValueError(f"excess err {err} for {ix, iw}")

# Plot the total of the I-spline family shown in Fig. 1 of
# `Ramsay (1988)`_, adding some linear extrapolation outside the
# mesh range:

xplot = numpy.linspace(-0.2, 1.2, 1000)
isplines_totalplot = Isplines_total(order, mesh, xplot)
df = pd.DataFrame({'x': xplot,
                   'Itotal': isplines_totalplot.Itotal(weights, 0)})
_ = df.plot(x='x', y='Itotal')
