import functools
import itertools
import numpy
import pandas as pd
import scipy.optimize
from dms_variants.ispline import Msplines

order = 3
mesh = [0.0, 0.3, 0.5, 0.6, 1.0]
x = numpy.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
msplines = Msplines(order, mesh, x)
msplines.order
# 3
msplines.mesh
# array([0. , 0.3, 0.5, 0.6, 1. ])
msplines.n
# 6
msplines.knots
# array([0. , 0. , 0. , 0.3, 0.5, 0.6, 1. , 1. , 1. ])
msplines.lower
# 0.0
msplines.upper
# 1.0

# Evaluate the M-splines at some selected points:

for i in range(1, msplines.n + 1):
    print(f"M{i}: {numpy.round(msplines.M(i), 2)}")
# doctest: +NORMALIZE_WHITESPACE
# M1: [10. 1.11 0.  0.   0.   0.  ]
# M2: [0.  3.73 2.4 0.6  0.   0.  ]
# M3: [0.  1.33 3.  3.67 0.   0.  ]
# M4: [0.  0.   0.  0.71 0.86 0.  ]
# M5: [0.  0.   0.  0.   3.3  0.  ]
# M6: [0.  0.   0.  0.   1.88 7.5 ]

# Check that the gradients are correct:

for i, xval in itertools.product(range(1, msplines.n + 1), x):
    xval = numpy.array([xval])
    def func(xval):
        return Msplines(order, mesh, xval).M(i)
    def dfunc(xval):
        return Msplines(order, mesh, xval).dM_dx(i)
    err = scipy.optimize.check_grad(func, dfunc, xval)
    if err > 1e-5:
        raise ValueError(f"excess err {err} for {i}, {xval}")

# Plot the M-splines in in Fig. 1 of `Ramsay (1988)`_:

xplot = numpy.linspace(0, 1, 1000, endpoint=False)
msplines_plot = Msplines(order, mesh, xplot)
data = {'x': xplot}
for i in range(1, msplines_plot.n + 1):
    data[f"M{i}"] = msplines_plot.M(i)
df = pd.DataFrame(data)
_ = df.plot(x='x')
