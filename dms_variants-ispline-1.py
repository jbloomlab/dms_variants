import itertools
import numpy
import pandas as pd
import scipy.optimize
from dms_variants.ispline import Isplines

order = 3
mesh = [0.0, 0.3, 0.5, 0.6, 1.0]
x = numpy.array([0, 0.2, 0.3, 0.4, 0.8, 0.99999])
isplines = Isplines(order, mesh, x)
isplines.order
# 3
isplines.mesh
# array([0. , 0.3, 0.5, 0.6, 1. ])
isplines.n
# 6
isplines.lower
# 0.0
isplines.upper
# 1.0

# Evaluate the I-splines at some selected points:

for i in range(1, isplines.n + 1):
    print(f"I{i}: {numpy.round(isplines.I(i), 2)}")
# doctest: +NORMALIZE_WHITESPACE
# I1: [0.   0.96 1.   1.   1.   1.  ]
# I2: [0.   0.52 0.84 0.98 1.   1.  ]
# I3: [0.   0.09 0.3  0.66 1.   1.  ]
# I4: [0.   0.   0.   0.02 0.94 1.  ]
# I5: [0.   0.   0.   0.   0.58 1.  ]
# I6: [0.   0.   0.   0.   0.13 1.  ]

# Check that gradients are correct for :meth:`Isplines.dI_dx`:

for i, xval in itertools.product(range(1, isplines.n + 1), x):
    xval = numpy.array([xval])
    def func(xval):
        return Isplines(order, mesh, xval).I(i)
    def dfunc(xval):
        return Isplines(order, mesh, xval).dI_dx(i)
    err = scipy.optimize.check_grad(func, dfunc, xval)
    if err > 1e-5:
        raise ValueError(f"excess err {err} for {i}, {xval}")

# Plot the I-splines in Fig. 1 of `Ramsay (1988)`_:

xplot = numpy.linspace(0, 1, 1000)
isplines_xplot = Isplines(order, mesh, xplot)
data = {'x': xplot}
for i in range(1, isplines.n + 1):
    data[f"I{i}"] = isplines_xplot.I(i)
df = pd.DataFrame(data)
_ = df.plot(x='x')
