import matplotlib.pyplot as plt
import numpy
import pandas as pd
from dms_variants.bottlenecks import estimateBottleneck

numpy.random.seed(1)  # seed for reproducible output

nvariants = 100000  # number of variants in library
depth = nvariants * 100  # sequencing depth 100X library size

# Initial counts are multinomial draw from Dirichlet-distributed freqs:

freqs_pre = numpy.random.dirichlet(numpy.full(nvariants, 2))
n_pre = numpy.random.multinomial(depth, freqs_pre)

# Create data frame with pre-selection counts and plot distribution:

df = pd.DataFrame({'n_pre': n_pre})
_ = df['n_pre'].plot.hist(bins=40,
                          title='pre-selection counts/variant')

# Simulate counts after bottlenecks of various sizes, simulated
# as re-normalized multinomial draws of bottleneck size used to
# to parameterize new multinomial draws of sequencing counts. Then
# estimate the bottlenecks on the simulated data and compare to actual
# value:

estimates = []
for n_per_variant in [0.5, 2, 10, 100]:
    n_bottle = numpy.random.multinomial(
                                   int(n_per_variant * nvariants),
                                   n_pre / n_pre.sum())
    freqs_bottle = n_bottle / n_bottle.sum()
    n_post = numpy.random.multinomial(depth, freqs_bottle)
    df['n_post'] = n_post
    _ = plt.figure()
    _ = df['n_post'].plot.hist(
                      bins=40,
                      title=f"post-selection, {n_per_variant:.1g}")
    n_per_variant_est = estimateBottleneck(df)
    estimates.append((n_per_variant, n_per_variant_est))
estimates_df = pd.DataFrame.from_records(
                                estimates,
                                columns=['actual', 'estimated'])
estimates_df  # doctest: +SKIP

# Confirm that estimates are good when bottleneck is small:

numpy.allclose(
        estimates_df.query('actual <= 10')['actual'],
        estimates_df.query('actual <= 10')['estimated'],
        rtol=0.1)
# True
