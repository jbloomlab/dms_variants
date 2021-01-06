#source https://jbloomlab.github.io/dms_variants/codonvariant_sim_data.html
#python test.py
#output Wildtype gene of 30 codons: GCTAACCAAATCGTAGGCTGCACCCGCAACATCCTGAACATAGCTGACATCGATTATAAATATGGGCCAAGCTTCCCAACCACCTCCGCA
import collections
import itertools
import random
import tempfile
import time
import warnings
import pandas as pd
from plotnine import *
import scipy
import dmslogo  # used for preference logo plots
import dms_variants.binarymap
import dms_variants.codonvarianttable
import dms_variants.globalepistasis
import dms_variants.plotnine_themes
import dms_variants.simulate
from dms_variants.constants import CBPALETTE, CODONS_NOSTOP
seed = 42  # random number seed
genelength = 30  # gene length in codons
libs = ['lib_1', 'lib_2']  # distinct libraries of gene
variants_per_lib = 500 * genelength  # variants per library
avgmuts = 2.0  # average codon mutations per variant
bclen = 16  # length of nucleotide barcode for each variant
variant_error_rate = 0.005  # rate at which variant sequence mis-called
avgdepth_per_variant = 200  # average per-variant sequencing depth
lib_uniformity = 5  # uniformity of library pre-selection
noise = 0.02  # random noise in selections
bottlenecks = {  # bottlenecks from pre- to post-selection
        'tight_bottle': variants_per_lib * 5,
        'loose_bottle': variants_per_lib * 100,
        }
random.seed(seed)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 500)
warnings.simplefilter('ignore')
theme_set(dms_variants.plotnine_themes.theme_graygrid())
geneseq = ''.join(random.choices(CODONS_NOSTOP, k=genelength))
print(f"Wildtype gene of {genelength} codons:\n{geneseq}")
