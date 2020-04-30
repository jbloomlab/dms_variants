"""Tests counting codon variants.

Runs on a snippet of real data for RecA.

This test doesn't test for correct output, but rather that output is the
same as a version of the code that we are confident is correct. Therefore,
if later changes break this code, this test will identify that fact.

Essentially, it tests `CodonVariantTable` and `IlluminBarcodeParser`.

Written by Jesse Bloom.
"""


import itertools
import os
import unittest

import Bio.SeqIO

import pandas as pd
from pandas.testing import assert_frame_equal

from dms_variants.codonvarianttable import CodonVariantTable
from dms_variants.illuminabarcodeparser import IlluminaBarcodeParser


class test_count_codonvariants(unittest.TestCase):
    """Test counting of codon variants."""

    def test_count_codonvariants(self):
        """Test counting of codon variants."""
        indir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'count_codonvariant_files/')
        outdir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              'test_count_codonvariants_files/')
        os.makedirs(outdir, exist_ok=True)

        barcode_variant_file = os.path.join(indir, 'barcode_variant_table.csv')

        ampliconfile = os.path.join(indir, 'PacBio_amplicon.gb')
        amplicon = Bio.SeqIO.read(ampliconfile, 'genbank')

        gene = [f for f in amplicon.features if f.type == 'gene']
        assert len(gene) == 1, "Failed to find exactly one gene feature"
        geneseq = str(gene[0].location.extract(amplicon).seq)

        variants = CodonVariantTable(
                            barcode_variant_file=barcode_variant_file,
                            geneseq=geneseq)
        variants2 = CodonVariantTable(
                            barcode_variant_file=barcode_variant_file,
                            geneseq=geneseq)

        n_variants_file = os.path.join(indir, 'n_variants.csv')
        assert_frame_equal(
                variants.n_variants_df(samples=None)
                        .assign(library=lambda x: x['library'].astype('str'),
                                sample=lambda x: x['sample'].astype('str')
                                ),
                pd.read_csv(n_variants_file),
                check_dtype=False
                )

        for mut_type in ['aa', 'codon']:
            _ = variants.plotNumMutsHistogram(mut_type, samples=None,
                                              max_muts=7)

        for variant_type in ['all', 'single']:
            _ = variants.plotNumCodonMutsByType(variant_type, samples=None,)

        _ = variants.plotNumCodonMutsByType('all', samples=None, min_support=2)

        for variant_type, mut_type in itertools.product(['all', 'single'],
                                                        ['aa', 'codon']):
            _ = variants.plotCumulMutCoverage(variant_type,
                                              mut_type,
                                              samples=None)

        for variant_type, mut_type in itertools.product(['all', 'single'],
                                                        ['aa', 'codon']):
            _ = variants.plotMutFreqs(variant_type, mut_type, samples=None)

        for mut_type in ['aa', 'codon']:
            _ = variants.plotMutHeatmap('all', mut_type,
                                        samples=None, libraries='all_only',
                                        widthscale=2)

        samples = (pd.DataFrame({
                        'library': ['library-1', 'library-2', 'library-1'],
                        'sample': ['plasmid', 'plasmid', 'uninduced'],
                        })
                   .assign(
                        run='run-1',
                        upstream='TTTTTAAGTTGTAAGGATATGCCATTCTAGA',
                        downstream='',
                        R1file=lambda x: (indir + x['library'] + '_' +
                                          x['sample'] + '_R1.fastq'),
                        R2file=lambda x: (indir + x['library'] + '_' +
                                          x['sample'] + '_R2.fastq'),
                        )
                   )

        barcode = [f for f in amplicon.features if f.type == 'barcode']
        assert len(barcode) == 1, "Failed to find exactly one barcode feature"
        bclen = len(barcode[0])

        fates = []
        counts_df = []
        for (lib, sample), runs in samples.groupby(['library', 'sample']):

            # read barcodes for all runs for library / sample
            barcodes = []
            for run_tup in runs.itertuples(index=False):
                parser = IlluminaBarcodeParser(
                            bclen=bclen,
                            upstream=run_tup.upstream,
                            downstream=run_tup.downstream,
                            valid_barcodes=variants.valid_barcodes(lib),
                            upstream_mismatch=3,
                            downstream_mismatch=3)
                run_barcodes, run_fates = parser.parse(
                                    r1files=run_tup.R1file,
                                    r2files=None)
                barcodes.append(run_barcodes)
                fates.append(run_fates.assign(library=lib,
                                              sample=sample,
                                              run=run_tup.run
                                              ))

            # combine barcodes read for each run
            barcodes = (pd.concat(barcodes, ignore_index=True, sort=False)
                        .groupby('barcode')
                        .aggregate({'count': 'sum'})
                        .reset_index()
                        )

            counts_df.append(barcodes.assign(library=lib, sample=sample))

            # add barcode counts to variant table
            variants.addSampleCounts(lib, sample, barcodes)

        # make sure adding by `add_sample_counts_df` is same
        counts_df = pd.concat(counts_df)
        variants2.add_sample_counts_df(counts_df)
        assert_frame_equal(variants.variant_count_df,
                           variants2.variant_count_df,
                           check_like=True)

        # concatenate read fates into one data frame
        fates = (pd.concat(fates, ignore_index=True, sort=False)
                 .assign(count=lambda x: x['count'].fillna(0).astype('int'))
                 )

        fatesfile = os.path.join(indir, 'fates.csv')
        assert_frame_equal(
                fates,
                pd.read_csv(fatesfile)
                )

        libs_to_analyze = ['library-1']

        for mut_type in ['aa', 'codon']:
            _ = variants.plotNumMutsHistogram(mut_type,
                                              libraries=libs_to_analyze,
                                              max_muts=7, orientation='v')

        for variant_type in ['all', 'single']:
            _ = variants.plotNumCodonMutsByType(variant_type,
                                                libraries=libs_to_analyze,
                                                orientation='v')

        for variant_type, mut_type in itertools.product(['all', 'single'],
                                                        ['aa', 'codon']):
            _ = variants.plotCumulMutCoverage(variant_type, mut_type,
                                              libraries=libs_to_analyze,
                                              orientation='v')

        for variant_type, mut_type in itertools.product(['all', 'single'],
                                                        ['aa', 'codon']):
            _ = variants.plotMutFreqs(variant_type, mut_type,
                                      libraries=libs_to_analyze,
                                      orientation='v')

        for mut_type in ['aa', 'codon']:
            _ = variants.plotMutHeatmap('all', mut_type,
                                        libraries=libs_to_analyze,
                                        widthscale=2)

        variant_count_file = os.path.join(indir, 'variant_counts.csv')
        assert_frame_equal(
            variants.variant_count_df
                    .assign(library=lambda x: x['library'].astype(str),
                            sample=lambda x: x['sample'].astype(str)
                            ),
            pd.read_csv(variant_count_file).fillna('')
            )

        countfiles = []
        for mut_type in ['single', 'all']:
            countsdir = os.path.join(outdir, f'{mut_type}_mutant_codoncounts')
            mut_type_countfiles = variants.writeCodonCounts(mut_type,
                                                            outdir=countsdir)
            countfiles.append(mut_type_countfiles
                              .assign(mutant_type=f'{mut_type} mutants')
                              )


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    unittest.main(testRunner=runner)
