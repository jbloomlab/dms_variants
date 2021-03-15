# Data used by `polyclonal.ipynb` notebook

It considers antibodies targeting four "epitopes" on the SARS-CoV-2 RBD using the classification scheme of [Barnes et al (2020)](https://www.nature.com/articles/s41586-020-2852-1):
 - *LY-CoV016*: a "class 1" epitope
 - *LY-CoV555*: a "class 2" epitope
 - *REGN10987*: a "class 3" epitope
 - *CR3022*: a "class 4" epitope

The file [mutation_escape_fractions.csv](mutation_escape_fractions.csv) contains the mutation-level escape fractions for each antibody measured using deep mutational scanning in the following papers, only including mutations for which measurements are available for all four antibodies:
  - *LY-CoV016* and *REGN10987*: [Starr et al (2021), Science](https://science.sciencemag.org/content/371/6531/850)
  - *LY-CoV555*: [Starr et al (2021), bioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.17.431683v1)
  - *CR3022*: [Greaney et al (2021), Cell Host & Microbe](https://www.sciencedirect.com/science/article/pii/S1931312820306247), but re-analyzed with the same expression and ACE2-binding cutoffs in [Starr et al (2021), Science](https://science.sciencemag.org/content/371/6531/850).

The file [RBD_seq.fasta](RBD_seq.fasta) is the coding sequence of the RBD used in the Bloom lab deep mutational scanning (optimized for yeast display).

The directory also contains [6M0J.pdb](6M0J.pdb), which is just a downloaded version of [PDB 6m0j](https://www.rcsb.org/structure/6M0J), which has the RBD in complex with ACE2.
