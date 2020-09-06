"""
============
constants
============

Defines constants used by package.

"""


import Bio.Data.IUPACData
import Bio.Seq


CBPALETTE = ('#999999', '#E69F00', '#56B4E9', '#009E73',
             '#F0E442', '#0072B2', '#D55E00', '#CC79A7')
"""tuple: Color-blind safe palette.

From http://bconnelly.net/2013/10/creating-colorblind-friendly-figures/
"""

AAS_NOSTOP = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
"""tuple: Amino-acid one-letter codes alphabetized, doesn't include stop."""

AAS_WITHSTOP = tuple(list(AAS_NOSTOP) + ['*'])
"""tuple: Amino-acid one-letter codes alphabetized plus stop as ``*``."""

NTS = ('A', 'C', 'G', 'T')
"""tuple: DNA nucleotide one-letter codes."""

NTS_AMBIGUOUS = ('A', 'B', 'C', 'D', 'G', 'H', 'K', 'M', 'N',
                 'R', 'S', 'T', 'V', 'W', 'Y')
"""tuple: DNA nucleotide one-letter codes including ambiguous ones."""

NT_COMPLEMENT = {_nt: str(Bio.Seq.Seq(_nt).reverse_complement()) for
                 _nt in NTS_AMBIGUOUS}
"""dict: Maps each nucleotide to its complement, including ambiguous ones."""

NT_TO_REGEXP = dict(map(lambda tup: ((tup[0], tup[1]) if len(tup[1]) == 1 else
                                     (tup[0], '[' + ''.join(tup[1]) + ']')),
                        Bio.Data.IUPACData.ambiguous_dna_values.items()
                        ))
"""dict: Maps nucleotide code to regular expression expansion."""

CODONS = tuple(f"{_n1}{_n2}{_n3}" for _n1 in NTS for _n2 in NTS for _n3 in NTS)
"""tuple: DNA codons, alphabetized."""

CODON_TO_AA = {_c: str(Bio.Seq.Seq(_c).translate()) for _c in CODONS}
"""dict: Maps codons to amino acids."""

AA_TO_CODONS = {_aa: [_c for _c in CODONS if CODON_TO_AA[_c] == _aa]
                for _aa in AAS_WITHSTOP}
"""dict: Reverse translate amino acid to list of encoding codons."""

CODONS_NOSTOP = tuple(_c for _c in CODONS if CODON_TO_AA[_c] != '*')
"""tuple: DNA codons except for stop codons, alphabetized."""

SINGLE_NT_AA_MUTS = {_c: {CODON_TO_AA[_c[: _i] + _nt + _c[_i + 1:]]
                          for _i in range(3) for _nt in NTS
                          if (CODON_TO_AA[_c[: _i] + _nt + _c[_i + 1:]] !=
                              CODON_TO_AA[_c])
                          }
                     for _c in CODONS
                     }
"""dict: Maps codons to all amino-acids accessible single-nucleotide change."""
