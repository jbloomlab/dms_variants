"""
=============
barcodeparser
=============

Parse barcode reads.

"""

import collections

import pandas as pd

import regex

import scipy

from dms_variants.utils import reverse_complement


class IlluminaBarcodeParser:
    """Parser for Illumina barcodes.

    Note
    ----
    Barcodes should be read by R1 and optionally R2. Expected arrangement is

        5'-[R2_start]-upstream-barcode-downstream-[R1_start]-3'

    R1 anneals downstream of barcode and reads backwards. If R2 is used,
    it anneals upstream of barcode and reads forward. There can be sequences
    (`upstream` and `downstream`) on either side of the barcode: `downstream`
    must fully cover region between R1 start and barcode, and if using R2
    then `upstream` must fully cover region between R2 start and barcode.
    However, it is fine if R1 reads backwards past `upstream`, and if `R2`
    reads forward past `downstream`.

    Parameters
    ----------
    bclen : int or `None`
        Barcode length; `None` if length determined from `valid_barcodes`.
    upstream : str
        Sequence upstream of the barcode; empty str if no such sequence.
    downstream : str
        Sequence downstream of barcode; empty str if no such sequence.
    upstream_mismatch : int
        Max number of mismatches allowed in `upstream`.
    downstream_mismatch : int
        Max number of mismatches allowed in `downstream`.
    valid_barcodes : None or iterable such as list
        If not `None`, only retain barcodes listed here. Use if you know
        the possible valid barcodes ahead of time.
    rc_barcode : bool
        Get reverse complement of the barcode (orientation read by R1).
    minq : int
        Require at least this quality score for all bases in barcode.
    chastity_filter : bool
        Drop any reads that fail Illumina chastity filter.
    list_all_valid_barcodes :  bool
        If using `valid_barcodes`, barcode sets returned by
        :meth:`IlluminaBarcodeParser.parse` include all valid barcodes
        even if no counts.

    Attributes
    ----------
    bclen : int
        Length of barcodes.
    upstream : str
        Sequence upstream of barcode.
    downstream : str
        Sequence downstream of barcode.
    upstream_mismatch : int
        Max number of mismatches allowed in `upstream`.
    downstream_mismatch : int
        Max number of mismatches allowed in `downstream`.
    valid_barcodes : None or set
        If not `None`, set of barcodes to retain.
    rc_barcode : bool
        Get reverse complement of the barcode (orientation read by R1).
    minq : int
        Require at least this quality score for all bases in barcode.
    chastity_filter : bool
        Drop any reads that fail Illumina chastity filter.
    list_all_valid_barcodes :  bool
        If using `valid_barcodes`, barcode sets returned by
        :meth:`IlluminaBarcodeParser.parse` include all valid barcodes
        even if no counts.

    """

    VALID_NTS = 'ACGTN'
    """str : Valid nucleotide characters in FASTQ files."""

    def __init__(self, *, bclen=None, upstream='', downstream='',
                 upstream_mismatch=0, downstream_mismatch=0,
                 valid_barcodes=None, rc_barcode=True, minq=20,
                 chastity_filter=True, list_all_valid_barcodes=True):
        """See main class doc string."""
        self.bclen = bclen
        if regex.match(f"^[{self.VALID_NTS}]*$", upstream):
            self.upstream = upstream
        else:
            raise ValueError(f"invalid chars in upstream {upstream}")
        if regex.match(f"^[{self.VALID_NTS}]*$", downstream):
            self.downstream = downstream
        else:
            raise ValueError(f"invalid chars in downstream {downstream}")
        self.upstream_mismatch = upstream_mismatch
        self.downstream_mismatch = downstream_mismatch
        self.valid_barcodes = valid_barcodes
        if self.valid_barcodes is not None:
            self.valid_barcodes = set(self.valid_barcodes)
            if len(self.valid_barcodes) < 1:
                raise ValueError('empty list for `valid_barcodes`')
            if self.bclen is None:
                self.bclen = len(list(self.valid_barcodes)[0])
            if any(len(bc) != self.bclen for bc in self.valid_barcodes):
                raise ValueError('`valid_barcodes` not all valid length')
        elif self.bclen is None:
            raise ValueError('must specify `bclen` or `valid_barcodes`')
        self.minq = minq
        self.rc_barcode = rc_barcode
        self.chastity_filter = chastity_filter
        self.list_all_valid_barcodes = list_all_valid_barcodes

        # specify information about R1 / R2 matches
        self._bcend = {
                'R1': self.bclen + len(self.downstream),
                'R2': self.bclen + len(self.upstream)
                }
        self._rcdownstream = reverse_complement(self.downstream)
        self._rcupstream = reverse_complement(self.upstream)
        self._matches = {'R1': {}, 'R2': {}}  # match objects by read length

    def parse(self, r1files, r2files=None):
        """Parse barcodes from files.

        Parameters
        ----------
        r1files : str or list
            Name of R1 FASTQ file, or list of such files. Can be gzipped.
        r2files : None, str, or list
            `None` or empty list if not using R2, or like `r1files` for R2.

        Returns
        -------
        tuple
            The 2-tuple `(barcodes, fates)`, where:
                - `barcodes` is pandas DataFrame giving number of observations
                  of each barcode (columns are "barcode" and "count").
                - `fates` is pandas DataFrame giving total number of reads with
                  each fate (columns "fate" and "count"). Possible fates:
                  - "valid barcode"
                  - "invalid barcode": not in barcode whitelist
                  - "R1 / R2 disagree"
                  - "low quality barcode": sequencing quality low
                  - "unparseable barcode": invalid flank sequence, N in barcode

        """
        if not r2files:
            reads = ['R1']
            r2files = None
        else:
            reads = ['R1', 'R2']

        if self.valid_barcodes and self.list_all_valid_barcodes:
            barcodes = {bc: 0 for bc in self.valid_barcodes}
        else:
            barcodes = collections.defaultdict(int)

        fates = collections.defaultdict(int)

        for _, r1, r2, q1, q2, fail in \
                dms_tools2.utils.iteratePairedFASTQ(r1files, r2files):

            if fail and self.chastity_filter:
                fates['failed chastity filter'] += 1
                continue

            matches = {}
            for read, r in zip(reads, [r1, r2]):
                rlen = len(r)

                # get or build matcher for read of this length
                len_past_bc = rlen - self._bcend[read]
                if len_past_bc < 0:
                    raise ValueError(f"{read} too short: {rlen}")
                elif rlen in self._matches[read]:
                    matcher = self._matches[read][rlen]
                else:
                    if read == 'R1':
                        match_str = (
                                f"^({self._rcdownstream})"
                                f"{{s<={self.downstream_mismatch}}}"
                                f"(?P<bc>[ACTG]{{{self.bclen}}})"
                                f"({self._rcupstream[: len_past_bc]})"
                                f"{{s<={self.upstream_mismatch}}}"
                                )
                    else:
                        assert read == 'R2'
                        match_str = (
                                f"^({self.upstream})"
                                f"{{s<={self.upstream_mismatch}}}"
                                f"(?P<bc>[ACTG]{{{self.bclen}}})"
                                f"({self.downstream[: len_past_bc]})"
                                f"{{s<={self.downstream_mismatch}}}"
                                )
                    matcher = regex.compile(match_str, flags=regex.BESTMATCH)
                    self._matches[read][rlen] = matcher

                m = matcher.match(r)
                if m:
                    matches[read] = m
                else:
                    break

            if len(matches) == len(reads):
                bc = {}
                bc_q = {}
                for read, q in zip(reads, [q1, q2]):
                    bc[read] = matches[read].group('bc')
                    bc_q[read] = scipy.array([
                                 ord(qi) - 33 for qi in
                                 q[matches[read].start('bc'):
                                   matches[read].end('bc')]],
                                 dtype='int')
                if self.rc_barcode and 'R2' in reads:
                    bc['R2'] = reverse_complement(bc['R2'])
                    bc_q['R2'] = scipy.flip(bc_q['R2'], axis=0)
                elif 'R2' in reads:
                    bc['R1'] = reverse_complement(bc['R1'])
                    bc_q['R1'] = scipy.flip(bc_q['R1'], axis=0)
                if len(reads) == 1:
                    if (bc_q['R1'] >= self.minq).all():
                        if self.valid_barcodes and (
                                bc['R1'] not in self.valid_barcodes):
                            fates['invalid barcode'] += 1
                        else:
                            barcodes[bc['R1']] += 1
                            fates['valid barcode'] += 1
                    else:
                        fates['low quality barcode'] += 1
                else:
                    if bc['R1'] == bc['R2']:
                        if self.valid_barcodes and (
                                bc['R1'] not in self.valid_barcodes):
                            fates['invalid barcode'] += 1
                        elif (scipy.maximum(bc_q['R1'], bc_q['R2'])
                                >= self.minq).all():
                            barcodes[bc['R1']] += 1
                            fates['valid barcode'] += 1
                        else:
                            fates['low quality barcode'] += 1
                    else:
                        fates['R1 / R2 disagree'] += 1
            else:
                # invalid flanking sequence or N in barcode
                fates['unparseable barcode'] += 1

        barcodes = (pd.DataFrame(
                        list(barcodes.items()),
                        columns=['barcode', 'count'])
                    .sort_values(['count', 'barcode'],
                                 ascending=[False, True])
                    .reset_index(drop=True)
                    )

        fates = (pd.DataFrame(
                    list(fates.items()),
                    columns=['fate', 'count'])
                 .sort_values(['count', 'fate'],
                              ascending=[False, True])
                 .reset_index(drop=True)
                 )

        return (barcodes, fates)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
