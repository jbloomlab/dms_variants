"""
=====================
illuminabarcodeparser
=====================

Defines :class:`IlluminaBarcodeParser` to parse barcodes from Illumina reads.

"""

import collections

import numpy

import pandas as pd

import regex

from dms_variants.fastq import (
    iterate_fastq,
    iterate_fastq_pair,
    qual_str_to_array,
)
from dms_variants.utils import reverse_complement


class IlluminaBarcodeParser:
    """Parser for Illumina barcodes.

    Note
    ----
    Barcodes should be read by R1 and optionally R2. Expected arrangement is

        5'-[R2_start]-upstream2-upstream-barcode-downstream-downstream2-[R1_start]-3'

    R1 anneals downstream of barcode and reads backwards. If R2 is used,
    it anneals upstream of barcode and reads forward. There can be sequences
    (`upstream` and `downstream`) on either side of the barcode: `downstream`
    must fully cover region between R1 start and barcode, and if using R2
    then `upstream` must fully cover region between R2 start and barcode.
    However, it is fine if R1 reads backwards past `upstream`, and if `R2`
    reads forward past `downstream`. The `upstream2` and `downstream2`
    can be used to require additional flanking sequences. Normally these
    would just be rolled into `upstream` and `downstream`, but you might
    specify separately if you are actually using these to parse additional
    indices that you might want to set different mismatch criteria for.

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
    bc_orientation : {'R1', 'R2'}
        Is the barcode defined in the orientation read by R1 or R2?
    minq : int
        Require >= this Q score for all bases in barcode for at least one read.
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
    upstream2 : str
        Second sequence upstream of barcode.
    downstream : str
        Sequence downstream of barcode.
    downstream2 : str
        Second sequence downstream of barcode
    upstream_mismatch : int
        Max number of mismatches allowed in `upstream`.
    upstream2_mismatch : int
        Max number of mismatches allowed in `upstream2`.
    downstream_mismatch : int
        Max number of mismatches allowed in `downstream`.
    downstream2_mismatch : int
        Max number of mismatches allowed in `downstream2`.
    valid_barcodes : None or set
        If not `None`, set of barcodes to retain.
    bc_orientation : {'R1', 'R2'}
        Is the barcode defined in the orientation read by R1 or R2?
    minq : int
        Require >= this Q score for all bases in barcode for at least one read.
    chastity_filter : bool
        Drop any reads that fail Illumina chastity filter.
    list_all_valid_barcodes :  bool
        If using `valid_barcodes`, barcode sets returned by
        :meth:`IlluminaBarcodeParser.parse` include all valid barcodes
        even if no counts.

    """

    VALID_NTS = "ACGTN"
    """str : Valid nucleotide characters in FASTQ files."""

    def __init__(
        self,
        *,
        bclen=None,
        upstream="",
        upstream2="",
        downstream="",
        downstream2="",
        upstream_mismatch=0,
        upstream2_mismatch=0,
        downstream_mismatch=0,
        downstream2_mismatch=0,
        valid_barcodes=None,
        bc_orientation="R1",
        minq=20,
        chastity_filter=True,
        list_all_valid_barcodes=True,
    ):
        """See main class doc string."""
        self.bclen = bclen
        for param_name, param_val in [
            ("upstream", upstream),
            ("downstream", downstream),
            ("upstream2", upstream2),
            ("downstream2", downstream2),
        ]:
            if regex.match(f"^[{self.VALID_NTS}]*$", param_val):
                setattr(self, param_name, param_val)
            else:
                raise ValueError(f"invalid chars in {param_name} {param_val}")
        self.upstream_mismatch = upstream_mismatch
        self.downstream_mismatch = downstream_mismatch
        self.upstream2_mismatch = upstream2_mismatch
        self.downstream2_mismatch = downstream2_mismatch
        self.valid_barcodes = valid_barcodes
        if self.valid_barcodes is not None:
            self.valid_barcodes = set(self.valid_barcodes)
            if len(self.valid_barcodes) < 1:
                raise ValueError("empty list for `valid_barcodes`")
            if self.bclen is None:
                self.bclen = len(list(self.valid_barcodes)[0])
            if any(len(bc) != self.bclen for bc in self.valid_barcodes):
                raise ValueError("`valid_barcodes` not all valid length")
        elif self.bclen is None:
            raise ValueError("must specify `bclen` or `valid_barcodes`")
        self.minq = minq
        if bc_orientation in {"R1", "R2"}:
            self.bc_orientation = bc_orientation
        else:
            raise ValueError(f"invalid `bc_orientation` {bc_orientation}")
        self.chastity_filter = chastity_filter
        self.list_all_valid_barcodes = list_all_valid_barcodes

        # specify information about R1 / R2 matches
        self._rcdownstream = reverse_complement(self.downstream)
        self._rcupstream = reverse_complement(self.upstream)
        self._rcdownstream2 = reverse_complement(self.downstream2)
        self._rcupstream2 = reverse_complement(self.upstream2)

        # build the regex read matches
        self._matchers = {
            "R1": regex.compile(
                f"({self._rcdownstream2})"
                + f"{{s<={self.downstream2_mismatch}}}"
                + f"({self._rcdownstream})"
                + f"{{s<={self.downstream_mismatch}}}"
                + f"(?P<bc>[ACTG]{{{self.bclen}}})"
                + f"({self._rcupstream})"
                + f"{{s<={self.upstream_mismatch}}}"
                + f"({self._rcupstream2})"
                + f"{{s<={self.upstream2_mismatch}}}"
            ),
            "R2": regex.compile(
                f"({self.upstream2})"
                + f"{{s<={self.upstream2_mismatch}}}"
                + f"({self.upstream})"
                + f"{{s<={self.upstream_mismatch}}}"
                + f"(?P<bc>[ACTG]{{{self.bclen}}})"
                + f"({self.downstream})"
                + f"{{s<={self.downstream_mismatch}}}"
                + f"({self.downstream2})"
                + f"{{s<={self.downstream2_mismatch}}}"
            ),
        }

        # build matchers that do not have upstream2 or downstream2 if needed
        self._has_flank2 = (len(self.upstream2) > 0) or (len(self.downstream2) > 0)
        self._matchers_no_flank2 = {
            "R1": regex.compile(
                f"[{self.VALID_NTS}]{{{len(self.downstream2)}}}"
                + f"({self._rcdownstream})"
                + f"{{s<={self.downstream_mismatch}}}"
                + f"(?P<bc>[ACTG]{{{self.bclen}}})"
                + f"({self._rcupstream})"
                + f"{{s<={self.upstream_mismatch}}}"
                + f"[{self.VALID_NTS}]{{{len(self.upstream2)}}}"
            ),
            "R2": regex.compile(
                f"[{self.VALID_NTS}]{{{len(self.upstream2)}}}"
                + f"^({self.upstream})"
                + f"{{s<={self.upstream_mismatch}}}"
                + f"(?P<bc>[ACTG]{{{self.bclen}}})"
                + f"({self.downstream})"
                + f"{{s<={self.downstream_mismatch}}}"
                + f"[{self.VALID_NTS}]{{{len(self.downstream2)}}}"
            ),
        }

    def parse(self, r1files, *, r2files=None, add_cols=None, outer_flank_fates=False):
        """Parse barcodes from files.

        Parameters
        ----------
        r1files : str or list
            Name of R1 FASTQ file, or list of such files. Can be gzipped.
        r2files : None, str, or list
            `None` or empty list if not using R2, or like `r1files` for R2.
        add_cols : None or dict
            If dict, specify names and values (i.e., sample or library names)
            to be aded to returned data frames.
        outer_flank_fates : bool
            If `True`, if using outer flanking regions then in the output fates
            specify reads that fail just the outer flanking regions (`upstream2` or
            `downstream2`). Otherwise, such failures will be grouped with the
            "unparseable barcode" fate.

        Returns
        -------
        tuple
            The 2-tuple `(barcodes, fates)`, where:
                - `barcodes` is pandas DataFrame giving number of observations
                  of each barcode (columns are "barcode" and "count").
                - `fates` is pandas DataFrame giving total number of reads with
                  each fate (columns "fate" and "count"). Possible fates:
                  - "failed chastity filter"
                  - "valid barcode"
                  - "invalid barcode": not in barcode whitelist
                  - "R1 / R2 disagree" (if using `r2files`)
                  - "low quality barcode": sequencing quality low
                  - "unparseable barcode": invalid flank sequence, N in barcode
                  - "read too short": read is too short to cover specified region
                  - "invalid outer flank" : if using `outer_flank_fates` and
                    `upstream2` or `downstream2` fails.

            Note that these data frames also include any columns specified by
            `add_cols`.

        """
        if isinstance(r1files, str):
            r1files = [r1files]
        if isinstance(r2files, str):
            r2files = [r2files]

        if not r2files:
            reads = ["R1"]
            r2files = None
            fileslist = [r1files]
            r1only = True
        else:
            reads = ["R1", "R2"]
            if len(r1files) != len(r2files):
                raise ValueError("`r1files` and `r2files` different length")
            fileslist = [r1files, r2files]
            r1only = False

        if self.valid_barcodes and self.list_all_valid_barcodes:
            barcodes = {bc: 0 for bc in self.valid_barcodes}
        else:
            barcodes = collections.defaultdict(int)

        fates = {
            "failed chastity filter": 0,
            "unparseable barcode": 0,
            "low quality barcode": 0,
            "invalid barcode": 0,
            "valid barcode": 0,
            "read too short": 0,
        }
        if not r1only:
            fates["R1 / R2 disagree"] = 0
        if outer_flank_fates and self._has_flank2:
            fates["invalid outer flank"] = 0

        # min length of interest for reads
        minlen = (
            self.bclen
            + len(self.upstream)
            + len(self.downstream)
            + len(self.upstream2)
            + len(self.downstream2)
        )

        for filetup in zip(*fileslist):
            if r1only:
                assert len(filetup) == 1
                iterator = iterate_fastq(filetup[0], check_pair=1, trim=minlen)
            else:
                assert len(filetup) == 2, f"{filetup}\n{fileslist}"
                iterator = iterate_fastq_pair(
                    filetup[0], filetup[1], r1trim=minlen, r2trim=minlen
                )

            for entry in iterator:
                if r1only:
                    readlist = [entry[1]]
                    qlist = [entry[2]]
                    fail = entry[3]

                else:
                    readlist = [entry[1], entry[2]]
                    qlist = [entry[3], entry[4]]
                    fail = entry[5]

                if fail and self.chastity_filter:
                    fates["failed chastity filter"] += 1
                    continue

                if any(len(r) < minlen for r in readlist):
                    fates["read too short"] += 1
                    continue

                assert all(len(r) == minlen for r in readlist)

                matches = {
                    read: self._matchers[read].fullmatch(r)
                    for (read, r) in zip(reads, readlist)
                }

                if all(m is not None for m in matches.values()):
                    bc = {}
                    bc_q = {}
                    for read, q in zip(reads, qlist):
                        bc[read] = matches[read].group("bc")
                        bc_q[read] = qual_str_to_array(
                            q[matches[read].start("bc") : matches[read].end("bc")]
                        )
                    if self.bc_orientation == "R1":
                        if not r1only:
                            bc["R2"] = reverse_complement(bc["R2"])
                            bc_q["R2"] = numpy.flip(bc_q["R2"], axis=0)
                    else:
                        assert self.bc_orientation == "R2"
                        bc["R1"] = reverse_complement(bc["R1"])
                        bc_q["R1"] = numpy.flip(bc_q["R1"], axis=0)
                    if r1only:
                        if (bc_q["R1"] >= self.minq).all():
                            if self.valid_barcodes and (
                                bc["R1"] not in self.valid_barcodes
                            ):
                                fates["invalid barcode"] += 1
                            else:
                                barcodes[bc["R1"]] += 1
                                fates["valid barcode"] += 1
                        else:
                            fates["low quality barcode"] += 1
                    else:
                        if bc["R1"] == bc["R2"]:
                            if self.valid_barcodes and (
                                bc["R1"] not in self.valid_barcodes
                            ):
                                fates["invalid barcode"] += 1
                            elif (
                                numpy.maximum(bc_q["R1"], bc_q["R2"]) >= self.minq
                            ).all():
                                barcodes[bc["R1"]] += 1
                                fates["valid barcode"] += 1
                            else:
                                fates["low quality barcode"] += 1
                        else:
                            fates["R1 / R2 disagree"] += 1
                elif (
                    outer_flank_fates
                    and self._has_flank2
                    and all(
                        self._matchers_no_flank2[read].fullmatch(r) is not None
                        for (read, r) in zip(reads, readlist)
                    )
                ):
                    fates["invalid outer flank"] += 1
                else:
                    # invalid flanking sequence or N in barcode
                    fates["unparseable barcode"] += 1

        if add_cols is None:
            add_cols = {}
        existing_cols = {"barcode", "count", "fate"}
        if set(add_cols).intersection(existing_cols):
            raise ValueError(f"`add_cols` cannot contain {existing_cols}")

        barcodes = (
            pd.DataFrame(list(barcodes.items()), columns=["barcode", "count"])
            .sort_values(["count", "barcode"], ascending=[False, True])
            .assign(**add_cols)
            .reset_index(drop=True)
        )

        fates = (
            pd.DataFrame(list(fates.items()), columns=["fate", "count"])
            .sort_values(["count", "fate"], ascending=[False, True])
            .assign(**add_cols)
            .reset_index(drop=True)
        )

        return (barcodes, fates)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
