"""
===========
fastq
===========

Tools for processing FASTQ files.

"""


import collections
import gzip
import itertools
import os
import tempfile  # noqa: F401

import numpy


def qual_str_to_array(q_str, *, offset=33):
    """Convert quality score string to array of integers.

    Parameters
    ----------
    q_str : str
        Quality score string.
    offset : int
        Offset in ASCII encoding of Q-scores.

    Returns
    -------
    numpy.ndarray
        Array of integer quality scores.

    Example
    -------
    >>> qual_str_to_array('!I:0G')
    array([ 0, 40, 25, 15, 38])

    """
    return numpy.array([ord(q) - offset for q in q_str],
                       dtype='int')


def iterate_fastq_pair(r1filename, r2filename, *,
                       r1trim=None, r2trim=None, qual_format='str'):
    r"""Iterate over paired R1 and R2 FASTQ files.

    Parameters
    ----------
    r1filename : str
        R1 FASTQ file name, can be gzipped (extension ``.gz``).
    r2filename : str
        R2 FASTQ file name, can be gzipped (extension ``.gz``).
    r1trim : int or None
        If not `None`, trim R1 reads and Q scores to be longer than this.
    r2trim : int or None
        If not `None`, trim R2 reads and Q scores to be longer than this.
    qual_format : {'str', 'array'}
        Return the quality scores as string of ASCII codes or array of numbers?

    Yields
    ------
    namedtuple
        The entries in the tuple are (in order):
         - `id` : read id
         - `r1_seq` : R1 read sequence
         - `r2_seq` : R2 read sequence
         - `r1_qs` : R1 Q scores (`qual_format` parameter determines format)
         - `r2_qs` : R2 Q scores (`qual_format` parameter determines format)
         - `fail` : did either read fail chastity filter? (`None` if no info)

    Example
    -------
    >>> f1 = tempfile.NamedTemporaryFile(mode='w')
    >>> _ = f1.write(
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984 1:N:0:CGATGT\n'
    ...         'ATGCAATTG\n'
    ...         '+\n'
    ...         'GGGGGIIII\n'
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985 1:Y:0:CGATGT\n'
    ...         'ACGCTATTC\n'
    ...         '+\n'
    ...         'GHGGGIKII\n'
    ...         )
    >>> f1.flush()
    >>> f2 = tempfile.NamedTemporaryFile(mode='w')
    >>> _ = f2.write(
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984 2:N:0:CGATGT\n'
    ...         'CAGCATA\n'
    ...         '+\n'
    ...         'AGGGGII\n'
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985 2:Y:0:CGATGT\n'
    ...         'CTGAATA\n'
    ...         '+\n'
    ...         'GHBGGIK\n'
    ...         )
    >>> f2.flush()

    >>> for tup in iterate_fastq_pair(f1.name, f2.name, r1trim=8, r2trim=5,
    ...                               qual_format='array'):
    ...     print(tup)
    ... # doctest: +NORMALIZE_WHITESPACE
    FastqPairEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984',
                   r1_seq='ATGCAATT',
                   r2_seq='CAGCA',
                   r1_qs=array([38, 38, 38, 38, 38, 40, 40, 40]),
                   r2_qs=array([32, 38, 38, 38, 38]),
                   fail=False)
    FastqPairEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985',
                   r1_seq='ACGCTATT',
                   r2_seq='CTGAA',
                   r1_qs=array([38, 39, 38, 38, 38, 40, 42, 40]),
                   r2_qs=array([38, 39, 33, 38, 38]),
                   fail=True)

    >>> for tup in iterate_fastq_pair(f1.name, f2.name, r1trim=8, r2trim=5):
    ...     print(tup)
    ... # doctest: +NORMALIZE_WHITESPACE
    FastqPairEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984',
                   r1_seq='ATGCAATT',
                   r2_seq='CAGCA',
                   r1_qs='GGGGGIII',
                   r2_qs='AGGGG',
                   fail=False)
    FastqPairEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985',
                   r1_seq='ACGCTATT',
                   r2_seq='CTGAA',
                   r1_qs='GHGGGIKI',
                   r2_qs='GHBGG',
                   fail=True)


    >>> f1.close()
    >>> f2.close()

    """
    FastqPairEntry = collections.namedtuple(
                            'FastqPairEntry',
                            'id r1_seq r2_seq r1_qs r2_qs fail')

    r1_iterator = iterate_fastq(r1filename,
                                trim=r1trim,
                                check_pair=1,
                                qual_format=qual_format)
    r2_iterator = iterate_fastq(r2filename,
                                trim=r2trim,
                                check_pair=2,
                                qual_format=qual_format)

    for r1_entry, r2_entry in itertools.zip_longest(r1_iterator, r2_iterator):

        if (r1_entry is None) or (r2_entry is None):
            raise IOError(f"{r1filename} and {r2filename} have unequal "
                          'number of entries')

        if r1_entry[0] != r2_entry[0]:
            raise IOError(f"{r1filename} and {r2filename} specify different "
                          f"read IDs:\n{r1_entry[0]}\n{r2_entry[0]}")

        yield FastqPairEntry(id=r1_entry[0],
                             r1_seq=r1_entry[1],
                             r2_seq=r2_entry[1],
                             r1_qs=r1_entry[2],
                             r2_qs=r2_entry[2],
                             fail=(r1_entry[3] or r2_entry[3]),
                             )


def iterate_fastq(filename, *, trim=None, check_pair=None, qual_format='str'):
    r"""Iterate over a FASTQ file.

    Parameters
    ----------
    filename : str
        FASTQ file name, can be gzipped (extension ``.gz``).
    trim : int or None
        If not `None`, trim reads and Q scores to be longer than this.
    check_pair : {1, 2, None}
        If not `None`, check reads are read 1 or read 2 if this info given.
        Assumes Casava 1.8 or SRA header format.
    qual_format : {'str', 'array'}
        Return the quality scores as string of ASCII codes or array of numbers?

    Yields
    ------
    namedtuple
        The entries in the tuple are (in order):
         - `id` : read id
         - `seq` : read sequence
         - `qs` : Q scores (`qual_format` parameter determines format)
         - `fail` : did read fail chastity filter? (`None` if no filter info)

    Example
    -------
    >>> f = tempfile.NamedTemporaryFile(mode='w')
    >>> _ = f.write(
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984 1:N:0:CGATGT\n'
    ...         'ATGCAATTG\n'
    ...         '+\n'
    ...         'GGGGGIIII\n'
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985 1:Y:0:CGATGT\n'
    ...         'ACGCTATTC\n'
    ...         '+\n'
    ...         'GHGGGIKII\n'
    ...         '@DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985 2:Y:0:CGATGT\n'
    ...         'ACGCTATTC\n'
    ...         '+\n'
    ...         'GHGGGIKII\n'
    ...         )
    >>> f.flush()

    >>> try:
    ...     for tup in iterate_fastq(f.name, trim=5, check_pair=1):
    ...         print(tup)
    ... except ValueError as e:
    ...    print(f"ValueError: {e}")
    ... # doctest: +NORMALIZE_WHITESPACE
    FastqEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984',
               seq='ATGCA',
               qs='GGGGG',
               fail=False)
    FastqEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985',
               seq='ACGCT',
               qs='GHGGG',
               fail=True)
    ValueError: header not for R1:
    @DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985 2:Y:0:CGATGT

    >>> for tup in iterate_fastq(f.name, qual_format='array'):
    ...    print(tup)
    ... # doctest: +NORMALIZE_WHITESPACE
    FastqEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1984',
               seq='ATGCAATTG',
               qs=array([38, 38, 38, 38, 38, 40, 40, 40, 40]),
               fail=False)
    FastqEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985',
               seq='ACGCTATTC',
               qs=array([38, 39, 38, 38, 38, 40, 42, 40, 40]),
               fail=True)
    FastqEntry(id='DH1DQQN1:933:HMLH5BCXY:1:1101:2165:1985',
               seq='ACGCTATTC',
               qs=array([38, 39, 38, 38, 38, 40, 42, 40, 40]),
               fail=True)

    >>> f.close()

    """
    FastqEntry = collections.namedtuple('FastqEntry',
                                        'id seq qs fail')

    if check_pair is not None:
        check_pair = str(check_pair)
        if check_pair not in {'1', '2'}:
            raise ValueError(f"invalid `check_pair` of {check_pair}")

    if not os.path.isfile(filename):
        raise IOError(f"no FASTQ file {filename}")

    if qual_format == 'array':
        qual_to_array = True
    elif qual_format == 'str':
        qual_to_array = False
    else:
        raise ValueError(f"invalid value for `qual_format`: {qual_format}")

    if os.path.splitext(filename)[1].lower() == '.gz':
        openfunc = gzip.open
    else:
        openfunc = open

    with openfunc(filename, mode='rt') as f:
        head = f.readline()
        while head:
            if head[0] != '@':
                raise IOError(f"id starts with {head[0]}, not @:\n{head}")
            else:
                head = head.rstrip()
                headspl = head[1:].split()
                read_id = headspl[0]
            seq = f.readline().rstrip()
            plusline = f.readline().rstrip()
            qs = f.readline().rstrip()
            if (not seq) or (len(seq) != len(qs)) or (plusline != '+'):
                raise IOError(f"invalid entry for {read_id} in {filename}:\n"
                              f"{head}\n{seq}\n{plusline}\n{qs}")

            # trim last two characters (needed for SRA downloads)
            if read_id[-2:] in {'.1', '.2'}:
                if check_pair and (read_id[-1] != check_pair):
                    raise ValueError(f"header not for R{check_pair}:\n{head}")
                read_id = read_id[: -2]

            if check_pair and len(headspl) > 1 and headspl[1][0] != check_pair:
                raise ValueError(f"header not for R{check_pair}:\n{head}")

            # parse chastity filter assuming CASAVA 1.8 header
            try:
                chastity = headspl[1][2]
                if chastity == 'Y':
                    fail = True
                elif chastity == 'N':
                    fail = False
                else:
                    raise ValueError(f"cannot parse chastity filter in {head}")
            except IndexError:
                fail = None  # header does not specify chastity filter

            if trim is not None:
                seq = seq[: trim]
                qs = qs[: trim]

            if qual_to_array:
                qs = qual_str_to_array(qs)

            yield FastqEntry(id=read_id,
                             seq=seq,
                             qs=qs,
                             fail=fail,
                             )

            head = f.readline()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
