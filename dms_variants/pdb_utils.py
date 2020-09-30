"""
===========
pdb_utils
===========

Functions to manipulate `PDB <https://www.rcsb.org/>`_ files.

"""


import collections  # noqa: F401
import os  # noqa: F401
import tempfile  # noqa: F401
import warnings

import Bio.PDB

import pandas as pd  # noqa: F401

import requests  # noqa: F401


def reassign_b_factor(input_pdbfile,
                      output_pdbfile,
                      df,
                      metric_col,
                      *,
                      site_col='site',
                      chain_col='chain',
                      missing_metric=0,
                      model_index=0,
                      ):
    r"""Reassign B factors in PDB file to some other metric.

    B-factor re-assignment is useful because PDB images can be colored
    by B factor using programs such as ``pymol`` using commands like::

        show surface, RBD; spectrum b, white red, RBD, minimum=0, maximum=1

    Parameters
    ----------
    input_pdbfile : str
        Path to input PDB file.
    output_pdbfile: str
        Name of created output PDB file with re-assigned B factors.
    df : pandas.DataFrame
        Data frame with metric used to re-assign B factor.
    metric_col : str
        Name of column in `df` that has the numerical metric that the B
        factor is re-assigned to.
    site_col : str
        Name of column in `df` with site numbers, which should map numbers
        used in PDB.
    chain_col : str
        Name of column in `df` with chain labels.
    missing_metric : float or dict
        How do we handl sites that are missing in `df`? If a float, reassign
        B factors for all missing sites to this value. If a dict, should be
        keyed by chain and assign all missing sites in each chain to
        indicated value.
    model_index : int
        Which model in the PDB to use. If a X-ray structure, there is
        probably just one model so you can use default of 0.

    Returns
    -------
    None

    Example
    -------
    Create data frame `df` that assigns metric to two sites in chain E:

    >>> df = pd.DataFrame({'chain': ['E', 'E'],
    ...                    'site': [333, 334],
    ...                    'metric': [0.5, 1.2]})

    Create dict `missing_metric` that assigns -1 to sites with missing
    metrics in chain E, and 0 to sites in other chains:

    >>> missing_metric = collections.defaultdict(lambda: 0)
    >>> missing_metric['E'] = -1

    Download PDB, do the re-assignment of B factors, read the lines
    from the resulting re-assigned PDB:

    >>> pdb_url = 'https://files.rcsb.org/download/6M0J.pdb'
    >>> r = requests.get(pdb_url)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...    original_pdbfile = os.path.join(tmpdir, 'original.pdb')
    ...    with open(original_pdbfile, 'wb') as f:
    ...        _ = f.write(r.content)
    ...    reassigned_pdbfile = os.path.join(tmpdir, 'reassigned.pdb')
    ...    reassign_b_factor(input_pdbfile=original_pdbfile,
    ...                      output_pdbfile=reassigned_pdbfile,
    ...                      df=df,
    ...                      metric_col='metric',
    ...                      missing_metric=missing_metric)
    ...    pdb_text = open(reassigned_pdbfile).readlines()

    Now spot check some key lines in the output PDB.
    Chain A has all sites with B factors (last entry) re-assigned to 0:

    >>> print(pdb_text[0].strip())
    ATOM      1  N   SER A  19     -31.455  49.474   2.505  1.00  0.00           N

    Chain E has sites 333 and 334 with B-factors assigned to values in `df`, and
    other sites (such as 335) assigned to -1:

    >>> print('\n'.join(line.strip() for line in pdb_text[5010: 5025]))
    ATOM   5010  O   THR E 333     -34.954  13.568  46.370  1.00  0.50           O
    ATOM   5011  CB  THR E 333     -33.695  14.409  48.627  1.00  0.50           C
    ATOM   5012  OG1 THR E 333     -34.797  14.149  49.507  1.00  0.50           O
    ATOM   5013  CG2 THR E 333     -32.495  14.879  49.438  1.00  0.50           C
    ATOM   5014  N   ASN E 334     -35.532  15.604  45.605  1.00  1.20           N
    ATOM   5015  CA  ASN E 334     -36.287  15.087  44.474  1.00  1.20           C
    ATOM   5016  C   ASN E 334     -35.475  15.204  43.182  1.00  1.20           C
    ATOM   5017  O   ASN E 334     -34.533  15.994  43.076  1.00  1.20           O
    ATOM   5018  CB  ASN E 334     -37.622  15.823  44.337  1.00  1.20           C
    ATOM   5019  CG  ASN E 334     -38.660  15.006  43.586  1.00  1.20           C
    ATOM   5020  OD1 ASN E 334     -38.568  13.776  43.514  1.00  1.20           O
    ATOM   5021  ND2 ASN E 334     -39.649  15.686  43.016  1.00  1.20           N
    ATOM   5022  N   LEU E 335     -35.849  14.391  42.194  1.00 -1.00           N
    ATOM   5023  CA  LEU E 335     -35.084  14.305  40.955  1.00 -1.00           C
    ATOM   5024  C   LEU E 335     -35.466  15.426  39.992  1.00 -1.00           C

    """  # noqa: E501
    # subset `df` to needed columns and error check it
    cols = [metric_col, site_col, chain_col]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"`df` lacks column {col}")
    df = df[cols].drop_duplicates()
    if len(df) != len(df.groupby([site_col, chain_col])):
        raise ValueError('non-unique metric for a site in a chain')

    if df[site_col].dtype != int:
        raise ValueError('function currently requires `site_col` to be int')

    # read PDB, catch warnings about discontinuous chains
    with warnings.catch_warnings():
        warnings.simplefilter(
                'ignore',
                category=Bio.PDB.PDBExceptions.PDBConstructionWarning)
        pdb = Bio.PDB.PDBParser().get_structure('_', input_pdbfile)

    # get the model out of the PDB
    model = list(pdb.get_models())[model_index]

    # make sure all chains in PDB
    missing_chains = set(df[chain_col]) - {chain.id for chain
                                           in model.get_chains()}
    if missing_chains:
        raise ValueError(f"`df` has chains not in PDB: {missing_chains}")

    # make missing_metric a dict if it isn't already
    if not isinstance(missing_metric, dict):
        missing_metric = {chain.id: missing_metric for chain
                          in model.get_chains()}

    # loop over all chains and do coloring
    for chain in model.get_chains():
        chain_id = chain.id
        site_to_val = (df
                       .query(f"{chain_col} == @chain_id")
                       .set_index(site_col)
                       [metric_col]
                       .to_dict()
                       )
        for residue in chain:
            site = residue.get_id()[1]
            try:
                metric_val = site_to_val[site]
            except KeyError:
                metric_val = missing_metric[chain_id]
            # for disordered residues, get list of them
            try:
                residuelist = residue.disordered_get_list()
            except AttributeError:
                residuelist = [residue]
            for r in residuelist:
                for atom in r:
                    # for disordered atoms, get list of them
                    try:
                        atomlist = atom.disordered_get_list()
                    except AttributeError:
                        atomlist = [atom]
                    for a in atomlist:
                        a.bfactor = metric_val

    # write PDB
    io = Bio.PDB.PDBIO()
    io.set_structure(pdb)
    io.save(output_pdbfile)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
