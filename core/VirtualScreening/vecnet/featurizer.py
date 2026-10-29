"""
Featurization for VecNet — SMILES -> 300d (Mol2vec), AA sequence -> 100d (ProtVec).

Produces RAW summed vectors (float64). Normalization by the four train-set
constants happens inside the model (model.py), where the constants live as frozen
buffers. Nothing here recomputes those constants.

mol2alt_sentence is the local RDKit-MorganGenerator reimplementation (verified
identifier-identical to the mol2vec package); the mol2vec package is not imported.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from gensim.models import word2vec


def mol2alt_sentence(mol, radius=2):
    """Morgan substructure identifiers as strings, atom-major then radius-minor.
    Uses the current MorganGenerator API (no deprecated call, no mol2vec dep)."""
    radii = list(range(radius + 1))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetSparseFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()                     # {identifier: ((atom_idx, radius), ...)}
    atoms = [a.GetIdx() for a in mol.GetAtoms()]
    per_atom = {a: {r: None for r in radii} for a in atoms}
    for ident, envs in info.items():
        for atom_idx, rad in envs:
            per_atom[atom_idx][rad] = ident
    out = []
    for a in per_atom:
        for r in radii:
            out.append(per_atom[a][r])
    return [str(x) for x in out if x is not None]


class Featurizer:
    """Loads the frozen pretrained encoders (Mol2vec, ProtVec) once."""

    def __init__(self, mol2vec_path, protvec_csv):
        self.mol2vec = word2vec.Word2Vec.load(mol2vec_path)
        pv = pd.read_csv(protvec_csv, delimiter="\t")
        vals = pv.iloc[:, 1:].to_numpy(dtype=np.float64)
        self.trigrams = {word: vals[i] for i, word in enumerate(pv["words"].values)}

    def drug_raw(self, smiles):
        """SMILES -> 300d raw Mol2vec sum (float64), or None if RDKit can't parse."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        v = np.zeros(300, dtype=np.float64)
        for sub in mol2alt_sentence(mol, 2):
            try:
                v += self.mol2vec.wv[sub]
            except KeyError:
                v += self.mol2vec.wv["UNK"]
        return v

    def target_raw(self, sequence):
        """AA sequence -> 100d raw ProtVec sum (float64)."""
        v = np.zeros(100, dtype=np.float64)
        for i in range(0, len(sequence), 3):
            tri = sequence[i:i + 3]
            if len(tri) == 2:
                tri = "X" + tri
            elif len(tri) == 1:
                tri = "XX" + tri
            hit = self.trigrams.get(tri)
            if hit is not None:
                v += hit
        return v
