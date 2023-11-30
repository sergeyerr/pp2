import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np


class ProteinDataset(Dataset):
    def __init__(self, chemical_shifts_df, prott5_file, prott5_res_file, prostt5_file):
        self.chemical_shifts_df = chemical_shifts_df
        self.prott5_embs = h5py.File(prott5_file, "r")
        self.prott5_res_embs = h5py.File(prott5_res_file, "r")
        self.prostt5_embs = h5py.File(prostt5_file, "r")

        self.avg_prostt5_embs = {key: np.mean(np.array(self.prostt5_embs[key]), axis=0) for key in self.prostt5_embs.keys()}

    def __len__(self):
        return len(self.chemical_shifts_df)

    def __getitem__(self, idx):
        row = self.chemical_shifts_df.iloc[idx]
        protein_id = row['ID']
        amino_acid_index = row['seq_index'] - 1  # Adjust for zero-based indexing

        # Fetch amino acid specific embeddings
        amino_acid_prott5_emb = self.prott5_res_embs[protein_id][amino_acid_index]
        amino_acid_prostt5_emb = self.prostt5_embs[protein_id][amino_acid_index]

        # Fetch averaged protein embeddings
        protein_prott5_emb = self.prott5_embs[protein_id]
        protein_prostt5_emb = self.avg_prostt5_embs[protein_id]

        # Get chemical shifts
        chemical_shifts = row[['C', 'CA', 'CB', 'HA', 'H', 'N', 'HB']].values.astype(np.float32)

        return (torch.tensor(amino_acid_prott5_emb, dtype=torch.float32),
                torch.tensor(amino_acid_prostt5_emb, dtype=torch.float32),
                torch.tensor(protein_prott5_emb, dtype=torch.float32),
                torch.tensor(protein_prostt5_emb, dtype=torch.float32),
                torch.tensor(chemical_shifts, dtype=torch.float32))