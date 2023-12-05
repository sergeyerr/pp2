import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class ProteinDataset(Dataset):
    def __init__(self, target_columns, chemical_shifts_df, prott5_file, prott5_res_file, prostt5_file, esm2_res_file, esm2_file):
        self.chemical_shifts_df = chemical_shifts_df
        self.prott5_embs = h5py.File(prott5_file, "r")
        self.prott5_res_embs = h5py.File(prott5_res_file, "r")
        self.prostt5_embs = h5py.File(prostt5_file, "r")
        #self.esm2_res = h5py.File(esm2_res_file, 'r')
        #self.esm2 = h5py.File(esm2_file, 'r')
        self.target_columns = target_columns
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
        #amino_acid_esm2_emb = self.esm2_res[protein_id][amino_acid_index]

        # Fetch averaged protein embeddings
        protein_prott5_emb = self.prott5_embs[protein_id]
        protein_prostt5_emb_stack = self.prostt5_embs[protein_id]
        #protein_esm2_emb = self.esm2[protein_id]
        # Get chemical shifts
        chemical_shifts = row[self.target_columns].values.astype(np.float32)

        return (torch.tensor(amino_acid_prott5_emb, dtype=torch.float32),
                torch.tensor(amino_acid_prostt5_emb, dtype=torch.float32),
                #torch.tensor(amino_acid_esm2_emb, dtype=torch.float32),
                torch.tensor(np.array(protein_prott5_emb), dtype=torch.float32).squeeze(),
                torch.tensor(np.array(protein_prostt5_emb_stack), dtype=torch.float32),
                #torch.tensor(protein_esm2_emb, dtype=torch.float32),
                torch.tensor(chemical_shifts, dtype=torch.float32))