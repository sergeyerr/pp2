import torch.nn as nn

class ChemicalShiftsPredictor(nn.Module):
    def __init__(self, use_prostt5=True, use_protein_mean=True):
        super(ChemicalShiftsPredictor, self).__init__()
        # Compute input size based on the selected options
        self.use_prostt5 = use_prostt5
        self.use_protein_mean = use_protein_mean

        input_size = 1024  # Default for amino_acid_prott5_emb
        if use_prostt5:
            input_size += 1024  # For amino_acid_prostt5_emb
        if use_protein_mean:
            input_size += 1024 * (2 if use_prostt5 else 1)  # For protein_prott5_emb and protein_prostt5_emb

        # Define the architecture of the MLP
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # Output size for chemical shifts
        )

    def forward(self, x):
        return self.fc_layers(x)

# Example usage
model = ChemicalShiftsPredictor(use_prostt5=False, use_protein_mean=False)
