import torch.nn as nn
import torch


class ChemicalShiftsPredictorAttention(nn.Module):
    def __init__(self, use_prostt5=True, use_attention=True, use_protein_mean=True):
        super(ChemicalShiftsPredictorAttention, self).__init__()
        # Compute input size based on the selected options
        self.use_prostt5 = use_prostt5
        self.use_protein_mean = use_protein_mean
        self.use_attention = use_attention
        if use_attention:
            self.light_attention = LightAttention(embeddings_dim=1024)

        input_size = 1024 * (3 if use_attention else 1)  # Default for amino_acid_prott5_emb
        if use_prostt5:
            input_size += 1024  # For amino_acid_prostt5_emb
        if use_protein_mean:
            input_size += 1024  # For protein_prott5_emb and protein_prostt5_emb
        input_size+=2560

        self.q_W = nn.Linear(input_size, 1024)
        self.k_W = nn.Linear(input_size, 1024)
        self.v_W = nn.Linear(input_size, 1024)

        self.att = nn.MultiheadAttention(1024, 1, 0)
        # Define the architecture of the MLP
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output size for chemical shifts
        )

    def forward(self, x, stack, mask):
        if self.use_attention:
            attended = self.light_attention(stack, mask)
            o = torch.cat([attended, x], dim=1)
        else:
            o = x
        Q, K, V = self.q_W(o), self.k_W(o), self.v_W(o)
        o, _ = self.att(Q, K, V)
        return self.fc_layers(o)
    


class ChemicalShiftsPredictor(nn.Module):
    def __init__(self, use_prostt5=True, use_attention=True, use_protein_mean=True):
        super(ChemicalShiftsPredictor, self).__init__()
        # Compute input size based on the selected options
        self.use_prostt5 = use_prostt5
        self.use_protein_mean = use_protein_mean
        self.use_attention = use_attention
        if use_attention:
            self.light_attention = LightAttention(embeddings_dim=1024)
        input_size = 1024 * (3 if use_attention else 1)  # Default for amino_acid_prott5_emb
        if use_prostt5:
            input_size += 1024  # For amino_acid_prostt5_emb
        if use_protein_mean:
            input_size += 1024  # For protein_prott5_emb and protein_prostt5_emb
        input_size+=2560
        # Define the architecture of the MLP
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            #nn.Linear(1024, 256),
            #nn.ReLU(),
            nn.Linear(input_size, 1)  # Output size for chemical shifts
        )

    def forward(self, x, stack, mask):
        if self.use_attention:
            attended = self.light_attention(stack, mask)
            o = torch.cat([attended, x], dim=1)
        else:
            o = x
        return self.fc_layers(o)
    

class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)


    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        #o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        return o