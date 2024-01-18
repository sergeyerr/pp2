import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .model import ChemicalShiftsPredictor, ChemicalShiftsPredictorAttention
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def packed_padded_collate(batch):
    """
    Takes list of tuples with embeddings of variable sizes and pads them with zeros
    Args:
        batch: list of tuples with embeddings and the corresponding label

    Returns: tuple of tensor of embeddings with [batchsize, length_of_longest_sequence, embeddings_dim]
    and tensor of labels [batchsize, labels_dim]

    """
    amino_acid_prott5_emb_list, amino_acid_prostt5_emb_list, amino_acid_esm2_emb_list, protein_prott5_emb_list, protein_prostt5_emb_list, targets_list = zip(*batch)

    amino_acid_prott5_emb = pad_sequence(amino_acid_prott5_emb_list, batch_first=True)
    amino_acid_prostt5_emb = pad_sequence(amino_acid_prostt5_emb_list, batch_first=True)
    protein_prott5_emb = pad_sequence(protein_prott5_emb_list, batch_first=True)
    amino_acid_esm2_emb = pad_sequence(amino_acid_esm2_emb_list, batch_first=True)
    targets = torch.stack(targets_list)

    embeddings = pad_sequence([item[4] for item in batch], batch_first=True)


    del amino_acid_prott5_emb_list
    del amino_acid_prostt5_emb_list
    del protein_prott5_emb_list
    del targets_list

    return amino_acid_prott5_emb, amino_acid_prostt5_emb, amino_acid_esm2_emb, protein_prott5_emb, embeddings.permute(0, 2, 1), targets

def train_model(train_dataset, val_dataset, learning_rate, weight_decay, patience, batch_size=32, use_prostt5=True, num_epochs=100, use_attention=True, use_protein_mean=True, scaler=None):
    # Set up TensorBoard writer
    writer = SummaryWriter()

    
    if use_attention:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=packed_padded_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, collate_fn=packed_padded_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChemicalShiftsPredictor(use_prostt5=use_prostt5, use_protein_mean=use_protein_mean, use_attention=use_attention).to(device)
    print(model)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    t_mean, t_std = torch.tensor(scaler.mean_).to(device), torch.tensor(scaler.scale_).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_actual_rmse = 0
        total_actual_rmse_val = 0
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # Determine which inputs to use based on the dataset options
            amino_acid_prott5_emb, amino_acid_prostt5_emb, amino_acid_esm2_emb, protein_prott5_emb, protein_prostt5_emb_stack, targets = inputs
            targets = targets.to(device)
            embeddings = [amino_acid_prott5_emb]
            if use_prostt5:
                embeddings.append(amino_acid_prostt5_emb)
            if use_protein_mean:
                embeddings.append(protein_prott5_emb)
            embeddings.append(amino_acid_esm2_emb)
            concatenated_embeddings = torch.cat(embeddings, dim=1).to(device)
            optimizer.zero_grad()
            if use_attention:
                sequence_lengths = protein_prostt5_emb_stack.abs().sum(dim=-1).nonzero()[:, 1].max(dim=-1, keepdim=True)[0] + 1

                # Create mask based on actual sequence lengths
                max_seq_len = protein_prostt5_emb_stack.size(2)
                mask = torch.arange(max_seq_len, device=device)[None, :] < sequence_lengths.to(device)
                outputs = model(concatenated_embeddings, protein_prostt5_emb_stack.to(device), mask.to(device))
            else:
                outputs = model(concatenated_embeddings, None, None)
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Denormalize predictions and targets
            denorm_outputs = outputs*t_std + t_mean
            denorm_targets = targets*t_std + t_mean 

            # Calculate actual MSE
            actual_rmse = ((denorm_outputs - denorm_targets) ** 2).mean() ** (1/2)
            total_actual_rmse += actual_rmse.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_actual_mse = total_actual_rmse / len(train_loader)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('RMSE/Train', avg_actual_mse, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                amino_acid_prott5_emb, amino_acid_prostt5_emb, amino_acid_esm2_emb, protein_prott5_emb, protein_prostt5_emb_stack, targets = [x.to(device) for x in inputs]

                embeddings = [amino_acid_prott5_emb]
                if use_prostt5:
                    embeddings.append(amino_acid_prostt5_emb)
                if use_protein_mean:
                    embeddings.append(protein_prott5_emb)
                    #if use_prostt5:
                        #embeddings.append(protein_prostt5_emb)
                embeddings.append(amino_acid_esm2_emb)
                concatenated_embeddings = torch.cat(embeddings, dim=1)
                optimizer.zero_grad()

                if use_attention:
                    sequence_lengths = protein_prostt5_emb_stack.abs().sum(dim=-1).nonzero()[:, 1].max(dim=-1, keepdim=True)[0] + 1

                    # Create mask based on actual sequence lengths
                    max_seq_len = protein_prostt5_emb_stack.size(2)
                    mask = torch.arange(max_seq_len, device=device)[None, :] < sequence_lengths.to(device)
                    outputs = model(concatenated_embeddings, protein_prostt5_emb_stack.to(device), mask.to(device))
                else:
                    outputs = model(concatenated_embeddings, None, None)
                val_loss += criterion(outputs, targets).item()
                denorm_outputs = outputs*t_std + t_mean
                denorm_targets = targets*t_std + t_mean 

                # Calculate actual MSE
                actual_rmse = ((denorm_outputs - denorm_targets) ** 2).mean() ** (1/2)
                total_actual_rmse_val += actual_rmse.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_actual_mse_val = total_actual_rmse_val / len(val_loader)


        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('RMSE/Validation', avg_actual_mse_val, epoch)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1}, Train RMSE: {avg_actual_mse:.4f}, Validation RMSE: {avg_actual_mse_val:.4f}")
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered. No improvement in {patience} consecutive epochs.")
                break

    writer.close()
    return model


def test_model(model, test_dataset, use_prostt5, use_protein_mean, use_attention, batch_size=32, scaler=None):
    if use_attention:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, collate_fn=packed_padded_collate)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChemicalShiftsPredictor(use_prostt5=use_prostt5, use_protein_mean=use_protein_mean, use_attention=use_attention).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0
    total_actual_rmse_test = 0
    t_mean, t_std = torch.tensor(scaler.mean_).to(device), torch.tensor(scaler.scale_).to(device)

    with torch.no_grad():
        for inputs in test_loader:
            amino_acid_prott5_emb, amino_acid_prostt5_emb, amino_acid_esm2_emb, protein_prott5_emb, protein_prostt5_emb_stack, targets = [x.to(device) for x in inputs]
            embeddings = [amino_acid_prott5_emb]
            if use_prostt5:
                embeddings.append(amino_acid_prostt5_emb)
            if use_protein_mean:
                embeddings.append(protein_prott5_emb)
                # if use_prostt5:
                #     embeddings.append(protein_prostt5_emb)
            embeddings.append(amino_acid_esm2_emb)
            concatenated_embeddings = torch.cat(embeddings, dim=1)
            if use_attention:
                sequence_lengths = protein_prostt5_emb_stack.abs().sum(dim=-1).nonzero()[:, 1].max(dim=-1, keepdim=True)[0] + 1

                # Create mask based on actual sequence lengths
                max_seq_len = protein_prostt5_emb_stack.size(2)
                mask = torch.arange(max_seq_len, device=device)[None, :] < sequence_lengths.to(device)
                outputs = model(concatenated_embeddings, protein_prostt5_emb_stack.to(device), mask.to(device))
            else:
                outputs = model(concatenated_embeddings, None, None)
            total_loss += criterion(outputs, targets).item()
            denorm_outputs = outputs*t_std + t_mean
            denorm_targets = targets*t_std + t_mean 

            # Calculate actual MSE
            actual_rmse = ((denorm_outputs - denorm_targets) ** 2).mean() ** (1/2)
            total_actual_rmse_test += actual_rmse.item()

    avg_test_loss = total_loss / len(test_loader)
    avg_actual_mse_test = total_actual_rmse_test / len(test_loader)


    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test RMSE: {avg_actual_mse_test:.4f}")
