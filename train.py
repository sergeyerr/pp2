import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from src.dataset import ProteinDataset
from src.utils import train_model, test_model


# Load and prepare data
csv_file = 'data/disorder/strict.csv'
prott5_file = 'data/disorder/embeddings/unfiltered_all_prott5.h5'
prott5_res_file = 'data/disorder/embeddings/unfiltered_all_prott5_res.h5'
prostt5_file = 'data/disorder/embeddings/prostt5.h5'
esm_file = 'data/disorder/embeddings/unfiltered_all_esm2_3b.h5'
esm_res_file = 'data/disorder/embeddings/strict_esm2_3b_res.h5'
chemical_shifts_df = pd.read_csv(csv_file)
chemical_shifts_df.describe()


#target_columns = ['C', 'CA', 'CB', 'HA', 'H', 'N', 'HB']
target_columns = ['N']
print(len(chemical_shifts_df))
chemical_shifts_df.dropna(inplace=True, subset=target_columns)
#chemical_shifts_df = chemical_shifts_df.sample(100)
print(len(chemical_shifts_df))

# Split data into train, validation, and test sets
train_df, test_df = train_test_split(chemical_shifts_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Normalize targets based on training data statistics
scaler = StandardScaler()
train_targets = train_df[target_columns]
scaler.fit(train_targets)

# Save the mean and std for later un-normalizing
means = scaler.mean_
stds = scaler.scale_

print('Standard scaler means:', means)
print('Standard scaler stds:', stds)
# Save the scaler object
joblib.dump(scaler, 'scaler.joblib')

# Apply normalization to the training targets
train_df[target_columns] = scaler.transform(train_targets)

# Apply the same normalization to validation and test sets
val_df[target_columns] = scaler.transform(val_df[target_columns])
test_df[target_columns] = scaler.transform(test_df[target_columns])

# Create datasets
train_dataset = ProteinDataset(target_columns, train_df, prott5_file, prott5_res_file, prostt5_file, esm_res_file, esm_file)
val_dataset = ProteinDataset(target_columns, val_df, prott5_file, prott5_res_file, prostt5_file, esm_res_file, esm_file)
test_dataset = ProteinDataset(target_columns, test_df, prott5_file, prott5_res_file, prostt5_file, esm_res_file, esm_file)

print('Trainng dataset length:', len(train_dataset))
print('Validation dataset length:', len(val_dataset))
print('Test dataset length:', len(test_dataset))

learning_rate = 5e-5
weight_decay = 0
patience = 10
batch_size = 64
num_epochs = 100


#trained_model = train_model(train_dataset, val_dataset, learning_rate=learning_rate, num_epochs=num_epochs, weight_decay=weight_decay, patience=patience, batch_size=batch_size, use_attention=True, use_prostt5=True, use_protein_mean=True, scaler=scaler)
test_model(None, test_dataset, batch_size=batch_size, use_prostt5=True, use_protein_mean=True, use_attention=True, scaler=scaler)
