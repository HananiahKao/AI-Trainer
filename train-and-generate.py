import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Helper Functions
def file_to_bytes(file_path):
    """ Reads a file as a sequence of bytes. """
    print(f"Reading file: {file_path}")
    with open(file_path, "rb") as f:
        data = list(f.read())  # Convert file content to a list of bytes
    print(f"File read complete: {file_path}, {len(data)} bytes loaded.")
    return data

def bytes_to_file(byte_sequence, output_path):
    """ Writes a sequence of bytes back to a file. """
    print(f"Writing file: {output_path}")
    with open(output_path, "wb") as f:
        f.write(bytes(byte_sequence))  # Convert list of ints to bytes
    print(f"File write complete: {output_path}, {len(byte_sequence)} bytes written.")

# Model Classes
class ByteEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        print("Initializing Encoder")
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        print(f"Encoding input of shape: {src.shape}")
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        print("Encoding complete.")
        return hidden, cell

class ByteDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        print("Initializing Decoder")
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        print(f"Decoding input of shape: {input.shape}")
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        print("Decoding complete.")
        return prediction, hidden, cell

class ByteSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        print("Initializing Seq2Seq Model")
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        print(f"Running forward pass with src shape {src.shape}, trg shape {trg.shape}")
        hidden, cell = self.encoder(src)
        outputs = []
        input = trg[:, 0]

        for t in range(1, trg.shape[1]):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs.append(output.unsqueeze(1))
            input = output.argmax(1)

        print("Forward pass complete.")
        return torch.cat(outputs, dim=1)

# Dataset and DataLoader
HOME_DIR = os.path.expanduser("~")  # '/Users/yourname' or '/home/yourname'
READMES_DIR = os.path.join(HOME_DIR, "READMEs")
REPO_DIR = os.path.join(HOME_DIR, "repos")

class RepoDataset(Dataset):
    def __init__(self, max_samples=100):
        print("Initializing dataset")
        self.repo_names = [f.replace(".md", "") for f in os.listdir(READMES_DIR) if f.endswith(".md")][:max_samples]
        print(f"Loaded {len(self.repo_names)} repo names.")

    def __len__(self):
        return len(self.repo_names)

    def __getitem__(self, idx):
        repo_name = self.repo_names[idx]
        readme_path = os.path.join(READMES_DIR, f"{repo_name}.md")
        repo_zip_path = os.path.join(REPO_DIR, f"{repo_name}.zip")

        if not os.path.exists(repo_zip_path):
            print(f"⚠️ Warning: Repo not found: {repo_zip_path}")
            return None  # Skip if zip file doesn't exist

        print(f"Loading dataset item: {repo_name}")
        readme_bytes = file_to_bytes(readme_path)
        zip_bytes = file_to_bytes(repo_zip_path)

        readme_tensor = torch.tensor(readme_bytes, dtype=torch.long)
        zip_tensor = torch.tensor(zip_bytes, dtype=torch.long)

        print(f"Dataset item loaded: {repo_name}")
        return readme_tensor, zip_tensor

def collate_fn(batch):
    """Collate function to pad sequences to the same length."""
    print("Collating batch")
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        print("⚠️ Warning: Empty batch! Skipping...")
        return None

    src, trg = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True, padding_value=0)

    print("Batch collated successfully.")
    return src, trg

# Training Function

def train(model, iterator, optimizer, criterion, device):
    print("Starting training...")
    model.train()
    epoch_loss = 0

    for batch_idx, (src, trg) in enumerate(iterator):
        print(f"Processing batch {batch_idx+1}")
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        print("Got output.")
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        print("Computes gradients for all model parameters.")
        loss.backward()
        print("Updates the model’s weights using the computed gradients.")
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Batch {batch_idx+1} loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "byte_seq2seq.pth")
        print("Model saved as byte_seq2seq.pth.")

    print("Training complete.")
    return epoch_loss / len(iterator)

def generate_zip_bytes(model, readme_bytes, device):
    model.eval()
    src = torch.tensor(readme_bytes, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(src, src)  # Predict byte sequence

    predicted_bytes = [int(tok) for tok in output.argmax(2).tolist()[0]]
    print(f"Predicted output size: {len(predicted_bytes)} bytes")
    return predicted_bytes


# Model Initialization
device = torch.device("cpu")
print(f"Using device: {device}")

INPUT_DIM = OUTPUT_DIM = 256  # Byte values range from 0-255
EMB_DIM = 128
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.2

encoder = ByteEncoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
decoder = ByteDecoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
model = ByteSeq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

dataset = RepoDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

print(f"Total samples in dataset: {len(dataset)}")

# Train the Model
loss = train(model, dataloader, optimizer, criterion, device)
print(f"Final Loss: {loss:.4f}")

# Save the Model
torch.save(model.state_dict(), "byte_seq2seq.pth")
print("Model saved as byte_seq2seq.pth.")

# Load Model for Prediction
model.load_state_dict(torch.load("byte_seq2seq.pth", map_location=device))

# Generate Zip File from README
readme_bytes = file_to_bytes("README.md")
predicted_zip_bytes = generate_zip_bytes(model, readme_bytes, device)
bytes_to_file(predicted_zip_bytes, "output.zip")
print("Generated output.zip successfully!")
