import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.nn as nn
from torch.optim import Adam

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Pretrained Wav2Vec2 model (feature extractor)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
backbone_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
backbone_model.to(device)
backbone_model.eval()  # We freeze the backbone for now

# Custom Classifier Head
class AudioClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x

classifier = AudioClassifier()
classifier.to(device)

# ASVspoof Dataset Class
class ASVspoofDataset(Dataset):
    def __init__(self, data_dir, protocol_file, feature_extractor, max_length=16000*4):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        
        self.filepaths = []
        self.labels = []

        # Map bonafide/spoof to 0/1
        self.label_map = {'bonafide': 0, 'spoof': 1}

        # Read protocol file
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                filename = parts[0] + ".flac"
                label = parts[-1]
                self.filepaths.append(filename)
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load audio
        filepath = os.path.join(self.data_dir, self.filepaths[idx])
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if needed
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        waveform = waveform.squeeze(0)  # remove channel dim

        # Truncate or pad
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_length - waveform.shape[0]))

        # Feature extraction
        inputs = self.feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        input_values = inputs.input_values.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        label = self.labels[idx]

        return input_values, attention_mask, torch.tensor(label)


# Set paths (Change these paths according to your setup)
data_dir = "/path/to/ASVspoof2019_LA_train/flac/"
protocol_file = "/path/to/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Create dataset
asvspoof_dataset = ASVspoofDataset(data_dir, protocol_file, feature_extractor)

# Create DataLoader
def collate_fn(batch):
    input_values = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return input_values, attention_mask, labels

loader = DataLoader(asvspoof_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Model Training Setup
optimizer = Adam(classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training Loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    classifier.train()
    
    for batch in loader:
        inputs, attention_mask, labels = batch
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
        
        with torch.no_grad():  # Extract features using Wav2Vec2
            features = backbone_model(inputs).last_hidden_state.mean(dim=1)  # mean pooling
        
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

print("Training complete!")

# Save model after training
torch.save(classifier.state_dict(), "audio_classifier.pth")
