import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Pretrained Wav2Vec2 model (feature extractor)
print("Loading Wav2Vec2FeatureExtractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
print("Loading Wav2Vec2Model...")
backbone_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
print("Moving model to device...")
backbone_model.to(device)
backbone_model.train()  # Enable training mode for fine-tuning
print("Model loaded and set to train mode")

# Custom Classifier Head
class AudioClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, features):
        x = self.fc1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("Initializing AudioClassifier...")
classifier = AudioClassifier()
classifier.to(device)
print("Classifier initialized and moved to device")

# ASVspoof Dataset Class
class ASVspoofDataset(Dataset):
    def __init__(self, data_dir, protocol_file, feature_extractor, max_length=16000*4):
        print(f"Initializing ASVspoofDataset with data_dir: {data_dir}, protocol_file: {protocol_file}")
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        
        self.filepaths = []
        self.labels = []

        # Map bonafide/spoof to 0/1
        self.label_map = {'bonafide': 0, 'spoof': 1}

        # Verify protocol file exists
        if not os.path.exists(protocol_file):
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
        print(f"Protocol file found: {protocol_file}")

        # Read protocol file
        with open(protocol_file, 'r') as f:
            lines = f.readlines()  # Use all lines
            print(f"Found {len(lines)} lines in protocol file")
            for i, line in enumerate(lines):
                parts = line.strip().split()
                filename = parts[1] + ".flac"
                label = parts[-1]
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    self.filepaths.append(filename)
                    self.labels.append(self.label_map[label])
                else:
                    print(f"Skipping missing file: {filepath}")
                if i % 1000 == 0:
                    print(f"Processed {i+1} protocol entries")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            print(f"Loading file {self.filepaths[idx]} (index {idx})")
            filepath = os.path.join(self.data_dir, self.filepaths[idx])
            waveform, sr = torchaudio.load(filepath)
            print(f"Loaded waveform with shape {waveform.shape}, sample rate {sr}")
            
            # Resample if needed
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

            waveform = waveform.squeeze(0)  # Remove channel dim
            print(f"Waveform after squeeze: {waveform.shape}")

            # Normalize amplitude
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Truncate or pad
            if waveform.shape[0] > self.max_length:
                waveform = waveform[:self.max_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.max_length - waveform.shape[0]))
            print(f"Waveform after padding/truncation: {waveform.shape}")

            # Feature extraction
            print("Extracting features...")
            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True
            )
            print(f"Feature extractor output keys: {list(inputs.keys())}")

            # Extract input_values
            if 'input_values' not in inputs:
                raise ValueError(f"Feature extractor did not return 'input_values' for file: {filepath}")
            input_values = inputs.input_values.squeeze(0)
            
            # Handle attention mask
            if 'attention_mask' in inputs:
                attention_mask = inputs.attention_mask.squeeze(0)
            else:
                print("No attention mask returned, creating dummy mask")
                attention_mask = torch.ones_like(input_values)

            label = self.labels[idx]
            print(f"Returning item with input_values shape {input_values.shape}, attention_mask shape {attention_mask.shape}, label {label}")

            return input_values, attention_mask, torch.tensor(label)

        except Exception as e:
            print(f"Error processing file {self.filepaths[idx]}: {str(e)}")
            raise

# Set paths (Change these paths according to your setup)
data_dir = "LA/ASVspoof2019_LA_train/flac/"
protocol_file = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Verify paths
print(f"Checking data directory: {data_dir}")
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
print(f"Checking protocol file: {protocol_file}")
if not os.path.exists(protocol_file):
    raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

# Create dataset
print("Creating dataset...")
asvspoof_dataset = ASVspoofDataset(data_dir, protocol_file, feature_extractor)
print(f"Dataset created with {len(asvspoof_dataset)} items")

# Split dataset into train and validation (80-20 split for simplicity)
train_size = int(0.8 * len(asvspoof_dataset))
val_size = len(asvspoof_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(asvspoof_dataset, [train_size, val_size])

# Define collate_fn
def collate_fn(batch):
    input_values = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return input_values, attention_mask, labels

# Create DataLoaders
print("Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
print("DataLoaders created")

# Compute class weights for imbalanced dataset
labels = [asvspoof_dataset.labels[i] for i in range(len(asvspoof_dataset))]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Model Training Setup
print("Setting up optimizer and criterion...")
optimizer = Adam([
    {'params': backbone_model.parameters(), 'lr': 1e-5},  # Lower LR for backbone
    {'params': classifier.parameters(), 'lr': 1e-4}      # Higher LR for classifier
])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Function to compute EER
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    return eer

# Training Loop
epochs = 20
for epoch in range(epochs):
    classifier.train()
    backbone_model.train()
    total_loss, total_acc = 0, 0
    
    print(f"Starting epoch {epoch+1}")
    for i, batch in enumerate(train_loader):
        inputs, attention_mask, labels = batch
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
        
        # Extract features using Wav2Vec2
        features = backbone_model(inputs, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_acc += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
    
    # Validation
    classifier.eval()
    backbone_model.eval()
    val_loss, val_acc, val_labels, val_scores = 0, 0, [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, attention_mask, labels = batch
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            features = backbone_model(inputs, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Spoof class scores
            val_labels.extend(labels.cpu().numpy())
            val_scores.extend(scores)
            
            preds = torch.argmax(outputs, dim=1)
            val_acc += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    eer = compute_eer(val_labels, val_scores)
    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}, Val EER: {eer:.4f}")
    
    scheduler.step(avg_val_loss)

# Save model after training
torch.save({
    'backbone_state_dict': backbone_model.state_dict(),
    'classifier_state_dict': classifier.state_dict()
}, "audio_classifier.pth")
print("Training complete! Model saved to audio_classifier.pth")