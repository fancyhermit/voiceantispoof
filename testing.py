import os
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.nn as nn
import torch.nn.functional as F

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the AudioClassifier class (must match the training script)
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

# Load pretrained Wav2Vec2 model and feature extractor
print("Loading Wav2Vec2FeatureExtractor...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
print("Loading Wav2Vec2Model...")
backbone_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
backbone_model.to(device)
backbone_model.eval()  # Set to evaluation mode
print("Wav2Vec2 model loaded and set to eval mode")

# Initialize and load the trained classifier
print("Initializing AudioClassifier...")
classifier = AudioClassifier()
classifier.to(device)
model_path = "audio_classifier.pth"  # Adjust path if needed
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
classifier.load_state_dict(torch.load(model_path, map_location=device))
classifier.eval()  # Set to evaluation mode
print("Classifier loaded and set to eval mode")

# Function to preprocess and classify an audio file
def classify_audio(audio_path, max_length=16000*4):
    try:
        # Load audio
        print(f"Loading audio file: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        waveform, sr = torchaudio.load(audio_path)
        print(f"Loaded waveform with shape {waveform.shape}, sample rate {sr}")

        # Convert stereo to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            print(f"Converting stereo (channels={waveform.shape[0]}) to mono...")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"Waveform after channel processing: {waveform.shape}")

        # Resample to 16kHz if needed
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz...")
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        
        # Remove channel dimension
        waveform = waveform.squeeze(0)
        print(f"Waveform after squeeze: {waveform.shape}")

        # Truncate or pad to match training max_length
        if waveform.shape[0] > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.shape[0]))
        print(f"Waveform after padding/truncation: {waveform.shape}")

        # Extract features using Wav2Vec2FeatureExtractor
        print("Extracting features...")
        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_length=max_length,
            truncation=True,
            return_attention_mask=True
        )
        print(f"Feature extractor output keys: {list(inputs.keys())}")

        # Move inputs to device
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else torch.ones_like(input_values).to(device)
        print(f"Input values shape: {input_values.shape}, Attention mask shape: {attention_mask.shape}")

        # Extract features using Wav2Vec2Model
        with torch.no_grad():
            features = backbone_model(input_values, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        print(f"Extracted features shape: {features.shape}")

        # Classify using AudioClassifier
        with torch.no_grad():
            logits = classifier(features)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

        # Map class index to label
        label_map = {0: "real (bonafide)", 1: "deepfake (spoof)"}
        predicted_label = label_map[predicted_class]

        return predicted_label, confidence

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {str(e)}")
        raise

# Path to your audio sample
audio_path = "C:/Users/Sez/Desktop/aispoof/fem.mp3"  # Update if needed

# Classify the audio
try:
    label, confidence = classify_audio(audio_path)
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.4f}")
except Exception as e:
    print(f"Failed to classify audio: {str(e)}")