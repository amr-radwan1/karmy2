import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random
from collections import defaultdict
import gc
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

class SimplifiedSiameseNetwork(nn.Module):
    """Simplified version for debugging"""
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
        super(SimplifiedSiameseNetwork, self).__init__()
        
        # Much simpler architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Simple similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward_one(self, x):
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # (batch, features)
        return self.feature_extractor(x)
        
    def forward(self, input1, input2):
        # Extract features
        feat1 = self.forward_one(input1)
        feat2 = self.forward_one(input2)
        
        # Simple absolute difference
        diff = torch.abs(feat1 - feat2)
        
        # Get similarity score
        similarity_logits = self.similarity_net(diff)
        similarity = torch.sigmoid(similarity_logits)
        
        return similarity, similarity_logits, None, None

class DebugDataset(Dataset):
    """Simplified dataset for debugging"""
    def __init__(self, csv_file, device='cuda', max_samples=1000):
        self.data = pd.read_csv(csv_file)
        self.device = device
        self.max_samples = max_samples
        
        # Use only a subset for debugging
        if len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42)
        
        # Time-series features
        self.array_features = [
            'angle', 'angular_velocity', 'angular_acceleration',
            'angle2', 'angular_velocity2', 'angular_acceleration2',
            'distance1', 'velocity1', 'acceleration1',
        ]
        
        # Determine sequence length
        self.sequence_length = 100  # Fixed shorter length for debugging
        
        # Parse video labels
        self._organize_videos_by_label()
        
        # Create simple pairs
        self._create_simple_pairs()
        
        print(f"Debug dataset: {len(self.pairs)} pairs from {len(self.data)} samples")
    
    def _organize_videos_by_label(self):
        """Organize videos by their labels"""
        self.real_videos = []
        self.fake_videos = []
        
        for video_name in self.data['video_name'].unique():
            if pd.isna(video_name):
                continue
                
            video_str = str(video_name)
            if 'REAL' in video_str.upper():
                self.real_videos.append(video_name)
            elif 'FAKE' in video_str.upper():
                self.fake_videos.append(video_name)
        
        print(f"Debug: {len(self.real_videos)} REAL, {len(self.fake_videos)} FAKE videos")
    
    def _create_simple_pairs(self):
        """Create a small, balanced set of pairs for debugging"""
        self.pairs = []
        
        # Limit to small numbers for debugging
        max_per_type = min(50, len(self.real_videos), len(self.fake_videos))
        
        # Same class pairs (label = 1)
        for i in range(min(max_per_type, len(self.real_videos))):
            for j in range(i + 1, min(max_per_type + 1, len(self.real_videos))):
                if j < len(self.real_videos):
                    self.pairs.append({
                        'video1': self.real_videos[i],
                        'video2': self.real_videos[j],
                        'label': 1
                    })
        
        for i in range(min(max_per_type, len(self.fake_videos))):
            for j in range(i + 1, min(max_per_type + 1, len(self.fake_videos))):
                if j < len(self.fake_videos):
                    self.pairs.append({
                        'video1': self.fake_videos[i],
                        'video2': self.fake_videos[j],
                        'label': 1
                    })
        
        # Different class pairs (label = 0) - same amount
        same_class_count = len(self.pairs)
        diff_pairs = []
        for real_video in self.real_videos[:max_per_type]:
            for fake_video in self.fake_videos[:max_per_type]:
                diff_pairs.append({
                    'video1': real_video,
                    'video2': fake_video,
                    'label': 0
                })
        
        # Balance the classes
        if len(diff_pairs) > same_class_count:
            diff_pairs = random.sample(diff_pairs, same_class_count)
        
        self.pairs.extend(diff_pairs)
        random.shuffle(self.pairs)
        
        # Print label distribution
        labels = [p['label'] for p in self.pairs]
        print(f"Label distribution: 0={labels.count(0)}, 1={labels.count(1)}")
    
    def _parse_array(self, value):
        """Parse array strings"""
        if isinstance(value, str):
            try:
                cleaned_value = value.replace('\n', '').strip()
                return np.array(ast.literal_eval(cleaned_value))
            except:
                try:
                    numbers = re.findall(r'-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|\d+', value)
                    if numbers:
                        return np.array([float(num) for num in numbers])
                except:
                    return np.array([])
        return np.array([])
    
    def extract_features(self, video_name):
        """Extract simplified features"""
        video_data = self.data[self.data['video_name'] == video_name]
        
        if video_data.empty:
            return torch.zeros((self.sequence_length, len(self.array_features)), dtype=torch.float32)
        
        # Extract features
        features_list = []
        for feature in self.array_features:
            if feature in video_data.columns:
                value = video_data[feature].values[0]
                feature_array = self._parse_array(value)
                
                if len(feature_array) > 0:
                    # Truncate or pad to fixed length
                    if len(feature_array) > self.sequence_length:
                        feature_array = feature_array[:self.sequence_length]
                    else:
                        feature_array = np.pad(feature_array, 
                                             (0, self.sequence_length - len(feature_array)), 
                                             'constant')
                    features_list.append(feature_array)
                else:
                    features_list.append(np.zeros(self.sequence_length))
            else:
                features_list.append(np.zeros(self.sequence_length))
        
        # Stack features
        try:
            all_features = np.stack(features_list, axis=1)
            all_features = all_features.astype(np.float32)
        except:
            all_features = np.zeros((self.sequence_length, len(self.array_features)), dtype=np.float32)
        
        # Replace NaNs and infs
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return torch.FloatTensor(all_features)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        features1 = self.extract_features(pair["video1"])
        features2 = self.extract_features(pair["video2"])
        label = float(pair["label"])
        
        return features1.to(self.device), features2.to(self.device), torch.FloatTensor([label]).to(self.device)

def debug_data_loading(csv_path):
    """Debug function to check data loading"""
    print("=== DEBUGGING DATA LOADING ===")
    
    # Load dataset
    dataset = DebugDataset(csv_path, max_samples=500)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Check first few batches
    for i, (x1, x2, labels) in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"  x1 shape: {x1.shape}")
        print(f"  x2 shape: {x2.shape}")
        print(f"  labels: {labels.flatten()}")
        print(f"  x1 stats: min={x1.min():.4f}, max={x1.max():.4f}, mean={x1.mean():.4f}")
        print(f"  x2 stats: min={x2.min():.4f}, max={x2.max():.4f}, mean={x2.mean():.4f}")
        print(f"  x1 has NaN: {torch.isnan(x1).any()}")
        print(f"  x2 has NaN: {torch.isnan(x2).any()}")
        
        if i >= 2:  # Check first 3 batches
            break
    
    return dataset

def debug_model_training(csv_path, epochs=10):
    """Debug version of training with extensive logging"""
    print("=== DEBUGGING MODEL TRAINING ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create debug dataset
    dataset = DebugDataset(csv_path, device=device, max_samples=500)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Get input size
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[2]
    print(f"Input size: {input_size}")
    
    # Create simplified model
    model = SimplifiedSiameseNetwork(input_size=input_size, hidden_size=32, dropout_rate=0.2).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Simple optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop with detailed logging
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_accs = []
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            # Forward pass
            similarity, logits, _, _ = model(x1, x2)
            loss = criterion(logits, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                print(f"  Logits: {logits.flatten()}")
                print(f"  Labels: {labels.flatten()}")
                return None
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            predicted = (similarity > 0.5).float()
            acc = (predicted == labels).float().mean().item()
            train_accs.append(acc)
            
            # Log every few batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: loss={loss.item():.4f}, acc={acc:.4f}, grad_norm={grad_norm:.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        val_accs = []
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                similarity, logits, _, _ = model(x1, x2)
                loss = criterion(logits, labels)
                
                val_losses.append(loss.item())
                predicted = (similarity > 0.5).float()
                acc = (predicted == labels).float().mean().item()
                val_accs.append(acc)
                
                val_outputs.extend(similarity.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        
        try:
            val_auc = roc_auc_score(val_labels, val_outputs)
        except:
            val_auc = 0.5
        
        print(f"Epoch {epoch+1:2d}/{epochs} - Train: {train_loss:.4f}/{train_acc:.4f}, "
              f"Val: {val_loss:.4f}/{val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Check if learning
        if epoch > 3 and abs(train_loss - 0.693) < 0.01:
            print("WARNING: Model not learning (loss stuck at log(2))")
    
    return model, dataset

def analyze_feature_quality(csv_path):
    """Analyze the quality of features in the dataset"""
    print("=== ANALYZING FEATURE QUALITY ===")
    
    data = pd.read_csv(csv_path)
    
    array_features = [
        'angle', 'angular_velocity', 'angular_acceleration',
        'angle2', 'angular_velocity2', 'angular_acceleration2',
        'distance1', 'velocity1', 'acceleration1',
    ]
    
    def parse_array(value):
        if isinstance(value, str):
            try:
                cleaned_value = value.replace('\n', '').strip()
                return np.array(ast.literal_eval(cleaned_value))
            except:
                return np.array([])
        return np.array([])
    
    for feature in array_features:
        if feature in data.columns:
            print(f"\n{feature}:")
            
            # Check how many are valid
            valid_count = 0
            lengths = []
            all_values = []
            
            for value in data[feature].values:
                arr = parse_array(value)
                if len(arr) > 0:
                    valid_count += 1
                    lengths.append(len(arr))
                    all_values.extend(arr)
            
            print(f"  Valid samples: {valid_count}/{len(data)} ({valid_count/len(data)*100:.1f}%)")
            
            if lengths:
                print(f"  Length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
            
            if all_values:
                all_values = np.array(all_values)
                print(f"  Value stats: min={np.min(all_values):.4f}, max={np.max(all_values):.4f}, "
                      f"mean={np.mean(all_values):.4f}, std={np.std(all_values):.4f}")
                print(f"  NaN/Inf values: {np.sum(~np.isfinite(all_values))}")
    
    # Check video name distribution
    print(f"\nVideo names:")
    real_count = 0
    fake_count = 0
    for video_name in data['video_name'].unique():
        if pd.isna(video_name):
            continue
        video_str = str(video_name)
        if 'REAL' in video_str.upper():
            real_count += 1
        elif 'FAKE' in video_str.upper():
            fake_count += 1
    
    print(f"  REAL videos: {real_count}")
    print(f"  FAKE videos: {fake_count}")
    print(f"  Total unique videos: {data['video_name'].nunique()}")

def quick_baseline_test(csv_path):
    """Test with an even simpler baseline model"""
    print("=== QUICK BASELINE TEST ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create very simple dataset
    dataset = DebugDataset(csv_path, device=device, max_samples=200)
    
    # Simple train/test split
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Get input size
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1] * sample_batch[0].shape[2]  # Flatten
    
    # Ultra-simple model
    class UltraSimpleModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size * 2, 64),  # Concatenate both inputs
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x1, x2):
            # Flatten inputs
            x1_flat = x1.view(x1.size(0), -1)
            x2_flat = x2.view(x2.size(0), -1)
            
            # Concatenate
            combined = torch.cat([x1_flat, x2_flat], dim=1)
            
            # Get output
            out = self.net(combined)
            return out, out, None, None
    
    model = UltraSimpleModel(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print(f"Ultra-simple model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train for a few epochs
    for epoch in range(5):
        model.train()
        train_losses = []
        
        for x1, x2, labels in train_loader:
            pred, _, _, _ = model(x1, x2)
            loss = criterion(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Test
        model.eval()
        test_accs = []
        with torch.no_grad():
            for x1, x2, labels in test_loader:
                pred, _, _, _ = model(x1, x2)
                acc = ((pred > 0.5).float() == labels).float().mean().item()
                test_accs.append(acc)
        
        print(f"Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, Test Acc={np.mean(test_accs):.4f}")

if __name__ == "__main__":
    csv_path = '../master_features.csv'
    
    # Run all diagnostic functions
    print("Starting comprehensive debugging...")
    
    # Step 1: Analyze feature quality
    analyze_feature_quality(csv_path)
    
    # Step 2: Debug data loading
    dataset = debug_data_loading(csv_path)
    
    # Step 3: Try ultra-simple baseline
    quick_baseline_test(csv_path)
    
    # Step 4: Try debug training
    model, dataset = debug_model_training(csv_path, epochs=20)