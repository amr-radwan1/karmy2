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

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5):
        super(SiameseNetwork, self).__init__()
        
        # Convolutional blocks for initial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size, hidden_size*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Shared feature extractor with LSTM (for time series data)
        self.lstm = nn.LSTM(
            hidden_size*2, hidden_size, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout_rate, 
            bidirectional=True
        )
        
        # Self-attention mechanism
        self.query = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.key = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.value = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),  # Concatenated features
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fully connected layers for similarity scoring
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Final activation is separate 
        self.final_activation = nn.Sigmoid()
        
    def self_attention(self, x):
        # Apply self-attention mechanism
        q = self.query(x)  # (batch, seq_len, hidden*2)
        k = self.key(x)    # (batch, seq_len, hidden*2)
        v = self.value(x)  # (batch, seq_len, hidden*2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)
        
        return context, attn_weights
    
    def forward_one(self, x):
        # Transpose for Conv1d (batch, seq_len, features) -> (batch, features, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Apply CNN feature extraction
        x_conv = self.conv_layers(x_conv)
        
        # Transpose back for LSTM (batch, features, seq_len) -> (batch, seq_len, features)
        x_conv = x_conv.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x_conv)
        
        # Apply self-attention
        context, attn_weights = self.self_attention(lstm_out)
        
        # Global pooling for sequence embedding
        global_max_pool = torch.max(context, dim=1)[0]
        global_avg_pool = torch.mean(context, dim=1)
        
        # Concatenate pooled features
        pooled_features = torch.cat([global_max_pool, global_avg_pool], dim=1)
        
        # Apply fusion
        fused = self.fusion(pooled_features)
        
        return fused, attn_weights
        
    def forward(self, input1, input2):
        # Extract features from both inputs
        output1, attn1 = self.forward_one(input1)
        output2, attn2 = self.forward_one(input2)
        
        # Calculate both absolute difference and elementwise product (for more signal)
        diff = torch.abs(output1 - output2)
        
        # Pass through similarity network
        similarity_logits = self.fc(diff)
        similarity = self.final_activation(similarity_logits)
        
        return similarity, similarity_logits, attn1, attn2

class AllDataPairDataset(Dataset):
    def __init__(self, csv_file, device='cuda', normalize=True, augment=False, 
                 pair_strategy='balanced', max_pairs_per_class=None):
        self.data = pd.read_csv(csv_file)
        self.device = device
        self.normalize = normalize
        self.augment = augment
        self.pair_strategy = pair_strategy
        self.max_pairs_per_class = max_pairs_per_class
        
        # Time-series features
        self.array_features = [
            'angle', 'angular_velocity', 'angular_acceleration',
            'angle2', 'angular_velocity2', 'angular_acceleration2',
            'angle3', 'angular_velocity3', 'angular_acceleration3',
            'angle4', 'angular_velocity4', 'angular_acceleration4',
            'distance1', 'velocity1', 'acceleration1',
        ]
        
        # Determine sequence length
        self._determine_sequence_length()
        
        # Parse video labels and organize data
        self._organize_videos_by_label()
        
        self.pairs = []     

        # Calculate normalization stats if needed
        if self.normalize:
            self._calculate_normalization_stats()
    
    def _determine_sequence_length(self):
        """Determine the mean sequence length from angle arrays"""
        angle_lengths = []
        for idx, row in self.data.iterrows():
            angle_value = row.get('angle')
            if isinstance(angle_value, str):
                try:
                    array = self._parse_array(angle_value)
                    if len(array) > 0:
                        angle_lengths.append(len(array))
                except:
                    pass
        
        if angle_lengths:
            self.mean_sequence_length = int(np.mean(angle_lengths))
            print(f"Mean sequence length: {self.mean_sequence_length}")
        else:
            self.mean_sequence_length = 350
            print(f"Using default sequence length: {self.mean_sequence_length}")
    
    def _organize_videos_by_label(self):
        """Organize videos by their labels (REAL vs FAKE)"""
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
        
        print(f"Found {len(self.real_videos)} REAL videos and {len(self.fake_videos)} FAKE videos")
    
    def create_pairs_for_split(self, real_videos, fake_videos):
        """Create pairs using only videos from a specific split"""
        pairs = []
        
        # Same-class pairs
        for i in range(len(real_videos)):
            for j in range(i + 1, len(real_videos)):
                pairs.append({'video1': real_videos[i], 'video2': real_videos[j], 'label': 1})
        
        for i in range(len(fake_videos)):
            for j in range(i + 1, len(fake_videos)):
                pairs.append({'video1': fake_videos[i], 'video2': fake_videos[j], 'label': 1})
        
        # Different-class pairs
        for real_video in real_videos:
            for fake_video in fake_videos:
                pairs.append({'video1': real_video, 'video2': fake_video, 'label': 0})
        
        random.shuffle(pairs)
        return pairs
    
    def _create_balanced_pairs(self):
        """Create balanced same-class and different-class pairs with stratified sampling"""
        # Same-class pairs (REAL-REAL and FAKE-FAKE) -> label = 1
        same_class_pairs = []
        
        # REAL-REAL pairs
        for i in range(len(self.real_videos)):
            for j in range(i + 1, len(self.real_videos)):
                same_class_pairs.append({
                    'video1': self.real_videos[i],
                    'video2': self.real_videos[j],
                    'label': 1,
                    'pair_type': 'real_real'  # Add pair type for stratification
                })
        
        # FAKE-FAKE pairs
        for i in range(len(self.fake_videos)):
            for j in range(i + 1, len(self.fake_videos)):
                same_class_pairs.append({
                    'video1': self.fake_videos[i],
                    'video2': self.fake_videos[j],
                    'label': 1,
                    'pair_type': 'fake_fake'
                })
        
        # Limit same-class pairs if specified
        if self.max_pairs_per_class:
            same_class_pairs = random.sample(same_class_pairs, 
                                        min(len(same_class_pairs), self.max_pairs_per_class))
        
        # Different-class pairs (REAL-FAKE) -> label = 0
        diff_class_pairs = []
        for real_video in self.real_videos:
            for fake_video in self.fake_videos:
                diff_class_pairs.append({
                    'video1': real_video,
                    'video2': fake_video,
                    'label': 0,
                    'pair_type': 'real_fake'
                })
        
        # Balance the number of different-class pairs with same-class pairs
        target_diff_pairs = len(same_class_pairs)
        if len(diff_class_pairs) > target_diff_pairs:
            diff_class_pairs = random.sample(diff_class_pairs, target_diff_pairs)
        
        # Combine all pairs
        self.pairs = same_class_pairs + diff_class_pairs
        
        # Shuffle the pairs
        random.shuffle(self.pairs)
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for normalization"""
        self.feature_means = {}
        self.feature_stds = {}
        
        for feature in self.array_features:
            if feature in self.data.columns:
                feature_values = []
                for value in self.data[feature].values:
                    if isinstance(value, str):
                        try:
                            array = self._parse_array(value)
                            if len(array) > 0:
                                feature_values.extend(array)
                        except:
                            pass
                
                if feature_values:
                    self.feature_means[feature] = np.mean(feature_values)
                    self.feature_stds[feature] = np.std(feature_values) + 1e-6
                else:
                    self.feature_means[feature] = 0.0
                    self.feature_stds[feature] = 1.0
    
    def _parse_array(self, value):
        """Parse array strings, including Python list format"""
        if isinstance(value, str):
            try:
                cleaned_value = value.replace('\n', '').strip()
                return np.array(ast.literal_eval(cleaned_value))
            except (ValueError, SyntaxError):
                try:
                    numbers = re.findall(r'-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|\d+', value)
                    if numbers:
                        return np.array([float(num) for num in numbers])
                except Exception:
                    return np.array([])
        elif isinstance(value, (list, np.ndarray)):
            return np.array(value)
        elif isinstance(value, (int, float)) and not pd.isna(value):
            return np.array([float(value)])
        else:
            return np.array([])
    
    def _process_feature(self, feature_data, feature_name):
        """Process and normalize a feature"""
        if self.normalize and feature_name in self.feature_means:
            feature_data = (feature_data - self.feature_means[feature_name]) / self.feature_stds[feature_name]
        
        if len(feature_data) > self.mean_sequence_length:
            feature_data = feature_data[:self.mean_sequence_length]
        else:
            feature_data = np.pad(feature_data, (0, self.mean_sequence_length - len(feature_data)), 'constant')
        
        return feature_data
    
    def _augment_time_series(self, features, noise_level=0.05):
        """Apply data augmentation to time series"""
        if not self.augment or np.random.random() > 0.4:
            return features
            
        seq_len, feat_dim = features.shape
        
        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, noise_level, size=features.shape)
            features = features + noise
            
        # Random masking
        if np.random.random() > 0.7:
            mask_len = np.random.randint(5, 20)
            mask_start = np.random.randint(0, seq_len - mask_len)
            features[mask_start:mask_start+mask_len, :] = 0
        
        return features
    
    def extract_features(self, video_name):
        """Extract features for a single video"""
        video_data = self.data[self.data['video_name'] == video_name]
        
        if video_data.empty:
            return torch.zeros((self.mean_sequence_length, len(self.array_features) * 2), dtype=torch.float32)
        
        # Extract array features
        array_features_list = []
        for feature in self.array_features:
            if feature in video_data.columns:
                value = video_data[feature].values[0] if len(video_data) > 0 else None
                
                try:
                    feature_array = self._parse_array(value)
                    if feature_array is not None and feature_array.size > 0:
                        processed_feature = self._process_feature(feature_array, feature)
                        array_features_list.append(processed_feature)
                    else:
                        array_features_list.append(np.zeros(self.mean_sequence_length))
                except Exception:
                    array_features_list.append(np.zeros(self.mean_sequence_length))
            else:
                array_features_list.append(np.zeros(self.mean_sequence_length))
        
        # Stack features
        try:
            all_features = np.stack(array_features_list, axis=1)
            all_features = all_features.astype(np.float32)
        except Exception:
            all_features = np.zeros((self.mean_sequence_length, len(self.array_features)), dtype=np.float32)
        
        # Calculate derivatives
        try:
            derivatives = np.zeros_like(all_features)
            for i in range(1, len(all_features)):
                derivatives[i] = all_features[i] - all_features[i-1]
            all_features = np.concatenate([all_features, derivatives], axis=1)
        except Exception:
            derivatives = np.zeros_like(all_features)
            all_features = np.concatenate([all_features, derivatives], axis=1)
        
        # Apply augmentation
        if self.augment:
            all_features = self._augment_time_series(all_features)
        
        # Replace NaNs
        all_features = np.nan_to_num(all_features, nan=0.0)
        
        return torch.FloatTensor(all_features)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        features1 = self.extract_features(pair["video1"])
        features2 = self.extract_features(pair["video2"])
        label = float(pair["label"])
        
        return features1.to(self.device), features2.to(self.device), torch.FloatTensor([label]).to(self.device)

def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def create_video_level_splits(real_videos, fake_videos, val_size=0.15, test_size=0.15, random_state=42):
    """Split videos first, then create pairs within each split"""
    random.seed(random_state)
    
    # Shuffle videos
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    # Calculate split indices
    n_real = len(real_videos)
    n_fake = len(fake_videos)
    
    real_test_idx = int(n_real * test_size)
    real_val_idx = int(n_real * val_size)
    
    fake_test_idx = int(n_fake * test_size)
    fake_val_idx = int(n_fake * val_size)
    
    return {
        'train': {
            'real': real_videos[real_val_idx + real_test_idx:],
            'fake': fake_videos[fake_val_idx + fake_test_idx:]
        },
        'val': {
            'real': real_videos[real_test_idx:real_test_idx + real_val_idx],
            'fake': fake_videos[fake_test_idx:fake_test_idx + fake_val_idx]
        },
        'test': {
            'real': real_videos[:real_test_idx],
            'fake': fake_videos[:fake_test_idx]
        }
    }


def train_enhanced_siamese_model(csv_path, batch_size=16, num_epochs=6, learning_rate=0.0001, 
                               hidden_size=128, device='cuda', val_size=0.15, test_size=0.15, 
                               pair_strategy='balanced', max_pairs_per_class=5000,
                               augment=True, find_threshold=True, log_dir='./runs'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create base dataset (without pairs initially)
    base_dataset = AllDataPairDataset(
        csv_path, 
        device=device, 
        normalize=True, 
        augment=False,  # We'll handle augmentation per split
        pair_strategy=pair_strategy,
        max_pairs_per_class=max_pairs_per_class
    )

    # Split videos first to prevent leakage
    video_splits = create_video_level_splits(
        base_dataset.real_videos, 
        base_dataset.fake_videos, 
        val_size=val_size, 
        test_size=test_size
    )

    # Create pairs for training split
    train_pairs = base_dataset.create_pairs_for_split(
        video_splits['train']['real'], 
        video_splits['train']['fake']
    )

    # Create pairs for validation split  
    val_pairs = base_dataset.create_pairs_for_split(
        video_splits['val']['real'], 
        video_splits['val']['fake']
    )

    # Create pairs for test split
    test_pairs = base_dataset.create_pairs_for_split(
        video_splits['test']['real'], 
        video_splits['test']['fake']
    )

    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Calculate label distribution from training data
    label_counts = {}
    for pair in train_pairs:
        label = pair['label']
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    print("\nTraining label distribution:")
    for label, count in label_counts.items():
        label_name = "Same class (consistent)" if label == 1 else "Different class (inconsistent)"
        print(f"  {label_name}: {count} samples ({count/len(train_pairs)*100:.1f}%)")
    
    # Create custom datasets for each split
    class PairSubset(Dataset):
        def __init__(self, original_dataset, pairs, augment=False):
            self.original_dataset = original_dataset
            self.pairs = pairs
            self.augment = augment
            # Temporarily override augmentation setting
            self.original_augment = original_dataset.augment
            
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            # Set augmentation for this batch
            self.original_dataset.augment = self.augment
            
            pair = self.pairs[idx]
            features1 = self.original_dataset.extract_features(pair["video1"])
            features2 = self.original_dataset.extract_features(pair["video2"])
            label = float(pair["label"])
            
            # Restore original augmentation setting
            self.original_dataset.augment = self.original_augment
            
            return features1.to(device), features2.to(device), torch.FloatTensor([label]).to(device)
    
    train_dataset = PairSubset(base_dataset, train_pairs, augment=augment)
    val_dataset = PairSubset(base_dataset, val_pairs, augment=False)
    test_dataset = PairSubset(base_dataset, test_pairs, augment=False)
    
    print(f"\nTrain samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    sample_data = next(iter(DataLoader(train_dataset, batch_size=1)))    
    input_size = sample_data[0].shape[2]
    print(f"Input size: {input_size} features")
    
    model = SiameseNetwork(input_size=input_size, hidden_size=hidden_size).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable out of {total_params:,} total")
    
    # Log model architecture to TensorBoard
    writer.add_text('Model/Architecture', str(model))
    writer.add_scalar('Model/Total_Parameters', total_params)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params)
    
    # Enhanced loss function with class balancing
    pos_weight = torch.tensor([label_counts[0] / label_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop with TensorBoard logging
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': [], 'lr': []}
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 5
    
    for epoch in range(num_epochs):
        clear_cuda_cache()
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch_idx, (x1, x2, labels) in enumerate(train_loop):
            similarity, logits, attn1, attn2 = model(x1, x2)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate batch accuracy
            predicted = (similarity > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            train_loop.set_postfix(loss=loss.item())
            
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            writer.add_scalar('Accuracy/Train_Batch', (predicted == labels).float().mean().item(), global_step)
            
            # Log attention weights periodically
            if batch_idx % 100 == 0:
                writer.add_histogram('Attention/Weights1', attn1[0].cpu().detach(), global_step)
                writer.add_histogram('Attention/Weights2', attn2[0].cpu().detach(), global_step)
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for x1, x2, labels in val_loop:
                similarity, logits, _, _ = model(x1, x2)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_outputs.extend(similarity.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Calculate validation accuracy
                predicted = (similarity > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                val_loop.set_postfix(loss=loss.item())
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels, val_outputs)
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_acc)
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
        writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
        writer.add_scalar('Accuracy/Train_Epoch', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation_Epoch', val_acc, epoch)
        writer.add_scalar('AUC/Validation', val_auc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Add model weights histogram
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param.cpu().detach(), epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad.cpu().detach(), epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model! AUC: {val_auc:.4f}")
            # Log best model metrics
            writer.add_scalar('Best/AUC', val_auc, epoch)
            writer.add_scalar('Best/Accuracy', val_acc, epoch)
        else:
            patience_counter += 1
            
        if patience_counter >= patience :
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
              f"AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
    
    model.load_state_dict(best_model_state)
    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    
    # Find optimal threshold
    threshold = 0.5
    if find_threshold:
        model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                similarity, _, _, _ = model(x1, x2)
                all_outputs.extend(similarity.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        thresholds = np.linspace(0.1, 0.9, 100)
        f1_scores = []
        accuracies = []
        
        for thresh in thresholds:
            pred = (np.array(all_outputs) > thresh).astype(int)
            f1 = f1_score(np.array(all_labels).astype(int), pred)
            acc = np.mean(pred == np.array(all_labels).astype(int))
            f1_scores.append(f1)
            accuracies.append(acc)
        
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
        print(f"Optimal threshold: {threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
        
        # Log threshold optimization to TensorBoard
        for i, (thresh, f1, acc) in enumerate(zip(thresholds, f1_scores, accuracies)):
            writer.add_scalar('Threshold_Optimization/F1_Score', f1, i)
            writer.add_scalar('Threshold_Optimization/Accuracy', acc, i)
            writer.add_scalar('Threshold_Optimization/Threshold', thresh, i)
        
        writer.add_scalar('Optimal/Threshold', threshold)
        writer.add_scalar('Optimal/F1_Score', f1_scores[best_idx])
    
    # Log final dataset statistics
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    total_real_videos = len(base_dataset.real_videos)
    total_fake_videos = len(base_dataset.fake_videos)

    writer.add_scalar('Dataset/Total_Samples', total_samples)
    writer.add_scalar('Dataset/Train_Samples', len(train_dataset))
    writer.add_scalar('Dataset/Val_Samples', len(val_dataset))
    writer.add_scalar('Dataset/Test_Samples', len(test_dataset))
    writer.add_scalar('Dataset/Total_Real_Videos', total_real_videos)
    writer.add_scalar('Dataset/Total_Fake_Videos', total_fake_videos)

    # Log video splits
    writer.add_scalar('Dataset/Train_Real_Videos', len(video_splits['train']['real']))
    writer.add_scalar('Dataset/Train_Fake_Videos', len(video_splits['train']['fake']))
    writer.add_scalar('Dataset/Val_Real_Videos', len(video_splits['val']['real']))
    writer.add_scalar('Dataset/Val_Fake_Videos', len(video_splits['val']['fake']))
    writer.add_scalar('Dataset/Test_Real_Videos', len(video_splits['test']['real']))
    writer.add_scalar('Dataset/Test_Fake_Videos', len(video_splits['test']['fake']))
    
    test_results = evaluate_on_test_set(model, test_dataset, threshold, device, writer)
    
    # Close TensorBoard writer
    writer.close()

    # Final evaluation
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, labels in val_loader:
            similarity, _, _, _ = model(x1, x2)
            all_outputs.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predicted = (np.array(all_outputs) > threshold).astype(int)
    true = np.array(all_labels).astype(int)
    
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print(f"AUC: {roc_auc_score(true, all_outputs):.4f}")
    print(f"Optimal threshold: {threshold:.4f}")
    print("\nClassification Report:")
    print(classification_report(true, predicted, target_names=['Different Class', 'Same Class']))
    
    cm = confusion_matrix(true, predicted)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Diff  Same")
    print(f"Actual Diff  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Same  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_auc'], label='Val AUC', color='green')
    plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate', color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history, threshold, base_dataset, video_splits, test_results

def evaluate_model_on_videos(model, base_dataset, video_split, threshold=0.5, device='cuda', max_comparisons_per_type=200):
    """
    Evaluate the model by comparing videos (optimized version)
    """
    model.eval()
    
    real_videos = video_split['real']
    fake_videos = video_split['fake']
    
    print(f"\nEvaluating on {len(real_videos)} REAL and {len(fake_videos)} FAKE videos")
    print(f"Limiting to {max_comparisons_per_type} comparisons per type for efficiency")
    
    results = {
        'real_vs_real': [],
        'fake_vs_fake': [],
        'real_vs_fake': []
    }
    
    # Real vs Real comparisons (sample randomly to limit)
    print("Evaluating REAL vs REAL pairs...")
    real_pairs = [(i, j) for i in range(len(real_videos)) for j in range(i + 1, len(real_videos))]
    if len(real_pairs) > max_comparisons_per_type:
        real_pairs = random.sample(real_pairs, max_comparisons_per_type)
    
    # Batch process for efficiency
    batch_size = 32
    for batch_start in tqdm(range(0, len(real_pairs), batch_size), desc="Real vs Real"):
        batch_pairs = real_pairs[batch_start:batch_start + batch_size]
        
        # Prepare batch
        features1_batch = []
        features2_batch = []
        
        for i, j in batch_pairs:
            features1 = base_dataset.extract_features(real_videos[i])
            features2 = base_dataset.extract_features(real_videos[j])
            features1_batch.append(features1)
            features2_batch.append(features2)
        
        if features1_batch:
            # Stack into batch tensors
            features1_tensor = torch.stack(features1_batch).to(device)
            features2_tensor = torch.stack(features2_batch).to(device)
            
            with torch.no_grad():
                similarities, _, _, _ = model(features1_tensor, features2_tensor)
                results['real_vs_real'].extend(similarities.cpu().numpy().flatten())
    
    # Fake vs Fake comparisons (sample randomly to limit)
    print("Evaluating FAKE vs FAKE pairs...")
    fake_pairs = [(i, j) for i in range(len(fake_videos)) for j in range(i + 1, len(fake_videos))]
    if len(fake_pairs) > max_comparisons_per_type:
        fake_pairs = random.sample(fake_pairs, max_comparisons_per_type)
    
    for batch_start in tqdm(range(0, len(fake_pairs), batch_size), desc="Fake vs Fake"):
        batch_pairs = fake_pairs[batch_start:batch_start + batch_size]
        
        features1_batch = []
        features2_batch = []
        
        for i, j in batch_pairs:
            features1 = base_dataset.extract_features(fake_videos[i])
            features2 = base_dataset.extract_features(fake_videos[j])
            features1_batch.append(features1)
            features2_batch.append(features2)
        
        if features1_batch:
            features1_tensor = torch.stack(features1_batch).to(device)
            features2_tensor = torch.stack(features2_batch).to(device)
            
            with torch.no_grad():
                similarities, _, _, _ = model(features1_tensor, features2_tensor)
                results['fake_vs_fake'].extend(similarities.cpu().numpy().flatten())
    
    # Real vs Fake comparisons (sample randomly)
    print("Evaluating REAL vs FAKE pairs...")
    real_fake_pairs = []
    for _ in range(max_comparisons_per_type):
        if real_videos and fake_videos:
            real_video = random.choice(real_videos)
            fake_video = random.choice(fake_videos)
            real_fake_pairs.append((real_video, fake_video))
    
    for batch_start in tqdm(range(0, len(real_fake_pairs), batch_size), desc="Real vs Fake"):
        batch_pairs = real_fake_pairs[batch_start:batch_start + batch_size]
        
        features1_batch = []
        features2_batch = []
        
        for real_video, fake_video in batch_pairs:
            features1 = base_dataset.extract_features(real_video)
            features2 = base_dataset.extract_features(fake_video)
            features1_batch.append(features1)
            features2_batch.append(features2)
        
        if features1_batch:
            features1_tensor = torch.stack(features1_batch).to(device)
            features2_tensor = torch.stack(features2_batch).to(device)
            
            with torch.no_grad():
                similarities, _, _, _ = model(features1_tensor, features2_tensor)
                results['real_vs_fake'].extend(similarities.cpu().numpy().flatten())
        
        # Clear cache periodically
        if batch_start % (batch_size * 4) == 0:
            clear_cuda_cache()
    
    # Print statistics
    print("\n" + "="*60)
    print("VIDEO COMPARISON RESULTS")
    print("="*60)
    
    for comparison_type, similarities in results.items():
        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            above_threshold = np.mean(np.array(similarities) > threshold)
            
            print(f"\n{comparison_type.upper().replace('_', ' ')} ({len(similarities)} comparisons):")
            print(f"  Mean similarity: {mean_sim:.4f} ± {std_sim:.4f}")
            print(f"  Above threshold ({threshold:.3f}): {above_threshold:.1%}")
            print(f"  Min: {np.min(similarities):.4f}, Max: {np.max(similarities):.4f}")
    
    # Plot similarity distributions (same as before)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    if results['real_vs_real']:
        plt.hist(results['real_vs_real'], bins=30, alpha=0.7, color='blue', density=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
        plt.title('REAL vs REAL\n(Should be HIGH similarity)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    if results['fake_vs_fake']:
        plt.hist(results['fake_vs_fake'], bins=30, alpha=0.7, color='green', density=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
        plt.title('FAKE vs FAKE\n(Should be HIGH similarity)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if results['real_vs_fake']:
        plt.hist(results['real_vs_fake'], bins=30, alpha=0.7, color='red', density=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
        plt.title('REAL vs FAKE\n(Should be LOW similarity)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def save_model(model, threshold, filepath='enhanced_siamese_model.pth'):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'model_config': {
            'input_size': model.conv_layers[0].in_channels,
            'hidden_size': 128,  # Default from initialization
            'dropout_rate': 0.5
        }
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device='cuda'):
    """Load a trained model"""
    checkpoint = torch.load(filepath, map_location=device)
    
    config = checkpoint['model_config']
    model = SiameseNetwork(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']
    
    print(f"Model loaded from {filepath}")
    print(f"Threshold: {threshold:.4f}")
    
    return model, threshold

def evaluate_on_test_set(model, test_dataset, threshold, device, writer=None):
    """Evaluate the final model on the test set with TensorBoard logging"""
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    model.eval()
    test_outputs = []
    test_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for x1, x2, labels in tqdm(test_loader, desc="Test evaluation"):
            similarity, _, _, _ = model(x1, x2)
            test_outputs.extend(similarity.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate test metrics
    test_auc = roc_auc_score(test_labels, test_outputs)
    test_pred = (np.array(test_outputs) > threshold).astype(int)
    test_acc = np.mean(test_pred == np.array(test_labels).astype(int))
    test_f1 = f1_score(np.array(test_labels).astype(int), test_pred)
    
    if writer is not None:
        writer.add_scalar('Test/AUC', test_auc)
        writer.add_scalar('Test/Accuracy', test_acc)
        writer.add_scalar('Test/F1_Score', test_f1)
        
        writer.add_histogram('Test/Predictions', np.array(test_outputs))
        writer.add_histogram('Test/Labels', np.array(test_labels))
    
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Using threshold: {threshold:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(np.array(test_labels).astype(int), test_pred, 
                              target_names=['Different Class', 'Same Class']))
    
    test_cm = confusion_matrix(np.array(test_labels).astype(int), test_pred)
    print(f"\nTest Set Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Diff  Same")
    print(f"Actual Diff  {test_cm[0,0]:4d}  {test_cm[0,1]:4d}")
    print(f"       Same  {test_cm[1,0]:4d}  {test_cm[1,1]:4d}")
    
    return {
        'auc': test_auc,
        'accuracy': test_acc,
        'f1_score': test_f1,
        'predictions': test_pred,
        'labels': np.array(test_labels).astype(int),
        'probabilities': np.array(test_outputs)
    }


if __name__ == "__main__":
    model, history, best_threshold, base_dataset, video_splits, test_results = train_enhanced_siamese_model(
        csv_path='../master_features.csv', 
        batch_size=16,
        num_epochs=100,
        learning_rate=0.0001,
        hidden_size=128,
        device='cuda',
        test_size=0.2,
        pair_strategy='balanced',
        max_pairs_per_class=5000,
        augment=True,
        find_threshold=True,
        log_dir=f"./runs/siamese_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    save_model(model, best_threshold, 'enhanced_siamese_model.pth')
    
    print(f"\nFinal Test Set Performance:")
    print(f"AUC: {test_results['auc']:.4f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print("\nTraining completed successfully!")

    video_results = evaluate_model_on_videos(model, base_dataset, video_splits['test'], threshold=best_threshold)
    
