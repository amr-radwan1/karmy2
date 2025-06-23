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

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
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
        
        # Final activation is separate (for easy modification of threshold)
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
        
        # Create pairs based on strategy
        self._create_pairs()
        
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
    
    def _create_pairs(self):
        """Create pairs based on the specified strategy"""
        self.pairs = []
        
        if self.pair_strategy == 'balanced':
            self._create_balanced_pairs()
        elif self.pair_strategy == 'all':
            self._create_all_pairs()
        elif self.pair_strategy == 'sampled':
            self._create_sampled_pairs()
        else:
            raise ValueError(f"Unknown pair strategy: {self.pair_strategy}")
        
        print(f"Created {len(self.pairs)} total pairs")
        
        # Count pairs by label
        same_class_pairs = sum(1 for p in self.pairs if p['label'] == 1)
        diff_class_pairs = sum(1 for p in self.pairs if p['label'] == 0)
        print(f"Same class pairs (label=1): {same_class_pairs}")
        print(f"Different class pairs (label=0): {diff_class_pairs}")
    
    def _create_balanced_pairs(self):
        """Create balanced same-class and different-class pairs"""
        # Same-class pairs (REAL-REAL and FAKE-FAKE) -> label = 1
        same_class_pairs = []
        
        # REAL-REAL pairs
        for i in range(len(self.real_videos)):
            for j in range(i + 1, len(self.real_videos)):
                same_class_pairs.append({
                    'video1': self.real_videos[i],
                    'video2': self.real_videos[j],
                    'label': 1
                })
        
        # FAKE-FAKE pairs
        for i in range(len(self.fake_videos)):
            for j in range(i + 1, len(self.fake_videos)):
                same_class_pairs.append({
                    'video1': self.fake_videos[i],
                    'video2': self.fake_videos[j],
                    'label': 1
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
                    'label': 0
                })
        
        # Balance the number of different-class pairs with same-class pairs
        target_diff_pairs = len(same_class_pairs)
        if len(diff_class_pairs) > target_diff_pairs:
            diff_class_pairs = random.sample(diff_class_pairs, target_diff_pairs)
        
        # Combine all pairs
        self.pairs = same_class_pairs + diff_class_pairs
        
        # Shuffle the pairs
        random.shuffle(self.pairs)
    
    def _create_all_pairs(self):
        """Create all possible pairs (memory intensive)"""
        all_videos = self.real_videos + self.fake_videos
        
        for i in range(len(all_videos)):
            for j in range(i + 1, len(all_videos)):
                video1 = all_videos[i]
                video2 = all_videos[j]
                
                # Determine label based on video types
                video1_is_real = 'REAL' in str(video1).upper()
                video2_is_real = 'REAL' in str(video2).upper()
                
                # Same class if both real or both fake
                label = 1 if video1_is_real == video2_is_real else 0
                
                self.pairs.append({
                    'video1': video1,
                    'video2': video2,
                    'label': label
                })
        
        random.shuffle(self.pairs)
    
    def _create_sampled_pairs(self, same_class_ratio=0.5, total_pairs=10000):
        """Create a sampled set of pairs with specified ratio"""
        target_same_class = int(total_pairs * same_class_ratio)
        target_diff_class = total_pairs - target_same_class
        
        # Create same-class pairs
        same_class_pairs = []
        
        # REAL-REAL pairs
        real_pairs_needed = target_same_class // 2
        for _ in range(real_pairs_needed):
            if len(self.real_videos) >= 2:
                video1, video2 = random.sample(self.real_videos, 2)
                same_class_pairs.append({
                    'video1': video1,
                    'video2': video2,
                    'label': 1
                })
        
        # FAKE-FAKE pairs
        fake_pairs_needed = target_same_class - len(same_class_pairs)
        for _ in range(fake_pairs_needed):
            if len(self.fake_videos) >= 2:
                video1, video2 = random.sample(self.fake_videos, 2)
                same_class_pairs.append({
                    'video1': video1,
                    'video2': video2,
                    'label': 1
                })
        
        # Create different-class pairs
        diff_class_pairs = []
        for _ in range(target_diff_class):
            if self.real_videos and self.fake_videos:
                real_video = random.choice(self.real_videos)
                fake_video = random.choice(self.fake_videos)
                diff_class_pairs.append({
                    'video1': real_video,
                    'video2': fake_video,
                    'label': 0
                })
        
        self.pairs = same_class_pairs + diff_class_pairs
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
        if not self.augment or np.random.random() > 0.5:
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

class TestSimilarityDataset(Dataset):
    def __init__(self, master_csv, test_csv, device='cuda', normalize=True, augment=False):
        self.master_dataset = AllDataPairDataset(master_csv, device=device, normalize=normalize, augment=augment, pair_strategy='all')
        self.test_data = pd.read_csv(test_csv)
        self.device = device
        self.normalize = normalize

        self.array_features = self.master_dataset.array_features
        self.mean_sequence_length = self.master_dataset.mean_sequence_length
        self.feature_means = self.master_dataset.feature_means
        self.feature_stds = self.master_dataset.feature_stds
        self.master_data = self.master_dataset.data
        
        # Create test-to-master pairs
        self.pairs = self._create_test_pairs()
    
    def _create_test_pairs(self):
        pairs = []
        for _, test_row in self.test_data.iterrows():
            test_video = test_row['video_name']
            for _, master_row in self.master_data.iterrows():
                master_video = master_row['video_name']
                pairs.append({
                    'test_row': test_row,
                    'master_row': master_row,
                    'test_video': test_video,
                    'master_video': master_video
                })
        return pairs
    
    def add_derivatives(self, tensor):
        """Compute first-order derivative and concatenate with original tensor."""
        derivatives = torch.zeros_like(tensor)
        derivatives[1:] = tensor[1:] - tensor[:-1]
        return torch.cat((tensor, derivatives), dim=1)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        test_row = pair['test_row']
        master_row = pair['master_row']

        def get_feature_vector(row):
            feature_vectors = []
            for feature_name in self.array_features:
                val = row.get(feature_name, None)
                array = self.master_dataset._parse_array(val)
                array = self.master_dataset._process_feature(array, feature_name)
                feature_vectors.append(array)
            combined = np.stack(feature_vectors, axis=1)  # Shape: [seq_len, num_features]
            return combined

        test_feat = get_feature_vector(test_row)
        master_feat = get_feature_vector(master_row)

        test_feat = torch.tensor(test_feat, dtype=torch.float32).to(self.device)
        master_feat = torch.tensor(master_feat, dtype=torch.float32).to(self.device)
        test_feat = self.add_derivatives(test_feat)
        master_feat = self.add_derivatives(master_feat)

        return test_feat, master_feat, pair['test_video'], pair['master_video']

from torch.nn import functional as F



test_dataset = TestSimilarityDataset(
    master_csv='../master_features.csv',
    test_csv='test.csv',
    device='cuda',
    normalize=True,
    augment=False
)

dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
sample_data = next(iter(DataLoader(test_dataset, batch_size=1)))
input_size = sample_data[0].shape[2]

model = SiameseNetwork(input_size=input_size)

checkpoint = torch.load('enhanced_siamese_model.pth', map_location='cuda')

# Load the model weights from the checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to('cuda')
model.eval()


results = []

with torch.no_grad():
    for test_feat, master_feat, test_video, master_video in dataloader:
        test_feat = test_feat.squeeze(0)  # [seq_len, feat_dim]
        master_feat = master_feat.squeeze(0)

        # Get embeddings
        test_emb = model(test_feat.unsqueeze(0))    # [1, emb_dim]
        master_emb = model(master_feat.unsqueeze(0))  # [1, emb_dim]

        # Cosine similarity
        similarity = F.cosine_similarity(test_emb, master_emb).item()

        results.append({
            'test_video': test_video[0],
            'master_video': master_video[0],
            'similarity': similarity
        })


from collections import defaultdict

grouped_results = defaultdict(list)
for r in results:
    grouped_results[r['test_video']].append(r)

# Sort similarities
for test_video in grouped_results:
    grouped_results[test_video].sort(key=lambda x: x['similarity'], reverse=True)
    print(f"\nTop matches for {test_video}:")
    for match in grouped_results[test_video][:5]:  # top 5 matches
        print(f"  ↳ {match['master_video']} - Similarity: {match['similarity']:.4f}")
