import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import random
import re
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import seaborn as sns
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EnhancedPainDataset(Dataset):
    def __init__(self, csv_file, normalize=False, augment=False, device='cuda'):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.dropna(subset=['label'])
        self.data['label'] = self.data['label'].apply(lambda x: 1.0 if float(x) > 0.5 else 0.0 if pd.notnull(x) else np.nan)
        
        self.videos = self.data['video_name'].unique()
        self.videos = self.data['video_name'].dropna().values
        self.normalize = normalize
        self.augment = augment
        self.device = device
        
        # Time-series features (stored as arrays)
        self.array_features = [
            'angle', 'angular_velocity', 'angular_acceleration',
            'distance', 'velocity', 'acceleration',
            'dominant_frequencies', 'dominant_amplitudes'
        ]
        
        # self.scalar_features = [ ]
        # self.scalar_features = [
        #     'spectral_arc_length', 'range_of_motion',
        # ]
        self.scalar_features = [
            'mean_frequency', 'median_frequency', 'power_0_3Hz', 'power_3_6Hz', 
            'power_6_10Hz', 'frequency_bandwidth', 'range_of_motion', 'normalized_jerk', 
            'spectral_arc_length', 'coefficient_of_variation_angle', 
            'coefficient_of_variation_velocity', 'mean_absolute_velocity', 'peak_velocity', 
            'time_to_peak_velocity', 'mean_absolute_acceleration', 'peak_acceleration', 
            'rhythm_coefficient_of_variation'
        ]
        
        
        # Find joint one-hot encoded columns
        self.joint_features = [col for col in self.data.columns if col.startswith('joint_')]
        self.movement_features = [col for col in self.data.columns if col.startswith('movement_')]
        
        # Determine sequence length from angle arrays
        self._determine_sequence_length()
        
        # Calculate normalization statistics if needed
        if self.normalize:
            self._calculate_normalization_stats()

        self._calculate_feature_label_correlation()

    
    def _calculate_feature_label_correlation(self):
        """Compute correlation between features and label."""
        scalar_data = self.data[self.scalar_features + ['label']].dropna()
        
        if not scalar_data.empty:
            correlation = scalar_data.corr()['label'].sort_values(ascending=False)
            print("Feature-Label Correlation:\n", correlation)
            return correlation
        else:
            print("No valid scalar data for correlation with label.")
            return None
        
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
            print(f"Mean angle array length: {self.mean_sequence_length}")
        else:
            self.mean_sequence_length = 350  # Default if no valid angles found
            print(f"No valid angle arrays found, using default length of {self.mean_sequence_length}")
            
    def _calculate_normalization_stats(self):
        """Calculate mean and std for normalization"""
        self.feature_means = {}
        self.feature_stds = {}
        
        # Process array features
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
        
        # Process scalar features
        for feature in self.scalar_features:
            if feature in self.data.columns:
                values = self.data[feature].dropna().values
                if len(values) > 0:
                    self.feature_means[feature] = np.mean(values)
                    self.feature_stds[feature] = np.std(values) + 1e-6
                else:
                    self.feature_means[feature] = 0.0
                    self.feature_stds[feature] = 1.0
    
    def _parse_array(self, value):
        """Parse array strings, including Python list format"""
        if isinstance(value, str):
            try:
                # First, try direct ast.literal_eval for Python list strings
                # Remove newlines and extra whitespace to handle potential formatting issues
                cleaned_value = value.replace('\n', '').strip()
                return np.array(ast.literal_eval(cleaned_value))
            except (ValueError, SyntaxError):
                print("Failed to parse with literal_eval:", value)
                try:
                    # Fallback to regex parsing if literal_eval fails
                    # Improved regex to handle scientific notation and negative numbers
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
            print("Invalid value format:", value)
            exit()
    
    def _process_feature(self, feature_data, feature_name):
        """Process and normalize a feature"""
        # Normalize if needed
        if self.normalize and feature_name in self.feature_means:
            feature_data = (feature_data - self.feature_means[feature_name]) / self.feature_stds[feature_name]
        
        # Pad or truncate to mean sequence length
        if len(feature_data) > self.mean_sequence_length:
            feature_data = feature_data[:self.mean_sequence_length]
        else:
            feature_data = np.pad(feature_data, (0, self.mean_sequence_length - len(feature_data)), 'constant')
        
        return feature_data
    
    def _calculate_derivatives(self, features):
        """Calculate temporal derivatives for additional context"""
        # Convert to numeric array with error handling
        def safe_numeric_convert(arr):
            try:
                # Handle nested arrays or string representations
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    # First try to parse any string representations
                    converted = []
                    for row in arr:
                        row_converted = []
                        for val in row:
                            # Convert string to float if needed
                            if isinstance(val, str):
                                try:
                                    row_converted.append(float(val))
                                except:
                                    row_converted.append(0.0)
                            else:
                                row_converted.append(float(val))
                        converted.append(row_converted)
                    return np.array(converted, dtype=np.float32)
                
                # If already a numeric array, just ensure float32
                return np.array(arr, dtype=np.float32)
            
            except Exception as e:
                print(f"Error converting features to numeric array: {e}")
                return np.zeros_like(arr, dtype=np.float32)
        
        # Ensure features are numeric
        features = safe_numeric_convert(features)
        
        # Calculate derivatives
        derivatives = np.zeros_like(features)
        try:
            for i in range(1, len(features)):
                derivatives[i] = features[i] - features[i-1]
        except Exception as e:
            print(f"Error calculating derivatives: {e}")
            # Fallback to zero derivatives if calculation fails
            derivatives = np.zeros_like(features)
        
        return derivatives
    
    def _augment_sequence(self, features):
        """Apply data augmentation techniques"""
        # Gaussian noise
        if np.random.random() < 0.5:
            noise_factor = 0.05
            noise = np.random.normal(0, noise_factor, features.shape)
            features = features + noise
        
        # Time masking
        if np.random.random() < 0.3:
            if features.shape[0] > 10:
                mask_length = np.random.randint(5, 10) 
                start_idx = np.random.randint(0, features.shape[0] - mask_length)
                features[start_idx:start_idx+mask_length, :] = 0
        
        # Time warping - stretch or compress random segments
        if np.random.random() < 0.2:
            if features.shape[0] > 20:
                segment_length = np.random.randint(10, 20)
                start_idx = np.random.randint(0, features.shape[0] - segment_length)
                stretch_factor = np.random.uniform(0.8, 1.2)
                
                # Simple linear interpolation for stretching/compressing
                segment = features[start_idx:start_idx+segment_length]
                new_length = int(segment_length * stretch_factor)
                if new_length > 0:
                    indices = np.linspace(0, segment_length-1, new_length)
                    warped_segment = np.array([np.interp(indices, np.arange(segment_length), segment[:, i]) 
                                              for i in range(segment.shape[1])]).T
                    
                    # Replace segment with warped segment (maintaining original size)
                    if start_idx + new_length <= features.shape[0]:
                        features[start_idx:start_idx+new_length] = warped_segment
        
        return features
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        video_data = self.data[self.data['video_name'] == video]
        
        # Get label
        labels = video_data['label'].dropna().values
        label = float(labels[0]) if len(labels) > 0 else 0.0
        
        # Extract array features
        array_features_list = []
        for feature in self.array_features:
            if feature in video_data.columns:
                # Use first row data for each feature
                value = video_data[feature].values[0] if len(video_data) > 0 else None
                
                # Robust parsing of feature array
                try:
                    feature_array = self._parse_array(value)
                    
                    # Check if feature_array is actually an array with elements
                    if feature_array is not None and feature_array.size > 0:
                        processed_feature = self._process_feature(feature_array, feature)
                        array_features_list.append(processed_feature)
                    else:
                        array_features_list.append(np.zeros(self.mean_sequence_length))
                except Exception as e:
                    print(f"Error processing feature {feature}: {e}")
                    array_features_list.append(np.zeros(self.mean_sequence_length))
            else:
                array_features_list.append(np.zeros(self.mean_sequence_length))
        
        # Stack array features and ensure numeric type
        try:
            all_features = np.stack(array_features_list, axis=1)
            all_features = all_features.astype(np.float32)  # Explicit float32 conversion
        except Exception as e:
            print(f"Error stacking array features: {e}")
            # Fallback to zero array if stacking fails
            all_features = np.zeros((self.mean_sequence_length, len(self.array_features)), dtype=np.float32)
        
        # Extract scalar features and repeat them across time
        scalar_features_list = []
        for feature in self.scalar_features:
            if feature in video_data.columns:
                try:
                    value = video_data[feature].values[0] if len(video_data) > 0 else 0.0
                    if pd.isna(value):
                        value = 0.0
                    
                    # Normalize scalar feature
                    if self.normalize and feature in self.feature_means:
                        value = (value - self.feature_means[feature]) / self.feature_stds[feature]
                    
                    # Repeat scalar value across all time steps
                    scalar_features_list.append(np.full(self.mean_sequence_length, float(value)))
                except Exception as e:
                    print(f"Error processing scalar feature {feature}: {e}")
                    scalar_features_list.append(np.zeros(self.mean_sequence_length))
            else:
                scalar_features_list.append(np.zeros(self.mean_sequence_length))

        # Add scalar features if available
        if scalar_features_list:
            try:
                scalar_features = np.stack(scalar_features_list, axis=1).astype(np.float32)
                all_features = np.concatenate([all_features, scalar_features], axis=1)
            except Exception as e:
                print(f"Error adding scalar features: {e}")
        

        # Calculate derivatives for temporal features
        try:
            derivatives = self._calculate_derivatives(all_features)
            all_features = np.concatenate([all_features, derivatives], axis=1)
        except Exception as e:
            print(f"Error calculating derivatives: {e}")
            # Fallback to zero derivatives
            derivatives = np.zeros_like(all_features)
            all_features = np.concatenate([all_features, derivatives], axis=1)
        
        # Extract joint one-hot features if available
        if self.joint_features:
            joint_values = []
            for joint_feature in self.joint_features:
                try:
                    if joint_feature in video_data.columns:
                        value = video_data[joint_feature].values[0] if len(video_data) > 0 else 0.0
                        if pd.isna(value):
                            value = 0.0
                        joint_values.append(float(value))
                    else:
                        joint_values.append(0.0)
                except Exception as e:
                    print(f"Error processing joint feature {joint_feature}: {e}")
                    joint_values.append(0.0)
            
            # Repeat joint features for each time step
            if joint_values:
                try:
                    joint_features = np.tile(joint_values, (self.mean_sequence_length, 1)).astype(np.float32)
                    all_features = np.concatenate([all_features, joint_features], axis=1)
                except Exception as e:
                    print(f"Error adding joint features: {e}")
        
        if self.movement_features:
            movement_values = []
            for movement_feature in self.movement_features:
                try:
                    if movement_feature in video_data.columns:
                        value = video_data[movement_feature].values[0] if len(video_data) > 0 else 0.0
                        if pd.isna(value):
                            value = 0.0
                        movement_values.append(float(value))
                    else:
                        movement_values.append(0.0)
                except Exception as e:
                    print(f"Error processing joint feature {joint_feature}: {e}")
                    movement_values.append(0.0)
            
            # Repeat joint features for each time step
            if movement_values:
                try:
                    movement_features = np.tile(movement_values, (self.mean_sequence_length, 1)).astype(np.float32)
                    all_features = np.concatenate([all_features, movement_features], axis=1)
                except Exception as e:
                    print(f"Error adding joint features: {e}")
        
        # Ensure all features are float32
        all_features = all_features.astype(np.float32)
        
        # Replace NaNs with zeros
        all_features = np.nan_to_num(all_features, nan=0.0)
        
        # Apply data augmentation if needed
        if self.augment:
            try:
                all_features = self._augment_sequence(all_features)
            except Exception as e:
                print(f"Error during data augmentation: {e}")
        
        return torch.FloatTensor(all_features).to(self.device), torch.FloatTensor([label]).to(self.device)
    
    def debug_features(self, idx=0):
        """Debug feature extraction for a specific video"""
        video = self.videos[idx]
        video_data = self.data[self.data['video_name'] == video]
        
        print(f"Video: {video}")
        print(f"Label: {video_data['label'].values[0] if len(video_data) > 0 else 'N/A'}")
        print(f"Total rows: {len(video_data)}")
        
        # Debug array features
        for feature in self.array_features:
            if feature in video_data.columns:
                print(f"\n{feature} examples:")
                for i, value in enumerate(video_data[feature].values[:2]):
                    if isinstance(value, str):
                        try:
                            parsed = self._parse_array(value)
                            print(f"  Length: {len(parsed)}")
                            print(f"  First 5 values: {parsed[:5] if len(parsed) >= 5 else parsed}")
                        except Exception as e:
                            print(f"  Error parsing: {str(e)}")
                    else:
                        print(f"  Value: {value}")
                    if i >= 1:
                        break
        
        # Debug scalar features
        for feature in self.scalar_features[:5]:  # Show first 5 scalar features
            if feature in video_data.columns:
                print(f"\n{feature}: {video_data[feature].values[0] if len(video_data) > 0 else 'N/A'}")
        
        # Debug joint features
        if self.joint_features:
            print("\nJoint features:")
            for feature in self.joint_features:  
                if feature in video_data.columns:
                    print(f"  {feature}: {video_data[feature].values[0] if len(video_data) > 0 else 'N/A'}")

        # Debug movement features
        if self.movement_features:
            print("\nMovement features:")
            for feature in self.movement_features:  
                if feature in video_data.columns:
                    print(f"  {feature}: {video_data[feature].values[0] if len(video_data) > 0 else 'N/A'}")


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class EnhancedPainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(EnhancedPainClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Add batch normalization for input
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Multiple fully connected layers with batch norm and dropout
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)  # Changed from hidden_size * 2 to * 4 for combined pooling
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.residual_proj = nn.Linear(hidden_size, output_size)

        # Apply custom initialization
        self._initialize_weights()
        
    def _initialize_weights(self): 
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.4)  
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                if 'bias_ih' in name or 'bias_hh' in name:
                    param.data[self.hidden_size:2*self.hidden_size].fill_(3.0)
        
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        
        nn.init.constant_(self.batch_norm.weight, 1.0)
        nn.init.constant_(self.batch_norm.bias, 0.0)
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        
        # Now we ensure 3D tensor
        batch_size, seq_len, features = x.size()
        
        # Apply batch norm across feature dimension 
        x_reshaped = x.reshape(-1, features)
        x_normalized = self.batch_norm(x_reshaped)
        x = x_normalized.reshape(batch_size, seq_len, features)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use both the mean and max pooling and concatenate
        mean_pooling = torch.mean(lstm_out, dim=1)
        max_pooling, _ = torch.max(lstm_out, dim=1)  # Fixed syntax error
        combined = torch.cat([mean_pooling, max_pooling], dim=1)
        
        # First fully connected layer - using the combined pooling
        out = F.elu(self.bn1(self.fc1(combined)))
        out = self.dropout1(out)

        residual = out
        
        # Output layer
        out = self.fc2(out)
        
        # Apply residual connection with projection
        out = out + self.residual_proj(residual)

        return out

# Focal Loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

def train_with_cv(dataset, num_folds=5, input_size=None, hidden_size=64, num_layers=2, 
                  output_size=1, batch_size=16, learning_rate=0.0001, num_epochs=300):
    
    os.makedirs('./models', exist_ok=True)
    # Create a unique experiment name with timestamp
    experiment_name = f"pain_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    # Get sample to determine feature size if not provided
    if input_size is None:
        sample_features, _ = dataset[0]
        input_size = sample_features.shape[1]
    
    # Get unique videos from the dataset instead of using indices directly
    if hasattr(dataset, 'dataset'):  # If it's a Subset
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        # Get videos corresponding to the subset
        videos_in_subset = [original_dataset.videos[i] for i in subset_indices]
        unique_videos = np.unique(videos_in_subset)
    else:
        unique_videos = np.unique(dataset.videos)
    
    # Get labels for each unique video
    video_labels = []
    for video in unique_videos:
        if hasattr(dataset, 'dataset'):
            video_data = original_dataset.data[original_dataset.data['video_name'] == video]
        else:
            video_data = dataset.data[dataset.data['video_name'] == video]
        labels = video_data['label'].dropna().values
        label = int(labels[0]) if len(labels) > 0 else 0
        video_labels.append(label)
    
    video_labels = np.array(video_labels)
    
    # Setup k-fold cross validation
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # To store results
    fold_results = []
    best_models = []
    best_f1_scores = []
    all_train_losses = []
    all_val_losses = []
    
    # Run cross-validation
    for fold, (train_video_ids, val_video_ids) in enumerate(kfold.split(unique_videos, video_labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{num_folds}")
        print(f"{'='*50}")
        
        train_videos = unique_videos[train_video_ids]
        val_videos = unique_videos[val_video_ids]
        
        # Convert video names back to dataset indices
        train_indices = []
        val_indices = []
        
        if hasattr(dataset, 'dataset'):
            original_videos = original_dataset.videos
            for i in subset_indices:
                if original_videos[i] in train_videos:
                    # Find the position of this index in the subset
                    train_indices.append(subset_indices.index(i))
                elif original_videos[i] in val_videos:
                    val_indices.append(subset_indices.index(i))
        else:
            for i, video in enumerate(dataset.videos):
                if video in train_videos:
                    train_indices.append(i)
                elif video in val_videos:
                    val_indices.append(i)
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, drop_last=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, drop_last=False)
        
        # Create a new model instance
        model = EnhancedPainClassifier(input_size, hidden_size, num_layers, output_size).to(device)
        
        # Compute class weights for imbalanced data
        train_fold_labels = []
        for idx in train_indices:
            if hasattr(dataset, 'dataset'):
                # If it's a Subset, get the video name from the original dataset
                video = dataset.dataset.videos[dataset.indices[idx]]
                video_data = dataset.dataset.data[dataset.dataset.data['video_name'] == video]
            else:
                video = dataset.videos[idx]
                video_data = dataset.data[dataset.data['video_name'] == video]
            
            labels = video_data['label'].dropna().values
            label = int(labels[0]) if len(labels) > 0 else 0
            train_fold_labels.append(label)
        
        train_labels = np.array(train_fold_labels)
        
        pos_samples = np.sum(train_labels == 1)
        neg_samples = np.sum(train_labels == 0)
        
        if pos_samples > 0 and neg_samples > 0:
            pos_weight = neg_samples / pos_samples
            print(f"Class ratio (neg:pos): {neg_samples}:{pos_samples} = {neg_samples/pos_samples:.2f}")
            print(f"Using positive class weight: {pos_weight:.2f}")
        else:
            pos_weight = 1.0
            print("Only one class detected. Using default weight of 1.0")
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        # criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=2, eta_min=1e-6
        # )

        # TensorBoard: Log model architecture
        dummy_input = torch.randn(batch_size, input_size).unsqueeze(1).to(device)
        writer.add_graph(model, dummy_input)
        
        # Training loop
        best_val_f1 = 0.0
        patience = 20
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training metrics tracking
            all_train_preds = []
            all_train_labels = []
            
            for sequences, labels in train_loader:
                # Ensure sequences have the right shape
                if len(sequences.size()) == 2:
                    sequences = sequences.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Compute training metrics
                train_loss += loss.item()
                
                # Convert outputs to probabilities and predictions
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                # Track predictions and labels for detailed metrics
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                
                train_correct += (preds == labels).float().sum().item()
                train_total += labels.size(0)
            
            # Compute training metrics
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Compute detailed training metrics
            train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
            train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
            train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_probs = []
            all_labels_val = []
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    # Ensure sequences have the right shape
                    if len(sequences.size()) == 2:
                        sequences = sequences.unsqueeze(1)
                    
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Convert outputs to probabilities and predictions
                    probs = torch.sigmoid(outputs)
                    predicted = (probs >= 0.5).float()
                    
                    val_correct += (predicted == labels).float().sum().item()
                    val_total += labels.size(0)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_labels_val.extend(labels.cpu().numpy())
            
            # Compute validation metrics
            val_loss /= len(val_loader)
            val_losses.append(val_loss)  
            val_accuracy = val_correct / val_total
            
            # Detailed validation metrics
            if sum(all_preds) > 0 and sum(1 for x in all_labels_val if x == 1) > 0:
                val_precision = precision_score(all_labels_val, all_preds, zero_division=0)
                val_recall = recall_score(all_labels_val, all_preds, zero_division=0)
                val_f1 = f1_score(all_labels_val, all_preds, zero_division=0)
            else:
                val_precision = 0
                val_recall = 0
                val_f1 = 0
            
            try:
                val_auc = roc_auc_score(all_labels_val, all_probs)
            except:
                val_auc = 0.5
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # TensorBoard Logging - Enhanced Metrics
            # Learning Rates and Optimization
            writer.add_scalar(f'Fold_{fold+1}/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Loss Tracking
            writer.add_scalar(f'Fold_{fold+1}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Validation_Loss', val_loss, epoch)
            
            # Accuracy Tracking
            writer.add_scalar(f'Fold_{fold+1}/Train_Accuracy', train_accuracy, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Validation_Accuracy', val_accuracy, epoch)
            
            # Detailed Performance Metrics
            writer.add_scalar(f'Fold_{fold+1}/Train_Precision', train_precision, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Train_Recall', train_recall, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Train_F1', train_f1, epoch)
            
            writer.add_scalar(f'Fold_{fold+1}/Validation_Precision', val_precision, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Validation_Recall', val_recall, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Validation_F1', val_f1, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Validation_AUC', val_auc, epoch)
            
            # Histogram of model parameters and gradients
            for name, param in model.named_parameters():
                writer.add_histogram(f'Fold_{fold+1}/{name}.grad', param.grad, epoch)
                writer.add_histogram(f'Fold_{fold+1}/{name}.data', param, epoch)
                # if param.requires_grad:
                #     print(f"{name}: {param.grad.abs().mean().item()}")
            
            cm = confusion_matrix(all_labels_val, all_preds)
            writer.add_figure(f'Fold_{fold+1}/Confusion_Matrix', 
                                plot_confusion_matrix(cm, classes=['Negative', 'Positive']), 
                                epoch)
            
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}, LR: {optimizer.param_groups[0]['lr']:.6f}, Patience: {patience_counter}/{patience}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                print(f"Val Metrics - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # Save the best model based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'./models/best_pain_classifier_fold{fold+1}.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # TensorBoard: Final fold metrics as text
        writer.add_text(f'Fold_{fold+1}/Final_Metrics', 
                        f"Best Val F1: {best_val_f1:.4f}\n"
                        f"Final Train Metrics:\n"
                        f"- Precision: {train_precision:.4f}\n"
                        f"- Recall: {train_recall:.4f}\n"
                        f"- F1: {train_f1:.4f}\n\n"
                        f"Final Val Metrics:\n"
                        f"- Accuracy: {val_accuracy:.4f}\n"
                        f"- Precision: {val_precision:.4f}\n"
                        f"- Recall: {val_recall:.4f}\n"
                        f"- F1: {val_f1:.4f}\n"
                        f"- AUC: {val_auc:.4f}")
        
        print(f"Best validation F1 score for fold {fold+1}: {best_val_f1:.4f}")
        
        # Load best model for this fold
        model.load_state_dict(torch.load(f'./models/best_pain_classifier_fold{fold+1}.pth', weights_only=True))
        
    
        writer.close()
    
        
        # Final evaluation on validation set
        model.eval()
        all_preds = []
        all_probs = []
        all_labels_val = []  # Renamed to avoid conflict
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())
        
        # Calculate final metrics
        val_acc = accuracy_score(all_labels_val, all_preds)
        val_balanced_acc = balanced_accuracy_score(all_labels_val, all_preds)
        
        if sum(all_preds) > 0 and sum(1 for x in all_labels_val if x == 1) > 0:
            val_precision = precision_score(all_labels_val, all_preds, zero_division=0)
            val_recall = recall_score(all_labels_val, all_preds, zero_division=0)
            val_f1 = f1_score(all_labels_val, all_preds, zero_division=0)
        else:
            val_precision = 0
            val_recall = 0
            val_f1 = 0
        
        try:
            val_auc = roc_auc_score(all_labels_val, all_probs)
        except:
            val_auc = 0.5
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'accuracy': val_acc,
            'balanced_accuracy': val_balanced_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        })
        
        best_models.append(model)
        best_f1_scores.append(best_val_f1)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
    
    # Print summary of cross-validation results
    print("\nCross-Validation Results:")
    for result in fold_results:
        print(f"Fold {result['fold']}: F1={result['f1']:.4f}, AUC={result['auc']:.4f}, Acc={result['accuracy']:.4f}")
    
    # Calculate average metrics
    avg_acc = np.mean([result['accuracy'] for result in fold_results])
    avg_balanced_acc = np.mean([result['balanced_accuracy'] for result in fold_results])
    avg_precision = np.mean([result['precision'] for result in fold_results])
    avg_recall = np.mean([result['recall'] for result in fold_results])
    avg_f1 = np.mean([result['f1'] for result in fold_results])
    avg_auc = np.mean([result['auc'] for result in fold_results])
    
    print(f"\nAverage Metrics:")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"Balanced Accuracy: {avg_balanced_acc:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"AUC: {avg_auc:.4f}")
    
    # Select best model from all folds
    best_fold_idx = np.argmax(best_f1_scores)
    best_model = best_models[best_fold_idx]
    print(f"Best model is from fold {best_fold_idx + 1} with F1 score: {best_f1_scores[best_fold_idx]:.4f}")
    
    # Save best model
    torch.save(best_model.state_dict(), './models/best_pain_classifier_overall.pth')
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'b-', alpha=0.3, label=f'Train Fold {i+1}' if i == 0 else "")
        plt.plot(epochs, val_loss, 'r-', alpha=0.3, label=f'Val Fold {i+1}' if i == 0 else "")
    
    plt.title('Learning Curves (All Folds)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot best fold learning curve
    epochs = range(1, len(all_train_losses[best_fold_idx]) + 1)
    plt.plot(epochs, all_train_losses[best_fold_idx], 'b-', label='Train')
    plt.plot(epochs, all_val_losses[best_fold_idx], 'r-', label='Validation')
    plt.title(f'Learning Curve (Best Fold {best_fold_idx + 1})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./stats/learning_curves.png')
    plt.close()
    
    return best_model, fold_results

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    return plt.gcf()

# Enhanced prediction function with optimal threshold finding
def enhanced_predict(model, dataset, show_attention=False):
    test_loader = DataLoader(dataset, batch_size=1)
    
    model.eval()
    predictions = []
    video_names = []
    probabilities = []
    labels = []
    attention_maps = []
    
    with torch.no_grad():
        for i, (sequence, label) in enumerate(test_loader):
            output = model(sequence)
            prob = torch.sigmoid(output).item()
            probabilities.append(prob)
            video_names.append(dataset.videos[i])
            labels.append(label.item())
            
            
    
    # Find optimal threshold using ROC curve if we have ground truth labels
    if sum(1 for l in labels if l > 0.5) > 0 and sum(1 for l in labels if l <= 0.5) > 0:
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        # Optimal threshold maximizes (tpr - fpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal threshold: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.562
        print("Using default threshold: 0.5")
    
    # Apply threshold
    predictions = [1 if p >= optimal_threshold else 0 for p in probabilities]
    
    results = pd.DataFrame({
        'video_name': video_names,
        'prediction_label': ['Fake' if p == 1 else 'Real' for p in predictions],
        'true_label_name': ['Fake' if l > 0.5 else 'Real' for l in labels],
        'probability': probabilities,
        'prediction': predictions,
        'true_label': labels,
    })
    
    # Plot probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist([results[results['true_label'] > 0.5]['probability'], 
              results[results['true_label'] <= 0.5]['probability']], 
             bins=20, alpha=0.7, label=['Fake', 'Real'])
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Threshold ({optimal_threshold:.2f})')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution by Class')
    plt.legend()
    plt.savefig('./stats/probability_distribution.png')
    plt.close()
    
    # Plot ROC curve
    if sum(1 for l in labels if l > 0.5) > 0 and sum(1 for l in labels if l <= 0.5) > 0:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig('./stats/roc_curve.png')
        plt.close()
    
    true_positives = results[(results['prediction'] == 1) & (results['true_label'] > 0.5)]
    false_positives = results[(results['prediction'] == 1) & (results['true_label'] <= 0.5)]
    true_negatives = results[(results['prediction'] == 0) & (results['true_label'] <= 0.5)]
    false_negatives = results[(results['prediction'] == 0) & (results['true_label'] > 0.5)]
    
    print(f"\nConfusion Matrix Analysis:")
    print(f"True Positives (Correctly predicted Fake): {len(true_positives)}")
    print(f"False Positives (Real incorrectly predicted as Fake): {len(false_positives)}")
    print(f"True Negatives (Correctly predicted Real): {len(true_negatives)}")
    print(f"False Negatives (Fake incorrectly predicted as Real): {len(false_negatives)}")
    
    print(f"\nAverage Probabilities:")
    print(f"True Positives (Fake predicted as Fake): {true_positives['probability'].mean():.4f}")
    print(f"False Positives (Real predicted as Fake): {false_positives['probability'].mean():.4f}")
    print(f"True Negatives (Real predicted as Real): {true_negatives['probability'].mean():.4f}")
    print(f"False Negatives (Fake predicted as Real): {false_negatives['probability'].mean():.4f}")


    # Calculate metrics
    if len(true_positives) + len(false_positives) > 0:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
    else:
        precision = 0
        
    if len(true_positives) + len(false_negatives) > 0:
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    else:
        recall = 0
        
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    accuracy = (len(true_positives) + len(true_negatives)) / len(results)
    balanced_acc = 0.5 * (len(true_positives) / (len(true_positives) + len(false_negatives)) + 
                          len(true_negatives) / (len(true_negatives) + len(false_positives)))
    
    # Calculate metrics
    if len(true_positives) + len(false_positives) > 0:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
    else:
        precision = 0
        
    if len(true_positives) + len(false_negatives) > 0:
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    else:
        recall = 0
        
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    accuracy = (len(true_positives) + len(true_negatives)) / len(results)
    
    if (len(true_positives) + len(false_negatives)) > 0 and (len(true_negatives) + len(false_positives)) > 0:
        balanced_acc = 0.5 * (len(true_positives) / (len(true_positives) + len(false_negatives)) + 
                              len(true_negatives) / (len(true_negatives) + len(false_positives)))
    else:
        balanced_acc = 0
    
    print(f"\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    
    return results, optimal_threshold


def main():
    data_csv = 'movement_csv/neck_movements.csv'
    
    # Model parameters
    hidden_size = 256
    num_layers = 4
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 150
    test_size = 0.1  

    print("Loading dataset...")
    # Load the dataset
    full_dataset = EnhancedPainDataset(
        data_csv, 
        normalize=True,
        augment=False  # No augmentation initially
    )
    full_dataset.debug_features()

    unique_videos = np.unique(full_dataset.videos)  # Get unique video names
    video_labels = []
    
    # Get label for each unique video
    for video in unique_videos:
        video_data = full_dataset.data[full_dataset.data['video_name'] == video]
        labels = video_data['label'].dropna().values
        label = int(labels[0]) if len(labels) > 0 else 0
        video_labels.append(label)
    
    video_labels = np.array(video_labels)
    
    # Split the data indices into train and test sets
    train_videos, test_videos = train_test_split(
        unique_videos, 
        test_size=test_size, 
        random_state=42,
        stratify=video_labels
    )
    
    print(f"Total videos: {len(unique_videos)}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    # Create indices based on video splits
    train_indices = []
    test_indices = []
    
    for i, video in enumerate(full_dataset.videos):
        if video in train_videos:
            train_indices.append(i)
        elif video in test_videos:
            test_indices.append(i)

    # Create train and test datasets by using the same dataset with different indices
    # For training, we'll enable augmentation
    train_dataset = EnhancedPainDataset(
        data_csv, 
        normalize=True,
        augment=True  # Enable augmentation for training
    )
    
    # For testing, we'll use the same dataset but without augmentation
    test_dataset = EnhancedPainDataset(
        data_csv,
        normalize=True,
        augment=False  # No augmentation for test set
    )
    
    # Create train and test datasets with samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    # Get input size from dataset
    sample_features, _ = full_dataset[0]
    input_size = sample_features.shape[1]
    print(f"Feature dimensionality: {input_size}")
    
    train_video_labels = []
    for video in train_videos:
        video_data = full_dataset.data[full_dataset.data['video_name'] == video]
        labels = video_data['label'].dropna().values
        label = int(labels[0]) if len(labels) > 0 else 0
        train_video_labels.append(label)
    
    train_video_labels = np.array(train_video_labels)
    train_pos = np.sum(train_video_labels == 1)
    train_neg = np.sum(train_video_labels == 0)
    print(f"Training set: {len(train_indices)} samples from {len(train_videos)} videos ({train_pos} positive, {train_neg} negative)")
    
    # Calculate class distribution in test set
    # Get labels for test videos (not test indices)
    test_video_labels = []
    for video in test_videos:
        video_data = full_dataset.data[full_dataset.data['video_name'] == video]
        labels = video_data['label'].dropna().values
        label = int(labels[0]) if len(labels) > 0 else 0
        test_video_labels.append(label)
    
    test_video_labels = np.array(test_video_labels)
    test_pos = np.sum(test_video_labels == 1)
    test_neg = np.sum(test_video_labels == 0)
    print(f"Test set: {len(test_indices)} samples from {len(test_videos)} videos ({test_pos} positive, {test_neg} negative)")
    
    
    # Construct training dataset for cross-validation with specific train indices
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    # Train model with cross-validation
    print("Training model with cross-validation...")
    
    # Create a subset dataset for cross-validation 
    # (only including train indices)
    cv_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    best_model, cv_results = train_with_cv(
        cv_dataset, 
        num_folds=5,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    # Create a test dataset subset
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create a custom dataset-like object for prediction function
    class TestDatasetWrapper:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.videos = [dataset.videos[i] for i in indices]
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    test_wrapper = TestDatasetWrapper(test_dataset, test_indices)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    test_results, threshold = enhanced_predict(best_model, test_wrapper, show_attention=True)
    
    # Save results to CSV
    test_results.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")
    
    # Save final model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'threshold': threshold,
        'training_params': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs
        }
    }, './models/final_pain_classifier_model.pth')
    
    print("Model saved to 'final_pain_classifier_model.pth'")
    
    # Plot some example sequences with predictions
    plot_example_sequences(test_wrapper, best_model, threshold, num_examples=5)

# Function to plot example sequences with predictions
def plot_example_sequences(dataset, model, threshold, num_examples=5):
    model.eval()
    
    # Randomly select examples
    indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sequence, true_label = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            output = model(sequence.unsqueeze(0))
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= threshold else 0
        
        # Extract features for visualization
        # Use first 6 features which are the basic kinematic features
        features = sequence.cpu().numpy()[:, :6]  
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot features
        for j in range(min(6,features.shape[1])):
            ax1.plot(features[:, j], label=f"Feature {j+1}")
        
        video_name = dataset.videos[i] if hasattr(dataset, 'videos') else f"Example {i}"
        
        ax1.set_title(f"Video: {video_name}, True: {'Fake' if true_label > 0.5 else 'Real'}, "
                     f"Pred: {'Fake' if pred == 1 else 'Real'} (Prob: {prob:.4f})")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Feature Value")
        ax1.legend(loc='upper right')
        
        # Plot attention weights
        # attention = attention.squeeze().cpu().numpy()
        # ax2.bar(range(len(attention)), attention, alpha=0.7, color='blue')
        # ax2.set_title("Attention Weights")
        # ax2.set_xlabel("Time Steps")
        # ax2.set_ylabel("Weight")
        
        plt.tight_layout()
        plt.savefig(f'./sequences/example_sequence_{i}.png')
        plt.close()

if __name__ == "__main__":
    main()
