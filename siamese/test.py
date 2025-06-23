import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ast
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
from itertools import combinations
import random


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
            nn.Linear(hidden_size * 4, hidden_size * 2),
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
        
        self.final_activation = nn.Sigmoid()
        
    def self_attention(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        return context, attn_weights
    
    def forward_one(self, x):
        x_conv = x.transpose(1, 2)
        x_conv = self.conv_layers(x_conv)
        x_conv = x_conv.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_conv)
        context, attn_weights = self.self_attention(lstm_out)
        
        global_max_pool = torch.max(context, dim=1)[0]
        global_avg_pool = torch.mean(context, dim=1)
        
        pooled_features = torch.cat([global_max_pool, global_avg_pool], dim=1)
        fused = self.fusion(pooled_features)
        
        return fused, attn_weights
        
    def forward(self, input1, input2):
        output1, attn1 = self.forward_one(input1)
        output2, attn2 = self.forward_one(input2)
        
        diff = torch.abs(output1 - output2)
        similarity_logits = self.fc(diff)
        similarity = self.final_activation(similarity_logits)
        
        return similarity, similarity_logits, attn1, attn2


class VideoDataset(Dataset):
    def __init__(self, video_pairs, labels, classifier):
        self.video_pairs = video_pairs
        self.labels = labels
        self.classifier = classifier
    
    def __len__(self):
        return len(self.video_pairs)
    
    def __getitem__(self, idx):
        video1, video2 = self.video_pairs[idx]
        features1 = self.classifier.extract_features(video1)
        features2 = self.classifier.extract_features(video2)
        label = self.labels[idx]
        return features1, features2, torch.tensor(label, dtype=torch.float32)
    
class VideoClassifier:
    def __init__(self, model_path, data_csv_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data = pd.read_csv(data_csv_path)
        self.mean_sequence_length = 350
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']
        
        self.model = SiameseNetwork(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.model.eval()
        
        # Time-series features
        self.array_features = [
            'angle', 'angular_velocity', 'angular_acceleration',
            'angle2', 'angular_velocity2', 'angular_acceleration2',
            'angle3', 'angular_velocity3', 'angular_acceleration3',
            'angle4', 'angular_velocity4', 'angular_acceleration4',
            'distance1', 'velocity1', 'acceleration1',
        ]
        
        # Create reference videos for comparison
        self._create_video_pairs()
        
        print(f"Model loaded successfully!")
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Device: {self.device}")
    
    def _create_video_pairs(self, num_pairs_per_class=100):
        """Create pairs of videos for evaluation"""
        real_videos = [v for v in self.data['video_name'].unique() 
                    if pd.notna(v) and 'REAL' in str(v).upper()]
        fake_videos = [v for v in self.data['video_name'].unique() 
                    if pd.notna(v) and 'FAKE' in str(v).upper()]
        
        pairs = []
        labels = []
        
        # Same class pairs (label = 1)
        # Real-Real pairs
        if len(real_videos) >= 2:
            real_pairs = list(combinations(real_videos, 2))
            real_pairs = random.sample(real_pairs, min(num_pairs_per_class, len(real_pairs)))
            pairs.extend(real_pairs)
            labels.extend([1] * len(real_pairs))
        
        # Fake-Fake pairs
        if len(fake_videos) >= 2:
            fake_pairs = list(combinations(fake_videos, 2))
            fake_pairs = random.sample(fake_pairs, min(num_pairs_per_class, len(fake_pairs)))
            pairs.extend(fake_pairs)
            labels.extend([1] * len(fake_pairs))
        
        # Different class pairs (label = 0)
        # Real-Fake pairs
        cross_pairs = [(r, f) for r in real_videos for f in fake_videos]
        cross_pairs = random.sample(cross_pairs, min(num_pairs_per_class * 2, len(cross_pairs)))
        pairs.extend(cross_pairs)
        labels.extend([0] * len(cross_pairs))
        
        return pairs, labels
    
    def _parse_array(self, value):
        """Parse array strings"""
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
    
    def _process_feature(self, feature_data):
        """Process feature to fixed length"""
        if len(feature_data) > self.mean_sequence_length:
            feature_data = feature_data[:self.mean_sequence_length]
        else:
            feature_data = np.pad(feature_data, (0, self.mean_sequence_length - len(feature_data)), 'constant')
        return feature_data
    
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
                        processed_feature = self._process_feature(feature_array)
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
        
        # Replace NaNs
        all_features = np.nan_to_num(all_features, nan=0.0)
        
        return torch.FloatTensor(all_features)
    
    def evaluate(self, num_pairs_per_class=100, batch_size=16):
        """
        Evaluate the model on video pairs like the evaluate_on_test_set function
        
        Args:
            num_pairs_per_class: Number of pairs per class to generate
            batch_size: Batch size for evaluation
        
        Returns:
            dict: Contains evaluation metrics
        """
        print(f"\nGenerating video pairs for evaluation...")
        
        # Create video pairs
        video_pairs, labels = self._create_video_pairs(num_pairs_per_class)
        
        if not video_pairs:
            return {'error': 'No valid video pairs could be created'}
        
        print(f"Created {len(video_pairs)} video pairs")
        print(f"Same class pairs: {sum(labels)}")
        print(f"Different class pairs: {len(labels) - sum(labels)}")
        
        # Create dataset and dataloader
        dataset = VideoDataset(video_pairs, labels, self)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=False)
        
        # Evaluate model
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for x1, x2, batch_labels in tqdm(dataloader, desc="Evaluation"):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                similarity, _, _, _ = self.model(x1, x2)
                all_outputs.extend(similarity.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels).astype(int)
        
        auc = roc_auc_score(all_labels, all_outputs)
        predictions = (all_outputs > self.threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Using threshold: {self.threshold:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, predictions, 
                                target_names=['Different Class', 'Same Class']))
        
        cm = confusion_matrix(all_labels, predictions)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Diff  Same")
        print(f"Actual Diff  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Same  {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'predictions': predictions,
            'labels': all_labels,
            'probabilities': all_outputs,
            'threshold': self.threshold
        }

def predict_video(model_path, data_csv_path, video_name, device='cuda'):
    """
    Quick function to predict a single video
    """
    classifier = VideoClassifier(model_path, data_csv_path, device)
    return classifier.predict(video_name)

if __name__ == "__main__":
    # Initialize classifier
    classifier = VideoClassifier(
        model_path='enhanced_siamese_model.pth',
        data_csv_path='../master_features.csv',
        device='cuda'
    )
    
    # Run evaluation instead of individual predictions
    results = classifier.evaluate(num_pairs_per_class=50, batch_size=16)
    
