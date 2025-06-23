import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ast
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random

# Import your model class (assuming it's in the same directory or available)
# from your_model_file import SiameseNetwork  # Uncomment if in separate file

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
        
        # Final activation is separate
        self.final_activation = nn.Sigmoid()
        
    def self_attention(self, x):
        import math
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
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

class VideoPredictor:
    def __init__(self, model_path, master_csv_path, device='cuda'):
        """
        Initialize the video predictor
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        master_csv_path : str
            Path to the master features CSV with reference videos
        device : str
            Device to use for prediction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.model, self.threshold = self.load_model(model_path)
        
        # Load reference data
        self.master_data = pd.read_csv(master_csv_path)
        
        # Organize reference videos
        self.real_videos = []
        self.fake_videos = []
        
        for video_name in self.master_data['video_name'].unique():
            if pd.isna(video_name):
                continue
            video_str = str(video_name)
            if 'REAL' in video_str.upper():
                self.real_videos.append(video_name)
            elif 'FAKE' in video_str.upper():
                self.fake_videos.append(video_name)
        
        print(f"Loaded {len(self.real_videos)} REAL reference videos")
        print(f"Loaded {len(self.fake_videos)} FAKE reference videos")
        
        # Feature configuration
        self.array_features = [
            'angle', 'angular_velocity', 'angular_acceleration',
            'angle2', 'angular_velocity2', 'angular_acceleration2',
            'angle3', 'angular_velocity3', 'angular_acceleration3',
            'angle4', 'angular_velocity4', 'angular_acceleration4',
            'distance1', 'velocity1', 'acceleration1',
        ]
        
        # Calculate sequence length and normalization stats
        self._determine_sequence_length()
        self._calculate_normalization_stats()
    
    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint['model_config']
        model = SiameseNetwork(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        threshold = checkpoint['threshold']
        
        model.eval()  # Set to evaluation mode
        
        print(f"Model loaded successfully!")
        print(f"Optimal threshold: {threshold:.4f}")
        
        return model, threshold
    
    def _determine_sequence_length(self):
        """Determine sequence length from master data"""
        angle_lengths = []
        for idx, row in self.master_data.iterrows():
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
        else:
            self.mean_sequence_length = 350
        
        print(f"Using sequence length: {self.mean_sequence_length}")
    
    def _calculate_normalization_stats(self):
        """Calculate normalization statistics from master data"""
        self.feature_means = {}
        self.feature_stds = {}
        
        for feature in self.array_features:
            if feature in self.master_data.columns:
                feature_values = []
                for value in self.master_data[feature].values:
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
    
    def _process_feature(self, feature_data, feature_name):
        """Process and normalize a feature"""
        # Normalize
        if feature_name in self.feature_means:
            feature_data = (feature_data - self.feature_means[feature_name]) / self.feature_stds[feature_name]
        
        # Pad or truncate to sequence length
        if len(feature_data) > self.mean_sequence_length:
            feature_data = feature_data[:self.mean_sequence_length]
        else:
            feature_data = np.pad(feature_data, (0, self.mean_sequence_length - len(feature_data)), 'constant')
        
        return feature_data
    
    def extract_features(self, video_data):
        """Extract features for a single video"""
        if isinstance(video_data, str):
            # If video_data is a video name, extract from master data
            video_df = self.master_data[self.master_data['video_name'] == video_data]
        else:
            # If video_data is a DataFrame row or series
            video_df = video_data if hasattr(video_data, 'columns') else pd.DataFrame([video_data])
        
        if video_df.empty:
            return torch.zeros((self.mean_sequence_length, len(self.array_features) * 2), dtype=torch.float32)
        
        # Extract array features
        array_features_list = []
        for feature in self.array_features:
            if feature in video_df.columns:
                value = video_df[feature].values[0] if len(video_df) > 0 else None
                
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
        
        # Replace NaNs
        all_features = np.nan_to_num(all_features, nan=0.0)
        
        return torch.FloatTensor(all_features)
    
    def predict_single_video(self, test_csv_path, num_reference_videos=10):
        # Load test video
        test_data = pd.read_csv(test_csv_path)
        
        if len(test_data) == 0:
            raise ValueError("Test CSV is empty")
        
        # Extract test video features
        test_video_name = test_data['video_name'].iloc[0] if 'video_name' in test_data.columns else "test_video"
        test_features = self.extract_features(test_data.iloc[0]).to(self.device)
        
        print(f"Analyzing video: {test_video_name}")
        print(f"Test video features shape: {test_features.shape}")
        
        # Sample reference videos for efficiency
        real_sample = random.sample(self.real_videos, min(num_reference_videos, len(self.real_videos)))
        fake_sample = random.sample(self.fake_videos, min(num_reference_videos, len(self.fake_videos)))
        
        results = {
            'video_name': test_video_name,
            'real_similarities': [],
            'fake_similarities': [],
            'reference_videos': {'real': real_sample, 'fake': fake_sample}
        }
        
        # Compare with REAL videos
        print(f"\nComparing with {len(real_sample)} REAL reference videos...")
        with torch.no_grad():
            for real_video in tqdm(real_sample, desc="Real comparisons"):
                real_features = self.extract_features(real_video).to(self.device)
                
                # Add batch dimension
                test_batch = test_features.unsqueeze(0)
                real_batch = real_features.unsqueeze(0)
                
                similarity, _, _, _ = self.model(test_batch, real_batch)
                results['real_similarities'].append(similarity.item())
        
        # Compare with FAKE videos
        print(f"Comparing with {len(fake_sample)} FAKE reference videos...")
        with torch.no_grad():
            for fake_video in tqdm(fake_sample, desc="Fake comparisons"):
                fake_features = self.extract_features(fake_video).to(self.device)
                
                # Add batch dimension
                test_batch = test_features.unsqueeze(0)
                fake_batch = fake_features.unsqueeze(0)
                
                similarity, _, _, _ = self.model(test_batch, fake_batch)
                results['fake_similarities'].append(similarity.item())
        
        # Calculate statistics
        real_similarities = np.array(results['real_similarities'])
        fake_similarities = np.array(results['fake_similarities'])
        
        results['statistics'] = {
            'real_mean_similarity': np.mean(real_similarities),
            'real_std_similarity': np.std(real_similarities),
            'real_max_similarity': np.max(real_similarities),
            'fake_mean_similarity': np.mean(fake_similarities),
            'fake_std_similarity': np.std(fake_similarities),
            'fake_max_similarity': np.max(fake_similarities),
            'threshold': self.threshold
        }
        
        # Make prediction
        real_score = np.mean(real_similarities)
        fake_score = np.mean(fake_similarities)
        
        # Simple voting: if more similar to real videos, predict real
        if real_score > fake_score:
            prediction = "REAL"
            confidence = real_score - fake_score
        else:
            prediction = "FAKE"
            confidence = fake_score - real_score
        
        results['prediction'] = {
            'class': prediction,
            'confidence': confidence,
            'real_score': real_score,
            'fake_score': fake_score
        }
        
        return results
    
    def visualize_results(self, results):
        """Visualize the prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Similarity distributions
        axes[0, 0].hist(results['real_similarities'], bins=20, alpha=0.7, color='blue', label='vs REAL', density=True)
        axes[0, 0].hist(results['fake_similarities'], bins=20, alpha=0.7, color='red', label='vs FAKE', density=True)
        axes[0, 0].axvline(self.threshold, color='black', linestyle='--', label=f'Threshold={self.threshold:.3f}')
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Similarity Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [results['real_similarities'], results['fake_similarities']]
        box_plot = axes[0, 1].boxplot(data_to_plot, labels=['vs REAL', 'vs FAKE'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        axes[0, 1].axhline(self.threshold, color='black', linestyle='--', label=f'Threshold={self.threshold:.3f}')
        axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].set_title('Similarity Score Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean comparison
        means = [results['statistics']['real_mean_similarity'], results['statistics']['fake_mean_similarity']]
        stds = [results['statistics']['real_std_similarity'], results['statistics']['fake_std_similarity']]
        x_pos = np.arange(len(means))
        
        bars = axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5, 
                             color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[1, 0].set_xlabel('Comparison Type')
        axes[1, 0].set_ylabel('Mean Similarity Score')
        axes[1, 0].set_title('Mean Similarity Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(['vs REAL', 'vs FAKE'])
        axes[1, 0].axhline(self.threshold, color='black', linestyle='--', label=f'Threshold={self.threshold:.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction summary
        axes[1, 1].text(0.1, 0.8, f"Video: {results['video_name']}", fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Prediction: {results['prediction']['class']}", fontsize=14, fontweight='bold', 
                       color='blue' if results['prediction']['class'] == 'REAL' else 'red', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Confidence: {results['prediction']['confidence']:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Real Score: {results['prediction']['real_score']:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Fake Score: {results['prediction']['fake_score']:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f"Threshold: {self.threshold:.4f}", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Prediction Summary')
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self, results):
        """Print detailed prediction results"""

        
        print(f"\nVideo: {results['video_name']}")
        print(f"Prediction: {results['prediction']['class']}")
        print(f"Confidence: {results['prediction']['confidence']:.4f}")
        
        print(f"\nSimilarity Scores:")
        print(f"  Average similarity to REAL videos: {results['statistics']['real_mean_similarity']:.4f} +- {results['statistics']['real_std_similarity']:.4f}")
        print(f"  Average similarity to FAKE videos: {results['statistics']['fake_mean_similarity']:.4f} +- {results['statistics']['fake_std_similarity']:.4f}")
        
        print(f"\nMax Similarities:")
        print(f"  Highest similarity to REAL video: {results['statistics']['real_max_similarity']:.4f}")
        print(f"  Highest similarity to FAKE video: {results['statistics']['fake_max_similarity']:.4f}")
        
        print(f"\nModel Threshold: {self.threshold:.4f}")
        
        # Interpretation
        real_above_threshold = np.mean(np.array(results['real_similarities']) > self.threshold)
        fake_above_threshold = np.mean(np.array(results['fake_similarities']) > self.threshold)
        
        print(f"  {real_above_threshold:.1%} of comparisons with REAL videos are above threshold")
        print(f"  {fake_above_threshold:.1%} of comparisons with FAKE videos are above threshold")
        
        if results['prediction']['class'] == 'REAL':
            print(f"Prediction: real pain")
        else:
            print(f"Prediction: fake pain")

def predict_video_authenticity(model_path, master_csv_path, test_csv_path, num_reference_videos=20):

    predictor = VideoPredictor(model_path, master_csv_path)
    
    results = predictor.predict_single_video(test_csv_path, num_reference_videos)
    
    predictor.print_detailed_results(results)
    predictor.visualize_results(results)
    
    return results

if __name__ == "__main__":
    # Example usage
    model_path = 'neck_model.pth'
    master_csv_path = './neck_features.csv'
    test_csv_path = 'test.csv'
    
    results = predict_video_authenticity(
        model_path=model_path,
        master_csv_path=master_csv_path, 
        test_csv_path=test_csv_path,
        num_reference_videos=300
    )