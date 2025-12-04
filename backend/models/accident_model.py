import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
import os
import shap

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on location-based features"""
    def __init__(self, hidden_size: int):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for focusing on time-based features"""
    def __init__(self, hidden_size: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights

class CNNBiLSTMAttention(nn.Module):
    """
    CNN-BiLSTM-Attention model for traffic accident risk prediction.
    Based on the research paper architecture:
    - CNN for spatial feature extraction
    - BiLSTM for temporal sequence learning
    - Spatial-Temporal Local Attention for focusing on key features
    """
    def __init__(
        self,
        input_size: int = 32,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(CNNBiLSTMAttention, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 1D CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        )
        
        # BiLSTM for temporal feature learning
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Attention mechanisms
        self.spatial_attention = SpatialAttention(lstm_hidden * 2)
        self.temporal_attention = TemporalAttention(lstm_hidden * 2)
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x shape: (batch, input_size)
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, 1, input_size)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, cnn_channels, seq_len-1)
        
        # Transpose for LSTM: (batch, seq_len, cnn_channels)
        cnn_out = cnn_out.transpose(1, 2)
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(cnn_out)  # (batch, seq_len, lstm_hidden*2)
        
        # Apply attention mechanisms
        spatial_context, spatial_weights = self.spatial_attention(lstm_out)
        temporal_context, temporal_weights = self.temporal_attention(lstm_out)
        
        # Concatenate attention outputs
        combined = torch.cat([spatial_context, temporal_context], dim=1)
        
        # Classification
        output = self.fc(combined)
        
        attention_weights = {
            'spatial': spatial_weights,
            'temporal': temporal_weights
        }
        
        return output, attention_weights

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get probability predictions"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()


class AccidentRiskPredictor:
    """
    High-level wrapper for the accident risk prediction model.
    Handles preprocessing, prediction, and SHAP-based explanations.
    """
    
    SEVERITY_LABELS = ['Slight', 'Serious', 'Fatal']
    
    FEATURE_NAMES = [
        'hour', 'day_of_week', 'month', 'year',
        'latitude', 'longitude', 'speed_limit', 'road_type',
        'junction_control', 'junction_detail', 'light_conditions',
        'weather_conditions', 'road_surface_conditions', 'urban_rural',
        'number_of_vehicles', 'number_of_casualties', 'police_force',
        'vehicle_type', 'vehicle_manoeuvre', 'engine_capacity',
        'age_of_vehicle', 'driver_age', 'driver_sex',
        'journey_purpose', 'pedestrian_crossing_human',
        'pedestrian_crossing_physical', 'special_conditions',
        'carriageway_hazards', 'first_road_class', 'second_road_class',
        'road_type_detail', 'did_police_attend'
    ]
    
    FEATURE_CATEGORIES = {
        'time': ['hour', 'day_of_week', 'month', 'year'],
        'spatial': ['latitude', 'longitude', 'road_type', 'junction_control',
                   'junction_detail', 'urban_rural', 'first_road_class',
                   'second_road_class', 'road_type_detail'],
        'environmental': ['light_conditions', 'weather_conditions',
                         'road_surface_conditions', 'special_conditions',
                         'carriageway_hazards'],
        'vehicle': ['number_of_vehicles', 'vehicle_type', 'vehicle_manoeuvre',
                   'engine_capacity', 'age_of_vehicle'],
        'personnel': ['number_of_casualties', 'driver_age', 'driver_sex',
                     'journey_purpose', 'pedestrian_crossing_human',
                     'pedestrian_crossing_physical', 'police_force',
                     'did_police_attend']
    }
    
    def __init__(self, model_path: str = None):
        self.model = CNNBiLSTMAttention()
        self.scaler = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess(self, features: Dict) -> np.ndarray:
        """Preprocess input features for prediction"""
        # Create feature vector in correct order
        feature_vector = []
        for name in self.FEATURE_NAMES:
            value = features.get(name, 0)
            # Normalize numerical values
            if name == 'hour':
                value = value / 24.0
            elif name == 'day_of_week':
                value = value / 7.0
            elif name == 'month':
                value = value / 12.0
            elif name == 'year':
                value = (value - 2005) / 15.0  # Normalize to 2005-2020 range
            elif name == 'speed_limit':
                value = value / 70.0  # Max typical speed limit
            elif name == 'latitude':
                value = (value - 50) / 10.0  # UK latitude range
            elif name == 'longitude':
                value = (value + 5) / 10.0  # UK longitude range
            elif name == 'engine_capacity':
                value = value / 5000.0  # Max engine capacity in cc
            elif name == 'age_of_vehicle':
                value = value / 30.0  # Max vehicle age
            elif name == 'driver_age':
                value = value / 100.0  # Max driver age
            elif name == 'number_of_vehicles':
                value = value / 10.0
            elif name == 'number_of_casualties':
                value = value / 10.0
            feature_vector.append(float(value))
        
        return np.array(feature_vector, dtype=np.float32)
    
    def predict(self, features: Dict) -> Dict:
        """Make a prediction for given features"""
        # Preprocess
        x = self.preprocess(features)
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits, attention = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        
        return {
            'severity': self.SEVERITY_LABELS[predicted_class],
            'risk_level': predicted_class + 1,
            'probabilities': {
                'slight': float(probs[0]),
                'serious': float(probs[1]),
                'fatal': float(probs[2])
            },
            'confidence': float(np.max(probs))
        }
    
    def get_feature_importance(self, features: Dict) -> List[Dict]:
        """Calculate feature importance using gradient-based approximation"""
        x = self.preprocess(features)
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        x_tensor.requires_grad = True
        
        self.model.eval()
        logits, _ = self.model(x_tensor)
        
        # Get gradients for predicted class
        predicted_class = torch.argmax(logits, dim=1)
        logits[0, predicted_class].backward()
        
        # Feature importance based on gradients
        gradients = x_tensor.grad.numpy()[0]
        importance = np.abs(gradients)
        
        # Normalize
        importance = importance / (np.sum(importance) + 1e-10)
        
        # Create importance list
        feature_importance = []
        for i, name in enumerate(self.FEATURE_NAMES):
            feature_importance.append({
                'feature': name,
                'importance': float(importance[i]),
                'value': float(x[i]),
                'category': self._get_feature_category(name)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Get the category of a feature"""
        for category, features in self.FEATURE_CATEGORIES.items():
            if feature_name in features:
                return category
        return 'other'
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save({
            'model_state': self.model.state_dict(),
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_trained = checkpoint.get('is_trained', True)
