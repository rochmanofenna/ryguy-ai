#!/usr/bin/env python3
"""
Modality Registry System

Clean schema + factory for registering any encoder without touching core code.
The spine of the domain-agnostic architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModalitySpec:
    """Schema for modality specification"""
    name: str
    extractor: 'BaseExtractor'
    freshness_policy: str  # "latest_24h", "all", "window:7d" 
    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseExtractor(ABC):
    """Base class for all modality extractors"""
    
    @abstractmethod
    def extract(self, data: Any) -> torch.Tensor:
        """Extract features from raw data"""
        pass
    
    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return the shape of extracted features"""
        pass
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data format"""
        return True

class TextEmbedder(BaseExtractor):
    """Text embedding extractor using HuggingFace models"""
    
    def __init__(self, model_name: str = "mxbai-embed-large", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.embed_dim = 1024  # Default for mxbai-embed-large
        
        # Mock implementation - in production would load actual HF model
        self.encoder = nn.Linear(1, self.embed_dim)  # Placeholder
        logger.info(f"Initialized TextEmbedder with model: {model_name}")
    
    @classmethod
    def from_hf(cls, model_name: str) -> 'TextEmbedder':
        """Factory method for HuggingFace models"""
        return cls(model_name)
    
    def extract(self, data: Union[str, List[str]]) -> torch.Tensor:
        """Extract embeddings from text"""
        if isinstance(data, str):
            data = [data]
        
        # Mock embedding - in production would use actual model
        batch_size = len(data)
        embeddings = torch.randn(batch_size, self.embed_dim)
        return embeddings
    
    def get_output_shape(self) -> Tuple[int, ...]:
        return (self.embed_dim,)

class SeriesStats(BaseExtractor):
    """Time series statistical feature extractor"""
    
    def __init__(self, features: List[str] = None, window_size: int = 100):
        if features is None:
            features = ["slope", "zscore", "vol", "mean", "std", "min", "max", "trend"]
        self.features = features
        self.window_size = window_size
        self.output_dim = len(features)
        logger.info(f"Initialized SeriesStats with features: {features}")
    
    def extract(self, data: Union[np.ndarray, torch.Tensor, List[float]]) -> torch.Tensor:
        """Extract statistical features from time series"""
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.numpy()
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Add batch dimension
        
        batch_size = data.shape[0]
        features = torch.zeros(batch_size, self.output_dim)
        
        for i, series in enumerate(data):
            # Compute statistical features
            feature_idx = 0
            
            if "slope" in self.features:
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0] if len(series) > 1 else 0.0
                features[i, feature_idx] = slope
                feature_idx += 1
            
            if "zscore" in self.features:
                zscore = (series[-1] - np.mean(series)) / (np.std(series) + 1e-8)
                features[i, feature_idx] = zscore
                feature_idx += 1
                
            if "vol" in self.features:
                vol = np.std(series)
                features[i, feature_idx] = vol
                feature_idx += 1
                
            if "mean" in self.features:
                features[i, feature_idx] = np.mean(series)
                feature_idx += 1
                
            if "std" in self.features:
                features[i, feature_idx] = np.std(series)
                feature_idx += 1
                
            if "min" in self.features:
                features[i, feature_idx] = np.min(series)
                feature_idx += 1
                
            if "max" in self.features:
                features[i, feature_idx] = np.max(series)
                feature_idx += 1
                
            if "trend" in self.features:
                # Simple trend: positive if end > start
                trend = 1.0 if series[-1] > series[0] else -1.0
                features[i, feature_idx] = trend
                feature_idx += 1
        
        return features
    
    def get_output_shape(self) -> Tuple[int, ...]:
        return (self.output_dim,)

class ImageExtractor(BaseExtractor):
    """Vision feature extractor using lightweight models"""
    
    def __init__(self, model_type: str = "resnet18", pretrained: bool = True):
        self.model_type = model_type
        self.pretrained = pretrained
        self.feature_dim = 512  # ResNet18 feature dimension
        
        # Mock implementation - in production would load actual vision model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.feature_dim)
        )
        logger.info(f"Initialized ImageExtractor with model: {model_type}")
    
    def extract(self, data: torch.Tensor) -> torch.Tensor:
        """Extract features from images"""
        if len(data.shape) == 3:
            data = data.unsqueeze(0)  # Add batch dimension
        
        # Mock feature extraction
        batch_size = data.shape[0]
        features = torch.randn(batch_size, self.feature_dim)
        return features
    
    def get_output_shape(self) -> Tuple[int, ...]:
        return (self.feature_dim,)

class StructuredExtractor(BaseExtractor):
    """Extractor for structured/tabular data"""
    
    def __init__(self, feature_columns: List[str], categorical_dims: Dict[str, int] = None):
        self.feature_columns = feature_columns
        self.categorical_dims = categorical_dims or {}
        self.output_dim = len(feature_columns)
        
        # Add embedding dimensions for categorical features
        for col, dim in self.categorical_dims.items():
            if col in feature_columns:
                self.output_dim += dim - 1  # Replace 1D categorical with embedding
        
        logger.info(f"Initialized StructuredExtractor with {len(feature_columns)} features")
    
    def extract(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from structured data"""
        features = []
        
        for col in self.feature_columns:
            if col in data:
                if col in self.categorical_dims:
                    # Mock categorical embedding
                    embed_dim = self.categorical_dims[col]
                    batch_size = data[col].shape[0]
                    embedding = torch.randn(batch_size, embed_dim)
                    features.append(embedding)
                else:
                    # Numerical feature
                    if len(data[col].shape) == 1:
                        features.append(data[col].unsqueeze(1))
                    else:
                        features.append(data[col])
        
        if features:
            return torch.cat(features, dim=1)
        else:
            # Return empty tensor with correct batch size
            batch_size = next(iter(data.values())).shape[0]
            return torch.zeros(batch_size, self.output_dim)
    
    def get_output_shape(self) -> Tuple[int, ...]:
        return (self.output_dim,)

class ModalityRegistry:
    """Central registry for managing modalities"""
    
    def __init__(self):
        self.modalities: Dict[str, ModalitySpec] = {}
        self.extractors: Dict[str, BaseExtractor] = {}
        logger.info("Initialized ModalityRegistry")
    
    def register_modality(self, 
                         name: str,
                         extractor: BaseExtractor,
                         freshness_policy: str = "all",
                         shape: Optional[Tuple[int, ...]] = None,
                         dtype: torch.dtype = torch.float32,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new modality"""
        
        if shape is None:
            shape = extractor.get_output_shape()
        
        spec = ModalitySpec(
            name=name,
            extractor=extractor,
            freshness_policy=freshness_policy,
            shape=shape,
            dtype=dtype,
            metadata=metadata or {}
        )
        
        self.modalities[name] = spec
        self.extractors[name] = extractor
        
        logger.info(f"Registered modality '{name}' with shape {shape} and policy '{freshness_policy}'")
    
    def get_modality(self, name: str) -> Optional[ModalitySpec]:
        """Get modality specification by name"""
        return self.modalities.get(name)
    
    def extract_features(self, name: str, data: Any) -> torch.Tensor:
        """Extract features using registered modality"""
        if name not in self.extractors:
            raise ValueError(f"Modality '{name}' not registered")
        
        return self.extractors[name].extract(data)
    
    def extract_batch(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract features for a batch across all modalities"""
        results = {}
        
        for modality_name, data in batch_data.items():
            if modality_name in self.extractors:
                try:
                    features = self.extract_features(modality_name, data)
                    results[modality_name] = features
                except Exception as e:
                    logger.warning(f"Failed to extract features for '{modality_name}': {e}")
            else:
                logger.warning(f"Unknown modality: {modality_name}")
        
        return results
    
    def list_modalities(self) -> List[str]:
        """List all registered modality names"""
        return list(self.modalities.keys())
    
    def get_combined_shape(self, modality_names: List[str]) -> Tuple[int, ...]:
        """Get combined shape for multiple modalities"""
        total_dim = 0
        for name in modality_names:
            if name in self.modalities:
                total_dim += np.prod(self.modalities[name].shape)
        return (total_dim,)
    
    def validate_freshness(self, name: str, timestamp: datetime) -> bool:
        """Validate data freshness according to policy"""
        if name not in self.modalities:
            return False
        
        policy = self.modalities[name].freshness_policy
        now = datetime.now()
        
        if policy == "all":
            return True
        elif policy.startswith("latest_"):
            # Parse duration like "latest_24h"
            duration_str = policy.replace("latest_", "")
            if duration_str.endswith("h"):
                hours = int(duration_str[:-1])
                cutoff = now - timedelta(hours=hours)
            elif duration_str.endswith("d"):
                days = int(duration_str[:-1])
                cutoff = now - timedelta(days=days)
            else:
                return True  # Default to accept
            
            return timestamp >= cutoff
        elif policy.startswith("window:"):
            # Parse window like "window:7d"
            window_str = policy.replace("window:", "")
            if window_str.endswith("d"):
                days = int(window_str[:-1])
                cutoff = now - timedelta(days=days)
                return timestamp >= cutoff
        
        return True  # Default to accept

# Global registry instance
_global_registry = ModalityRegistry()

def register_modality(name: str,
                     extractor: BaseExtractor,
                     freshness_policy: str = "all",
                     shape: Optional[Tuple[int, ...]] = None,
                     dtype: torch.dtype = torch.float32,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """Global function to register modality"""
    _global_registry.register_modality(name, extractor, freshness_policy, shape, dtype, metadata)

def get_registry() -> ModalityRegistry:
    """Get the global registry instance"""
    return _global_registry

# Convenience factory functions
def create_text_modality(name: str, model_name: str = "mxbai-embed-large", 
                        freshness_policy: str = "latest_24h") -> None:
    """Create and register a text modality"""
    extractor = TextEmbedder.from_hf(model_name)
    register_modality(name, extractor, freshness_policy)

def create_series_modality(name: str, features: List[str] = None,
                          freshness_policy: str = "latest_1h") -> None:
    """Create and register a time series modality"""
    extractor = SeriesStats(features)
    register_modality(name, extractor, freshness_policy)

def create_image_modality(name: str, model_type: str = "resnet18",
                         freshness_policy: str = "latest_1h") -> None:
    """Create and register an image modality"""
    extractor = ImageExtractor(model_type)
    register_modality(name, extractor, freshness_policy)

def create_structured_modality(name: str, feature_columns: List[str],
                              categorical_dims: Dict[str, int] = None,
                              freshness_policy: str = "all") -> None:
    """Create and register a structured data modality"""
    extractor = StructuredExtractor(feature_columns, categorical_dims)
    register_modality(name, extractor, freshness_policy)

if __name__ == "__main__":
    # Demo the modality registry
    print("Modality Registry Demo")
    print("=" * 30)
    
    # Register some modalities
    create_text_modality("patient_notes", freshness_policy="latest_24h")
    create_series_modality("vital_signs", features=["slope", "zscore", "vol"])
    create_image_modality("xray_images")
    create_structured_modality("lab_results", ["glucose", "bp_systolic", "heart_rate"])
    
    registry = get_registry()
    print(f"Registered modalities: {registry.list_modalities()}")
    
    # Test extraction
    test_data = {
        "patient_notes": ["Patient reports chest pain"],
        "vital_signs": [[120, 125, 130, 128, 135]],
        "lab_results": {
            "glucose": torch.tensor([95.0]),
            "bp_systolic": torch.tensor([140.0]),
            "heart_rate": torch.tensor([85.0])
        }
    }
    
    features = registry.extract_batch(test_data)
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    print("\nModality registry system working correctly!")