#!/usr/bin/env python3
"""
Dataset utilities for evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

def create_synthetic_dataset(n_samples: int = 1000, 
                           n_features: int = 32,
                           n_modalities: int = 2,
                           contradiction_rate: float = 0.3,
                           noise_level: float = 0.1) -> Dict[str, Any]:
    """Create synthetic dataset with controllable contradiction patterns"""
    
    np.random.seed(42)
    
    # Generate base features
    features = []
    labels = []
    regimes = []
    metadata = []
    
    modality_dim = n_features // n_modalities
    
    for i in range(n_samples):
        # Determine if this sample should have contradiction
        has_contradiction = np.random.random() < contradiction_rate
        
        # Generate modality features
        sample_features = []
        base_features = None
        
        for mod in range(n_modalities):
            if mod == 0:
                # Base modality
                mod_features = np.random.randn(modality_dim)
                base_features = mod_features.copy()
            else:
                # Other modalities - either aligned or contradictory
                if has_contradiction:
                    # Generate contradictory features
                    mod_features = -base_features + noise_level * np.random.randn(modality_dim)
                else:
                    # Generate aligned features
                    mod_features = base_features + noise_level * np.random.randn(modality_dim)
            
            sample_features.extend(mod_features)
        
        # Pad to target dimension
        while len(sample_features) < n_features:
            sample_features.append(0.0)
        
        sample_features = np.array(sample_features[:n_features])
        
        # Generate label based on modality alignment
        if has_contradiction:
            label = 0.0  # Contradiction = negative outcome
            regime = "contradictory"
        else:
            label = 1.0  # Alignment = positive outcome  
            regime = "aligned"
        
        features.append(torch.tensor(sample_features, dtype=torch.float32))
        labels.append(label)
        regimes.append(regime)
        metadata.append({
            'has_contradiction': has_contradiction,
            'generated_at': i
        })
    
    dataset = {
        'features': features,
        'labels': labels,
        'regime': regimes,
        'metadata': metadata,
        'n_features': n_features,
        'n_modalities': n_modalities,
        'modalities': [f'modality_{i}' for i in range(n_modalities)],
        'task_type': 'classification',
        'contradiction_rate': contradiction_rate
    }
    
    logger.info(f"Created synthetic dataset: {n_samples} samples, {contradiction_rate:.1%} contradiction rate")
    
    return dataset

class DatasetLoader:
    """Loader for various datasets"""
    
    def __init__(self):
        self.available_datasets = {
            'synthetic': create_synthetic_dataset,
            'healthcare_demo': self._create_healthcare_demo,
            'sre_demo': self._create_sre_demo,
            'robotics_demo': self._create_robotics_demo
        }
    
    def load_dataset(self, name: str, **kwargs) -> Dict[str, Any]:
        """Load dataset by name"""
        
        if name not in self.available_datasets:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.available_datasets.keys())}")
        
        return self.available_datasets[name](**kwargs)
    
    def _create_healthcare_demo(self, n_samples: int = 500) -> Dict[str, Any]:
        """Create healthcare demonstration dataset"""
        
        features = []
        labels = []
        regimes = {}
        metadata = {}
        
        for i in range(n_samples):
            # Symptom severity (0-10 scale, normalized)
            symptom_severity = np.random.uniform(0, 1, 8)  # 8 different symptoms
            
            # Biomarker levels (0-1 scale) 
            biomarker_levels = np.random.uniform(0, 1, 8)  # 8 different biomarkers
            
            # Create contradiction scenarios
            if np.random.random() < 0.3:
                # High symptoms, low biomarkers (functional disorder)
                symptom_severity = np.random.uniform(0.7, 1.0, 8)
                biomarker_levels = np.random.uniform(0.0, 0.3, 8)
                label = 0.0  # Poor outcome prediction
                regime = "functional_disorder"
            elif np.random.random() < 0.2:
                # Low symptoms, high biomarkers (underreporting)
                symptom_severity = np.random.uniform(0.0, 0.3, 8)
                biomarker_levels = np.random.uniform(0.7, 1.0, 8)
                label = 1.0  # Actually needs attention
                regime = "underreported"
            else:
                # Aligned symptoms and biomarkers
                correlation = np.random.uniform(0.7, 0.9)
                biomarker_levels = symptom_severity * correlation + 0.1 * np.random.randn(8)
                biomarker_levels = np.clip(biomarker_levels, 0, 1)
                label = np.mean(symptom_severity)
                regime = "standard_case"
            
            combined_features = np.concatenate([symptom_severity, biomarker_levels])
            features.append(torch.tensor(combined_features, dtype=torch.float32))
            labels.append(float(label))
            regimes[i] = regime
            metadata[i] = {
                'symptom_mean': float(np.mean(symptom_severity)),
                'biomarker_mean': float(np.mean(biomarker_levels))
            }
        
        return {
            'features': features,
            'labels': labels,
            'regime': regimes,
            'metadata': metadata,
            'n_features': 16,
            'n_modalities': 2,
            'modalities': ['symptoms', 'biomarkers'],
            'task_type': 'regression'
        }
    
    def _create_sre_demo(self, n_samples: int = 500) -> Dict[str, Any]:
        """Create SRE/ops demonstration dataset"""
        
        features = []
        labels = []
        regimes = {}
        metadata = {}
        
        for i in range(n_samples):
            # Metrics (error rates, latency, throughput)
            metrics = np.random.uniform(0, 1, 8)
            
            # User reports (ticket volume, complaint severity)
            user_reports = np.random.uniform(0, 1, 8)
            
            # Create incident scenarios
            if np.random.random() < 0.2:
                # Metrics look good but users complaining (hidden issue)
                metrics = np.random.uniform(0.0, 0.3, 8)  # Good metrics
                user_reports = np.random.uniform(0.7, 1.0, 8)  # High complaints
                label = 1.0  # Incident
                regime = "hidden_incident"
            elif np.random.random() < 0.15:
                # Metrics bad but no user reports (not user-facing)
                metrics = np.random.uniform(0.7, 1.0, 8)  # Bad metrics
                user_reports = np.random.uniform(0.0, 0.3, 8)  # Low reports
                label = 0.5  # Internal issue
                regime = "internal_issue"
            else:
                # Normal correlation
                correlation = np.random.uniform(0.6, 0.8)
                user_reports = metrics * correlation + 0.2 * np.random.randn(8)
                user_reports = np.clip(user_reports, 0, 1)
                label = (np.mean(metrics) + np.mean(user_reports)) / 2
                regime = "normal_ops"
            
            combined_features = np.concatenate([metrics, user_reports])
            features.append(torch.tensor(combined_features, dtype=torch.float32))
            labels.append(float(label))
            regimes[i] = regime
            metadata[i] = {
                'metrics_mean': float(np.mean(metrics)),
                'reports_mean': float(np.mean(user_reports))
            }
        
        return {
            'features': features,
            'labels': labels,
            'regime': regimes,
            'metadata': metadata,
            'n_features': 16,
            'n_modalities': 2,
            'modalities': ['metrics', 'user_reports'],
            'task_type': 'regression'
        }
    
    def _create_robotics_demo(self, n_samples: int = 500) -> Dict[str, Any]:
        """Create robotics demonstration dataset"""
        
        features = []
        labels = []
        regimes = {}
        metadata = {}
        
        for i in range(n_samples):
            # Vision detection confidence
            vision_conf = np.random.uniform(0, 1, 6)
            
            # Lidar distance measurements  
            lidar_dist = np.random.uniform(0, 1, 6)
            
            # Proprioceptive state
            proprio_state = np.random.uniform(0, 1, 4)
            
            # Create safety scenarios
            if np.random.random() < 0.1:
                # Vision says clear, lidar detects obstacle
                vision_conf = np.random.uniform(0.0, 0.2, 6)  # Low obstacle detection
                lidar_dist = np.random.uniform(0.8, 1.0, 6)   # Close obstacles detected
                label = 0.0  # Stop/unsafe
                regime = "vision_lidar_conflict"
            elif np.random.random() < 0.05:
                # Vision detects obstacle, lidar says clear
                vision_conf = np.random.uniform(0.8, 1.0, 6)   # High obstacle detection
                lidar_dist = np.random.uniform(0.0, 0.2, 6)    # Far/no obstacles
                label = 0.5  # Investigate
                regime = "lidar_vision_conflict"
            else:
                # Normal operation
                correlation = np.random.uniform(0.7, 0.9)
                lidar_dist = (1 - vision_conf) * correlation + 0.1 * np.random.randn(6)
                lidar_dist = np.clip(lidar_dist, 0, 1)
                label = 1.0 - np.mean(vision_conf)  # Safe to proceed if no obstacles
                regime = "normal_operation"
            
            combined_features = np.concatenate([vision_conf, lidar_dist, proprio_state])
            features.append(torch.tensor(combined_features, dtype=torch.float32))
            labels.append(float(label))
            regimes[i] = regime
            metadata[i] = {
                'vision_mean': float(np.mean(vision_conf)),
                'lidar_mean': float(np.mean(lidar_dist)),
                'proprio_mean': float(np.mean(proprio_state))
            }
        
        return {
            'features': features,
            'labels': labels,
            'regime': regimes,
            'metadata': metadata,
            'n_features': 16,
            'n_modalities': 3,
            'modalities': ['vision', 'lidar', 'proprioception'],
            'task_type': 'regression'
        }