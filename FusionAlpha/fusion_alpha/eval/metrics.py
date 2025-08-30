#!/usr/bin/env python3
"""
Metrics calculator for evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculator for various evaluation metrics"""
    
    def __init__(self):
        self.supported_metrics = {
            'auroc', 'accuracy', 'precision', 'recall', 'f1',
            'mse', 'mae', 'rmse', 'ece', 'mce', 'brier_score',
            'coverage', 'selective_risk', 'abstain_rate'
        }
    
    def calculate_metrics(self, 
                         predictions: List[float],
                         labels: List[float], 
                         confidences: List[float],
                         metrics: List[str]) -> Dict[str, float]:
        """Calculate specified metrics"""
        
        results = {}
        
        predictions = np.array(predictions)
        labels = np.array(labels) 
        confidences = np.array(confidences)
        
        # Classification metrics
        if any(m in metrics for m in ['auroc', 'accuracy', 'precision', 'recall', 'f1']):
            pred_binary = (predictions > 0.5).astype(int)
            labels_binary = (labels > 0.5).astype(int)
            
            if 'auroc' in metrics and len(np.unique(labels_binary)) > 1:
                try:
                    results['auroc'] = roc_auc_score(labels_binary, predictions)
                except:
                    results['auroc'] = 0.5
            
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(labels_binary, pred_binary)
            
            if any(m in metrics for m in ['precision', 'recall', 'f1']):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels_binary, pred_binary, average='binary', zero_division=0
                )
                if 'precision' in metrics:
                    results['precision'] = precision
                if 'recall' in metrics:
                    results['recall'] = recall
                if 'f1' in metrics:
                    results['f1'] = f1
        
        # Regression metrics
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(labels, predictions)
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(labels, predictions)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(labels, predictions))
        
        # Calibration metrics would be calculated here
        # (Implementation depends on specific requirements)
        
        return results
    
    def calculate_per_group_metrics(self,
                                  predictions: List[float],
                                  labels: List[float], 
                                  groups: List[str],
                                  metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per group/regime"""
        
        group_results = {}
        unique_groups = set(groups)
        
        for group in unique_groups:
            group_mask = np.array(groups) == group
            group_predictions = np.array(predictions)[group_mask]
            group_labels = np.array(labels)[group_mask]
            group_confidences = np.ones(len(group_predictions))  # Placeholder
            
            if len(group_predictions) > 0:
                group_results[group] = self.calculate_metrics(
                    group_predictions.tolist(),
                    group_labels.tolist(), 
                    group_confidences.tolist(),
                    metrics
                )
        
        return group_results
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics"""
        return list(self.supported_metrics)