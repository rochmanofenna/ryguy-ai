#!/usr/bin/env python3
"""
Evaluation Harness

One command evaluation: `fa eval --gate mlp --dataset robotics_nav --metrics auroc,ece,coverage --report out/`
Always reports: per-regime AUROC, calibration, risk-coverage, gate confusion
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
from datetime import datetime
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

from ..core.router import FusionAlpha
from ..core.modality_registry import get_registry
from ..core.calibrate import CalibrationMetrics
from .datasets import DatasetLoader, create_synthetic_dataset
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    gate_type: str = "rule"
    dataset: str = "synthetic"
    metrics: List[str] = None
    output_dir: str = "eval_results"
    num_trials: int = 1
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    device: str = "cpu"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["auroc", "ece", "coverage", "gate_confusion"]

@dataclass
class EvalResult:
    """Complete evaluation result"""
    config: EvalConfig
    overall_metrics: Dict[str, float]
    per_regime_metrics: Dict[str, Dict[str, float]]
    calibration_metrics: Dict[str, float]
    gate_confusion: Dict[str, Any]
    coverage_analysis: Dict[str, float]
    expert_performance: Dict[str, Dict[str, float]]
    predictions: List[Dict[str, Any]]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class EvalHarness:
    """Main evaluation harness"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator()
        
        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        logger.info(f"Initialized EvalHarness with config: {config}")
    
    def run_evaluation(self) -> EvalResult:
        """Run complete evaluation"""
        
        logger.info(f"Starting evaluation with {self.config.gate_type} gate on {self.config.dataset}")
        
        # Load dataset
        train_loader, val_loader, test_loader, dataset_info = self._load_dataset()
        
        # Create and train FusionAlpha
        fusion_alpha = self._create_fusion_alpha(dataset_info)
        
        if train_loader:
            fusion_alpha.fit(train_loader, val_loader)
        
        # Run evaluation on test set
        predictions = []
        for batch in test_loader:
            batch_predictions = self._evaluate_batch(fusion_alpha, batch)
            predictions.extend(batch_predictions)
        
        # Calculate metrics
        overall_metrics = self._calculate_overall_metrics(predictions)
        per_regime_metrics = self._calculate_per_regime_metrics(predictions)
        calibration_metrics = self._calculate_calibration_metrics(predictions)
        gate_confusion = self._analyze_gate_confusion(predictions)
        coverage_analysis = self._analyze_coverage(predictions)
        expert_performance = self._analyze_expert_performance(predictions, fusion_alpha)
        
        # Create result
        result = EvalResult(
            config=self.config,
            overall_metrics=overall_metrics,
            per_regime_metrics=per_regime_metrics,
            calibration_metrics=calibration_metrics,
            gate_confusion=gate_confusion,
            coverage_analysis=coverage_analysis,
            expert_performance=expert_performance,
            predictions=predictions,
            timestamp=datetime.now().isoformat()
        )
        
        # Save results
        self._save_results(result)
        
        logger.info(f"Evaluation completed. Results saved to {self.config.output_dir}")
        
        return result
    
    def _load_dataset(self) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """Load and split dataset"""
        
        if self.config.dataset == "synthetic":
            # Create synthetic dataset
            n_modalities = 2
            modality_dim = 8
            total_features = n_modalities * modality_dim
            dataset = create_synthetic_dataset(
                n_samples=1000,
                n_features=total_features,
                n_modalities=n_modalities,
                contradiction_rate=0.3
            )
        else:
            # Load real dataset
            loader = DatasetLoader()
            dataset = loader.load_dataset(self.config.dataset)
        
        # Split dataset
        n_total = len(dataset['features'])
        n_train = int(self.config.train_split * n_total)
        n_val = int(self.config.val_split * n_total)
        n_test = n_total - n_train - n_val
        
        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        def create_loader(indices):
            regime_data = dataset.get('regime', [])
            metadata_data = dataset.get('metadata', [])
            
            return [{
                'features': dataset['features'][i],
                'labels': dataset['labels'][i] if 'labels' in dataset else 0,
                'regime': regime_data[i] if isinstance(regime_data, list) and i < len(regime_data) else regime_data.get(i, 'unknown') if isinstance(regime_data, dict) else 'unknown',
                'metadata': metadata_data[i] if isinstance(metadata_data, list) and i < len(metadata_data) else metadata_data.get(i, {}) if isinstance(metadata_data, dict) else {}
            } for i in indices]
        
        train_loader = create_loader(train_idx) if n_train > 0 else None
        val_loader = create_loader(val_idx) if n_val > 0 else None
        test_loader = create_loader(test_idx)
        
        dataset_info = {
            'n_features': dataset.get('n_features', 32),
            'n_modalities': dataset.get('n_modalities', 2),
            'modalities': dataset.get('modalities', ['modality_a', 'modality_b']),
            'task_type': dataset.get('task_type', 'classification')
        }
        
        logger.info(f"Dataset loaded: {n_train} train, {n_val} val, {n_test} test samples")
        
        return train_loader, val_loader, test_loader, dataset_info
    
    def _create_fusion_alpha(self, dataset_info: Dict[str, Any]) -> FusionAlpha:
        """Create FusionAlpha instance"""
        
        # Register modalities if not already registered
        registry = get_registry()
        for modality in dataset_info.get('modalities', []):
            if modality not in registry.list_modalities():
                from ..core.modality_registry import create_series_modality
                create_series_modality(modality)
        
        fusion_alpha = FusionAlpha(
            modalities=dataset_info.get('modalities'),
            gate=self.config.gate_type,
            contradiction_scores=["cosine", "delta", "magnitude_ratio"],
            calibrator="temperature",
            device=self.config.device
        )
        
        return fusion_alpha
    
    def _evaluate_batch(self, fusion_alpha: FusionAlpha, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate single batch"""
        
        # Handle different batch formats
        if isinstance(batch, list):
            results = []
            for item in batch:
                result = fusion_alpha.predict(item['features'], return_info=True)
                prediction, info = result
                
                results.append({
                    'prediction': float(prediction),
                    'confidence': info['uncertainty'],
                    'expert_used': info['chosen_expert'],
                    'gate_scores': info['routing_scores'],
                    'contradiction_scores': info['scores'],
                    'abstained': info['abstained'],
                    'true_label': item.get('labels', 0),
                    'regime': item.get('regime', 'unknown'),
                    'metadata': item.get('metadata', {})
                })
            return results
        else:
            result = fusion_alpha.predict(batch['features'], return_info=True)
            prediction, info = result
            
            return [{
                'prediction': float(prediction),
                'confidence': info['uncertainty'],
                'expert_used': info['chosen_expert'],
                'gate_scores': info['routing_scores'],
                'contradiction_scores': info['scores'],
                'abstained': info['abstained'],
                'true_label': batch.get('labels', 0),
                'regime': batch.get('regime', 'unknown'),
                'metadata': batch.get('metadata', {})
            }]
    
    def _calculate_overall_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        
        metrics = {}
        
        # Extract predictions and labels
        y_pred = [p['prediction'] for p in predictions]
        y_true = [p['true_label'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        # Convert to numpy for easier computation
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        confidences = np.array(confidences)
        
        # Binary classification metrics (assuming threshold at 0.5)
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_true_binary = (y_true > 0.5).astype(int) if any(y > 1 for y in y_true) else y_true.astype(int)
        
        if "auroc" in self.config.metrics:
            try:
                if len(np.unique(y_true_binary)) > 1:
                    metrics['auroc'] = roc_auc_score(y_true_binary, y_pred)
                else:
                    metrics['auroc'] = 0.5  # Random baseline for single class
            except:
                metrics['auroc'] = 0.5
        
        if "accuracy" in self.config.metrics:
            metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        
        if "precision" in self.config.metrics or "recall" in self.config.metrics or "f1" in self.config.metrics:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_binary, y_pred_binary, average='binary', zero_division=0
            )
            if "precision" in self.config.metrics:
                metrics['precision'] = precision
            if "recall" in self.config.metrics:
                metrics['recall'] = recall
            if "f1" in self.config.metrics:
                metrics['f1'] = f1
        
        # Add custom metrics
        metrics['abstain_rate'] = np.mean([p['abstained'] for p in predictions])
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['prediction_variance'] = np.var(y_pred)
        
        return metrics
    
    def _calculate_per_regime_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per regime/domain"""
        
        regime_metrics = {}
        
        # Group by regime
        regime_groups = {}
        for pred in predictions:
            regime = pred['regime']
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(pred)
        
        # Calculate metrics for each regime
        for regime, group in regime_groups.items():
            if len(group) < 5:  # Skip regimes with too few samples
                continue
            
            y_pred = np.array([p['prediction'] for p in group])
            y_true = np.array([p['true_label'] for p in group])
            
            regime_metrics[regime] = {}
            
            if len(np.unique(y_true)) > 1:
                try:
                    regime_metrics[regime]['auroc'] = roc_auc_score(y_true, y_pred)
                except:
                    regime_metrics[regime]['auroc'] = 0.5
            else:
                regime_metrics[regime]['auroc'] = 0.5
            
            y_pred_binary = (y_pred > 0.5).astype(int)
            y_true_binary = y_true.astype(int)
            regime_metrics[regime]['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
            regime_metrics[regime]['abstain_rate'] = np.mean([p['abstained'] for p in group])
        
        return regime_metrics
    
    def _calculate_calibration_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate calibration metrics"""
        
        confidences = np.array([p['confidence'] for p in predictions])
        predictions_binary = np.array([p['prediction'] > 0.5 for p in predictions])
        labels = np.array([p['true_label'] > 0.5 if p['true_label'] > 1 else p['true_label'] 
                          for p in predictions]).astype(int)
        
        # Calculate calibration metrics
        calibration_metrics = {}
        
        if "ece" in self.config.metrics:
            accuracies = (predictions_binary == labels).astype(float)
            calibration_metrics['ece'] = CalibrationMetrics.expected_calibration_error(confidences, accuracies)
            calibration_metrics['mce'] = CalibrationMetrics.maximum_calibration_error(confidences, accuracies)
        
        calibration_metrics['brier_score'] = CalibrationMetrics.brier_score(
            np.array([p['prediction'] for p in predictions]), labels
        )
        
        return calibration_metrics
    
    def _analyze_gate_confusion(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gate routing decisions"""
        
        gate_analysis = {
            'expert_usage': {},
            'regime_routing': {},
            'routing_accuracy': {}
        }
        
        # Count expert usage
        expert_counts = {}
        for pred in predictions:
            expert = pred['expert_used']
            expert_counts[expert] = expert_counts.get(expert, 0) + 1
        
        total_predictions = len(predictions)
        gate_analysis['expert_usage'] = {
            expert: count / total_predictions 
            for expert, count in expert_counts.items()
        }
        
        # Analyze routing by regime
        regime_routing = {}
        for pred in predictions:
            regime = pred['regime']
            expert = pred['expert_used']
            
            if regime not in regime_routing:
                regime_routing[regime] = {}
            if expert not in regime_routing[regime]:
                regime_routing[regime][expert] = 0
            regime_routing[regime][expert] += 1
        
        # Normalize by regime
        for regime in regime_routing:
            total = sum(regime_routing[regime].values())
            regime_routing[regime] = {
                expert: count / total 
                for expert, count in regime_routing[regime].items()
            }
        
        gate_analysis['regime_routing'] = regime_routing
        
        # Estimate routing "accuracy" (how often the chosen expert performed well)
        routing_performance = {}
        for pred in predictions:
            expert = pred['expert_used']
            # Simple performance metric: prediction close to label
            performance = 1.0 - abs(pred['prediction'] - pred['true_label'])
            
            if expert not in routing_performance:
                routing_performance[expert] = []
            routing_performance[expert].append(performance)
        
        gate_analysis['routing_accuracy'] = {
            expert: np.mean(performances) 
            for expert, performances in routing_performance.items()
        }
        
        return gate_analysis
    
    def _analyze_coverage(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze coverage at different confidence thresholds"""
        
        confidences = np.array([p['confidence'] for p in predictions])
        errors = np.array([abs(p['prediction'] - p['true_label']) for p in predictions])
        
        coverage_analysis = {}
        
        # Test different coverage levels
        for coverage_target in [0.8, 0.9, 0.95]:
            risk, actual_coverage = CalibrationMetrics.selective_risk(
                confidences, errors, coverage_target
            )
            
            coverage_analysis[f'risk_at_{int(coverage_target*100)}'] = risk
            coverage_analysis[f'actual_coverage_{int(coverage_target*100)}'] = actual_coverage
        
        return coverage_analysis
    
    def _analyze_expert_performance(self, predictions: List[Dict[str, Any]], 
                                  fusion_alpha: FusionAlpha) -> Dict[str, Dict[str, float]]:
        """Analyze individual expert performance"""
        
        expert_stats = fusion_alpha.get_expert_stats()
        
        # Add performance metrics from predictions
        expert_performance = {}
        expert_predictions = {}
        
        for pred in predictions:
            expert = pred['expert_used']
            if expert not in expert_predictions:
                expert_predictions[expert] = []
            expert_predictions[expert].append(pred)
        
        for expert, preds in expert_predictions.items():
            y_pred = np.array([p['prediction'] for p in preds])
            y_true = np.array([p['true_label'] for p in preds])
            
            performance = {
                'n_predictions': len(preds),
                'mse': np.mean((y_pred - y_true) ** 2),
                'mae': np.mean(np.abs(y_pred - y_true)),
                'abstain_rate': np.mean([p['abstained'] for p in preds])
            }
            
            # Add from expert stats
            if expert in expert_stats:
                performance.update(expert_stats[expert])
            
            expert_performance[expert] = performance
        
        return expert_performance
    
    def _save_results(self, result: EvalResult):
        """Save evaluation results"""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = output_dir / f"eval_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save summary report
        self._save_summary_report(result, output_dir / f"eval_summary_{timestamp}.txt")
        
        # Save detailed metrics
        self._save_detailed_metrics(result, output_dir / f"eval_detailed_{timestamp}.csv")
        
        logger.info(f"Results saved to {output_dir}")
    
    def _save_summary_report(self, result: EvalResult, filename: Path):
        """Save human-readable summary report"""
        
        with open(filename, 'w') as f:
            f.write("FusionAlpha Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Gate Type: {result.config.gate_type}\n")
            f.write(f"  Dataset: {result.config.dataset}\n")
            f.write(f"  Metrics: {', '.join(result.config.metrics)}\n")
            f.write(f"  Timestamp: {result.timestamp}\n\n")
            
            f.write("Overall Metrics:\n")
            for metric, value in result.overall_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Calibration Metrics:\n")
            for metric, value in result.calibration_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Expert Usage:\n")
            expert_usage = result.gate_confusion.get('expert_usage', {})
            for expert, usage in expert_usage.items():
                f.write(f"  {expert}: {usage:.1%}\n")
            f.write("\n")
            
            f.write("Per-Regime Performance:\n")
            for regime, metrics in result.per_regime_metrics.items():
                f.write(f"  {regime}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
            f.write("\n")
    
    def _save_detailed_metrics(self, result: EvalResult, filename: Path):
        """Save detailed metrics as CSV"""
        
        # Convert predictions to DataFrame
        df = pd.DataFrame(result.predictions)
        df.to_csv(filename, index=False)

def run_evaluation_from_args(args: argparse.Namespace) -> EvalResult:
    """Run evaluation from command line arguments"""
    
    config = EvalConfig(
        gate_type=args.gate,
        dataset=args.dataset,
        metrics=args.metrics.split(',') if args.metrics else None,
        output_dir=args.report,
        num_trials=args.trials,
        seed=args.seed,
        device=args.device
    )
    
    harness = EvalHarness(config)
    return harness.run_evaluation()

def main():
    """Command line interface"""
    
    parser = argparse.ArgumentParser(description="FusionAlpha Evaluation Harness")
    parser.add_argument("--gate", type=str, default="rule", 
                       choices=["rule", "mlp", "bandit", "rl"],
                       help="Gate type to evaluate")
    parser.add_argument("--dataset", type=str, default="synthetic",
                       help="Dataset to evaluate on")
    parser.add_argument("--metrics", type=str, default="auroc,ece,coverage",
                       help="Comma-separated list of metrics")
    parser.add_argument("--report", type=str, default="eval_results",
                       help="Output directory for results")
    parser.add_argument("--trials", type=int, default=1,
                       help="Number of evaluation trials")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    result = run_evaluation_from_args(args)
    
    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {args.report}")
    print(f"Overall AUROC: {result.overall_metrics.get('auroc', 0):.4f}")
    print(f"ECE: {result.calibration_metrics.get('ece', 0):.4f}")
    print(f"Abstain Rate: {result.overall_metrics.get('abstain_rate', 0):.1%}")

if __name__ == "__main__":
    main()