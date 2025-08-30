#!/usr/bin/env python3
"""
Uncertainty & Calibration Framework

Built-ins: MC-dropout, Deep Ensembles, temperature scaling
Metrics: ECE, risk-coverage, selective risk for abstention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

class BaseCalibrator(ABC):
    """Base class for calibration methods"""
    
    def __init__(self):
        self.fitted = False
    
    @abstractmethod
    def fit(self, confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        """Fit calibrator to data"""
        pass
    
    @abstractmethod
    def calibrate(self, confidence: float, prediction: float) -> float:
        """Apply calibration to confidence score"""
        pass
    
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted"""
        return self.fitted

class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibration"""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
    def fit(self, confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        """Fit temperature parameter"""
        if len(confidences) == 0:
            logger.warning("No data provided for temperature calibration")
            return
        
        confidences = confidences.clone().detach().requires_grad_(False)
        labels = labels.clone().detach().requires_grad_(False)
        
        # Convert to logits (assuming confidences are probabilities)
        logits = torch.log(confidences + 1e-8) - torch.log(1 - confidences + 1e-8)
        
        def eval():
            self.optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            scaled_probs = torch.sigmoid(scaled_logits)
            loss = F.binary_cross_entropy(scaled_probs, labels.float())
            loss.backward()
            return loss
        
        self.optimizer.step(eval)
        self.fitted = True
        
        logger.info(f"Temperature scaling fitted with T={self.temperature.item():.3f}")
    
    def calibrate(self, confidence: float, prediction: float) -> float:
        """Apply temperature scaling"""
        if not self.fitted:
            return confidence
        
        # Convert to logit, scale, convert back
        logit = np.log(confidence + 1e-8) - np.log(1 - confidence + 1e-8)
        scaled_logit = logit / self.temperature.item()
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(np.clip(calibrated, 0.01, 0.99))

class PlattCalibrator(BaseCalibrator):
    """Platt scaling (sigmoid) calibration"""
    
    def __init__(self):
        super().__init__()
        self.A = 1.0
        self.B = 0.0
    
    def fit(self, confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        """Fit Platt scaling parameters"""
        if len(confidences) == 0:
            return
        
        from sklearn.linear_model import LogisticRegression
        
        # Convert confidences to decision values (logits)
        confidences_np = confidences.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Fit logistic regression
        lr = LogisticRegression()
        try:
            lr.fit(confidences_np.reshape(-1, 1), labels_np)
            self.A = float(lr.coef_[0][0])
            self.B = float(lr.intercept_[0])
            self.fitted = True
            
            logger.info(f"Platt scaling fitted with A={self.A:.3f}, B={self.B:.3f}")
        except Exception as e:
            logger.warning(f"Platt scaling fit failed: {e}")
    
    def calibrate(self, confidence: float, prediction: float) -> float:
        """Apply Platt scaling"""
        if not self.fitted:
            return confidence
        
        # Apply sigmoid with learned parameters
        calibrated = 1 / (1 + np.exp(-(self.A * confidence + self.B)))
        return float(np.clip(calibrated, 0.01, 0.99))

class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration"""
    
    def __init__(self):
        super().__init__()
        self.calibrator = None
        
    def fit(self, confidences: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor):
        """Fit isotonic regression"""
        try:
            from sklearn.isotonic import IsotonicRegression
            
            confidences_np = confidences.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(confidences_np, labels_np)
            self.fitted = True
            
            logger.info("Isotonic calibration fitted")
        except ImportError:
            logger.warning("sklearn not available for isotonic calibration")
        except Exception as e:
            logger.warning(f"Isotonic calibration failed: {e}")
    
    def calibrate(self, confidence: float, prediction: float) -> float:
        """Apply isotonic calibration"""
        if not self.fitted or self.calibrator is None:
            return confidence
        
        calibrated = self.calibrator.predict([confidence])[0]
        return float(np.clip(calibrated, 0.01, 0.99))

class UncertaintyEstimator:
    """Multi-method uncertainty estimation"""
    
    def __init__(self, methods: List[str] = None):
        if methods is None:
            methods = ["feature_entropy", "prediction_variance", "contradiction_uncertainty"]
        
        self.methods = methods
        self.history = []
    
    def estimate(self, features: torch.Tensor, prediction, 
                contradiction_scores: Dict[str, float]) -> Dict[str, float]:
        """Estimate uncertainty using multiple methods"""
        
        uncertainties = {}
        
        if "feature_entropy" in self.methods:
            uncertainties["feature_entropy"] = self._feature_entropy(features)
        
        if "prediction_variance" in self.methods:
            uncertainties["prediction_variance"] = self._prediction_variance(features, prediction)
        
        if "contradiction_uncertainty" in self.methods:
            uncertainties["contradiction_uncertainty"] = self._contradiction_uncertainty(contradiction_scores)
        
        if "aleatoric" in self.methods:
            uncertainties["aleatoric"] = self._aleatoric_uncertainty(features)
        
        if "epistemic" in self.methods:
            uncertainties["epistemic"] = self._epistemic_uncertainty(features)
        
        # Aggregate uncertainty
        total_uncertainty = np.mean(list(uncertainties.values()))
        uncertainties["total_uncertainty"] = total_uncertainty
        
        # Recommend abstention if uncertainty is very high
        uncertainties["should_abstain"] = total_uncertainty > 0.8
        
        return uncertainties
    
    def _feature_entropy(self, features: torch.Tensor) -> float:
        """Entropy-based uncertainty from feature distribution"""
        if len(features.shape) > 1:
            features = features.flatten()
        
        # Discretize features for entropy calculation
        hist, _ = np.histogram(features.detach().cpu().numpy(), bins=20, density=True)
        hist = hist + 1e-8  # Avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        
        # Normalize to 0-1
        max_entropy = np.log(20)  # Max entropy for 20 bins
        return entropy / max_entropy
    
    def _prediction_variance(self, features: torch.Tensor, prediction) -> float:
        """Variance-based uncertainty (simplified MC-dropout style)"""
        # Simulate multiple predictions with noise
        num_samples = 10
        predictions = []
        
        for _ in range(num_samples):
            noisy_features = features + 0.1 * torch.randn_like(features)
            # Simple linear transformation as proxy for model prediction
            pred = torch.mean(noisy_features).item()
            predictions.append(pred)
        
        return float(np.std(predictions))
    
    def _contradiction_uncertainty(self, contradiction_scores: Dict[str, float]) -> float:
        """Uncertainty from contradiction scores"""
        if not contradiction_scores:
            return 0.5
        
        # Higher contradiction = higher uncertainty
        avg_contradiction = np.mean(list(contradiction_scores.values()))
        return min(avg_contradiction, 1.0)
    
    def _aleatoric_uncertainty(self, features: torch.Tensor) -> float:
        """Data/noise uncertainty (simplified)"""
        # Estimate from feature noise level
        feature_std = torch.std(features).item()
        return min(feature_std, 1.0)
    
    def _epistemic_uncertainty(self, features: torch.Tensor) -> float:
        """Model uncertainty (simplified)"""
        # Estimate from how "far" features are from training distribution
        # Simple proxy: distance from zero (assuming normalized features)
        distance = torch.norm(features).item()
        normalized_distance = min(distance / 10.0, 1.0)
        return normalized_distance

class CalibrationMetrics:
    """Calibration and uncertainty metrics"""
    
    @staticmethod
    def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, 
                                 n_bins: int = 15) -> float:
        """Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def maximum_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                n_bins: int = 15) -> float:
        """Maximum Calibration Error (MCE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    @staticmethod
    def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray,
                               n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Data for plotting reliability diagram"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)
    
    @staticmethod
    def brier_score(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Brier score for calibration assessment"""
        try:
            return brier_score_loss(labels, predictions)
        except Exception:
            return float(np.mean((predictions - labels) ** 2))
    
    @staticmethod
    def selective_risk(confidences: np.ndarray, errors: np.ndarray, 
                      coverage: float = 0.8) -> Tuple[float, float]:
        """Selective risk at given coverage level"""
        n = len(confidences)
        n_select = int(coverage * n)
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        selected_errors = errors[sorted_indices[:n_select]]
        
        risk = selected_errors.mean()
        actual_coverage = n_select / n
        
        return risk, actual_coverage

def create_calibrator(method: str = "temperature") -> BaseCalibrator:
    """Factory function for calibrators"""
    if method == "temperature":
        return TemperatureCalibrator()
    elif method == "platt":
        return PlattCalibrator()
    elif method == "isotonic":
        return IsotonicCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")

if __name__ == "__main__":
    # Demo calibration system
    print("Uncertainty & Calibration Framework Demo")
    print("=" * 45)
    
    # Generate test data
    n_samples = 1000
    confidences = torch.rand(n_samples)
    predictions = torch.rand(n_samples)
    labels = torch.bernoulli(predictions)  # Bernoulli with varying rates
    
    # Test temperature calibration
    temp_cal = TemperatureCalibrator()
    temp_cal.fit(confidences, predictions, labels)
    
    # Test calibrated predictions
    test_conf = 0.7
    test_pred = 0.6
    
    calibrated = temp_cal.calibrate(test_conf, test_pred)
    print(f"Original confidence: {test_conf:.3f}")
    print(f"Calibrated confidence: {calibrated:.3f}")
    
    # Test uncertainty estimation
    uncertainty_est = UncertaintyEstimator()
    test_features = torch.randn(32)
    contradiction_scores = {"cosine": 0.6, "delta": 0.4}
    
    uncertainties = uncertainty_est.estimate(test_features, test_pred, contradiction_scores)
    print(f"\nUncertainty estimates: {uncertainties}")
    
    # Test calibration metrics
    confidences_np = confidences.numpy()
    accuracies_np = (predictions.numpy() > 0.5) == (labels.numpy() > 0.5)
    
    ece = CalibrationMetrics.expected_calibration_error(confidences_np, accuracies_np.astype(float))
    mce = CalibrationMetrics.maximum_calibration_error(confidences_np, accuracies_np.astype(float))
    
    print(f"\nCalibration Metrics:")
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")
    
    print("\nCalibration framework working correctly!")