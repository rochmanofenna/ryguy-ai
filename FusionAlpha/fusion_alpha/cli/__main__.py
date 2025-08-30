#!/usr/bin/env python3
"""
FusionAlpha CLI

Usage:
    fa eval --gate mlp --dataset healthcare_demo --metrics auroc,ece,coverage --report out/
    fa demo sre
    fa demo healthcare  
    fa demo robotics
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fusion_alpha.eval.harness import run_evaluation_from_args
from fusion_alpha.core.router import create_healthcare_fusion, create_sre_fusion, create_robotics_fusion
from fusion_alpha.core.modality_registry import (
    create_text_modality, create_series_modality, create_structured_modality
)

def run_demo(domain: str):
    """Run a quick demo for specified domain"""
    
    print(f"FusionAlpha {domain.title()} Demo")
    print("=" * 40)
    
    # Create domain-specific fusion system
    if domain == "healthcare":
        # Register modalities
        create_text_modality("patient_notes")
        create_series_modality("vital_signs")
        create_structured_modality("lab_results", ["glucose", "bp", "heart_rate"])
        
        fa = create_healthcare_fusion()
        print("Created healthcare fusion system")
        print("Modalities: patient notes, vital signs, lab results")
        
    elif domain == "sre":
        # Register modalities  
        create_series_modality("metrics_ts")
        create_text_modality("logs_text")
        create_series_modality("user_reports")
        
        fa = create_sre_fusion()
        print("Created SRE/ops fusion system")
        print("Modalities: metrics time series, log text, user reports")
        
    elif domain == "robotics":
        # Register modalities
        create_structured_modality("vision_det", ["obj_conf", "bbox", "class_prob"])
        create_series_modality("lidar_vec")
        create_series_modality("proprio_state")
        
        fa = create_robotics_fusion()
        print("Created robotics fusion system") 
        print("Modalities: vision detection, lidar vectors, proprioceptive state")
        
    else:
        print(f"Unknown domain: {domain}")
        return
    
    # Generate test data with appropriate dimensions for the domain
    import torch
    
    if domain == "healthcare":
        # patient_notes (1024) + lab_results (3) + vital_signs (8) = 1035
        test_features = torch.randn(1035)
    elif domain == "sre":
        # metrics_ts (8) + logs_text (1024) + user_reports (8) = 1040  
        test_features = torch.randn(1040)
    elif domain == "robotics":
        # vision_det (3) + lidar_vec (8) + proprio_state (8) = 19
        test_features = torch.randn(19)
    else:
        test_features = torch.randn(32)  # Default dimension
    
    # Make prediction
    result = fa.predict(test_features)
    
    print(f"\nTest Prediction:")
    print(f"  Prediction: {result.prediction:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Expert used: {result.expert_used}")
    print(f"  Gate type: {result.metadata['gate_type']}")
    print(f"  Contradiction scores: {result.contradiction_scores}")
    print(f"  Abstained: {result.abstained}")
    
    # Show performance metrics
    metrics = fa.get_performance_metrics()
    print(f"\nSystem Metrics:")
    print(f"  Total predictions: {metrics['total_predictions']}")
    print(f"  Abstain rate: {metrics['abstain_rate']:.1%}")
    print(f"  Expert usage: {metrics['expert_usage_pct']}")
    
    print(f"\n✅ {domain.title()} demo completed successfully!")

def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(description="FusionAlpha CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Eval subcommand
    eval_parser = subparsers.add_parser('eval', help='Run evaluation')
    eval_parser.add_argument("--gate", type=str, default="rule",
                           choices=["rule", "mlp", "bandit", "rl"],
                           help="Gate type to evaluate")
    eval_parser.add_argument("--dataset", type=str, default="synthetic",
                           help="Dataset to evaluate on")
    eval_parser.add_argument("--metrics", type=str, default="auroc,ece,coverage",
                           help="Comma-separated metrics")
    eval_parser.add_argument("--report", type=str, default="eval_results",
                           help="Output directory")
    eval_parser.add_argument("--trials", type=int, default=1,
                           help="Number of trials")
    eval_parser.add_argument("--seed", type=int, default=42,
                           help="Random seed")
    eval_parser.add_argument("--device", type=str, default="cpu",
                           help="Device (cpu/cuda)")
    
    # Demo subcommand
    demo_parser = subparsers.add_parser('demo', help='Run domain demo')
    demo_parser.add_argument('domain', choices=['healthcare', 'sre', 'robotics'],
                           help='Domain to demo')
    
    # Version subcommand
    version_parser = subparsers.add_parser('version', help='Show version')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.command == 'eval':
        print("Running FusionAlpha evaluation...")
        result = run_evaluation_from_args(args)
        
        print(f"\n🎯 Evaluation Results:")
        print(f"  Overall AUROC: {result.overall_metrics.get('auroc', 0):.4f}")
        print(f"  Calibration ECE: {result.calibration_metrics.get('ece', 0):.4f}")
        print(f"  Abstain Rate: {result.overall_metrics.get('abstain_rate', 0):.1%}")
        print(f"  Results saved to: {args.report}")
        
    elif args.command == 'demo':
        run_demo(args.domain)
        
    elif args.command == 'version':
        print("FusionAlpha v1.0.0 - Universal Contradiction Detection Framework")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()