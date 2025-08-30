#!/usr/bin/env python3
"""
Manufacturing Quality Control Example

This example demonstrates how to use FusionAlpha for industrial applications,
detecting contradictions between design specifications and actual measurements,
sensor readings and expected values, and quality control inconsistencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to sys.path to import fusion_alpha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ManufacturingContradictionDetector:
    """
    Manufacturing-specific contradiction detection using FusionAlpha framework.
    
    Detects contradictions between:
    - Design specifications vs. actual measurements
    - Multi-sensor readings for the same parameters
    - Expected quality metrics vs. inspection results
    - Process parameters vs. output characteristics
    """
    
    def __init__(self, tolerance_threshold=0.05, confidence_threshold=0.7):
        self.tolerance_threshold = tolerance_threshold  # 5% tolerance
        self.confidence_threshold = confidence_threshold
        
    def detect_specification_contradiction(self, design_specs, actual_measurements,
                                         spec_names, measurement_names, batch_id=None):
        """
        Detect contradictions between design specifications and actual measurements.
        
        Args:
            design_specs: Target specification values (normalized 0-1)
            actual_measurements: Measured values (normalized 0-1)
            spec_names: Names of specifications
            measurement_names: Names of measurements
            batch_id: Production batch identifier
            
        Returns:
            ContradictionResult with quality assessment details
        """
        
        specs_array = np.array(design_specs)
        measurements_array = np.array(actual_measurements)
        
        # Calculate quality metrics
        spec_target = np.mean(specs_array)
        measurement_actual = np.mean(measurements_array)
        
        # Tolerance analysis
        absolute_deviation = np.abs(specs_array - measurements_array)
        max_deviation = np.max(absolute_deviation)
        avg_deviation = np.mean(absolute_deviation)
        
        # Quality assessment
        within_tolerance = max_deviation <= self.tolerance_threshold
        quality_score = max(0, 1 - avg_deviation / self.tolerance_threshold)
        
        # Contradiction scoring
        contradiction_score = min(1.0, avg_deviation / self.tolerance_threshold)
        is_contradictory = contradiction_score > self.confidence_threshold
        
        # Classify quality issue type
        if max_deviation > self.tolerance_threshold * 3:
            quality_issue = "CRITICAL_DEVIATION"
        elif avg_deviation > self.tolerance_threshold:
            quality_issue = "TOLERANCE_EXCEEDED"
        elif np.std(absolute_deviation) > self.tolerance_threshold:
            quality_issue = "INCONSISTENT_QUALITY"
        else:
            quality_issue = "WITHIN_SPECIFICATIONS"
            
        return {
            'is_contradictory': is_contradictory,
            'confidence': contradiction_score,
            'quality_issue': quality_issue,
            'spec_target': spec_target,
            'measurement_actual': measurement_actual,
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation,
            'quality_score': quality_score,
            'within_tolerance': within_tolerance,
            'batch_id': batch_id,
            'timestamp': datetime.now(),
            'details': {
                'specifications': dict(zip(spec_names, design_specs)),
                'measurements': dict(zip(measurement_names, actual_measurements))
            }
        }
    
    def detect_sensor_contradiction(self, sensor_readings, sensor_names, 
                                  expected_correlation=0.8, equipment_id=None):
        """
        Detect contradictions between multiple sensors measuring related parameters.
        
        Args:
            sensor_readings: List of sensor reading arrays
            sensor_names: Names of sensors
            expected_correlation: Expected correlation between related sensors
            equipment_id: Equipment identifier
            
        Returns:
            ContradictionResult with sensor validation details
        """
        
        # Calculate cross-sensor correlations
        sensor_arrays = [np.array(readings) for readings in sensor_readings]
        
        correlations = []
        sensor_pairs = []
        
        for i in range(len(sensor_arrays)):
            for j in range(i+1, len(sensor_arrays)):
                if len(sensor_arrays[i]) > 1 and len(sensor_arrays[j]) > 1:
                    corr = np.corrcoef(sensor_arrays[i], sensor_arrays[j])[0, 1]
                    correlations.append(abs(corr))
                    sensor_pairs.append((sensor_names[i], sensor_names[j]))
        
        # Sensor agreement analysis
        avg_correlation = np.mean(correlations) if correlations else 1.0
        min_correlation = np.min(correlations) if correlations else 1.0
        
        # Detect sensor contradictions
        correlation_gap = expected_correlation - min_correlation
        sensor_disagreement = max(0, correlation_gap)
        
        confidence = min(1.0, sensor_disagreement * 2)
        is_contradictory = confidence > self.confidence_threshold
        
        # Classify sensor issue
        if min_correlation < 0.3:
            sensor_issue = "SENSOR_MALFUNCTION"
        elif avg_correlation < expected_correlation - 0.2:
            sensor_issue = "CALIBRATION_DRIFT"
        elif np.std(correlations) > 0.3:
            sensor_issue = "INCONSISTENT_SENSORS"
        else:
            sensor_issue = "SENSORS_ALIGNED"
            
        return {
            'is_contradictory': is_contradictory,
            'confidence': confidence,
            'sensor_issue': sensor_issue,
            'avg_correlation': avg_correlation,
            'min_correlation': min_correlation,
            'correlation_gap': correlation_gap,
            'sensor_agreement': 1.0 - sensor_disagreement,
            'equipment_id': equipment_id,
            'timestamp': datetime.now(),
            'sensor_pairs': list(zip(sensor_pairs, correlations)) if correlations else [],
            'details': dict(zip(sensor_names, sensor_readings))
        }

def simulate_manufacturing_scenarios():
    """Generate realistic manufacturing quality control scenarios."""
    
    # Scenario 1: Automotive part - Tolerance exceeded
    automotive_part = {
        'batch_id': 'AUTO_BATCH_001',
        'scenario': 'Automotive Engine Component - Tolerance Issues',
        'product_type': 'Precision Engine Part',
        'specifications': {
            'dimension_length': [0.95, 0.96, 0.95, 0.94],    # Target dimensions
            'surface_finish': [0.90, 0.91, 0.90, 0.89],      # Surface quality target
            'hardness': [0.85, 0.86, 0.85, 0.84],            # Material hardness target
            'weight': [0.88, 0.89, 0.88, 0.87]               # Weight specifications
        },
        'measurements': {
            'actual_length': [0.92, 0.88, 0.93, 0.87],       # Below specification
            'actual_finish': [0.75, 0.73, 0.76, 0.74],       # Poor surface finish
            'actual_hardness': [0.80, 0.79, 0.81, 0.78],     # Slightly low hardness
            'actual_weight': [0.85, 0.84, 0.86, 0.83]        # Within acceptable range
        },
        'sensors': {
            'laser_measure': [0.92, 0.88, 0.93, 0.87],       # Laser measurement
            'contact_probe': [0.91, 0.87, 0.92, 0.86],       # Contact measurement
            'optical_scan': [0.93, 0.89, 0.94, 0.88]         # Optical measurement
        }
    }
    
    # Scenario 2: Electronics - Sensor malfunction
    electronics_board = {
        'batch_id': 'ELEC_BATCH_002',
        'scenario': 'Circuit Board Manufacturing - Sensor Malfunction',
        'product_type': 'PCB Assembly',
        'specifications': {
            'trace_width': [0.98, 0.97, 0.98, 0.97],         # High precision traces
            'solder_quality': [0.95, 0.94, 0.95, 0.94],      # Solder joint quality
            'component_placement': [0.96, 0.95, 0.96, 0.95], # Component accuracy
            'electrical_test': [0.92, 0.91, 0.92, 0.91]      # Electrical performance
        },
        'measurements': {
            'measured_width': [0.97, 0.96, 0.97, 0.96],      # Close to spec
            'solder_inspection': [0.93, 0.92, 0.93, 0.92],   # Good solder quality
            'placement_check': [0.94, 0.93, 0.94, 0.93],     # Good placement
            'electrical_result': [0.90, 0.89, 0.90, 0.89]    # Acceptable performance
        },
        'sensors': {
            'vision_system_1': [0.97, 0.96, 0.97, 0.96],     # Primary vision system
            'vision_system_2': [0.45, 0.44, 0.46, 0.43],     # Malfunctioning sensor
            'xray_inspection': [0.95, 0.94, 0.95, 0.94]      # X-ray system
        }
    }
    
    # Scenario 3: Pharmaceutical - Consistent quality
    pharmaceutical_tablet = {
        'batch_id': 'PHARMA_BATCH_003',
        'scenario': 'Pharmaceutical Tablet - Consistent Quality',
        'product_type': 'Medication Tablets',
        'specifications': {
            'tablet_weight': [0.90, 0.91, 0.90, 0.89],       # Weight consistency
            'active_ingredient': [0.95, 0.94, 0.95, 0.94],   # API concentration
            'dissolution_rate': [0.88, 0.87, 0.88, 0.87],    # Dissolution profile
            'tablet_hardness': [0.85, 0.84, 0.85, 0.84]      # Tablet integrity
        },
        'measurements': {
            'actual_weight': [0.89, 0.90, 0.89, 0.88],       # Good weight control
            'api_content': [0.94, 0.93, 0.94, 0.93],         # API within spec
            'dissolution_test': [0.87, 0.86, 0.87, 0.86],    # Good dissolution
            'hardness_test': [0.84, 0.83, 0.84, 0.83]        # Adequate hardness
        },
        'sensors': {
            'analytical_balance': [0.89, 0.90, 0.89, 0.88],  # Weight measurement
            'hplc_system': [0.94, 0.93, 0.94, 0.93],         # API analysis
            'dissolution_tester': [0.87, 0.86, 0.87, 0.86]   # Dissolution testing
        }
    }
    
    # Scenario 4: Aerospace - Critical deviation
    aerospace_component = {
        'batch_id': 'AERO_BATCH_004',
        'scenario': 'Aerospace Component - Critical Safety Issue',
        'product_type': 'Aircraft Engine Part',
        'specifications': {
            'critical_dimension': [0.99, 0.98, 0.99, 0.98],  # Ultra-high precision
            'material_strength': [0.95, 0.94, 0.95, 0.94],   # Strength requirements
            'surface_integrity': [0.97, 0.96, 0.97, 0.96],   # Surface condition
            'fatigue_resistance': [0.93, 0.92, 0.93, 0.92]   # Fatigue properties
        },
        'measurements': {
            'measured_dimension': [0.85, 0.83, 0.86, 0.84],  # Critical deviation
            'strength_test': [0.88, 0.87, 0.89, 0.86],       # Below requirements
            'surface_analysis': [0.92, 0.91, 0.93, 0.90],    # Acceptable surface
            'fatigue_test': [0.89, 0.88, 0.90, 0.87]         # Marginal fatigue
        },
        'sensors': {
            'cmm_machine': [0.85, 0.83, 0.86, 0.84],         # Coordinate measuring
            'tensile_tester': [0.88, 0.87, 0.89, 0.86],      # Strength testing
            'surface_profiler': [0.92, 0.91, 0.93, 0.90]     # Surface measurement
        }
    }
    
    return [automotive_part, electronics_board, pharmaceutical_tablet, aerospace_component]

def analyze_manufacturing_contradictions():
    """Analyze contradictions across multiple manufacturing scenarios."""
    
    detector = ManufacturingContradictionDetector(tolerance_threshold=0.05, confidence_threshold=0.6)
    scenarios = simulate_manufacturing_scenarios()
    
    results = []
    
    print("Manufacturing Quality Control Contradiction Analysis")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\nBatch: {scenario['batch_id']} - {scenario['scenario']}")
        print(f"Product: {scenario['product_type']}")
        print("-" * 50)
        
        # Specification vs. measurement analysis
        spec_values = []
        measurement_values = []
        spec_names = []
        measurement_names = []
        
        for spec_name, values in scenario['specifications'].items():
            spec_values.extend(values)
            spec_names.extend([f"{spec_name}_{i}" for i in range(len(values))])
        
        for meas_name, values in scenario['measurements'].items():
            measurement_values.extend(values)
            measurement_names.extend([f"{meas_name}_{i}" for i in range(len(values))])
        
        # Ensure equal length
        min_length = min(len(spec_values), len(measurement_values))
        spec_values = spec_values[:min_length]
        measurement_values = measurement_values[:min_length]
        spec_names = spec_names[:min_length]
        measurement_names = measurement_names[:min_length]
        
        spec_result = detector.detect_specification_contradiction(
            design_specs=spec_values,
            actual_measurements=measurement_values,
            spec_names=spec_names,
            measurement_names=measurement_names,
            batch_id=scenario['batch_id']
        )
        
        # Sensor correlation analysis
        sensor_readings = list(scenario['sensors'].values())
        sensor_names = list(scenario['sensors'].keys())
        
        sensor_result = detector.detect_sensor_contradiction(
            sensor_readings=sensor_readings,
            sensor_names=sensor_names,
            expected_correlation=0.8,
            equipment_id=f"EQUIP_{scenario['batch_id']}"
        )
        
        # Store combined results
        combined_result = {
            'batch_id': scenario['batch_id'],
            'scenario': scenario['scenario'],
            'product_type': scenario['product_type'],
            'specification_check': spec_result,
            'sensor_check': sensor_result
        }
        results.append(combined_result)
        
        # Display results
        print(f"🔧 Specification Analysis:")
        print(f"   Quality issue detected: {spec_result['is_contradictory']}")
        print(f"   Confidence: {spec_result['confidence']:.3f}")
        print(f"   Quality issue: {spec_result['quality_issue']}")
        print(f"   Quality score: {spec_result['quality_score']:.3f}")
        print(f"   Max deviation: {spec_result['max_deviation']:.3f}")
        print(f"   Within tolerance: {spec_result['within_tolerance']}")
        
        print(f"\n📡 Sensor Validation:")
        print(f"   Sensor issue detected: {sensor_result['is_contradictory']}")
        print(f"   Confidence: {sensor_result['confidence']:.3f}")
        print(f"   Sensor issue: {sensor_result['sensor_issue']}")
        print(f"   Sensor agreement: {sensor_result['sensor_agreement']:.3f}")
        print(f"   Min correlation: {sensor_result['min_correlation']:.3f}")
        
        # Quality control recommendations
        if spec_result['is_contradictory'] or sensor_result['is_contradictory']:
            print(f"\n⚠️  QUALITY CONTROL ALERT:")
            
            if spec_result['is_contradictory']:
                print(f"   📏 Specification Issue: {spec_result['quality_issue']}")
                if spec_result['quality_issue'] == 'CRITICAL_DEVIATION':
                    print("      🚨 CRITICAL: Stop production immediately")
                    print("      🔍 Required: Root cause analysis")
                    print("      📋 Action: Quarantine all recent production")
                elif spec_result['quality_issue'] == 'TOLERANCE_EXCEEDED':
                    print("      ⚠️  WARNING: Adjust process parameters")
                    print("      🔧 Action: Equipment calibration check")
                    print("      📊 Monitor: Increased sampling frequency")
                
            if sensor_result['is_contradictory']:
                print(f"   🔧 Sensor Issue: {sensor_result['sensor_issue']}")
                if sensor_result['sensor_issue'] == 'SENSOR_MALFUNCTION':
                    print("      🚨 URGENT: Replace malfunctioning sensor")
                    print("      🔍 Action: Validate with backup systems")
                elif sensor_result['sensor_issue'] == 'CALIBRATION_DRIFT':
                    print("      🔧 Action: Recalibrate measurement systems")
                    print("      📅 Schedule: Preventive maintenance")
        else:
            print("✅ Quality within acceptable parameters")
    
    return results

def quality_dashboard_analysis(results):
    """Generate quality dashboard metrics and insights."""
    
    print("\n" + "="*60)
    print("MANUFACTURING QUALITY DASHBOARD")
    print("="*60)
    
    # Overall quality metrics
    total_batches = len(results)
    spec_issues = sum(r['specification_check']['is_contradictory'] for r in results)
    sensor_issues = sum(r['sensor_check']['is_contradictory'] for r in results)
    
    avg_quality_score = np.mean([r['specification_check']['quality_score'] for r in results])
    avg_sensor_agreement = np.mean([r['sensor_check']['sensor_agreement'] for r in results])
    
    print(f"\n📊 Overall Quality Metrics:")
    print(f"   Batches analyzed: {total_batches}")
    print(f"   Specification issues: {spec_issues} ({spec_issues/total_batches:.1%})")
    print(f"   Sensor issues: {sensor_issues} ({sensor_issues/total_batches:.1%})")
    print(f"   Average quality score: {avg_quality_score:.3f}")
    print(f"   Average sensor agreement: {avg_sensor_agreement:.3f}")
    
    # Quality issue breakdown
    spec_issues_types = [r['specification_check']['quality_issue'] for r in results 
                        if r['specification_check']['is_contradictory']]
    
    if spec_issues_types:
        issue_counts = {issue: spec_issues_types.count(issue) for issue in set(spec_issues_types)}
        print(f"\n🔧 Quality Issue Types:")
        for issue_type, count in issue_counts.items():
            print(f"   {issue_type}: {count} batches")
    
    # Sensor issue breakdown
    sensor_issues_types = [r['sensor_check']['sensor_issue'] for r in results 
                          if r['sensor_check']['is_contradictory']]
    
    if sensor_issues_types:
        sensor_counts = {issue: sensor_issues_types.count(issue) for issue in set(sensor_issues_types)}
        print(f"\n📡 Sensor Issue Types:")
        for issue_type, count in sensor_counts.items():
            print(f"   {issue_type}: {count} instances")
    
    # Critical alerts
    critical_batches = [r for r in results 
                       if r['specification_check']['quality_issue'] == 'CRITICAL_DEVIATION']
    
    if critical_batches:
        print(f"\n🚨 CRITICAL ALERTS:")
        for batch in critical_batches:
            print(f"   Batch {batch['batch_id']}: {batch['specification_check']['quality_issue']}")
            print(f"      Max deviation: {batch['specification_check']['max_deviation']:.3f}")
    
    return {
        'total_batches': total_batches,
        'quality_rate': (total_batches - spec_issues) / total_batches,
        'sensor_reliability': (total_batches - sensor_issues) / total_batches,
        'avg_quality_score': avg_quality_score,
        'critical_issues': len(critical_batches)
    }

def visualize_manufacturing_analysis(results):
    """Create visualizations for manufacturing quality analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    batch_ids = [r['batch_id'] for r in results]
    quality_scores = [r['specification_check']['quality_score'] for r in results]
    sensor_agreements = [r['sensor_check']['sensor_agreement'] for r in results]
    max_deviations = [r['specification_check']['max_deviation'] for r in results]
    has_spec_issue = [r['specification_check']['is_contradictory'] for r in results]
    has_sensor_issue = [r['sensor_check']['is_contradictory'] for r in results]
    
    # Plot 1: Quality scores by batch
    colors = ['red' if issue else 'green' for issue in has_spec_issue]
    ax1.bar(range(len(batch_ids)), quality_scores, color=colors, alpha=0.7)
    ax1.set_ylabel('Quality Score')
    ax1.set_title('Quality Score by Batch')
    ax1.set_xticks(range(len(batch_ids)))
    ax1.set_xticklabels([f'B{i+1}' for i in range(len(batch_ids))])
    ax1.axhline(y=0.8, color='orange', linestyle='--', label='Quality Threshold')
    ax1.legend()
    
    # Plot 2: Quality vs. Sensor Agreement
    for i, (qual, sensor) in enumerate(zip(quality_scores, sensor_agreements)):
        if has_spec_issue[i] and has_sensor_issue[i]:
            color = 'red'
            label = 'Both Issues'
        elif has_spec_issue[i]:
            color = 'orange'
            label = 'Spec Issue'
        elif has_sensor_issue[i]:
            color = 'yellow'
            label = 'Sensor Issue'
        else:
            color = 'green'
            label = 'Good Quality'
        ax2.scatter(qual, sensor, c=color, s=100, alpha=0.7)
    
    ax2.set_xlabel('Quality Score')
    ax2.set_ylabel('Sensor Agreement')
    ax2.set_title('Quality vs. Sensor Reliability')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Plot 3: Deviation distribution
    ax3.hist(max_deviations, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(x=0.05, color='red', linestyle='--', label='Tolerance Limit')
    ax3.set_xlabel('Maximum Deviation')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Maximum Deviations')
    ax3.legend()
    
    # Plot 4: Quality trend over time (simulated)
    times = pd.date_range(start='2024-01-01', periods=len(results), freq='D')
    ax4.plot(times, quality_scores, 'o-', color='blue', linewidth=2, markersize=6, label='Quality Score')
    ax4.plot(times, sensor_agreements, 's-', color='green', linewidth=2, markersize=6, label='Sensor Agreement')
    ax4.set_ylabel('Score')
    ax4.set_title('Quality Metrics Over Time')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    # Highlight problem periods
    for i, time in enumerate(times):
        if has_spec_issue[i] or has_sensor_issue[i]:
            ax4.axvspan(time - timedelta(hours=12), time + timedelta(hours=12), 
                       alpha=0.3, color='red')
    
    plt.tight_layout()
    plt.savefig('manufacturing_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def predictive_maintenance_demo():
    """Demonstrate predictive maintenance using contradiction detection."""
    
    print("\n" + "="*60)
    print("PREDICTIVE MAINTENANCE DEMO")
    print("="*60)
    
    detector = ManufacturingContradictionDetector(tolerance_threshold=0.03, confidence_threshold=0.5)
    
    print("\n🏭 Monitoring equipment health...")
    print("Equipment: CNC Machining Center")
    print("Product: Precision Aerospace Components")
    
    # Simulate equipment degradation over time
    time_points = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    
    for i, time_point in enumerate(time_points):
        print(f"\n📅 {time_point}:")
        
        # Equipment specifications (constant)
        specs = [0.95, 0.94, 0.96, 0.95]  # Target performance
        
        # Actual performance (degrading over time)
        performance = [0.94 - i*0.02, 0.93 - i*0.025, 0.95 - i*0.02, 0.94 - i*0.03]
        
        # Sensor readings (becoming inconsistent)
        sensor1 = [0.94 - i*0.02, 0.93 - i*0.025, 0.95 - i*0.02, 0.94 - i*0.03]
        sensor2 = [0.93 - i*0.01, 0.92 - i*0.03, 0.94 - i*0.025, 0.93 - i*0.035]
        sensor3 = [0.95 - i*0.03, 0.94 - i*0.02, 0.96 - i*0.015, 0.95 - i*0.025]
        
        print(f"Target performance: {np.mean(specs):.3f}")
        print(f"Actual performance: {np.mean(performance):.3f}")
        print(f"Performance degradation: {(np.mean(specs) - np.mean(performance)):.3f}")
        
        # Specification analysis
        spec_result = detector.detect_specification_contradiction(
            design_specs=specs,
            actual_measurements=performance,
            spec_names=['spec_1', 'spec_2', 'spec_3', 'spec_4'],
            measurement_names=['perf_1', 'perf_2', 'perf_3', 'perf_4'],
            batch_id=f'EQUIP_MONITOR_{time_point}'
        )
        
        # Sensor analysis
        sensor_result = detector.detect_sensor_contradiction(
            sensor_readings=[sensor1, sensor2, sensor3],
            sensor_names=['Primary_Sensor', 'Secondary_Sensor', 'Backup_Sensor'],
            expected_correlation=0.9,
            equipment_id='CNC_001'
        )
        
        print(f"🔧 Equipment health: {1 - spec_result['confidence']:.3f}")
        print(f"📡 Sensor health: {sensor_result['sensor_agreement']:.3f}")
        
        # Maintenance recommendations
        if spec_result['confidence'] > 0.3:
            print(f"⚠️  MAINTENANCE ALERT:")
            print(f"   Performance degradation detected")
            if spec_result['confidence'] > 0.7:
                print(f"   🚨 URGENT: Schedule immediate maintenance")
                print(f"   📋 Actions: Tool replacement, calibration check")
            else:
                print(f"   📅 Schedule: Preventive maintenance within 1 week")
        
        if sensor_result['confidence'] > 0.3:
            print(f"   📡 Sensor calibration needed")
            print(f"   🔧 Action: Verify sensor alignment and calibration")
        
        if spec_result['confidence'] <= 0.3 and sensor_result['confidence'] <= 0.3:
            print("✅ Equipment operating within normal parameters")

if __name__ == "__main__":
    print("FusionAlpha Manufacturing Quality Control")
    print("=" * 50)
    
    # Run main quality analysis
    results = analyze_manufacturing_contradictions()
    
    # Quality dashboard analysis
    dashboard_metrics = quality_dashboard_analysis(results)
    
    # Create visualizations
    visualize_manufacturing_analysis(results)
    
    # Demonstrate predictive maintenance
    predictive_maintenance_demo()
    
    print(f"\n✅ Manufacturing analysis complete!")
    print(f"   Analyzed {len(results)} production batches")
    print(f"   Quality rate: {dashboard_metrics['quality_rate']:.1%}")
    print(f"   Sensor reliability: {dashboard_metrics['sensor_reliability']:.1%}")
    print(f"   Critical issues: {dashboard_metrics['critical_issues']}")
    
    print(f"\n🏭 Manufacturing Applications:")
    print(f"   • Real-time quality control")
    print(f"   • Predictive maintenance")
    print(f"   • Process optimization")
    print(f"   • Sensor validation and calibration")
    print(f"   • Supply chain quality assurance")