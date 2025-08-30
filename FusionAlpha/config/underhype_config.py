#!/usr/bin/env python3
"""
Universal Contradiction Detection Pipeline Configuration

Configuration settings for the universal contradiction detection system.
Supports multiple domains including finance, healthcare, cybersecurity, media, and manufacturing.
"""

import os

# Domain-specific entity identifiers for priority processing
DOMAIN_ENTITIES = {
    'finance': ['AAPL', 'GOOGL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'META', 'NFLX', 'ORCL', 'CRM'],
    'healthcare': ['PATIENT_001', 'PATIENT_002', 'ICU_MONITOR', 'LAB_SYSTEM'],
    'cybersecurity': ['USER_001', 'SERVER_CLUSTER', 'NETWORK_ZONE_A', 'ENDPOINT_FLEET'],
    'manufacturing': ['FACTORY_LINE_1', 'QC_STATION_A', 'SENSOR_ARRAY_1', 'BATCH_PROCESSOR'],
    'media': ['NEWS_SOURCE_1', 'SOCIAL_PLATFORM', 'FACT_CHECKER', 'CONTENT_MONITOR']
}

# Backward compatibility
PRIORITY_TICKERS = DOMAIN_ENTITIES['finance']

def get_production_config(domain: str = 'finance'):
    """Get production configuration for specified domain"""
    return {
        'pipeline': {
            'domain': domain,
            'device': 'cuda' if os.environ.get('USE_CUDA', 'true').lower() == 'true' else 'cpu',
            'enable_bicep': True,
            'enable_enn': True,
            'enable_graph': True,
            'confidence_threshold': float(os.environ.get('CONFIDENCE_THRESHOLD', 2.5)),
            'bicep': {
                'n_paths': int(os.environ.get('BICEP_PATHS', 100)),
                'n_steps': int(os.environ.get('BICEP_STEPS', 50)),
                'scenarios_per_entity': int(os.environ.get('SCENARIOS_PER_ENTITY', 20))
            },
            'enn': {
                'num_neurons': int(os.environ.get('ENN_NEURONS', 128)),
                'num_states': int(os.environ.get('ENN_STATES', 8)),
                'entanglement_dim': 16,
                'memory_length': 10,
                'dropout_rate': 0.1
            },
            'risk_management': {
                'max_exposure': float(os.environ.get('MAX_EXPOSURE', 3.0)),
                'base_allocation': float(os.environ.get('BASE_ALLOCATION', 0.02)),
                'dynamic_adjustment': True
            }
        },
        'monitoring': {
            'update_interval': 1.0,
            'gpu_monitoring': True,
            'websocket_port': int(os.environ.get('WS_PORT', 8765))
        },
        'data': {
            'entities': DOMAIN_ENTITIES.get(domain, DOMAIN_ENTITIES['finance']),
            'update_interval': int(os.environ.get('UPDATE_INTERVAL', 60)),
            'lookback_period': int(os.environ.get('LOOKBACK_PERIOD', 90))
        },
        'domain_specific': get_domain_specific_config(domain)
    }

def get_domain_specific_config(domain: str):
    """Get domain-specific configuration settings"""
    configs = {
        'finance': {
            'signal_a_type': 'sentiment',
            'signal_b_type': 'price_movement',
            'contradiction_type': 'underhype',
            'success_metric': 'return_percentage'
        },
        'healthcare': {
            'signal_a_type': 'symptom_severity',
            'signal_b_type': 'biomarker_levels',
            'contradiction_type': 'symptom_biomarker_mismatch',
            'success_metric': 'diagnostic_accuracy'
        },
        'cybersecurity': {
            'signal_a_type': 'stated_behavior',
            'signal_b_type': 'actual_behavior',
            'contradiction_type': 'behavioral_anomaly',
            'success_metric': 'threat_detection_rate'
        },
        'manufacturing': {
            'signal_a_type': 'design_specifications',
            'signal_b_type': 'actual_measurements',
            'contradiction_type': 'specification_deviation',
            'success_metric': 'quality_score'
        },
        'media': {
            'signal_a_type': 'claim_strength',
            'signal_b_type': 'evidence_support',
            'contradiction_type': 'claim_evidence_mismatch',
            'success_metric': 'fact_check_accuracy'
        }
    }
    return configs.get(domain, configs['finance'])

def get_development_config(domain: str = 'finance'):
    """Get development configuration for specified domain"""
    config = get_production_config(domain)
    # Override for development
    config['pipeline']['bicep']['n_paths'] = 50
    config['data']['entities'] = config['data']['entities'][:5]  # Limit for dev
    return config

# Backward compatibility functions
def get_underhype_config():
    """Backward compatibility for underhype-specific configuration"""
    return get_production_config('finance')