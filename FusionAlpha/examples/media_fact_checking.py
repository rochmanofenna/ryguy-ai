#!/usr/bin/env python3
"""
Media & Information Fact-Checking Example

This example demonstrates how to use FusionAlpha for media analysis applications,
detecting contradictions between claims and facts, sentiment and engagement patterns,
and identifying potential misinformation or manipulation campaigns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import sys
import os

# Add the parent directory to sys.path to import fusion_alpha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MediaContradictionDetector:
    """
    Media-specific contradiction detection using FusionAlpha framework.
    
    Detects contradictions between:
    - Factual claims vs. verified evidence
    - Sentiment signals vs. engagement patterns
    - Source credibility vs. viral spreading
    - Narrative consistency vs. contradictory details
    """
    
    def __init__(self, confidence_threshold=0.65, viral_threshold=1000):
        self.confidence_threshold = confidence_threshold
        self.viral_threshold = viral_threshold
        
    def detect_fact_contradiction(self, claims_confidence, evidence_support,
                                claim_details, evidence_details, source_id=None):
        """
        Detect contradictions between claims and supporting evidence.
        
        Args:
            claims_confidence: Confidence/strength of claims being made (0-1)
            evidence_support: Strength of supporting evidence (0-1)
            claim_details: Descriptive details about claims
            evidence_details: Descriptive details about evidence
            source_id: Source identifier
            
        Returns:
            ContradictionResult with fact-check details
        """
        
        claims_array = np.array(claims_confidence)
        evidence_array = np.array(evidence_support)
        
        # Calculate contradiction metrics
        claim_strength = np.mean(claims_array)
        evidence_strength = np.mean(evidence_array)
        
        # Fact-checking analysis
        contradiction_gap = abs(claim_strength - evidence_strength)
        correlation = np.corrcoef(claims_array, evidence_array)[0, 1] if len(claims_array) > 1 else 0
        
        # Misinformation scoring
        if claim_strength > evidence_strength + 0.3:
            misinformation_type = "OVERSTATED_CLAIMS"
        elif evidence_strength > claim_strength + 0.3:
            misinformation_type = "UNDERSTATED_EVIDENCE" 
        elif correlation < -0.3:
            misinformation_type = "CONTRADICTORY_NARRATIVE"
        else:
            misinformation_type = "INSUFFICIENT_EVIDENCE"
            
        confidence = min(1.0, contradiction_gap * 2)
        is_contradictory = confidence > self.confidence_threshold
        
        return {
            'is_contradictory': is_contradictory,
            'confidence': confidence,
            'misinformation_type': misinformation_type,
            'claim_strength': claim_strength,
            'evidence_strength': evidence_strength,
            'contradiction_gap': contradiction_gap,
            'correlation': correlation,
            'source_id': source_id,
            'timestamp': datetime.now(),
            'credibility_score': 1.0 - confidence,  # Inverse of contradiction
            'details': {
                'claims': claim_details,
                'evidence': evidence_details
            }
        }
    
    def detect_engagement_manipulation(self, sentiment_scores, engagement_metrics,
                                     content_quality, source_credibility, post_id=None):
        """
        Detect artificial manipulation through sentiment vs. engagement contradictions.
        
        Args:
            sentiment_scores: Natural sentiment analysis scores (0-1)
            engagement_metrics: Engagement rates (likes, shares, comments) (0-1)
            content_quality: Assessed content quality scores (0-1)
            source_credibility: Source credibility ratings (0-1)
            post_id: Post identifier
            
        Returns:
            ContradictionResult with manipulation details
        """
        
        sentiment_array = np.array(sentiment_scores)
        engagement_array = np.array(engagement_metrics)
        quality_array = np.array(content_quality)
        credibility_array = np.array(source_credibility)
        
        # Calculate manipulation indicators
        sentiment_avg = np.mean(sentiment_array)
        engagement_avg = np.mean(engagement_array)
        quality_avg = np.mean(quality_array)
        credibility_avg = np.mean(credibility_array)
        
        # Detect manipulation patterns
        # High engagement + low quality/credibility = potential manipulation
        manipulation_score = max(0, engagement_avg - (quality_avg + credibility_avg) / 2)
        
        # Sentiment-engagement mismatch
        sentiment_engagement_gap = abs(sentiment_avg - engagement_avg)
        
        # Overall manipulation confidence
        confidence = min(1.0, manipulation_score + sentiment_engagement_gap)
        is_manipulated = confidence > self.confidence_threshold
        
        # Classify manipulation type
        if engagement_avg > 0.7 and quality_avg < 0.3:
            manipulation_type = "BOT_AMPLIFICATION"
        elif sentiment_avg < 0.3 and engagement_avg > 0.7:
            manipulation_type = "ASTROTURFING"
        elif credibility_avg < 0.3 and engagement_avg > 0.6:
            manipulation_type = "COORDINATED_INAUTHENTIC"
        else:
            manipulation_type = "ORGANIC_VIRAL"
            
        return {
            'is_manipulated': is_manipulated,
            'confidence': confidence,
            'manipulation_type': manipulation_type,
            'sentiment_avg': sentiment_avg,
            'engagement_avg': engagement_avg,
            'quality_avg': quality_avg,
            'credibility_avg': credibility_avg,
            'manipulation_score': manipulation_score,
            'post_id': post_id,
            'timestamp': datetime.now(),
            'authenticity_score': 1.0 - confidence,
            'viral_potential': engagement_avg > (self.viral_threshold / 10000)  # Normalized
        }

def simulate_media_scenarios():
    """Generate realistic media analysis scenarios."""
    
    # Scenario 1: Health misinformation - Strong claims, weak evidence
    health_misinfo = {
        'content_id': 'POST_001',
        'scenario': 'Health Misinformation - Miracle Cure Claims',
        'topic': 'Alternative Medicine',
        'claims': {
            'effectiveness': [0.9, 0.95, 0.9, 0.85],      # Very strong claims
            'safety': [0.9, 0.9, 0.95, 0.9],              # Claims completely safe
            'scientific_backing': [0.8, 0.85, 0.8, 0.9]   # Claims scientific support
        },
        'evidence': {
            'peer_review': [0.1, 0.0, 0.1, 0.0],          # No peer review
            'clinical_trials': [0.0, 0.1, 0.0, 0.0],      # No clinical trials
            'expert_consensus': [0.2, 0.1, 0.2, 0.1]      # Expert disagreement
        },
        'engagement': {
            'sentiment': [0.8, 0.7, 0.8, 0.9],            # Positive sentiment
            'viral_sharing': [0.9, 0.8, 0.9, 0.7],        # High viral spread
            'quality_score': [0.2, 0.3, 0.2, 0.3],        # Low quality content
            'source_credibility': [0.3, 0.2, 0.3, 0.2]    # Low credibility source
        }
    }
    
    # Scenario 2: Political astroturfing - Artificial engagement patterns
    political_astroturf = {
        'content_id': 'POST_002',
        'scenario': 'Political Astroturfing - Coordinated Campaign',
        'topic': 'Election Information',
        'claims': {
            'poll_accuracy': [0.6, 0.7, 0.6, 0.65],       # Moderate claims
            'candidate_support': [0.8, 0.7, 0.8, 0.75],   # Strong support claims
            'policy_impact': [0.7, 0.6, 0.7, 0.8]         # Policy benefit claims
        },
        'evidence': {
            'polling_data': [0.4, 0.5, 0.4, 0.45],        # Weak polling support
            'expert_analysis': [0.3, 0.4, 0.3, 0.35],     # Limited expert support
            'historical_precedent': [0.5, 0.4, 0.5, 0.6]  # Mixed historical evidence
        },
        'engagement': {
            'sentiment': [0.3, 0.4, 0.3, 0.35],           # Negative organic sentiment
            'viral_sharing': [0.9, 0.8, 0.9, 0.85],       # Artificially high sharing
            'quality_score': [0.4, 0.5, 0.4, 0.45],       # Moderate quality
            'source_credibility': [0.6, 0.5, 0.6, 0.55]   # Moderate credibility
        }
    }
    
    # Scenario 3: Legitimate news - Consistent patterns
    legitimate_news = {
        'content_id': 'POST_003',
        'scenario': 'Legitimate News - Well-Sourced Reporting',
        'topic': 'Scientific Discovery',
        'claims': {
            'research_findings': [0.7, 0.6, 0.7, 0.65],   # Moderate claims
            'methodology': [0.8, 0.7, 0.8, 0.75],         # Good methodology
            'significance': [0.6, 0.7, 0.6, 0.7]          # Appropriate significance
        },
        'evidence': {
            'peer_review': [0.9, 0.8, 0.9, 0.85],         # Strong peer review
            'replication': [0.7, 0.6, 0.7, 0.65],         # Good replication
            'expert_consensus': [0.8, 0.7, 0.8, 0.75]     # Expert agreement
        },
        'engagement': {
            'sentiment': [0.6, 0.7, 0.6, 0.65],           # Moderate positive sentiment
            'viral_sharing': [0.5, 0.6, 0.5, 0.55],       # Organic sharing
            'quality_score': [0.8, 0.7, 0.8, 0.75],       # High quality content
            'source_credibility': [0.9, 0.8, 0.9, 0.85]   # High credibility
        }
    }
    
    # Scenario 4: Conspiracy theory - Strong claims, no evidence
    conspiracy_theory = {
        'content_id': 'POST_004',
        'scenario': 'Conspiracy Theory - Unfounded Claims',
        'topic': 'Government Conspiracy',
        'claims': {
            'cover_up_evidence': [0.95, 0.9, 0.95, 0.9],  # Very strong claims
            'insider_knowledge': [0.9, 0.95, 0.9, 0.85],  # Claims inside info
            'widespread_impact': [0.85, 0.9, 0.85, 0.9]   # Claims major impact
        },
        'evidence': {
            'verifiable_sources': [0.0, 0.1, 0.0, 0.1],   # No verifiable sources
            'documentation': [0.1, 0.0, 0.1, 0.0],        # No documentation
            'expert_validation': [0.0, 0.0, 0.1, 0.0]     # No expert validation
        },
        'engagement': {
            'sentiment': [0.7, 0.8, 0.7, 0.75],           # High emotional sentiment
            'viral_sharing': [0.8, 0.9, 0.8, 0.85],       # High viral spread
            'quality_score': [0.1, 0.2, 0.1, 0.15],       # Very low quality
            'source_credibility': [0.2, 0.1, 0.2, 0.15]   # Very low credibility
        }
    }
    
    return [health_misinfo, political_astroturf, legitimate_news, conspiracy_theory]

def analyze_media_contradictions():
    """Analyze contradictions across multiple media scenarios."""
    
    detector = MediaContradictionDetector(confidence_threshold=0.6)
    scenarios = simulate_media_scenarios()
    
    results = []
    
    print("Media Contradiction & Fact-Checking Analysis")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\nContent: {scenario['content_id']} - {scenario['scenario']}")
        print(f"Topic: {scenario['topic']}")
        print("-" * 50)
        
        # Fact-checking analysis
        claims_values = []
        evidence_values = []
        
        for claim_type, values in scenario['claims'].items():
            claims_values.extend(values)
        
        for evidence_type, values in scenario['evidence'].items():
            evidence_values.extend(values)
        
        fact_result = detector.detect_fact_contradiction(
            claims_confidence=claims_values,
            evidence_support=evidence_values,
            claim_details=scenario['claims'],
            evidence_details=scenario['evidence'],
            source_id=scenario['content_id']
        )
        
        # Engagement manipulation analysis
        engagement_values = []
        quality_values = []
        
        engagement_values = scenario['engagement']['viral_sharing']
        sentiment_values = scenario['engagement']['sentiment']
        quality_values = scenario['engagement']['quality_score']
        credibility_values = scenario['engagement']['source_credibility']
        
        manipulation_result = detector.detect_engagement_manipulation(
            sentiment_scores=sentiment_values,
            engagement_metrics=engagement_values,
            content_quality=quality_values,
            source_credibility=credibility_values,
            post_id=scenario['content_id']
        )
        
        # Store combined results
        combined_result = {
            'content_id': scenario['content_id'],
            'scenario': scenario['scenario'],
            'topic': scenario['topic'],
            'fact_check': fact_result,
            'manipulation_check': manipulation_result
        }
        results.append(combined_result)
        
        # Display results
        print(f"📰 Fact-Check Results:")
        print(f"   Misinformation detected: {fact_result['is_contradictory']}")
        print(f"   Confidence: {fact_result['confidence']:.3f}")
        print(f"   Type: {fact_result['misinformation_type']}")
        print(f"   Credibility score: {fact_result['credibility_score']:.3f}")
        
        print(f"\n🎭 Manipulation Analysis:")
        print(f"   Manipulation detected: {manipulation_result['is_manipulated']}")
        print(f"   Confidence: {manipulation_result['confidence']:.3f}")
        print(f"   Type: {manipulation_result['manipulation_type']}")
        print(f"   Authenticity score: {manipulation_result['authenticity_score']:.3f}")
        
        # Content recommendations
        if fact_result['is_contradictory'] or manipulation_result['is_manipulated']:
            print(f"\n⚠️  CONTENT MODERATION ALERT:")
            if fact_result['is_contradictory']:
                print(f"   • Fact-check flag: {fact_result['misinformation_type']}")
                print(f"   • Credibility: {fact_result['credibility_score']:.1%}")
                
            if manipulation_result['is_manipulated']:
                print(f"   • Manipulation flag: {manipulation_result['manipulation_type']}")
                print(f"   • Authenticity: {manipulation_result['authenticity_score']:.1%}")
                
            print(f"   📋 Recommended actions:")
            print(f"      - Add fact-check warning labels")
            print(f"      - Reduce algorithmic distribution")
            print(f"      - Flag for human review")
            print(f"      - Monitor engagement patterns")
        else:
            print("✅ Content appears authentic and well-supported")
    
    return results

def media_intelligence_analysis(results):
    """Analyze media manipulation patterns across content."""
    
    print("\n" + "="*60)
    print("MEDIA INTELLIGENCE ANALYSIS")
    print("="*60)
    
    # Aggregate statistics
    total_content = len(results)
    misinfo_detected = sum(r['fact_check']['is_contradictory'] for r in results)
    manipulation_detected = sum(r['manipulation_check']['is_manipulated'] for r in results)
    
    avg_credibility = np.mean([r['fact_check']['credibility_score'] for r in results])
    avg_authenticity = np.mean([r['manipulation_check']['authenticity_score'] for r in results])
    
    print(f"\n📊 Content Analysis Summary:")
    print(f"   Content analyzed: {total_content}")
    print(f"   Misinformation detected: {misinfo_detected} ({misinfo_detected/total_content:.1%})")
    print(f"   Manipulation detected: {manipulation_detected} ({manipulation_detected/total_content:.1%})")
    print(f"   Average credibility: {avg_credibility:.3f}")
    print(f"   Average authenticity: {avg_authenticity:.3f}")
    
    # Misinformation type analysis
    misinfo_types = [r['fact_check']['misinformation_type'] for r in results 
                    if r['fact_check']['is_contradictory']]
    
    if misinfo_types:
        type_counts = {t: misinfo_types.count(t) for t in set(misinfo_types)}
        print(f"\n🚨 Misinformation Types:")
        for misinfo_type, count in type_counts.items():
            print(f"   {misinfo_type}: {count} instances")
    
    # Manipulation type analysis
    manipulation_types = [r['manipulation_check']['manipulation_type'] for r in results 
                         if r['manipulation_check']['is_manipulated']]
    
    if manipulation_types:
        manip_counts = {t: manipulation_types.count(t) for t in set(manipulation_types)}
        print(f"\n🎭 Manipulation Types:")
        for manip_type, count in manip_counts.items():
            print(f"   {manip_type}: {count} instances")
    
    return {
        'total_content': total_content,
        'misinfo_rate': misinfo_detected/total_content,
        'manipulation_rate': manipulation_detected/total_content,
        'avg_credibility': avg_credibility,
        'avg_authenticity': avg_authenticity
    }

def visualize_media_analysis(results):
    """Create visualizations for media analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    content_ids = [r['content_id'] for r in results]
    credibility_scores = [r['fact_check']['credibility_score'] for r in results]
    authenticity_scores = [r['manipulation_check']['authenticity_score'] for r in results]
    is_misinfo = [r['fact_check']['is_contradictory'] for r in results]
    is_manipulated = [r['manipulation_check']['is_manipulated'] for r in results]
    
    # Plot 1: Credibility vs. Authenticity
    for i, (cred, auth) in enumerate(zip(credibility_scores, authenticity_scores)):
        if is_misinfo[i] and is_manipulated[i]:
            color = 'red'
            label = 'Both Issues'
        elif is_misinfo[i]:
            color = 'orange'
            label = 'Misinformation'
        elif is_manipulated[i]:
            color = 'yellow'
            label = 'Manipulation'
        else:
            color = 'green'
            label = 'Authentic'
        ax1.scatter(cred, auth, c=color, s=100, alpha=0.7)
    
    ax1.set_xlabel('Credibility Score')
    ax1.set_ylabel('Authenticity Score')
    ax1.set_title('Content Quality Assessment')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Plot 2: Detection confidence by content
    fact_confidences = [r['fact_check']['confidence'] for r in results]
    manip_confidences = [r['manipulation_check']['confidence'] for r in results]
    
    x = np.arange(len(content_ids))
    width = 0.35
    
    ax2.bar(x - width/2, fact_confidences, width, label='Fact-Check', alpha=0.7)
    ax2.bar(x + width/2, manip_confidences, width, label='Manipulation', alpha=0.7)
    ax2.set_xlabel('Content')
    ax2.set_ylabel('Detection Confidence')
    ax2.set_title('Contradiction Detection Results')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{i+1}' for i in range(len(content_ids))])
    ax2.legend()
    
    # Plot 3: Content type distribution
    scenarios = [r['scenario'].split(' - ')[0] for r in results]
    scenario_counts = {s: scenarios.count(s) for s in set(scenarios)}
    ax3.pie(scenario_counts.values(), labels=scenario_counts.keys(), autopct='%1.1f%%')
    ax3.set_title('Content Type Distribution')
    
    # Plot 4: Quality correlation analysis
    claim_strengths = [r['fact_check']['claim_strength'] for r in results]
    evidence_strengths = [r['fact_check']['evidence_strength'] for r in results]
    
    ax4.scatter(claim_strengths, evidence_strengths, c=credibility_scores, 
               cmap='RdYlGn', s=100, alpha=0.7)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Evidence')
    ax4.set_xlabel('Claim Strength')
    ax4.set_ylabel('Evidence Strength')
    ax4.set_title('Claims vs. Evidence Analysis')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Credibility Score')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('media_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def social_media_monitoring_demo():
    """Demonstrate real-time social media monitoring."""
    
    print("\n" + "="*60)
    print("REAL-TIME SOCIAL MEDIA MONITORING DEMO")
    print("="*60)
    
    detector = MediaContradictionDetector(confidence_threshold=0.6)
    
    print("\n📱 Monitoring viral health claim...")
    print("Platform: Social Media")
    print("Topic: COVID-19 Treatment")
    
    # Simulate viral spread of health misinformation
    time_points = ['Hour 1', 'Hour 6', 'Hour 12', 'Hour 24']
    
    for i, time_point in enumerate(time_points):
        print(f"\n⏰ {time_point}:")
        
        # Strong health claims
        claims = [0.9, 0.85, 0.9, 0.95]  # Consistent strong claims
        
        # Weak/contradicting evidence (varies over time)
        evidence = [0.1 - i*0.02, 0.05, 0.08 - i*0.01, 0.03]  # Weakening evidence
        
        # Viral engagement pattern
        engagement = [0.3 + i*0.2, 0.4 + i*0.15, 0.5 + i*0.1, 0.6 + i*0.08]
        sentiment = [0.7, 0.8, 0.75, 0.8]  # High positive sentiment
        quality = [0.2, 0.1, 0.15, 0.1]     # Low quality
        credibility = [0.3, 0.2, 0.25, 0.2] # Low credibility
        
        print(f"Claim strength: {np.mean(claims):.2f}")
        print(f"Evidence support: {np.mean(evidence):.2f}")
        print(f"Viral engagement: {np.mean(engagement):.2f}")
        
        # Fact-check analysis
        fact_result = detector.detect_fact_contradiction(
            claims_confidence=claims,
            evidence_support=evidence,
            claim_details={'health_claims': claims},
            evidence_details={'scientific_evidence': evidence},
            source_id=f'VIRAL_POST_{time_point}'
        )
        
        # Manipulation analysis
        manip_result = detector.detect_engagement_manipulation(
            sentiment_scores=sentiment,
            engagement_metrics=engagement,
            content_quality=quality,
            source_credibility=credibility,
            post_id=f'VIRAL_POST_{time_point}'
        )
        
        print(f"🔍 Misinformation confidence: {fact_result['confidence']:.3f}")
        print(f"🎭 Manipulation confidence: {manip_result['confidence']:.3f}")
        
        if fact_result['is_contradictory'] or manip_result['is_manipulated']:
            print(f"🚨 CONTENT MODERATION ALERT!")
            if fact_result['is_contradictory']:
                print(f"   Misinformation type: {fact_result['misinformation_type']}")
            if manip_result['is_manipulated']:
                print(f"   Manipulation type: {manip_result['manipulation_type']}")
            print(f"   Recommended: Immediate fact-check label and reduced distribution")
        else:
            print("✅ Content within acceptable parameters")

if __name__ == "__main__":
    print("FusionAlpha Media Contradiction & Fact-Checking")
    print("=" * 50)
    
    # Run main media analysis
    results = analyze_media_contradictions()
    
    # Media intelligence analysis
    intel_summary = media_intelligence_analysis(results)
    
    # Create visualizations
    visualize_media_analysis(results)
    
    # Demonstrate real-time monitoring
    social_media_monitoring_demo()
    
    print(f"\n✅ Media analysis complete!")
    print(f"   Analyzed {len(results)} pieces of content")
    print(f"   Misinformation rate: {intel_summary['misinfo_rate']:.1%}")
    print(f"   Manipulation rate: {intel_summary['manipulation_rate']:.1%}")
    
    print(f"\n📰 Media Applications:")
    print(f"   • Automated fact-checking")
    print(f"   • Misinformation detection")
    print(f"   • Bot and astroturfing identification")
    print(f"   • Content quality assessment")
    print(f"   • Real-time content moderation")