#!/usr/bin/env python3
"""
Test script to verify mock data integration with the manual control implementation.
This tests that all three mock data functions are properly integrated.
"""

import asyncio
from datetime import datetime

from k8s_anomaly_detection_manual_control import (
    K8sAnomalyDetectionOrchestrator,
    EntityType,
    AnomalyDetectionResult,
    AnalysisFailure
)

async def test_mock_data_integration():
    """Test that all mock data functions are properly integrated."""
    print("🧪 TESTING MOCK DATA INTEGRATION")
    print("=" * 50)
    
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    # Use exact mock data time ranges
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    print(f"📅 Baseline: {baseline_start} to {baseline_end}")
    print(f"📅 Anomaly:  {anomaly_start} to {anomaly_end}")
    print()
    
    # Test Pod Analysis
    print("🔍 Testing Pod Analysis with Mock Data:")
    print("-" * 40)
    
    result = await orchestrator.analyze_entity(
        entity_name="frontend-6d8f4f79f7-kxzpl",
        entity_type=EntityType.POD,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end
    )
    
    if isinstance(result, AnomalyDetectionResult):
        print("✅ Pod analysis successful!")
        print(f"📊 Entity: {result.entity_name}")
        print(f"🏷️  Labels: {result.entity_metadata.labels}")
        print(f"🖥️  Node: {result.entity_metadata.node_name}")
        print()
        
        print("📈 BASELINE METRICS:")
        print(f"   CPU: {result.baseline_cpu.average_utilization:.2f}% avg, {result.baseline_cpu.peak_utilization:.2f}% peak")
        print(f"   Logs: {result.baseline_logs.error_count} errors, {result.baseline_logs.warning_count} warnings")
        print(f"   Events: {result.baseline_logs.anomalous_events}")
        print()
        
        print("🚨 ANOMALY METRICS:")
        print(f"   CPU: {result.anomaly_cpu.average_utilization:.2f}% avg, {result.anomaly_cpu.peak_utilization:.2f}% peak")
        print(f"   Logs: {result.anomaly_logs.error_count} errors, {result.anomaly_logs.warning_count} warnings")
        print(f"   Critical Patterns: {result.anomaly_logs.critical_patterns}")
        print(f"   Events: {result.anomaly_logs.anomalous_events}")
        print()
        
        print("🎯 ANOMALY DETECTION:")
        print(f"   Detected: {result.anomaly_detection.has_anomaly}")
        print(f"   Confidence: {result.anomaly_detection.confidence_score:.3f}")
        print(f"   Severity: {result.anomaly_detection.severity}")
        print(f"   Description: {result.anomaly_detection.description}")
        
        # Show that we're using real mock data
        cpu_increase = result.anomaly_cpu.average_utilization - result.baseline_cpu.average_utilization
        print(f"   CPU Change: +{cpu_increase:.2f}% (baseline: {result.baseline_cpu.average_utilization:.2f}% → anomaly: {result.anomaly_cpu.average_utilization:.2f}%)")
        
        if result.node_analysis:
            print(f"🖥️  Node Analysis: {result.node_analysis.anomaly_detection.has_anomaly} (confidence: {result.node_analysis.anomaly_detection.confidence_score:.3f})")
        
    else:
        print(f"❌ Pod analysis failed: {result.error_message}")
    
    print("\n" + "=" * 50)
    print("🎉 Mock Data Integration Test Complete!")
    print("\nExpected Behavior:")
    print("• Baseline period should show normal CPU usage (10-30%)")
    print("• Anomaly period should show high CPU usage (80-95%)")
    print("• Baseline logs should be minimal (info events)")
    print("• Anomaly logs should show unhealthy/backoff events")
    print("• LLM should detect significant anomaly with high confidence")

if __name__ == "__main__":
    asyncio.run(test_mock_data_integration()) 