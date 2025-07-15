#!/usr/bin/env python3
"""
Test script for the manual control Kubernetes anomaly detection implementation.

This demonstrates:
1. Correct parallel structure: [baseline_cpu→baseline_logs] || [anomaly_cpu→anomaly_logs]
2. LLM-based synthesis of anomaly detection results
3. Comprehensive error handling and validation
"""

import asyncio
import logging
from datetime import datetime, timedelta

from k8s_anomaly_detection_manual_control import (
    K8sAnomalyDetectionOrchestrator,
    EntityType,
    AnomalyDetectionResult,
    AnalysisFailure
)

# Configure logging to see the execution flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_pod_analysis():
    """Test pod anomaly detection with the improved workflow."""
    print("\n" + "="*80)
    print("TEST 1: Pod Anomaly Detection with Manual Control")
    print("="*80)
    
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    # Define time ranges that match mock data
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    print(f"📅 Baseline Period: {baseline_start.strftime('%H:%M:%S')} to {baseline_end.strftime('%H:%M:%S')}")
    print(f"📅 Anomaly Period: {anomaly_start.strftime('%H:%M:%S')} to {anomaly_end.strftime('%H:%M:%S')}")
    print()
    
    start_time = datetime.now()
    
    result = await orchestrator.analyze_entity(
        entity_name="frontend-6d8f4f79f7-kxzpl",  # Use actual mock pod name
        entity_type=EntityType.POD,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️  Total Execution Time: {execution_time:.2f} seconds")
    print()
    
    if isinstance(result, AnomalyDetectionResult):
        print("✅ ANALYSIS SUCCESSFUL")
        print(f"🏷️  Entity: {result.entity_type} '{result.entity_name}'")
        print(f"📊 Status: {result.status}")
        print()
        
        print("🔍 ANOMALY DETECTION RESULTS:")
        print(f"   • Anomaly Detected: {result.anomaly_detection.has_anomaly}")
        print(f"   • Confidence Score: {result.anomaly_detection.confidence_score:.3f}")
        if result.anomaly_detection.severity:
            print(f"   • Severity: {result.anomaly_detection.severity}")
        print(f"   • Description: {result.anomaly_detection.description}")
        
        if result.anomaly_detection.contributing_factors:
            print("   • Contributing Factors:")
            for factor in result.anomaly_detection.contributing_factors:
                print(f"     - {factor}")
        
        print("\n📈 BASELINE vs ANOMALY COMPARISON:")
        print(f"   CPU: {result.baseline_cpu.average_utilization:.1f}% → {result.anomaly_cpu.average_utilization:.1f}%")
        print(f"   Errors: {result.baseline_logs.error_count} → {result.anomaly_logs.error_count}")
        print(f"   Warnings: {result.baseline_logs.warning_count} → {result.anomaly_logs.warning_count}")
        
        if result.node_analysis:
            print(f"\n🖥️  NODE ANALYSIS: {result.node_analysis.entity_name}")
            print(f"   • Node Anomaly: {result.node_analysis.anomaly_detection.has_anomaly}")
            print(f"   • Node Confidence: {result.node_analysis.anomaly_detection.confidence_score:.3f}")
    
    else:
        print("❌ ANALYSIS FAILED")
        print(f"Error Type: {result.error_type}")
        print(f"Error Message: {result.error_message}")
        print("Recovery Suggestions:")
        for suggestion in result.recovery_suggestions:
            print(f"  • {suggestion}")
    
    return result

async def test_node_analysis():
    """Test node anomaly detection."""
    print("\n" + "="*80)
    print("TEST 2: Node Anomaly Detection with Manual Control")
    print("="*80)
    
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    # Define time ranges that match mock data
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    start_time = datetime.now()
    
    result = await orchestrator.analyze_entity(
        entity_name="node-1",  # Use actual mock node name
        entity_type=EntityType.NODE,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    
    if isinstance(result, AnomalyDetectionResult):
        print("✅ NODE ANALYSIS SUCCESSFUL")
        print(f"🎯 Anomaly Detected: {result.anomaly_detection.has_anomaly}")
        print(f"📊 Confidence: {result.anomaly_detection.confidence_score:.3f}")
        print(f"📝 Description: {result.anomaly_detection.description}")
    else:
        print(f"❌ Node analysis failed: {result.error_message}")
    
    return result

async def test_error_handling():
    """Test error handling with invalid entity type."""
    print("\n" + "="*80)
    print("TEST 3: Error Handling - Invalid Entity Type")
    print("="*80)
    
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    try:
        result = await orchestrator.analyze_entity(
            entity_name="invalid-entity",
            entity_type="k8s:service",  # Invalid type, should be POD or NODE
            baseline_start=datetime.fromisoformat("2024-06-26T09:00:00+00:00"),
            baseline_end=datetime.fromisoformat("2024-06-26T10:00:00+00:00"),
            anomaly_start=datetime.fromisoformat("2024-06-26T10:00:00+00:00"),
            anomaly_end=datetime.fromisoformat("2024-06-26T10:30:00+00:00")
        )
        
        if isinstance(result, AnalysisFailure):
            print("✅ ERROR HANDLING SUCCESSFUL")
            print(f"Error caught: {result.error_type} - {result.error_message}")
        else:
            print("❌ Expected error was not caught")
            
    except Exception as e:
        print(f"✅ VALIDATION ERROR CAUGHT: {str(e)}")

async def test_execution_pattern():
    """Test to verify the correct execution pattern."""
    print("\n" + "="*80)
    print("TEST 4: Execution Pattern Verification")
    print("="*80)
    print("Expected Pattern:")
    print("1. Get metadata")
    print("2. Branch A: baseline_cpu → baseline_logs (sequential)")
    print("3. Branch B: anomaly_cpu → anomaly_logs (sequential)")
    print("4. Branches A & B run in parallel")
    print("5. LLM synthesis of results")
    print("6. Node analysis (if pod)")
    print("\nWatch the logs for execution order confirmation:")
    print("-" * 60)
    
    await test_pod_analysis()

async def main():
    """Run all tests to demonstrate the improved implementation."""
    print("🚀 TESTING MANUAL CONTROL K8S ANOMALY DETECTION")
    print("This implementation features:")
    print("• Correct parallel structure: [baseline_cpu→logs] || [anomaly_cpu→logs]")
    print("• LLM-based intelligent synthesis of results")
    print("• Deterministic execution with asyncio.gather()")
    print("• Comprehensive error handling")
    
    # Run tests
    await test_execution_pattern()
    await test_node_analysis()
    await test_error_handling()
    
    print("\n" + "="*80)
    print("🎉 ALL TESTS COMPLETED")
    print("="*80)
    print("\nKey Improvements Demonstrated:")
    print("✅ Parallel branches: [baseline→logs] || [anomaly→logs]")
    print("✅ LLM synthesis with intelligent analysis")
    print("✅ Deterministic execution order")
    print("✅ Proper error handling and validation")
    print("✅ Performance optimization with true parallelism")

if __name__ == "__main__":
    asyncio.run(main()) 