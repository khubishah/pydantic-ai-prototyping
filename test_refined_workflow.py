#!/usr/bin/env python3
"""
Test script for the refined Kubernetes anomaly detection workflow.

This demonstrates the correct flow:
1. Parallel tool calls for baseline and anomaly analysis
2. LLM merge and analyze outputs from 2 paths
3. Conditional node analysis (if pod)
4. Final LLM summarization

Uses specialized agent architecture with separation of concerns.
"""

import asyncio
import logging
from datetime import datetime

from k8s_anomaly_detection_refined import (
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

async def test_refined_workflow():
    """Test the refined workflow with correct flow and specialized agents."""
    print("🔬 TESTING REFINED WORKFLOW")
    print("=" * 70)
    print("Workflow Steps:")
    print("1. Parallel analysis paths: [baseline_cpu→logs] || [anomaly_cpu→logs]")
    print("2. LLM merge and analyze outputs from 2 paths")
    print("3. Conditional node analysis (if pod)")
    print("4. Final LLM comprehensive summarization")
    print("=" * 70)
    
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    # Use exact mock data time ranges
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    print(f"\n📅 Analysis Periods:")
    print(f"   Baseline: {baseline_start.strftime('%H:%M')} to {baseline_end.strftime('%H:%M')}")
    print(f"   Anomaly:  {anomaly_start.strftime('%H:%M')} to {anomaly_end.strftime('%H:%M')}")
    print()
    
    # Test Pod Analysis
    print("🔍 TESTING POD ANALYSIS:")
    print("-" * 50)
    
    start_time = datetime.now()
    
    result = await orchestrator.analyze_entity(
        entity_name="frontend-6d8f4f79f7-kxzpl",
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
        print("✅ REFINED ANALYSIS SUCCESSFUL")
        print()
        
        print("📊 ENTITY INFORMATION:")
        print(f"   Name: {result.entity_name}")
        print(f"   Type: {result.entity_type}")
        print(f"   Labels: {result.entity_metadata.labels}")
        print(f"   Node: {result.entity_metadata.node_name}")
        print()
        
        print("📈 BASELINE PATH RESULTS:")
        print(f"   CPU: {result.baseline_analysis.cpu_metrics.average_utilization:.2f}% avg, {result.baseline_analysis.cpu_metrics.peak_utilization:.2f}% peak")
        print(f"   Logs: {result.baseline_analysis.log_analysis.error_count} errors, {result.baseline_analysis.log_analysis.warning_count} warnings")
        print(f"   Events: {result.baseline_analysis.log_analysis.anomalous_events}")
        print()
        
        print("🚨 ANOMALY PATH RESULTS:")
        print(f"   CPU: {result.anomaly_analysis.cpu_metrics.average_utilization:.2f}% avg, {result.anomaly_analysis.cpu_metrics.peak_utilization:.2f}% peak")
        print(f"   Logs: {result.anomaly_analysis.log_analysis.error_count} errors, {result.anomaly_analysis.log_analysis.warning_count} warnings")
        print(f"   Critical Patterns: {result.anomaly_analysis.log_analysis.critical_patterns}")
        print(f"   Events: {result.anomaly_analysis.log_analysis.anomalous_events}")
        print()
        
        print("🔄 LLM MERGED ANALYSIS:")
        print(f"   Significant Change: {result.merged_analysis.has_significant_change}")
        print(f"   Confidence: {result.merged_analysis.confidence_score:.3f}")
        print(f"   Severity: {result.merged_analysis.severity}")
        print(f"   Key Findings:")
        for finding in result.merged_analysis.key_findings:
            print(f"     • {finding}")
        print(f"   CPU Summary: {result.merged_analysis.cpu_change_summary}")
        print(f"   Log Summary: {result.merged_analysis.log_change_summary}")
        print()
        
        if result.node_analysis:
            print("🖥️  NODE ANALYSIS PERFORMED:")
            print(f"   Node: {result.node_analysis.entity_name}")
            print(f"   Node Anomaly: {result.node_analysis.final_summary.anomaly_detected}")
            print(f"   Node Confidence: {result.node_analysis.final_summary.confidence_score:.3f}")
            print(f"   Node Severity: {result.node_analysis.final_summary.severity}")
            print()
        else:
            print("🖥️  NODE ANALYSIS: Not performed or not needed")
            print()
        
        print("🎯 FINAL COMPREHENSIVE SUMMARY:")
        print(f"   Overall Conclusion: {result.final_summary.overall_conclusion}")
        print(f"   Anomaly Detected: {result.final_summary.anomaly_detected}")
        print(f"   Final Confidence: {result.final_summary.confidence_score:.3f}")
        print(f"   Final Severity: {result.final_summary.severity}")
        print(f"   Primary Evidence:")
        for evidence in result.final_summary.primary_evidence:
            print(f"     • {evidence}")
        print(f"   Recommendations:")
        for rec in result.final_summary.recommendations:
            print(f"     • {rec}")
        
        # Show the data progression
        cpu_change = result.anomaly_analysis.cpu_metrics.average_utilization - result.baseline_analysis.cpu_metrics.average_utilization
        cpu_change_pct = (cpu_change / result.baseline_analysis.cpu_metrics.average_utilization) * 100
        print()
        print("📊 DATA PROGRESSION:")
        print(f"   CPU Change: +{cpu_change:.2f}% ({cpu_change_pct:.1f}% increase)")
        print(f"   Log Change: {result.baseline_analysis.log_analysis.warning_count} → {result.anomaly_analysis.log_analysis.warning_count} warnings")
        
    else:
        print(f"❌ Refined analysis failed: {result.error_message}")
    
    print("\n" + "=" * 70)
    print("🎉 REFINED WORKFLOW TEST COMPLETE!")
    print("\nArchitecture Highlights:")
    print("✅ Specialized agents for merge analysis and final summary")
    print("✅ Correct workflow sequence with conditional node analysis")
    print("✅ Real mock data integration")
    print("✅ Comprehensive multi-stage LLM analysis")
    print("✅ Clear separation of concerns between analysis stages")

async def test_node_direct():
    """Test direct node analysis."""
    print("\n" + "=" * 70)
    print("🖥️  TESTING DIRECT NODE ANALYSIS")
    print("=" * 70)
    
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    start_time = datetime.now()
    
    result = await orchestrator.analyze_entity(
        entity_name="node-1",
        entity_type=EntityType.NODE,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end
    )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"⏱️  Node Analysis Time: {execution_time:.2f} seconds")
    
    if isinstance(result, AnomalyDetectionResult):
        print("✅ NODE ANALYSIS SUCCESSFUL")
        print(f"🔄 Merged Analysis Confidence: {result.merged_analysis.confidence_score:.3f}")
        print(f"🎯 Final Conclusion: {result.final_summary.anomaly_detected}")
        print(f"📊 Final Confidence: {result.final_summary.confidence_score:.3f}")
        print(f"📝 Key Findings: {result.merged_analysis.key_findings}")
    else:
        print(f"❌ Node analysis failed: {result.error_message}")

async def main():
    """Run all tests for the refined workflow."""
    await test_refined_workflow()
    await test_node_direct()
    
    print("\n" + "=" * 70)
    print("🏆 ALL REFINED WORKFLOW TESTS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main()) 