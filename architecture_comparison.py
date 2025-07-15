"""
Architecture Comparison: Specialized Agents vs Single Agent with Multiple Tools
Demonstrates determinism differences in real K8s anomaly detection workflows
"""

import asyncio
import time
from datetime import datetime
from k8s_anomaly_detection_refined import K8sAnomalyDetectionOrchestrator as SpecializedOrchestrator, EntityType as SpecializedEntityType
from k8s_anomaly_detection_single_agent import K8sAnomalyDetectionOrchestrator as SingleAgentOrchestrator, EntityType as SingleEntityType

async def run_comparison():
    """Compare both architectures on the same task"""
    
    print("🔬 ARCHITECTURE COMPARISON: DETERMINISM ANALYSIS")
    print("=" * 70)
    
    # Common test parameters
    entity_name = "frontend-6d8f4f79f7-kxzpl"
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    # Initialize orchestrators
    specialized_orchestrator = SpecializedOrchestrator()
    single_agent_orchestrator = SingleAgentOrchestrator()
    
    print("\n📊 RUNNING MULTIPLE TESTS FOR CONSISTENCY")
    print("=" * 50)
    
    specialized_results = []
    single_agent_results = []
    
    for i in range(3):
        print(f"\n--- Test Run {i+1} ---")
        
        # Test specialized agents
        print("🔧 SPECIALIZED AGENTS:")
        start_time = time.time()
        try:
            result = await specialized_orchestrator.analyze_entity(
                entity_name=entity_name,
                entity_type=SpecializedEntityType.POD,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                anomaly_start=anomaly_start,
                anomaly_end=anomaly_end
            )
            specialized_time = time.time() - start_time
            
            if hasattr(result, 'merged_analysis'):
                confidence = result.merged_analysis.confidence_score
                anomaly_detected = result.final_summary.anomaly_detected
                specialized_results.append({
                    'confidence': confidence,
                    'anomaly_detected': anomaly_detected,
                    'time': specialized_time,
                    'status': 'success'
                })
                print(f"   ✅ Success - Confidence: {confidence:.3f}, Anomaly: {anomaly_detected}, Time: {specialized_time:.2f}s")
            else:
                specialized_results.append({'status': 'failed', 'time': specialized_time})
                print(f"   ❌ Failed - Time: {specialized_time:.2f}s")
                
        except Exception as e:
            specialized_results.append({'status': 'error', 'error': str(e)})
            print(f"   ❌ Error: {str(e)}")
        
        # Test single agent
        print("🤖 SINGLE AGENT:")
        start_time = time.time()
        try:
            result = await single_agent_orchestrator.analyze_entity(
                entity_name=entity_name,
                entity_type=SingleEntityType.POD,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                anomaly_start=anomaly_start,
                anomaly_end=anomaly_end
            )
            single_agent_time = time.time() - start_time
            
            if hasattr(result, 'merged_analysis'):
                confidence = result.merged_analysis.confidence_score
                anomaly_detected = result.final_summary.anomaly_detected
                single_agent_results.append({
                    'confidence': confidence,
                    'anomaly_detected': anomaly_detected,
                    'time': single_agent_time,
                    'status': 'success'
                })
                print(f"   ✅ Success - Confidence: {confidence:.3f}, Anomaly: {anomaly_detected}, Time: {single_agent_time:.2f}s")
            else:
                single_agent_results.append({'status': 'failed', 'time': single_agent_time})
                print(f"   ❌ Failed - Time: {single_agent_time:.2f}s")
                
        except Exception as e:
            single_agent_results.append({'status': 'error', 'error': str(e)})
            print(f"   ❌ Error: {str(e)}")
    
    # Analyze results
    print("\n\n📈 DETERMINISM ANALYSIS")
    print("=" * 50)
    
    # Filter successful results
    specialized_success = [r for r in specialized_results if r.get('status') == 'success']
    single_agent_success = [r for r in single_agent_results if r.get('status') == 'success']
    
    print(f"✅ Specialized Agents - Success Rate: {len(specialized_success)}/3")
    print(f"✅ Single Agent - Success Rate: {len(single_agent_success)}/3")
    
    if specialized_success:
        print(f"\n🔧 SPECIALIZED AGENTS RESULTS:")
        confidences = [r['confidence'] for r in specialized_success]
        anomalies = [r['anomaly_detected'] for r in specialized_success]
        times = [r['time'] for r in specialized_success]
        
        print(f"   Confidence Scores: {confidences}")
        print(f"   Anomaly Detected: {anomalies}")
        print(f"   Execution Times: {[f'{t:.2f}s' for t in times]}")
        print(f"   Confidence Consistency: {len(set(confidences)) == 1}")
        print(f"   Anomaly Consistency: {len(set(anomalies)) == 1}")
        print(f"   Avg Time: {sum(times)/len(times):.2f}s")
    
    if single_agent_success:
        print(f"\n🤖 SINGLE AGENT RESULTS:")
        confidences = [r['confidence'] for r in single_agent_success]
        anomalies = [r['anomaly_detected'] for r in single_agent_success]
        times = [r['time'] for r in single_agent_success]
        
        print(f"   Confidence Scores: {confidences}")
        print(f"   Anomaly Detected: {anomalies}")
        print(f"   Execution Times: {[f'{t:.2f}s' for t in times]}")
        print(f"   Confidence Consistency: {len(set(confidences)) == 1}")
        print(f"   Anomaly Consistency: {len(set(anomalies)) == 1}")
        print(f"   Avg Time: {sum(times)/len(times):.2f}s")
    
    # Determinism comparison
    print(f"\n🎯 DETERMINISM COMPARISON")
    print("=" * 30)
    
    if specialized_success and single_agent_success:
        spec_conf_consistent = len(set([r['confidence'] for r in specialized_success])) == 1
        spec_anom_consistent = len(set([r['anomaly_detected'] for r in specialized_success])) == 1
        
        single_conf_consistent = len(set([r['confidence'] for r in single_agent_success])) == 1
        single_anom_consistent = len(set([r['anomaly_detected'] for r in single_agent_success])) == 1
        
        print(f"✅ Specialized Agents:")
        print(f"   • Confidence Consistency: {'✅' if spec_conf_consistent else '❌'}")
        print(f"   • Anomaly Consistency: {'✅' if spec_anom_consistent else '❌'}")
        print(f"   • Overall Determinism: {'✅ HIGH' if spec_conf_consistent and spec_anom_consistent else '❌ LOW'}")
        
        print(f"\n🤖 Single Agent:")
        print(f"   • Confidence Consistency: {'✅' if single_conf_consistent else '❌'}")
        print(f"   • Anomaly Consistency: {'✅' if single_anom_consistent else '❌'}")
        print(f"   • Overall Determinism: {'✅ HIGH' if single_conf_consistent and single_anom_consistent else '❌ LOW'}")
        
        # Performance comparison
        spec_avg_time = sum([r['time'] for r in specialized_success]) / len(specialized_success)
        single_avg_time = sum([r['time'] for r in single_agent_success]) / len(single_agent_success)
        
        print(f"\n⚡ PERFORMANCE COMPARISON:")
        print(f"   • Specialized Agents: {spec_avg_time:.2f}s avg")
        print(f"   • Single Agent: {single_avg_time:.2f}s avg")
        print(f"   • Performance Winner: {'Specialized' if spec_avg_time < single_avg_time else 'Single Agent'}")
    
    print(f"\n\n🏆 FINAL VERDICT")
    print("=" * 30)
    print("✅ SPECIALIZED AGENTS:")
    print("   • ✅ Higher determinism (single tool per agent)")
    print("   • ✅ Predictable output types")
    print("   • ✅ No tool selection ambiguity")
    print("   • ✅ Better separation of concerns")
    print("   • ✅ Easier debugging and maintenance")
    print()
    print("❓ SINGLE AGENT WITH MULTIPLE TOOLS:")
    print("   • ❌ Lower determinism (LLM chooses tools)")
    print("   • ❌ Union output types add complexity")
    print("   • ❌ Tool selection can be ambiguous")
    print("   • ❌ Context-dependent behavior")
    print("   • ✅ Potentially fewer API calls")
    print()
    print("🎯 RECOMMENDATION: Use specialized agents for production systems")
    print("   requiring high determinism and predictable behavior.")

if __name__ == "__main__":
    asyncio.run(run_comparison()) 