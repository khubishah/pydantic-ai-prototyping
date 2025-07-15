"""
Determinism Stress Test: Ambiguous Prompts
Shows when multi-tool agents become non-deterministic
"""

import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from typing import Union

# Mock data models
class EntityMetadata(BaseModel):
    name: str
    namespace: str

class CPUMetrics(BaseModel):
    avg_cpu: float
    peak_cpu: float

class LogAnalysis(BaseModel):
    error_count: int
    warning_count: int

# Mock tools
async def get_metadata(ctx: RunContext, entity_name: str) -> dict:
    print(f"🔧 TOOL CALLED: get_metadata for {entity_name}")
    return {"name": entity_name, "namespace": "default"}

async def get_cpu_metrics(ctx: RunContext, entity_name: str) -> dict:
    print(f"🔧 TOOL CALLED: get_cpu_metrics for {entity_name}")
    return {"avg_cpu": 25.5, "peak_cpu": 45.2}

async def get_log_analysis(ctx: RunContext, entity_name: str) -> dict:
    print(f"🔧 TOOL CALLED: get_log_analysis for {entity_name}")
    return {"error_count": 2, "warning_count": 5}

# Specialized Agents
metadata_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=EntityMetadata,
    system_prompt="You retrieve K8s metadata. Always use get_metadata tool."
)

# Multi-tool Agent
unified_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Union[EntityMetadata, CPUMetrics, LogAnalysis],
    system_prompt="""
    You are a K8s analysis tool. Based on the request, choose the appropriate tool:
    - get_metadata for entity information
    - get_cpu_metrics for CPU analysis  
    - get_log_analysis for log analysis
    
    If the request is ambiguous, make your best guess.
    """
)

# Register tools
metadata_agent.tool(get_metadata)
unified_agent.tool(get_metadata)
unified_agent.tool(get_cpu_metrics) 
unified_agent.tool(get_log_analysis)

async def test_ambiguous_prompts():
    """Test with ambiguous prompts that could match multiple tools"""
    
    print("🧪 TESTING WITH AMBIGUOUS PROMPTS")
    print("=" * 50)
    
    # Ambiguous prompts that could trigger different tools
    ambiguous_prompts = [
        "Analyze pod frontend-123",  # Could be metadata, CPU, or logs
        "Check frontend-123 status",  # Could be metadata or logs
        "Get frontend-123 information",  # Could be any tool
        "Investigate frontend-123",  # Very ambiguous
        "Tell me about frontend-123"  # Very ambiguous
    ]
    
    for prompt in ambiguous_prompts:
        print(f"\n📝 PROMPT: '{prompt}'")
        print("-" * 40)
        
        # Specialized agent - always deterministic
        print("✅ SPECIALIZED AGENT:")
        try:
            result = await metadata_agent.run(prompt)
            print(f"   Result: {type(result.output).__name__}")
        except Exception as e:
            print(f"   ERROR: {e}")
        
        # Multi-tool agent - potentially non-deterministic
        print("❓ MULTI-TOOL AGENT:")
        try:
            result = await unified_agent.run(prompt)
            print(f"   Result: {type(result.output).__name__}")
        except Exception as e:
            print(f"   ERROR: {e}")

async def test_multiple_runs():
    """Test same ambiguous prompt multiple times"""
    
    print("\n\n🔄 TESTING CONSISTENCY ACROSS MULTIPLE RUNS")
    print("=" * 50)
    
    ambiguous_prompt = "Analyze pod frontend-123 performance and status"
    
    print(f"📝 PROMPT: '{ambiguous_prompt}'")
    print("Running 5 times to check consistency...")
    
    specialized_results = []
    unified_results = []
    
    for i in range(5):
        print(f"\n--- Run {i+1} ---")
        
        # Specialized agent
        try:
            result = await metadata_agent.run(ambiguous_prompt)
            specialized_results.append(type(result.output).__name__)
            print(f"Specialized: {type(result.output).__name__}")
        except Exception as e:
            specialized_results.append("ERROR")
            print(f"Specialized: ERROR")
        
        # Multi-tool agent
        try:
            result = await unified_agent.run(ambiguous_prompt)
            unified_results.append(type(result.output).__name__)
            print(f"Multi-tool: {type(result.output).__name__}")
        except Exception as e:
            unified_results.append("ERROR")
            print(f"Multi-tool: ERROR")
    
    print(f"\n📊 CONSISTENCY ANALYSIS:")
    print(f"✅ Specialized Agent Results: {specialized_results}")
    print(f"   Consistency: {len(set(specialized_results)) == 1}")
    print(f"❓ Multi-Tool Agent Results: {unified_results}")
    print(f"   Consistency: {len(set(unified_results)) == 1}")
    
    if len(set(specialized_results)) == 1:
        print("✅ Specialized agent is 100% deterministic")
    else:
        print("❌ Specialized agent showed inconsistency")
    
    if len(set(unified_results)) == 1:
        print("✅ Multi-tool agent was consistent")
    else:
        print("❌ Multi-tool agent showed non-deterministic behavior")

async def main():
    print("🔬 DETERMINISM STRESS TEST")
    print("=" * 60)
    
    await test_ambiguous_prompts()
    await test_multiple_runs()
    
    print("\n\n🎯 CONCLUSION:")
    print("=" * 30)
    print("✅ Specialized agents are MORE deterministic because:")
    print("   • No tool selection ambiguity")
    print("   • Focused system prompts")
    print("   • Guaranteed output types")
    print("   • Predictable behavior")
    print()
    print("❌ Multi-tool agents are LESS deterministic because:")
    print("   • LLM must choose between multiple tools")
    print("   • Ambiguous prompts can trigger different tools")
    print("   • Union output types add complexity")
    print("   • Context-dependent behavior")

if __name__ == "__main__":
    asyncio.run(main()) 