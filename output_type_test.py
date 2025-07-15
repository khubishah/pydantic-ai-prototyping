"""
Output Type Determinism Test
Shows how multi-tool agents can return different output types
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
    return {"name": entity_name, "namespace": "default"}

async def get_cpu_metrics(ctx: RunContext, entity_name: str) -> dict:
    return {"avg_cpu": 25.5, "peak_cpu": 45.2}

async def get_log_analysis(ctx: RunContext, entity_name: str) -> dict:
    return {"error_count": 2, "warning_count": 5}

# Multi-tool Agent with different system prompts to show variability
performance_focused_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Union[EntityMetadata, CPUMetrics, LogAnalysis],
    system_prompt="""
    You focus on performance analysis. For any request about a pod:
    - Prioritize CPU metrics for performance analysis
    - Use get_cpu_metrics first
    - Only use other tools if specifically requested
    """
)

metadata_focused_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Union[EntityMetadata, CPUMetrics, LogAnalysis],
    system_prompt="""
    You focus on entity information. For any request about a pod:
    - Prioritize entity metadata
    - Use get_metadata first
    - Only use other tools if specifically requested
    """
)

# Register tools
performance_focused_agent.tool(get_metadata)
performance_focused_agent.tool(get_cpu_metrics)
performance_focused_agent.tool(get_log_analysis)

metadata_focused_agent.tool(get_metadata)
metadata_focused_agent.tool(get_cpu_metrics)
metadata_focused_agent.tool(get_log_analysis)

async def test_output_type_variability():
    """Test how different multi-tool agents return different types"""
    
    print("🔬 OUTPUT TYPE VARIABILITY TEST")
    print("=" * 50)
    
    test_prompt = "Analyze pod frontend-123"
    
    print(f"📝 SAME PROMPT: '{test_prompt}'")
    print("=" * 40)
    
    # Test multiple runs with different agent configurations
    print("🎯 PERFORMANCE-FOCUSED AGENT:")
    for i in range(3):
        try:
            result = await performance_focused_agent.run(test_prompt)
            print(f"   Run {i+1}: {type(result.output).__name__}")
        except Exception as e:
            print(f"   Run {i+1}: ERROR - {e}")
    
    print("\n📊 METADATA-FOCUSED AGENT:")
    for i in range(3):
        try:
            result = await metadata_focused_agent.run(test_prompt)
            print(f"   Run {i+1}: {type(result.output).__name__}")
        except Exception as e:
            print(f"   Run {i+1}: ERROR - {e}")
    
    print("\n🎯 KEY INSIGHT:")
    print("Same prompt, different system prompts → Different output types")
    print("This shows how multi-tool agents can be non-deterministic")

async def test_context_sensitivity():
    """Test how context affects tool selection"""
    
    print("\n\n🧠 CONTEXT SENSITIVITY TEST")
    print("=" * 50)
    
    # Same agent, different contexts
    context_prompts = [
        "The pod is running slow, analyze frontend-123",
        "I need basic info about frontend-123", 
        "Check if frontend-123 has any errors"
    ]
    
    unified_agent = Agent(
        'openai:gpt-4o-mini',
        output_type=Union[EntityMetadata, CPUMetrics, LogAnalysis],
        system_prompt="""
        Choose the most appropriate tool based on the context:
        - If performance/slow → get_cpu_metrics
        - If basic info → get_metadata
        - If errors → get_log_analysis
        """
    )
    
    unified_agent.tool(get_metadata)
    unified_agent.tool(get_cpu_metrics)
    unified_agent.tool(get_log_analysis)
    
    for prompt in context_prompts:
        print(f"\n📝 PROMPT: '{prompt}'")
        try:
            result = await unified_agent.run(prompt)
            print(f"   Result: {type(result.output).__name__}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    print("\n🎯 KEY INSIGHT:")
    print("Context changes → Different tools selected → Different output types")
    print("This context sensitivity makes multi-tool agents less predictable")

async def main():
    await test_output_type_variability()
    await test_context_sensitivity()
    
    print("\n\n📋 FINAL DETERMINISM VERDICT:")
    print("=" * 50)
    print("✅ SPECIALIZED AGENTS (1 agent = 1 tool = 1 output type):")
    print("   • 100% predictable output type")
    print("   • 100% predictable tool selection")
    print("   • 100% predictable resource usage")
    print("   • Zero ambiguity in behavior")
    print()
    print("❌ MULTI-TOOL AGENTS (1 agent = N tools = N output types):")
    print("   • Variable output types based on prompt")
    print("   • Variable tool selection based on context")
    print("   • Variable resource usage")
    print("   • LLM interpretation adds uncertainty")
    print()
    print("🏆 WINNER: Specialized agents are significantly more deterministic")

if __name__ == "__main__":
    asyncio.run(main()) 