"""
Determinism Test: Specialized Agents vs Multi-Tool Agent
Demonstrates why specialized agents are more deterministic
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

# APPROACH 1: Specialized Agents (More Deterministic)
print("=" * 60)
print("APPROACH 1: SPECIALIZED AGENTS")
print("=" * 60)

metadata_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=EntityMetadata,
    system_prompt="You retrieve K8s metadata. Always use get_metadata tool."
)

cpu_agent = Agent(
    'openai:gpt-4o-mini', 
    output_type=CPUMetrics,
    system_prompt="You analyze CPU metrics. Always use get_cpu_metrics tool."
)

log_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=LogAnalysis,
    system_prompt="You analyze logs. Always use get_log_analysis tool."
)

# Register tools with specialized agents
metadata_agent.tool(get_metadata)
cpu_agent.tool(get_cpu_metrics)
log_agent.tool(get_log_analysis)

# APPROACH 2: Single Multi-Tool Agent (Less Deterministic)
print("\nAPPROACH 2: SINGLE MULTI-TOOL AGENT")
print("=" * 60)

unified_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Union[EntityMetadata, CPUMetrics, LogAnalysis],
    system_prompt="""
    You are a K8s analysis tool. Based on the request, choose the appropriate tool:
    - get_metadata for entity information
    - get_cpu_metrics for CPU analysis  
    - get_log_analysis for log analysis
    """
)

# Register all tools with unified agent
unified_agent.tool(get_metadata)
unified_agent.tool(get_cpu_metrics) 
unified_agent.tool(get_log_analysis)

async def test_determinism():
    """Test determinism of both approaches"""
    
    print("\n🧪 TESTING DETERMINISM")
    print("=" * 40)
    
    # Test same request multiple times
    test_requests = [
        "Get metadata for pod frontend-123",
        "Get metadata for pod frontend-123", 
        "Get metadata for pod frontend-123"
    ]
    
    print("\n1️⃣ SPECIALIZED AGENT RESULTS:")
    for i, request in enumerate(test_requests):
        try:
            result = await metadata_agent.run(request)
            print(f"   Run {i+1}: {type(result.output).__name__} - {result.output}")
        except Exception as e:
            print(f"   Run {i+1}: ERROR - {e}")
    
    print("\n2️⃣ MULTI-TOOL AGENT RESULTS:")
    for i, request in enumerate(test_requests):
        try:
            result = await unified_agent.run(request)
            print(f"   Run {i+1}: {type(result.output).__name__} - {result.output}")
        except Exception as e:
            print(f"   Run {i+1}: ERROR - {e}")
    
    print("\n📊 DETERMINISM ANALYSIS:")
    print("✅ Specialized Agent: Always returns EntityMetadata")
    print("❌ Multi-Tool Agent: Could return any of the 3 types")
    print("❌ Multi-Tool Agent: LLM must interpret and choose tools")
    print("✅ Specialized Agent: No tool selection ambiguity")

if __name__ == "__main__":
    asyncio.run(test_determinism()) 