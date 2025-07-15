"""
Kubernetes Observability Agent using PydanticAI
Converts mock_data.py functions into agent tools for workflow orchestration
"""

import asyncio
from datetime import datetime
from typing import Any, List, Dict, Optional
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Import the mock data functions
from mock_data import get_entities as mock_get_entities
from mock_data import get_cpu_utilization as mock_get_cpu_utilization
from mock_data import get_logs as mock_get_logs

# Configure logfire
logfire.configure(
    send_to_logfire='if-token-present',
    console=logfire.ConsoleOptions(min_log_level='info'),
)
logfire.instrument_pydantic_ai()

# =============================================
# Pydantic Models for Structured Data
# =============================================

class EntityMetadata(BaseModel):
    """Structured metadata for K8s entities"""
    entity_type: str
    name: str
    metadata: Dict[str, Any]

class CPUDataPoint(BaseModel):
    """Single CPU utilization data point"""
    timestamp: str
    cpu_percent: float

class CPUUtilizationResponse(BaseModel):
    """CPU utilization time series response"""
    entity_type: str
    entity_name: str
    start_time: str
    end_time: str
    data_points: List[CPUDataPoint]

class LogEvent(BaseModel):
    """Kubernetes log event"""
    type: str
    reason: str
    message: str
    timestamp: str

class LogsResponse(BaseModel):
    """Logs query response"""
    entity_type: str
    entity_name: str
    start_time: str
    end_time: str
    events: List[LogEvent]

class ObservabilityAnalysis(BaseModel):
    """Analysis result from observability data"""
    entity_name: str
    entity_type: str
    status: str = Field(description="healthy, warning, critical")
    summary: str
    recommendations: List[str]
    metrics_analyzed: bool = False
    logs_analyzed: bool = False

# =============================================
# Dependencies for Agent Context
# =============================================

@dataclass
class K8sObservabilityDeps:
    """Dependencies for K8s observability operations"""
    cluster_name: str = "production-cluster"
    monitoring_enabled: bool = True
    user_permissions: List[str] = None
    
    def __post_init__(self):
        if self.user_permissions is None:
            self.user_permissions = ["read:pods", "read:nodes", "read:metrics", "read:logs"]

# =============================================
# Kubernetes Observability Agent
# =============================================

k8s_agent = Agent(
    'openai:gpt-4o',
    result_type=ObservabilityAnalysis,
    system_prompt="""You are a Kubernetes observability expert agent.

Your role is to:
1. Retrieve entity metadata for pods and nodes
2. Analyze CPU utilization metrics over time
3. Examine logs for issues and patterns
4. Provide comprehensive analysis with actionable recommendations

Available tools:
- get_k8s_entity: Get metadata for pods or nodes
- get_cpu_metrics: Retrieve CPU utilization time series
- get_k8s_logs: Fetch logs for troubleshooting

Analysis guidelines:
- CPU > 80% = warning, CPU > 95% = critical
- Look for error/warning patterns in logs
- Correlate metrics with log events
- Provide specific, actionable recommendations
- Consider entity relationships (pod -> node)

Always provide a structured analysis with clear status and recommendations.
""",
    deps_type=K8sObservabilityDeps,
)

@k8s_agent.tool
async def get_k8s_entity(
    ctx: RunContext[K8sObservabilityDeps],
    entity_type: str,
    entity_name: str,
) -> EntityMetadata:
    """
    Retrieve metadata for a Kubernetes entity (pod or node).
    
    Args:
        entity_type: Type of entity ('k8s:pod' or 'k8s:node')
        entity_name: Name of the specific entity
    
    Returns:
        EntityMetadata with complete entity information
    """
    with logfire.span('get_k8s_entity', entity_type=entity_type, entity_name=entity_name):
        try:
            # Check permissions
            required_perm = f"read:{entity_type.split(':')[1]}s"
            if required_perm not in ctx.deps.user_permissions:
                raise PermissionError(f"Missing permission: {required_perm}")
            
            # Call the mock function
            result = mock_get_entities(entity_type, entity_name)
            
            logfire.info('Entity retrieved successfully', 
                        entity=entity_name, 
                        type=entity_type)
            
            return EntityMetadata(**result)
            
        except Exception as e:
            logfire.error('Failed to retrieve entity', 
                         entity=entity_name, 
                         error=str(e))
            raise

@k8s_agent.tool
async def get_cpu_metrics(
    ctx: RunContext[K8sObservabilityDeps],
    entity_type: str,
    entity_name: str,
    start_time: str,
    end_time: str,
) -> CPUUtilizationResponse:
    """
    Retrieve CPU utilization metrics for a Kubernetes entity over a time range.
    
    Args:
        entity_type: Type of entity ('k8s:pod' or 'k8s:node')
        entity_name: Name of the specific entity
        start_time: Start time in ISO format (e.g., '2024-06-26T09:00:00+00:00')
        end_time: End time in ISO format
    
    Returns:
        CPUUtilizationResponse with time series data
    """
    with logfire.span('get_cpu_metrics', 
                     entity_type=entity_type, 
                     entity_name=entity_name,
                     start_time=start_time,
                     end_time=end_time):
        try:
            # Check permissions
            if "read:metrics" not in ctx.deps.user_permissions:
                raise PermissionError("Missing permission: read:metrics")
            
            # Call the mock function
            data_points = mock_get_cpu_utilization(entity_type, entity_name, start_time, end_time)
            
            # Convert to structured format
            cpu_points = [CPUDataPoint(**point) for point in data_points]
            
            response = CPUUtilizationResponse(
                entity_type=entity_type,
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time,
                data_points=cpu_points
            )
            
            avg_cpu = sum(point.cpu_percent for point in cpu_points) / len(cpu_points)
            max_cpu = max(point.cpu_percent for point in cpu_points)
            
            logfire.info('CPU metrics retrieved',
                        entity=entity_name,
                        avg_cpu=avg_cpu,
                        max_cpu=max_cpu,
                        data_points=len(cpu_points))
            
            return response
            
        except Exception as e:
            logfire.error('Failed to retrieve CPU metrics',
                         entity=entity_name,
                         error=str(e))
            raise

@k8s_agent.tool
async def get_k8s_logs(
    ctx: RunContext[K8sObservabilityDeps],
    entity_type: str,
    entity_name: str,
    start_time: str,
    end_time: str,
) -> LogsResponse:
    """
    Retrieve logs for a Kubernetes entity over a time range.
    
    Args:
        entity_type: Type of entity ('k8s:pod' or 'k8s:node')
        entity_name: Name of the specific entity
        start_time: Start time in ISO format
        end_time: End time in ISO format
    
    Returns:
        LogsResponse with filtered log events
    """
    with logfire.span('get_k8s_logs',
                     entity_type=entity_type,
                     entity_name=entity_name,
                     start_time=start_time,
                     end_time=end_time):
        try:
            # Check permissions
            if "read:logs" not in ctx.deps.user_permissions:
                raise PermissionError("Missing permission: read:logs")
            
            # Call the mock function
            events = mock_get_logs(entity_type, entity_name, start_time, end_time)
            
            # Convert to structured format
            log_events = [LogEvent(**event) for event in events]
            
            response = LogsResponse(
                entity_type=entity_type,
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time,
                events=log_events
            )
            
            warning_count = sum(1 for event in log_events if event.type == "Warning")
            error_count = sum(1 for event in log_events if event.type == "Error")
            
            logfire.info('Logs retrieved',
                        entity=entity_name,
                        total_events=len(log_events),
                        warnings=warning_count,
                        errors=error_count)
            
            return response
            
        except Exception as e:
            logfire.error('Failed to retrieve logs',
                         entity=entity_name,
                         error=str(e))
            raise

# =============================================
# Workflow Orchestration Functions
# =============================================

async def analyze_pod_health(pod_name: str, time_range: tuple[str, str]) -> ObservabilityAnalysis:
    """
    Comprehensive pod health analysis workflow
    """
    deps = K8sObservabilityDeps(
        cluster_name="production-cluster",
        monitoring_enabled=True
    )
    
    start_time, end_time = time_range
    
    with logfire.span('analyze_pod_health', pod_name=pod_name):
        prompt = f"""
        Analyze the health of pod '{pod_name}' from {start_time} to {end_time}.
        
        Please:
        1. Get the pod metadata first
        2. Retrieve CPU utilization metrics for the time range
        3. Fetch logs for the same period
        4. Provide a comprehensive analysis with recommendations
        
        Focus on identifying any performance issues, errors, or anomalies.
        """
        
        result = await k8s_agent.run(prompt, deps=deps)
        return result.data

async def analyze_node_health(node_name: str, time_range: tuple[str, str]) -> ObservabilityAnalysis:
    """
    Comprehensive node health analysis workflow
    """
    deps = K8sObservabilityDeps(
        cluster_name="production-cluster",
        monitoring_enabled=True
    )
    
    start_time, end_time = time_range
    
    with logfire.span('analyze_node_health', node_name=node_name):
        prompt = f"""
        Analyze the health of node '{node_name}' from {start_time} to {end_time}.
        
        Please:
        1. Get the node metadata first
        2. Retrieve CPU utilization metrics for the time range
        3. Fetch logs for the same period
        4. Provide a comprehensive analysis with recommendations
        
        Pay special attention to resource pressure and node conditions.
        """
        
        result = await k8s_agent.run(prompt, deps=deps)
        return result.data

async def comparative_analysis(entities: List[tuple[str, str]], time_range: tuple[str, str]) -> str:
    """
    Compare multiple entities (pods/nodes) for the same time period
    """
    deps = K8sObservabilityDeps()
    start_time, end_time = time_range
    
    entity_list = ", ".join([f"{etype}:{ename}" for etype, ename in entities])
    
    with logfire.span('comparative_analysis', entities=entity_list):
        prompt = f"""
        Perform a comparative analysis of these entities from {start_time} to {end_time}:
        {entity_list}
        
        For each entity:
        1. Get metadata and metrics
        2. Analyze performance patterns
        3. Compare CPU utilization across entities
        4. Identify any correlations or patterns
        
        Provide insights on relative performance and any systemic issues.
        """
        
        result = await k8s_agent.run(prompt, deps=deps)
        return result.data.summary

# =============================================
# Interactive CLI
# =============================================

async def main():
    """Interactive demo of the K8s observability agent"""
    
    print("🔍 Kubernetes Observability Agent Demo")
    print("=" * 50)
    
    # Demo scenarios
    scenarios = {
        "1": {
            "name": "Pod Health Analysis (Normal Period)",
            "func": analyze_pod_health,
            "args": ("frontend-6d8f4f79f7-kxzpl", ("2024-06-26T09:00:00+00:00", "2024-06-26T10:00:00+00:00"))
        },
        "2": {
            "name": "Pod Health Analysis (High CPU Period)", 
            "func": analyze_pod_health,
            "args": ("frontend-6d8f4f79f7-kxzpl", ("2024-06-26T10:00:00+00:00", "2024-06-26T10:30:00+00:00"))
        },
        "3": {
            "name": "Node Health Analysis (Normal Period)",
            "func": analyze_node_health,
            "args": ("node-1", ("2024-06-26T09:00:00+00:00", "2024-06-26T10:00:00+00:00"))
        },
        "4": {
            "name": "Node Health Analysis (High CPU Period)",
            "func": analyze_node_health,
            "args": ("node-1", ("2024-06-26T10:00:00+00:00", "2024-06-26T10:30:00+00:00"))
        }
    }
    
    for key, scenario in scenarios.items():
        print(f"{key}. {scenario['name']}")
    
    choice = input("\nSelect a scenario (1-4): ").strip()
    
    if choice in scenarios:
        scenario = scenarios[choice]
        print(f"\n🚀 Running: {scenario['name']}")
        print("-" * 40)
        
        try:
            result = await scenario['func'](*scenario['args'])
            
            print(f"\n📊 Analysis Results:")
            print(f"Entity: {result.entity_name} ({result.entity_type})")
            print(f"Status: {result.status}")
            print(f"Summary: {result.summary}")
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
            print(f"\nMetrics Analyzed: {result.metrics_analyzed}")
            print(f"Logs Analyzed: {result.logs_analyzed}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    asyncio.run(main()) 