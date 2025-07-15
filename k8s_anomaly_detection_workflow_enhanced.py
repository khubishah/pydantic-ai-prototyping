"""
Enhanced Kubernetes Anomaly Detection Workflow using PydanticAI

This implementation follows PydanticAI best practices with:
- Single agent with Union output types for flexible responses
- Enhanced structured outputs with better validation and documentation
- Enforced entity types (k8s:pod or k8s:node only)
- Separate CPU metrics and log analysis tools
- Parallel execution of baseline and anomaly timerange analysis
- Better error handling with ModelRetry patterns
- Streamlined dependency injection
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import logfire
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry

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
# Enhanced Data Models with Strict Validation
# =============================================

class EntityType(str, Enum):
    """Supported Kubernetes entity types - strictly enforced"""
    POD = "k8s:pod"
    NODE = "k8s:node"

class AnalysisStatus(str, Enum):
    """Analysis status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

class AnomalySeverity(str, Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EntityMetadata(BaseModel, use_attribute_docstrings=True):
    """Kubernetes entity metadata with comprehensive information"""
    entity_type: EntityType
    """The type of Kubernetes entity (must be k8s:pod or k8s:node)"""
    name: str
    """The name of the entity"""
    metadata: Dict[str, Any]
    """Additional metadata including labels, annotations, and runtime info"""
    
    @field_validator('entity_type')
    @classmethod
    def validate_entity_type(cls, v):
        if v not in [EntityType.POD, EntityType.NODE]:
            raise ValueError(f"Entity type must be {EntityType.POD} or {EntityType.NODE}, got: {v}")
        return v

class CPUMetrics(BaseModel, use_attribute_docstrings=True):
    """CPU utilization metrics with statistical analysis"""
    entity_type: EntityType
    """The entity type these metrics belong to"""
    entity_name: str
    """The entity name these metrics belong to"""
    start_time: str
    """Start time of the measurement period"""
    end_time: str
    """End time of the measurement period"""
    data_points: List[Dict[str, Any]]
    """Raw CPU data points with timestamps"""
    avg_cpu: float = Field(ge=0.0, le=100.0)
    """Average CPU utilization percentage"""
    max_cpu: float = Field(ge=0.0, le=100.0)
    """Maximum CPU utilization percentage"""
    min_cpu: float = Field(ge=0.0, le=100.0)
    """Minimum CPU utilization percentage"""

class LogAnalysis(BaseModel, use_attribute_docstrings=True):
    """Log analysis results with event categorization"""
    entity_type: EntityType
    """The entity type these logs belong to"""
    entity_name: str
    """The entity name these logs belong to"""
    start_time: str
    """Start time of the log analysis period"""
    end_time: str
    """End time of the log analysis period"""
    events: List[Dict[str, Any]]
    """All log events in the time period"""
    warning_count: int = Field(ge=0)
    """Number of warning-level events"""
    error_count: int = Field(ge=0)
    """Number of error-level events"""
    critical_events: List[str]
    """List of critical event descriptions"""

class TimeRangeAnalysis(BaseModel, use_attribute_docstrings=True):
    """Comprehensive analysis for a specific time range"""
    time_range_type: str
    """Type of analysis: 'baseline' or 'anomaly'"""
    cpu_metrics: CPUMetrics
    """CPU performance metrics for the time range"""
    log_analysis: LogAnalysis
    """Log analysis results for the time range"""
    summary: str
    """Human-readable summary of the analysis"""
    status: AnalysisStatus
    """Overall health status assessment"""
    key_findings: List[str]
    """Important observations and insights"""

class AnomalyDetectionResult(BaseModel, use_attribute_docstrings=True):
    """Final comprehensive anomaly detection result"""
    entity_name: str
    """Name of the analyzed entity"""
    entity_type: EntityType
    """Type of the analyzed entity"""
    entity_metadata: EntityMetadata
    """Complete entity metadata"""
    baseline_analysis: TimeRangeAnalysis
    """Analysis of the baseline period"""
    anomaly_analysis: TimeRangeAnalysis
    """Analysis of the anomaly period"""
    node_analysis: Optional[TimeRangeAnalysis] = None
    """Node analysis if entity is a pod"""
    comparison_summary: str
    """Summary comparing baseline vs anomaly periods"""
    anomaly_detected: bool
    """Whether an anomaly was detected"""
    severity: AnomalySeverity
    """Severity level of detected anomaly"""
    recommendations: List[str]
    """Actionable recommendations for investigation"""
    confidence_score: float = Field(ge=0.0, le=1.0)
    """Confidence score of the anomaly detection (0.0-1.0)"""

class AnalysisFailure(BaseModel, use_attribute_docstrings=True):
    """Represents an analysis failure with context"""
    error_type: str
    """Type of error that occurred"""
    explanation: str
    """Detailed explanation of the failure"""
    entity_name: str
    """Entity that failed analysis"""
    entity_type: Optional[str] = None
    """Entity type if known"""
    recovery_suggestions: List[str]
    """Suggestions for recovering from the failure"""

# =============================================
# Streamlined Dependencies
# =============================================

@dataclass
class K8sAnalysisDeps:
    """Streamlined dependencies for Kubernetes analysis"""
    cluster_name: str = "production-cluster"
    monitoring_enabled: bool = True
    user_permissions: List[str] = field(default_factory=lambda: [
        "read:pods", "read:nodes", "read:metrics", "read:logs"
    ])
    analysis_context: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.user_permissions
    
    def validate_entity_type(self, entity_type: str) -> EntityType:
        """Validate and return EntityType enum, raising ModelRetry for invalid types"""
        if entity_type == EntityType.POD or entity_type == "k8s:pod":
            return EntityType.POD
        elif entity_type == EntityType.NODE or entity_type == "k8s:node":
            return EntityType.NODE
        else:
            raise ModelRetry(
                f"Invalid entity_type: {entity_type}. Must be exactly '{EntityType.POD}' or '{EntityType.NODE}'"
            )

# =============================================
# Enhanced Anomaly Detection Agent
# =============================================

anomaly_detection_agent = Agent(
    'openai:gpt-4o',
    deps_type=K8sAnalysisDeps,
    output_type=Union[AnomalyDetectionResult, AnalysisFailure],
    system_prompt="""You are an expert Kubernetes anomaly detection system.

Your task is to analyze Kubernetes entities (pods/nodes) for anomalies by comparing baseline and anomaly periods.

IMPORTANT CONSTRAINTS:
- Entity types must be EXACTLY 'k8s:pod' or 'k8s:node' (no other formats accepted)
- Use separate tools for CPU metrics and log analysis
- CRITICAL TOOL ORDERING: Within each time period, ALWAYS call analyze_cpu_metrics BEFORE analyze_entity_logs
- Baseline and anomaly period analyses run in parallel, but each maintains sequential tool ordering

WORKFLOW EXECUTION PATTERN:
1. Get entity metadata using get_entity_metadata tool
2. Run two parallel branches (baseline and anomaly), each with sequential tool calls:
   
   BASELINE BRANCH (sequential):
   - FIRST: analyze_cpu_metrics for baseline period
   - THEN: analyze_entity_logs for baseline period
   
   ANOMALY BRANCH (parallel to baseline, but sequential internally):
   - FIRST: analyze_cpu_metrics for anomaly period  
   - THEN: analyze_entity_logs for anomaly period

3. If entity is a pod, repeat the same pattern for the host node
4. Compare all results and detect anomalies
5. Provide confidence score and actionable recommendations

ANOMALY DETECTION CRITERIA:
- CPU increase >20% indicates potential issues
- New error patterns in anomaly period are concerning
- Consider correlation between metrics and logs
- Factor in node-level changes for pod analysis

Always provide comprehensive analysis with confidence scores and clear explanations.""",
)

# =============================================
# Separate Tool Definitions
# =============================================

@anomaly_detection_agent.tool
async def get_entity_metadata(
    ctx: RunContext[K8sAnalysisDeps],
    entity_type: str,
    entity_name: str,
) -> EntityMetadata:
    """
    Retrieve comprehensive metadata for a Kubernetes entity.
    
    Args:
        entity_type: Must be exactly 'k8s:pod' or 'k8s:node'
        entity_name: Name of the specific entity
    
    Returns:
        Complete entity metadata including labels, annotations, and runtime info
    """
    with logfire.span('get_entity_metadata', entity_type=entity_type, entity_name=entity_name):
        try:
            # Strict validation of entity type
            validated_type = ctx.deps.validate_entity_type(entity_type)
            
            # Check permissions
            entity_part = validated_type.value.split(':')[1]  # Extract 'pod' or 'node'
            required_perm = f"read:{entity_part}s"
            if not ctx.deps.has_permission(required_perm):
                raise ModelRetry(f"Missing permission: {required_perm}")
            
            # Get metadata using mock function
            result = mock_get_entities(validated_type.value, entity_name)
            
            # Create EntityMetadata with validated type
            metadata = EntityMetadata(
                entity_type=validated_type,
                name=entity_name,
                metadata=result['metadata']
            )
            
            logfire.info('Entity metadata retrieved successfully', 
                        entity=entity_name, 
                        type=validated_type.value)
            
            return metadata
            
        except Exception as e:
            logfire.error('Failed to retrieve entity metadata', 
                         entity=entity_name, 
                         error=str(e))
            raise

@anomaly_detection_agent.tool
async def analyze_cpu_metrics(
    ctx: RunContext[K8sAnalysisDeps],
    entity_type: str,
    entity_name: str,
    start_time: str,
    end_time: str,
) -> CPUMetrics:
    """
    Analyze CPU utilization metrics for a specific time period.
    
    Args:
        entity_type: Must be exactly 'k8s:pod' or 'k8s:node'
        entity_name: Name of the specific entity
        start_time: Start time in ISO format
        end_time: End time in ISO format
    
    Returns:
        CPU metrics with statistical analysis
    """
    with logfire.span('analyze_cpu_metrics', 
                     entity_type=entity_type, 
                     entity_name=entity_name):
        try:
            # Strict validation of entity type
            validated_type = ctx.deps.validate_entity_type(entity_type)
            
            # Check permissions
            if not ctx.deps.has_permission("read:metrics"):
                raise ModelRetry("Missing permission: read:metrics")
            
            # Get raw CPU data
            cpu_data = mock_get_cpu_utilization(validated_type.value, entity_name, start_time, end_time)
            
            if not cpu_data:
                raise ModelRetry(f"No CPU metrics available for {entity_name} in time range {start_time} to {end_time}")
            
            # Calculate statistics
            cpu_values = [point['cpu_percent'] for point in cpu_data]
            avg_cpu = sum(cpu_values) / len(cpu_values)
            max_cpu = max(cpu_values)
            min_cpu = min(cpu_values)
            
            metrics = CPUMetrics(
                entity_type=validated_type,
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time,
                data_points=cpu_data,
                avg_cpu=round(avg_cpu, 2),
                max_cpu=round(max_cpu, 2),
                min_cpu=round(min_cpu, 2)
            )
            
            logfire.info('CPU metrics analyzed successfully',
                        entity=entity_name,
                        avg_cpu=avg_cpu,
                        max_cpu=max_cpu,
                        data_points=len(cpu_data))
            
            return metrics
            
        except Exception as e:
            logfire.error('Failed to analyze CPU metrics',
                         entity=entity_name,
                         error=str(e))
            raise

@anomaly_detection_agent.tool
async def analyze_entity_logs(
    ctx: RunContext[K8sAnalysisDeps],
    entity_type: str,
    entity_name: str,
    start_time: str,
    end_time: str,
) -> LogAnalysis:
    """
    Analyze logs for a Kubernetes entity over a specific time period.
    
    Args:
        entity_type: Must be exactly 'k8s:pod' or 'k8s:node'
        entity_name: Name of the specific entity
        start_time: Start time in ISO format
        end_time: End time in ISO format
    
    Returns:
        Log analysis with event categorization and critical events
    """
    with logfire.span('analyze_entity_logs',
                     entity_type=entity_type,
                     entity_name=entity_name):
        try:
            # Strict validation of entity type
            validated_type = ctx.deps.validate_entity_type(entity_type)
            
            # Check permissions
            if not ctx.deps.has_permission("read:logs"):
                raise ModelRetry("Missing permission: read:logs")
            
            # Get raw log events
            log_events = mock_get_logs(validated_type.value, entity_name, start_time, end_time)
            
            # Analyze log patterns
            warning_count = sum(1 for event in log_events if event.get('type') == 'Warning')
            error_count = sum(1 for event in log_events if event.get('type') == 'Error')
            
            # Identify critical events
            critical_events = [
                f"{event.get('reason', 'Unknown')}: {event.get('message', 'No message')}"
                for event in log_events if event.get('type') in ['Warning', 'Error']
            ]
            
            analysis = LogAnalysis(
                entity_type=validated_type,
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time,
                events=log_events,
                warning_count=warning_count,
                error_count=error_count,
                critical_events=critical_events
            )
            
            logfire.info('Log analysis completed successfully',
                        entity=entity_name,
                        total_events=len(log_events),
                        warnings=warning_count,
                        errors=error_count)
            
            return analysis
            
        except Exception as e:
            logfire.error('Failed to analyze entity logs',
                         entity=entity_name,
                         error=str(e))
            raise

# =============================================
# Enhanced Workflow Functions
# =============================================

async def run_enhanced_anomaly_detection_workflow(
    entity_type: str,
    entity_name: str,
    anomaly_time_range_start: str,
    anomaly_time_range_end: str,
    baseline_time_range_start: str,
    baseline_time_range_end: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[AnomalyDetectionResult, AnalysisFailure]:
    """
    Execute the enhanced anomaly detection workflow with parallel timerange analysis
    """
    
    if context is None:
        context = {}
    
    deps = K8sAnalysisDeps(
        cluster_name="production-cluster",
        monitoring_enabled=True,
        analysis_context=context
    )
    
    # Validate entity type upfront
    try:
        validated_entity_type = deps.validate_entity_type(entity_type)
    except ModelRetry as e:
        return AnalysisFailure(
            error_type="invalid_entity_type",
            explanation=str(e),
            entity_name=entity_name,
            entity_type=entity_type,
            recovery_suggestions=[
                f"Use exactly '{EntityType.POD}' for pod analysis",
                f"Use exactly '{EntityType.NODE}' for node analysis",
                "Check entity type spelling and format"
            ]
        )
    
    workflow_prompt = f"""
    Perform comprehensive anomaly detection for {validated_entity_type.value} '{entity_name}':
    
    ANALYSIS PERIODS:
    - Baseline: {baseline_time_range_start} to {baseline_time_range_end}
    - Anomaly: {anomaly_time_range_start} to {anomaly_time_range_end}
    
    REQUIRED STEPS (follow exactly):
    1. Get entity metadata using get_entity_metadata tool with entity_type='{validated_entity_type.value}'
    
    2. Run baseline and anomaly analysis IN PARALLEL, but within each analysis maintain this SEQUENTIAL order:
    
       BASELINE ANALYSIS (sequential within this branch):
       - FIRST: Call analyze_cpu_metrics for baseline period ({baseline_time_range_start} to {baseline_time_range_end})
       - THEN: Call analyze_entity_logs for baseline period ({baseline_time_range_start} to {baseline_time_range_end})
       - Combine into baseline TimeRangeAnalysis
       
       ANOMALY ANALYSIS (sequential within this branch, but parallel to baseline):
       - FIRST: Call analyze_cpu_metrics for anomaly period ({anomaly_time_range_start} to {anomaly_time_range_end})
       - THEN: Call analyze_entity_logs for anomaly period ({anomaly_time_range_start} to {anomaly_time_range_end})
       - Combine into anomaly TimeRangeAnalysis
       
       CRITICAL: CPU metrics MUST be analyzed before logs in each branch. The two branches (baseline vs anomaly) run in parallel.
    
    3. If entity is a pod (k8s:pod), analyze the host node:
       - Extract node name from pod metadata
       - Repeat step 2 for the node using the same parallel structure with sequential ordering within each branch
    
    4. Compare all results and detect anomalies:
       - CPU increase >20% indicates potential issues
       - New error patterns in anomaly period are concerning
       - Consider correlation between metrics and logs
       - Factor in node-level changes for pod analysis
    
    5. Provide comprehensive AnomalyDetectionResult with:
       - Confidence score (0.0-1.0)
       - Severity assessment
       - Actionable recommendations
    
    CONTEXT: {context}
    
    IMPORTANT: 
    - Use entity_type='{validated_entity_type.value}' in ALL tool calls
    - ALWAYS call analyze_cpu_metrics BEFORE analyze_entity_logs within each time period
    - Baseline and anomaly analyses run in parallel, but each maintains internal sequential order
    """
    
    with logfire.span('enhanced_anomaly_detection_workflow', 
                     entity_type=validated_entity_type.value, 
                     entity_name=entity_name):
        
        logfire.info('Starting enhanced anomaly detection workflow',
                    entity=entity_name,
                    entity_type=validated_entity_type.value,
                    baseline_period=f"{baseline_time_range_start} to {baseline_time_range_end}",
                    anomaly_period=f"{anomaly_time_range_start} to {anomaly_time_range_end}")
        
        try:
            result = await anomaly_detection_agent.run(workflow_prompt, deps=deps)
            
            if isinstance(result.output, AnomalyDetectionResult):
                logfire.info('Enhanced anomaly detection completed successfully',
                            entity=entity_name,
                            entity_type=validated_entity_type.value,
                            anomaly_detected=result.output.anomaly_detected,
                            severity=result.output.severity,
                            confidence=result.output.confidence_score)
            else:
                logfire.warning('Enhanced anomaly detection failed',
                               entity=entity_name,
                               entity_type=validated_entity_type.value,
                               error_type=result.output.error_type)
            
            return result.output
            
        except Exception as e:
            logfire.error('Enhanced workflow execution failed', 
                         entity=entity_name, 
                         entity_type=validated_entity_type.value,
                         error=str(e))
            return AnalysisFailure(
                error_type="workflow_execution_error",
                explanation=f"Failed to execute enhanced anomaly detection workflow: {str(e)}",
                entity_name=entity_name,
                entity_type=validated_entity_type.value,
                recovery_suggestions=[
                    "Check entity name exists in the cluster",
                    "Verify time ranges are valid and not in the future",
                    "Ensure monitoring data is available for the time periods",
                    "Check user permissions for metrics and logs access",
                    "Verify entity type is exactly 'k8s:pod' or 'k8s:node'"
                ]
            )

# =============================================
# Convenience Functions
# =============================================

async def analyze_pod_anomaly_enhanced(
    pod_name: str,
    anomaly_start: str,
    anomaly_end: str,
    baseline_start: str,
    baseline_end: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[AnomalyDetectionResult, AnalysisFailure]:
    """Analyze pod for anomalies with enhanced features (includes node analysis)"""
    return await run_enhanced_anomaly_detection_workflow(
        entity_type=EntityType.POD,
        entity_name=pod_name,
        anomaly_time_range_start=anomaly_start,
        anomaly_time_range_end=anomaly_end,
        baseline_time_range_start=baseline_start,
        baseline_time_range_end=baseline_end,
        context=context
    )

async def analyze_node_anomaly_enhanced(
    node_name: str,
    anomaly_start: str,
    anomaly_end: str,
    baseline_start: str,
    baseline_end: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[AnomalyDetectionResult, AnalysisFailure]:
    """Analyze node for anomalies with enhanced features"""
    return await run_enhanced_anomaly_detection_workflow(
        entity_type=EntityType.NODE,
        entity_name=node_name,
        anomaly_time_range_start=anomaly_start,
        anomaly_time_range_end=anomaly_end,
        baseline_time_range_start=baseline_start,
        baseline_time_range_end=baseline_end,
        context=context
    )

# =============================================
# Enhanced Interactive Demo
# =============================================

async def main():
    """Enhanced demo showcasing all improvements"""
    
    print("🚀 Enhanced Kubernetes Anomaly Detection Workflow")
    print("=" * 65)
    print("✅ NEW FEATURES:")
    print("  • Strict entity type validation (k8s:pod or k8s:node only)")
    print("  • Separate CPU metrics and log analysis tools")
    print("  • Parallel baseline and anomaly timerange analysis")
    print("  • Enhanced error handling with recovery suggestions")
    print("  • Union output types for flexible responses")
    print("  • Comprehensive structured outputs with validation")
    print("=" * 65)
    
    scenarios = {
        "1": {
            "name": "Pod Anomaly Detection (Enhanced with Parallel Analysis)",
            "func": analyze_pod_anomaly_enhanced,
            "args": (
                "frontend-6d8f4f79f7-kxzpl",
                "2024-06-26T10:00:00+00:00",  # anomaly start
                "2024-06-26T10:30:00+00:00",  # anomaly end
                "2024-06-26T09:00:00+00:00",  # baseline start
                "2024-06-26T10:00:00+00:00",  # baseline end
                {
                    "alert_id": "CPU-001", 
                    "severity": "high", 
                    "source": "prometheus",
                    "parallel_analysis": True
                }
            )
        },
        "2": {
            "name": "Node Anomaly Detection (Enhanced with Parallel Analysis)",
            "func": analyze_node_anomaly_enhanced,
            "args": (
                "node-1",
                "2024-06-26T10:00:00+00:00",  # anomaly start
                "2024-06-26T10:30:00+00:00",  # anomaly end
                "2024-06-26T09:00:00+00:00",  # baseline start
                "2024-06-26T10:00:00+00:00",  # baseline end
                {
                    "alert_id": "NODE-001", 
                    "cluster": "prod", 
                    "source": "node-exporter",
                    "parallel_analysis": True
                }
            )
        },
        "3": {
            "name": "Invalid Entity Type Test (Error Handling Demo)",
            "func": run_enhanced_anomaly_detection_workflow,
            "args": (
                "invalid-type",  # This will trigger error handling
                "test-entity",
                "2024-06-26T10:00:00+00:00",
                "2024-06-26T10:30:00+00:00",
                "2024-06-26T09:00:00+00:00",
                "2024-06-26T10:00:00+00:00",
                {"test": "error_handling"}
            )
        }
    }
    
    for key, scenario in scenarios.items():
        print(f"{key}. {scenario['name']}")
    
    choice = input("\nSelect a scenario (1-3): ").strip()
    
    if choice in scenarios:
        scenario = scenarios[choice]
        print(f"\n🚀 Running: {scenario['name']}")
        print("-" * 60)
        
        try:
            result = await scenario['func'](*scenario['args'])
            
            if isinstance(result, AnomalyDetectionResult):
                print(f"\n✅ ANOMALY DETECTION SUCCESS:")
                print(f"📊 Entity: {result.entity_name} ({result.entity_type})")
                print(f"🔍 Anomaly Detected: {result.anomaly_detected}")
                print(f"⚠️  Severity: {result.severity}")
                print(f"📈 Confidence Score: {result.confidence_score:.2f}")
                
                print(f"\n📊 BASELINE ANALYSIS:")
                print(f"  Status: {result.baseline_analysis.status}")
                print(f"  CPU Avg: {result.baseline_analysis.cpu_metrics.avg_cpu}%")
                print(f"  CPU Range: {result.baseline_analysis.cpu_metrics.min_cpu}% - {result.baseline_analysis.cpu_metrics.max_cpu}%")
                print(f"  Warnings: {result.baseline_analysis.log_analysis.warning_count}")
                print(f"  Errors: {result.baseline_analysis.log_analysis.error_count}")
                
                print(f"\n🚨 ANOMALY PERIOD ANALYSIS:")
                print(f"  Status: {result.anomaly_analysis.status}")
                print(f"  CPU Avg: {result.anomaly_analysis.cpu_metrics.avg_cpu}%")
                print(f"  CPU Range: {result.anomaly_analysis.cpu_metrics.min_cpu}% - {result.anomaly_analysis.cpu_metrics.max_cpu}%")
                print(f"  Warnings: {result.anomaly_analysis.log_analysis.warning_count}")
                print(f"  Errors: {result.anomaly_analysis.log_analysis.error_count}")
                
                if result.node_analysis:
                    print(f"\n🖥️  NODE ANALYSIS:")
                    print(f"  Status: {result.node_analysis.status}")
                    print(f"  CPU Avg: {result.node_analysis.cpu_metrics.avg_cpu}%")
                    print(f"  CPU Range: {result.node_analysis.cpu_metrics.min_cpu}% - {result.node_analysis.cpu_metrics.max_cpu}%")
                    print(f"  Warnings: {result.node_analysis.log_analysis.warning_count}")
                
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"  {i}. {rec}")
                
                print(f"\n📋 COMPARISON SUMMARY:")
                print(f"  {result.comparison_summary}")
                
                print(f"\n🔍 KEY FINDINGS:")
                print(f"  Baseline: {', '.join(result.baseline_analysis.key_findings)}")
                print(f"  Anomaly: {', '.join(result.anomaly_analysis.key_findings)}")
            
            elif isinstance(result, AnalysisFailure):
                print(f"\n❌ ANALYSIS FAILED:")
                print(f"🚫 Error Type: {result.error_type}")
                print(f"📝 Explanation: {result.explanation}")
                print(f"🎯 Entity: {result.entity_name}")
                if result.entity_type:
                    print(f"🏷️  Entity Type: {result.entity_type}")
                print(f"\n🔧 RECOVERY SUGGESTIONS:")
                for i, suggestion in enumerate(result.recovery_suggestions, 1):
                    print(f"  {i}. {suggestion}")
                
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
            print(f"🐛 This indicates a bug in the implementation")
    else:
        print("❌ Invalid choice! Please select 1, 2, or 3.")

if __name__ == "__main__":
    asyncio.run(main()) 