"""
Kubernetes Anomaly Detection Workflow using PydanticAI

This implementation follows PydanticAI best practices with:
- Single agent with output functions for different analysis steps
- Enhanced structured outputs with Union types and better validation
- Cleaner dependency injection and context management
- Optimized tool definitions with reduced duplication
- Better error handling with ModelRetry patterns
- Modern async workflow orchestration
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import logfire
from pydantic import BaseModel, Field
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
# Enhanced Data Models with Better Validation
# =============================================

class EntityType(str, Enum):
    """Supported Kubernetes entity types"""
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
    entity_type: str
    """The type of Kubernetes entity (e.g., k8s:pod, k8s:node)"""
    name: str
    """The name of the entity"""
    metadata: Dict[str, Any]
    """Additional metadata including labels, annotations, and runtime info"""

class CPUMetrics(BaseModel, use_attribute_docstrings=True):
    """CPU utilization metrics with statistical analysis"""
    entity_type: str
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
    entity_type: str
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
    entity_type: str
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

# =============================================
# Analysis Failure Models
# =============================================

class AnalysisFailure(BaseModel, use_attribute_docstrings=True):
    """Represents an analysis failure with context"""
    error_type: str
    """Type of error that occurred"""
    explanation: str
    """Detailed explanation of the failure"""
    entity_name: str
    """Entity that failed analysis"""
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
    
    def normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to include k8s: prefix"""
        if entity_type in ["pod", "node"]:
            return f"k8s:{entity_type}"
        return entity_type

# =============================================
# Single Agent with Output Functions
# =============================================

# Define output functions for different analysis steps
def get_entity_metadata_func(entity_type: str, entity_name: str) -> EntityMetadata:
    """Retrieve metadata for a Kubernetes entity"""
    try:
        # Normalize entity_type
        if entity_type in ["pod", "node"]:
            entity_type = f"k8s:{entity_type}"
        
        result = mock_get_entities(entity_type, entity_name)
        return EntityMetadata(**result)
    except Exception as e:
        raise ModelRetry(f"Failed to retrieve metadata for {entity_name}: {str(e)}")

def analyze_time_range_func(
    entity_type: str,
    entity_name: str,
    start_time: str,
    end_time: str,
    analysis_type: str
) -> TimeRangeAnalysis:
    """Analyze CPU metrics and logs for a specific time range"""
    try:
        # Normalize entity_type
        if entity_type in ["pod", "node"]:
            entity_type = f"k8s:{entity_type}"
        
        # Get CPU metrics
        cpu_data = mock_get_cpu_utilization(entity_type, entity_name, start_time, end_time)
        if not cpu_data:
            raise ModelRetry(f"No CPU data available for {entity_name}")
        
        cpu_values = [point['cpu_percent'] for point in cpu_data]
        cpu_metrics = CPUMetrics(
            entity_type=entity_type,
            entity_name=entity_name,
            start_time=start_time,
            end_time=end_time,
            data_points=cpu_data,
            avg_cpu=round(sum(cpu_values) / len(cpu_values), 2),
            max_cpu=round(max(cpu_values), 2),
            min_cpu=round(min(cpu_values), 2)
        )
        
        # Get log analysis
        log_events = mock_get_logs(entity_type, entity_name, start_time, end_time)
        warning_count = sum(1 for event in log_events if event.get('type') == 'Warning')
        error_count = sum(1 for event in log_events if event.get('type') == 'Error')
        critical_events = [
            f"{event.get('reason', 'Unknown')}: {event.get('message', 'No message')}"
            for event in log_events if event.get('type') in ['Warning', 'Error']
        ]
        
        log_analysis = LogAnalysis(
            entity_type=entity_type,
            entity_name=entity_name,
            start_time=start_time,
            end_time=end_time,
            events=log_events,
            warning_count=warning_count,
            error_count=error_count,
            critical_events=critical_events
        )
        
        # Determine status
        if cpu_metrics.avg_cpu > 80 or error_count > 0:
            status = AnalysisStatus.CRITICAL
        elif cpu_metrics.avg_cpu > 60 or warning_count > 2:
            status = AnalysisStatus.WARNING
        else:
            status = AnalysisStatus.HEALTHY
        
        # Generate key findings
        key_findings = []
        if cpu_metrics.avg_cpu > 70:
            key_findings.append(f"High CPU utilization: {cpu_metrics.avg_cpu}%")
        if warning_count > 0:
            key_findings.append(f"Found {warning_count} warnings")
        if error_count > 0:
            key_findings.append(f"Found {error_count} errors")
        if not key_findings:
            key_findings.append("No significant issues detected")
        
        summary = f"{analysis_type.title()} period analysis: {status.value} status with {cpu_metrics.avg_cpu}% avg CPU"
        
        return TimeRangeAnalysis(
            time_range_type=analysis_type,
            cpu_metrics=cpu_metrics,
            log_analysis=log_analysis,
            summary=summary,
            status=status,
            key_findings=key_findings
        )
    except Exception as e:
        raise ModelRetry(f"Failed to analyze {analysis_type} period for {entity_name}: {str(e)}")

# Main anomaly detection agent with Union output types
anomaly_detection_agent = Agent(
    'openai:gpt-4o',
    deps_type=K8sAnalysisDeps,
    output_type=Union[AnomalyDetectionResult, AnalysisFailure],
    system_prompt="""You are an expert Kubernetes anomaly detection system.

Your task is to analyze Kubernetes entities (pods/nodes) for anomalies by comparing baseline and anomaly periods.

Key responsibilities:
1. Retrieve entity metadata
2. Analyze baseline and anomaly time periods
3. Compare metrics and logs between periods
4. Detect anomalies and assess severity
5. Provide actionable recommendations

Use the available tools to gather data and perform comprehensive analysis.
Always provide confidence scores and clear explanations.""",
)

# =============================================
# Enhanced Tool Definitions
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
        entity_type: Type of entity ('pod', 'node', 'k8s:pod', or 'k8s:node')
        entity_name: Name of the specific entity
    
    Returns:
        Complete entity metadata including labels, annotations, and runtime info
    """
    with logfire.span('get_entity_metadata', entity_type=entity_type, entity_name=entity_name):
        try:
            # Normalize and validate entity type
            normalized_type = ctx.deps.normalize_entity_type(entity_type)
            entity_parts = normalized_type.split(':')
            
            if len(entity_parts) != 2 or entity_parts[0] != 'k8s':
                raise ModelRetry(f"Invalid entity_type: {entity_type}. Use 'pod', 'node', 'k8s:pod', or 'k8s:node'")
            
            # Check permissions
            required_perm = f"read:{entity_parts[1]}s"
            if not ctx.deps.has_permission(required_perm):
                raise ModelRetry(f"Missing permission: {required_perm}")
            
            result = get_entity_metadata_func(normalized_type, entity_name)
            
            logfire.info('Entity metadata retrieved successfully', 
                        entity=entity_name, 
                        type=normalized_type)
            
            return result
            
        except Exception as e:
            logfire.error('Failed to retrieve entity metadata', 
                         entity=entity_name, 
                         error=str(e))
            raise

@anomaly_detection_agent.tool
async def analyze_time_period(
    ctx: RunContext[K8sAnalysisDeps],
    entity_type: str,
    entity_name: str,
    start_time: str,
    end_time: str,
    analysis_type: str,
) -> TimeRangeAnalysis:
    """
    Perform comprehensive analysis of CPU metrics and logs for a time period.
    
    Args:
        entity_type: Type of entity ('pod', 'node', 'k8s:pod', or 'k8s:node')
        entity_name: Name of the specific entity
        start_time: Start time in ISO format
        end_time: End time in ISO format
        analysis_type: Type of analysis ('baseline' or 'anomaly')
    
    Returns:
        Complete time range analysis with metrics, logs, and assessment
    """
    with logfire.span('analyze_time_period', 
                     entity_type=entity_type, 
                     entity_name=entity_name,
                     analysis_type=analysis_type):
        try:
            # Validate permissions
            if not ctx.deps.has_permission("read:metrics"):
                raise ModelRetry("Missing permission: read:metrics")
            if not ctx.deps.has_permission("read:logs"):
                raise ModelRetry("Missing permission: read:logs")
            
            # Normalize entity type
            normalized_type = ctx.deps.normalize_entity_type(entity_type)
            
            result = analyze_time_range_func(
                normalized_type, entity_name, start_time, end_time, analysis_type
            )
            
            logfire.info('Time period analysis completed',
                        entity=entity_name,
                        analysis_type=analysis_type,
                        status=result.status,
                        avg_cpu=result.cpu_metrics.avg_cpu)
            
            return result
            
        except Exception as e:
            logfire.error('Failed to analyze time period',
                         entity=entity_name,
                         analysis_type=analysis_type,
                         error=str(e))
            raise

# =============================================
# Streamlined Workflow Functions
# =============================================

async def run_anomaly_detection_workflow(
    entity_type: str,
    entity_name: str,
    anomaly_time_range_start: str,
    anomaly_time_range_end: str,
    baseline_time_range_start: str,
    baseline_time_range_end: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[AnomalyDetectionResult, AnalysisFailure]:
    """
    Execute the complete anomaly detection workflow using a single agent approach
    """
    
    if context is None:
        context = {}
    
    deps = K8sAnalysisDeps(
        cluster_name="production-cluster",
        monitoring_enabled=True,
        analysis_context=context
    )
    
    workflow_prompt = f"""
    Perform comprehensive anomaly detection for {entity_type} '{entity_name}':
    
    ANALYSIS PERIODS:
    - Baseline: {baseline_time_range_start} to {baseline_time_range_end}
    - Anomaly: {anomaly_time_range_start} to {anomaly_time_range_end}
    
    REQUIRED STEPS:
    1. Get entity metadata using get_entity_metadata tool
    2. Analyze baseline period using analyze_time_period tool
    3. Analyze anomaly period using analyze_time_period tool
    4. If entity is a pod, also analyze the host node for both periods
    5. Compare all results and detect anomalies
    6. Provide confidence score and recommendations
    
    ANOMALY DETECTION CRITERIA:
    - CPU increase >20% indicates potential issues
    - New error patterns in anomaly period are concerning
    - Consider correlation between metrics and logs
    - Factor in node-level changes for pod analysis
    
    CONTEXT: {context}
    
    Return a comprehensive AnomalyDetectionResult with all analysis data.
    """
    
    with logfire.span('anomaly_detection_workflow', 
                     entity_type=entity_type, 
                     entity_name=entity_name):
        
        logfire.info('Starting streamlined anomaly detection workflow',
                    entity=entity_name,
                    baseline_period=f"{baseline_time_range_start} to {baseline_time_range_end}",
                    anomaly_period=f"{anomaly_time_range_start} to {anomaly_time_range_end}")
        
        try:
            result = await anomaly_detection_agent.run(workflow_prompt, deps=deps)
            
            if isinstance(result.output, AnomalyDetectionResult):
                logfire.info('Anomaly detection completed successfully',
                            entity=entity_name,
                            anomaly_detected=result.output.anomaly_detected,
                            severity=result.output.severity,
                            confidence=result.output.confidence_score)
            else:
                logfire.warning('Anomaly detection failed',
                               entity=entity_name,
                               error_type=result.output.error_type)
            
            return result.output
            
        except Exception as e:
            logfire.error('Workflow execution failed', entity=entity_name, error=str(e))
            return AnalysisFailure(
                error_type="workflow_execution_error",
                explanation=f"Failed to execute anomaly detection workflow: {str(e)}",
                entity_name=entity_name,
                recovery_suggestions=[
                    "Check entity name and type are correct",
                    "Verify time ranges are valid",
                    "Ensure monitoring data is available",
                    "Check user permissions for metrics and logs"
                ]
            )

# =============================================
# Convenience Functions (Unchanged)
# =============================================

async def analyze_pod_anomaly(
    pod_name: str,
    anomaly_start: str,
    anomaly_end: str,
    baseline_start: str,
    baseline_end: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[AnomalyDetectionResult, AnalysisFailure]:
    """Analyze pod for anomalies (includes node analysis)"""
    return await run_anomaly_detection_workflow(
        entity_type="k8s:pod",
        entity_name=pod_name,
        anomaly_time_range_start=anomaly_start,
        anomaly_time_range_end=anomaly_end,
        baseline_time_range_start=baseline_start,
        baseline_time_range_end=baseline_end,
        context=context
    )

async def analyze_node_anomaly(
    node_name: str,
    anomaly_start: str,
    anomaly_end: str,
    baseline_start: str,
    baseline_end: str,
    context: Optional[Dict[str, Any]] = None
) -> Union[AnomalyDetectionResult, AnalysisFailure]:
    """Analyze node for anomalies"""
    return await run_anomaly_detection_workflow(
        entity_type="k8s:node",
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
    """Enhanced demo of the streamlined anomaly detection workflow"""
    
    print("🔍 Enhanced Kubernetes Anomaly Detection Workflow")
    print("=" * 60)
    print("Features:")
    print("✓ Single agent with output functions")
    print("✓ Enhanced structured outputs with Union types")
    print("✓ Improved dependency injection")
    print("✓ Better error handling with ModelRetry")
    print("✓ Streamlined workflow orchestration")
    print("=" * 60)
    
    scenarios = {
        "1": {
            "name": "Pod Anomaly Detection (Enhanced Analysis)",
            "func": analyze_pod_anomaly,
            "args": (
                "frontend-6d8f4f79f7-kxzpl",
                "2024-06-26T10:00:00+00:00",  # anomaly start
                "2024-06-26T10:30:00+00:00",  # anomaly end
                "2024-06-26T09:00:00+00:00",  # baseline start
                "2024-06-26T10:00:00+00:00",  # baseline end
                {"alert_id": "CPU-001", "severity": "high", "source": "prometheus"}
            )
        },
        "2": {
            "name": "Node Anomaly Detection (Enhanced Analysis)",
            "func": analyze_node_anomaly,
            "args": (
                "node-1",
                "2024-06-26T10:00:00+00:00",  # anomaly start
                "2024-06-26T10:30:00+00:00",  # anomaly end
                "2024-06-26T09:00:00+00:00",  # baseline start
                "2024-06-26T10:00:00+00:00",  # baseline end
                {"alert_id": "NODE-001", "cluster": "prod", "source": "node-exporter"}
            )
        }
    }
    
    for key, scenario in scenarios.items():
        print(f"{key}. {scenario['name']}")
    
    choice = input("\nSelect a scenario (1-2): ").strip()
    
    if choice in scenarios:
        scenario = scenarios[choice]
        print(f"\n🚀 Running: {scenario['name']}")
        print("-" * 50)
        
        try:
            result = await scenario['func'](*scenario['args'])
            
            if isinstance(result, AnomalyDetectionResult):
                print(f"\n📊 Anomaly Detection Results:")
                print(f"Entity: {result.entity_name} ({result.entity_type})")
                print(f"Anomaly Detected: {result.anomaly_detected}")
                print(f"Severity: {result.severity}")
                print(f"Confidence Score: {result.confidence_score:.2f}")
                
                print(f"\n📈 Baseline Analysis:")
                print(f"  Status: {result.baseline_analysis.status}")
                print(f"  CPU Avg: {result.baseline_analysis.cpu_metrics.avg_cpu}%")
                print(f"  Warnings: {result.baseline_analysis.log_analysis.warning_count}")
                
                print(f"\n🚨 Anomaly Period Analysis:")
                print(f"  Status: {result.anomaly_analysis.status}")
                print(f"  CPU Avg: {result.anomaly_analysis.cpu_metrics.avg_cpu}%")
                print(f"  Warnings: {result.anomaly_analysis.log_analysis.warning_count}")
                
                if result.node_analysis:
                    print(f"\n🖥️  Node Analysis:")
                    print(f"  Status: {result.node_analysis.status}")
                    print(f"  CPU Avg: {result.node_analysis.cpu_metrics.avg_cpu}%")
                
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(result.recommendations, 1):
                    print(f"  {i}. {rec}")
                
                print(f"\n📋 Summary:")
                print(f"  {result.comparison_summary}")
            
            elif isinstance(result, AnalysisFailure):
                print(f"\n❌ Analysis Failed:")
                print(f"Error Type: {result.error_type}")
                print(f"Explanation: {result.explanation}")
                print(f"Entity: {result.entity_name}")
                print(f"\n🔧 Recovery Suggestions:")
                for i, suggestion in enumerate(result.recovery_suggestions, 1):
                    print(f"  {i}. {suggestion}")
                
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    asyncio.run(main()) 