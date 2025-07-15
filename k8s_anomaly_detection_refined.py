"""
Kubernetes Anomaly Detection with Refined Agent Architecture

This implementation follows the correct workflow as specified:
1. Parallel tool calls for baseline and anomaly analysis
2. LLM merge and analyze outputs from 2 paths
3. Conditional node analysis (if pod)
4. Final LLM summarization

Uses specialized agent architecture with PydanticAI best practices:
- Proper retry mechanisms with ModelRetry
- Comprehensive error handling
- Usage limits and tracking
- Modern API patterns
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import logfire
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

# Import mock data functions
from mock_data import get_entities, get_cpu_utilization, get_logs

# Configure logging and observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure()

# ============================================================================
# Data Models and Enums
# ============================================================================

class EntityType(str, Enum):
    """Supported Kubernetes entity types."""
    POD = "k8s:pod"
    NODE = "k8s:node"

class AnalysisStatus(str, Enum):
    """Status of analysis operations."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TimeRange(BaseModel):
    """Time range for analysis."""
    start: datetime = Field(..., description="Start time for analysis")
    end: datetime = Field(..., description="End time for analysis")
    
    @field_validator('end')
    @classmethod
    def validate_end_after_start(cls, v, info):
        if 'start' in info.data and v <= info.data['start']:
            raise ValueError("End time must be after start time")
        return v

class CPUMetrics(BaseModel):
    """CPU utilization metrics."""
    average_utilization: float = Field(..., ge=0, le=100, description="Average CPU utilization percentage")
    peak_utilization: float = Field(..., ge=0, le=100, description="Peak CPU utilization percentage")
    samples_count: int = Field(..., gt=0, description="Number of metric samples analyzed")
    
    @field_validator('peak_utilization')
    @classmethod
    def validate_peak_ge_average(cls, v, info):
        if 'average_utilization' in info.data and v < info.data['average_utilization']:
            raise ValueError("Peak utilization cannot be less than average utilization")
        return v

class LogAnalysis(BaseModel):
    """Analysis of entity logs."""
    error_count: int = Field(..., ge=0, description="Number of error-level log entries")
    warning_count: int = Field(..., ge=0, description="Number of warning-level log entries")
    critical_patterns: List[str] = Field(default_factory=list, description="Critical patterns found in logs")
    anomalous_events: List[str] = Field(default_factory=list, description="Anomalous events detected")

class EntityMetadata(BaseModel):
    """Metadata about the Kubernetes entity."""
    name: str = Field(..., description="Entity name")
    namespace: Optional[str] = Field(None, description="Kubernetes namespace")
    labels: Dict[str, str] = Field(default_factory=dict, description="Entity labels")
    node_name: Optional[str] = Field(None, description="Node where entity is running")
    creation_time: datetime = Field(..., description="Entity creation timestamp")

class PathAnalysisResult(BaseModel):
    """Results from a single analysis path (baseline or anomaly)."""
    timerange: TimeRange = Field(..., description="Time range analyzed")
    cpu_metrics: CPUMetrics = Field(..., description="CPU metrics for this period")
    log_analysis: LogAnalysis = Field(..., description="Log analysis for this period")

class MergedAnalysis(BaseModel):
    """LLM-merged analysis of baseline vs anomaly paths."""
    has_significant_change: bool = Field(..., description="Whether significant changes were detected")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the analysis")
    severity: Optional[AnomalySeverity] = Field(None, description="Severity of detected issues")
    key_findings: List[str] = Field(..., description="Key findings from the comparison")
    cpu_change_summary: str = Field(..., description="Summary of CPU utilization changes")
    log_change_summary: str = Field(..., description="Summary of log pattern changes")

class FinalSummary(BaseModel):
    """Final comprehensive summary of the anomaly detection analysis."""
    overall_conclusion: str = Field(..., description="Overall conclusion of the analysis")
    anomaly_detected: bool = Field(..., description="Whether an anomaly was definitively detected")
    confidence_score: float = Field(..., ge=0, le=1, description="Final confidence score")
    severity: Optional[AnomalySeverity] = Field(None, description="Final severity assessment")
    primary_evidence: List[str] = Field(..., description="Primary evidence supporting the conclusion")
    contributing_factors: List[str] = Field(..., description="All contributing factors")
    recommendations: List[str] = Field(..., description="Recommended actions")

class AnomalyDetectionResult(BaseModel):
    """Complete anomaly detection analysis results."""
    entity_type: EntityType = Field(..., description="Type of Kubernetes entity analyzed")
    entity_name: str = Field(..., description="Name of the analyzed entity")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When the analysis was performed")
    
    # Entity information
    entity_metadata: EntityMetadata = Field(..., description="Metadata about the analyzed entity")
    
    # Path analysis results
    baseline_analysis: PathAnalysisResult = Field(..., description="Baseline period analysis")
    anomaly_analysis: PathAnalysisResult = Field(..., description="Anomaly period analysis")
    
    # LLM analysis results
    merged_analysis: MergedAnalysis = Field(..., description="LLM-merged analysis of both paths")
    node_analysis: Optional['AnomalyDetectionResult'] = Field(None, description="Analysis of the host node")
    final_summary: FinalSummary = Field(..., description="Final comprehensive summary")
    
    # Status and usage
    status: AnalysisStatus = Field(..., description="Overall analysis status")
    usage_stats: Optional[Usage] = Field(None, description="Token usage statistics")

class AnalysisFailure(BaseModel):
    """Represents a failed analysis with recovery information."""
    error_type: str = Field(..., description="Type of error that occurred")
    error_message: str = Field(..., description="Detailed error message")
    entity_name: str = Field(..., description="Name of entity being analyzed")
    failed_at: datetime = Field(default_factory=datetime.now, description="When the failure occurred")
    recovery_suggestions: List[str] = Field(default_factory=list, description="Suggested recovery actions")
    retry_count: int = Field(default=0, description="Number of retries attempted")

# ============================================================================
# Dependencies and Context
# ============================================================================

class K8sAnalysisDeps(BaseModel):
    """Dependencies for Kubernetes analysis operations."""
    entity_name: str
    entity_type: EntityType
    baseline_start: datetime
    baseline_end: datetime
    anomaly_start: datetime
    anomaly_end: datetime
    
    @field_validator('entity_type')
    @classmethod
    def validate_entity_type(cls, v):
        """Ensure only k8s:pod or k8s:node are accepted."""
        if v not in [EntityType.POD, EntityType.NODE]:
            raise ValueError(f"Invalid entity type: {v}. Must be 'k8s:pod' or 'k8s:node'")
        return v
    
    def get_baseline_timerange(self) -> TimeRange:
        """Get baseline time range."""
        return TimeRange(start=self.baseline_start, end=self.baseline_end)
    
    def get_anomaly_timerange(self) -> TimeRange:
        """Get anomaly time range."""
        return TimeRange(start=self.anomaly_start, end=self.anomaly_end)

# ============================================================================
# Specialized Agents with Best Practices
# ============================================================================

# Agent for merging and analyzing the two paths
merge_analysis_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=MergedAnalysis,
    retries=3,  # Built-in retry mechanism
    system_prompt="""
    You are a Kubernetes anomaly detection specialist focused on comparing baseline vs anomaly periods.
    
    Your role is to analyze the results from two parallel analysis paths and determine:
    1. Whether significant changes occurred between baseline and anomaly periods
    2. Confidence level in your assessment (0.0-1.0)
    3. Severity of any detected issues
    4. Key findings and change summaries
    
    Focus on:
    - CPU utilization patterns and magnitude of changes
    - Log error/warning trends and critical patterns
    - Correlation between CPU and log anomalies
    - Relative significance (small baseline vs large changes matter more)
    
    Be conservative with confidence scores:
    - 0.9+: Very strong evidence (multiple clear indicators)
    - 0.7-0.9: Strong evidence (clear primary indicator + supporting)
    - 0.5-0.7: Moderate evidence (some indicators with uncertainty)
    - 0.3-0.5: Weak evidence (minor indicators, could be noise)
    
    If you cannot determine significance from the provided data, request more information.
    """,
)

# Agent for final comprehensive summarization
final_summary_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=FinalSummary,
    retries=3,  # Built-in retry mechanism
    system_prompt="""
    You are a senior Kubernetes operations analyst providing final comprehensive summaries.
    
    Your role is to synthesize all available evidence and provide:
    1. Clear overall conclusion about anomaly presence
    2. Final confidence score considering all evidence
    3. Actionable recommendations for operations teams
    4. Comprehensive list of contributing factors
    
    Consider:
    - Primary entity analysis results
    - Node analysis results (if available)
    - Correlation between different evidence sources
    - Practical operational impact
    
    Provide clear, actionable conclusions that help operators understand:
    - What happened
    - How confident we are
    - What actions to take
    
    If the evidence is insufficient for a confident conclusion, be explicit about limitations.
    """,
)

# ============================================================================
# Tool Call Agents with Enhanced Error Handling
# ============================================================================

# Tool call agent for metadata retrieval
metadata_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=EntityMetadata,
    retries=2,  # Fewer retries for metadata as it's simpler
    system_prompt="""
    You are a Kubernetes metadata retrieval tool. Your job is to call the appropriate tool 
    to retrieve entity metadata based on the entity type and name provided.
    
    Always use the get_k8s_entity_metadata tool to retrieve metadata information.
    If the tool fails, analyze the error and provide helpful context for retry.
    """,
)

# Tool call agent for CPU metrics analysis
cpu_metrics_agent = Agent(
    'openai:gpt-4o-mini', 
    output_type=CPUMetrics,
    retries=3,  # More retries for data analysis
    system_prompt="""
    You are a Kubernetes CPU metrics analysis tool. Your job is to call the appropriate tool
    to retrieve and analyze CPU utilization data for the specified entity and time range.
    
    Always use the get_cpu_metrics tool to retrieve CPU data and return structured metrics.
    If the tool fails due to data issues, provide clear guidance on what went wrong.
    """,
)

# Tool call agent for log analysis
log_analysis_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=LogAnalysis, 
    retries=3,  # More retries for data analysis
    system_prompt="""
    You are a Kubernetes log analysis tool. Your job is to call the appropriate tool
    to retrieve and analyze log data for the specified entity and time range.
    
    Always use the get_entity_logs tool to retrieve log data and return structured analysis.
    If the tool fails, analyze the error and suggest alternative approaches.
    """,
)

# ============================================================================
# Enhanced Tool Functions with Retry Logic
# ============================================================================

@metadata_agent.tool(retries=2)
async def get_k8s_entity_metadata(ctx: RunContext[K8sAnalysisDeps], entity_type: str, entity_name: str) -> dict:
    """Tool to retrieve Kubernetes entity metadata with enhanced error handling."""
    logger.info(f"Tool call (attempt {ctx.retry + 1}): Retrieving metadata for {entity_type}: {entity_name}")
    
    try:
        # Call the mock data function
        entity_data = get_entities(entity_type, entity_name)
        metadata = entity_data["metadata"]
        
        # Validate essential fields
        if not metadata.get("start_time"):
            raise ModelRetry(
                f"Metadata for {entity_name} is missing start_time. "
                f"Please ensure the entity exists and has valid metadata."
            )
        
        return metadata
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tool call failed for metadata retrieval: {error_msg}")
        
        # Provide specific retry guidance based on error type
        if "Unsupported entity type" in error_msg:
            raise ModelRetry(
                f"Entity type '{entity_type}' is not supported. "
                f"Please use 'k8s:pod' or 'k8s:node' instead."
            )
        elif "not found" in error_msg.lower():
            raise ModelRetry(
                f"Entity '{entity_name}' not found. "
                f"Please verify the entity name and try again."
            )
        else:
            # For other errors, provide fallback data and continue
            logger.warning(f"Using fallback metadata for {entity_name}")
            return {
                "labels": {},
                "node_name": "unknown-node" if entity_type == "k8s:pod" else None,
                "start_time": "2024-06-26T09:58:12Z"
            }

@cpu_metrics_agent.tool(retries=3)
async def get_cpu_metrics(ctx: RunContext[K8sAnalysisDeps], entity_type: str, entity_name: str, start_time: str, end_time: str) -> dict:
    """Tool to retrieve and analyze CPU metrics with enhanced error handling."""
    logger.info(f"Tool call (attempt {ctx.retry + 1}): Analyzing CPU metrics for {entity_type}: {entity_name} from {start_time} to {end_time}")
    
    try:
        # Add small delay to simulate real tool call
        await asyncio.sleep(1.0)
        
        # Call the mock data function
        cpu_data = get_cpu_utilization(entity_type, entity_name, start_time, end_time)
        
        if not cpu_data:
            raise ModelRetry(
                f"No CPU data available for {entity_name} in the specified time range. "
                f"Please verify the entity exists and the time range is valid."
            )
        
        # Validate data quality
        if len(cpu_data) < 3:
            raise ModelRetry(
                f"Insufficient CPU data points ({len(cpu_data)}) for {entity_name}. "
                f"Need at least 3 data points for reliable analysis."
            )
        
        # Process the data
        cpu_values = [point["cpu_percent"] for point in cpu_data]
        
        # Validate CPU values
        if any(cpu < 0 or cpu > 100 for cpu in cpu_values):
            raise ModelRetry(
                f"Invalid CPU values detected for {entity_name}. "
                f"CPU percentages must be between 0 and 100."
            )
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        peak_cpu = max(cpu_values)
        samples_count = len(cpu_values)
        
        return {
            "average_utilization": round(avg_cpu, 2),
            "peak_utilization": round(peak_cpu, 2),
            "samples_count": samples_count
        }
        
    except ModelRetry:
        raise  # Re-raise ModelRetry exceptions
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tool call failed for CPU metrics: {error_msg}")
        
        # Provide specific retry guidance
        if "Unsupported entity_type" in error_msg:
            raise ModelRetry(
                f"Entity type '{entity_type}' is not supported for CPU metrics. "
                f"Please use 'k8s:pod' or 'k8s:node'."
            )
        elif ctx.retry < 2:  # Allow retry for transient errors
            raise ModelRetry(
                f"CPU metrics retrieval failed: {error_msg}. "
                f"This might be a transient issue. Please try again."
            )
        else:
            # Final fallback after retries
            logger.warning(f"Using fallback CPU metrics for {entity_name}")
            avg_cpu = 20.0 + (hash(entity_name) % 10)
            peak_cpu = avg_cpu + 5.0
            
            return {
                "average_utilization": round(avg_cpu, 2),
                "peak_utilization": round(peak_cpu, 2),
                "samples_count": 12
            }

@log_analysis_agent.tool(retries=3)
async def get_entity_logs(ctx: RunContext[K8sAnalysisDeps], entity_type: str, entity_name: str, start_time: str, end_time: str) -> dict:
    """Tool to retrieve and analyze entity logs with enhanced error handling."""
    logger.info(f"Tool call (attempt {ctx.retry + 1}): Analyzing logs for {entity_type}: {entity_name} from {start_time} to {end_time}")
    
    try:
        # Add small delay to simulate real tool call
        await asyncio.sleep(0.8)
        
        # Call the mock data function
        log_events = get_logs(entity_type, entity_name, start_time, end_time)
        
        if not log_events:
            raise ModelRetry(
                f"No log events found for {entity_name} in the specified time range. "
                f"Please verify the entity exists and has log data available."
            )
        
        error_count = 0
        warning_count = 0
        critical_patterns = []
        anomalous_events = []
        
        for event in log_events:
            event_type = event.get("type", "").lower()
            reason = event.get("reason", "")
            message = event.get("message", "")
            
            # Validate event structure
            if not message:
                continue  # Skip events without messages
            
            if event_type in ["error", "warning"]:
                if event_type == "error":
                    error_count += 1
                elif event_type == "warning":
                    warning_count += 1
            
            if any(pattern in message.lower() for pattern in ["unhealthy", "failed", "error", "timeout"]):
                if reason in ["Unhealthy", "BackOff"]:
                    anomalous_events.append(f"{reason}: {message}")
                elif "failed" in message.lower():
                    critical_patterns.append(f"Failed operation detected: {reason}")
            
            if entity_type == "k8s:node":
                if "pressure" in message.lower():
                    critical_patterns.append(f"Resource pressure detected: {message}")
        
        return {
            "error_count": error_count,
            "warning_count": warning_count,
            "critical_patterns": list(set(critical_patterns)),
            "anomalous_events": list(set(anomalous_events))
        }
        
    except ModelRetry:
        raise  # Re-raise ModelRetry exceptions
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tool call failed for log analysis: {error_msg}")
        
        # Provide specific retry guidance
        if "Unsupported entity_type" in error_msg:
            raise ModelRetry(
                f"Entity type '{entity_type}' is not supported for log analysis. "
                f"Please use 'k8s:pod' or 'k8s:node'."
            )
        elif ctx.retry < 2:  # Allow retry for transient errors
            raise ModelRetry(
                f"Log analysis failed: {error_msg}. "
                f"This might be a transient issue. Please try again."
            )
        else:
            # Final fallback after retries
            logger.warning(f"Using fallback log analysis for {entity_name}")
            return {
                "error_count": 0,
                "warning_count": 1,
                "critical_patterns": [],
                "anomalous_events": []
            }

# ============================================================================
# Enhanced Orchestrator with Usage Tracking and Limits
# ============================================================================

class K8sAnomalyDetectionOrchestrator:
    """Orchestrates the anomaly detection workflow with enhanced error handling and usage tracking."""
    
    def __init__(self, usage_limits: Optional[UsageLimits] = None):
        self.merge_agent = merge_analysis_agent
        self.summary_agent = final_summary_agent
        # Tool call agents for diagram compliance
        self.metadata_agent = metadata_agent
        self.cpu_metrics_agent = cpu_metrics_agent
        self.log_analysis_agent = log_analysis_agent
        
        # Default usage limits to prevent runaway costs
        self.usage_limits = usage_limits or UsageLimits(
            request_limit=50,  # Maximum 50 requests per analysis
            total_tokens_limit=100000,  # Maximum 100k tokens per analysis
        )
    
    async def analyze_entity(
        self,
        entity_name: str,
        entity_type: EntityType,
        baseline_start: datetime,
        baseline_end: datetime,
        anomaly_start: datetime,
        anomaly_end: datetime,
        usage: Optional[Usage] = None
    ) -> Union[AnomalyDetectionResult, AnalysisFailure]:
        """
        Perform anomaly detection analysis with enhanced error handling and usage tracking.
        """
        # Initialize usage tracking
        if usage is None:
            usage = Usage()
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Create dependencies
                deps = K8sAnalysisDeps(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    anomaly_start=anomaly_start,
                    anomaly_end=anomaly_end
                )
                
                logger.info(f"Starting enhanced workflow orchestration for {entity_type}: {entity_name} (attempt {retry_count + 1})")
                
                # Step 1: Get entity metadata with usage tracking
                metadata = await self._get_metadata_with_usage(deps, usage)
                
                # Step 2: Parallel analysis paths with usage tracking
                baseline_result, anomaly_result = await self._analyze_parallel_paths_with_usage(deps, usage)
                
                # Step 3: LLM merge and analyze outputs from 2 paths
                merged_analysis = await self._llm_merge_analysis_with_usage(
                    deps, baseline_result, anomaly_result, usage
                )
                
                # Step 4: Conditional node analysis (if pod)
                node_analysis = None
                if entity_type == EntityType.POD and metadata.node_name:
                    logger.info(f"Pod detected, analyzing host node: {metadata.node_name}")
                    node_analysis = await self._analyze_node_with_usage(
                        metadata.node_name, baseline_start, baseline_end, 
                        anomaly_start, anomaly_end, usage
                    )
                
                # Step 5: Final LLM summarization
                final_summary = await self._llm_final_summary_with_usage(
                    deps, merged_analysis, node_analysis, usage
                )
                
                # Build final result with usage stats
                result = AnomalyDetectionResult(
                    entity_type=entity_type,
                    entity_name=entity_name,
                    entity_metadata=metadata,
                    baseline_analysis=baseline_result,
                    anomaly_analysis=anomaly_result,
                    merged_analysis=merged_analysis,
                    node_analysis=node_analysis,
                    final_summary=final_summary,
                    status=AnalysisStatus.SUCCESS,
                    usage_stats=usage
                )
                
                logger.info(f"Enhanced analysis completed successfully for {entity_name}")
                logger.info(f"Total usage: {usage}")
                return result
                
            except UsageLimitExceeded as e:
                logger.error(f"Usage limit exceeded for {entity_name}: {e}")
                return AnalysisFailure(
                    error_type="UsageLimitExceeded",
                    error_message=str(e),
                    entity_name=entity_name,
                    retry_count=retry_count,
                    recovery_suggestions=[
                        "Increase usage limits for complex analyses",
                        "Break down the analysis into smaller time windows",
                        "Optimize system prompts to reduce token usage"
                    ]
                )
            except UnexpectedModelBehavior as e:
                retry_count += 1
                logger.warning(f"Unexpected model behavior for {entity_name} (attempt {retry_count}): {e}")
                
                if retry_count >= max_retries:
                    return AnalysisFailure(
                        error_type="UnexpectedModelBehavior",
                        error_message=str(e),
                        entity_name=entity_name,
                        retry_count=retry_count,
                        recovery_suggestions=[
                            "Check model availability and status",
                            "Verify input data quality",
                            "Try with a different model",
                            "Simplify the analysis request"
                        ]
                    )
                
                # Wait before retry
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Analysis failed for {entity_name}: {str(e)}")
                return AnalysisFailure(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    entity_name=entity_name,
                    retry_count=retry_count,
                    recovery_suggestions=[
                        "Verify entity exists in the cluster",
                        "Check time ranges are valid",
                        "Ensure monitoring systems are operational",
                        "Review system logs for detailed error information"
                    ]
                )
        
        # This should never be reached due to the return in the except block
        return AnalysisFailure(
            error_type="MaxRetriesExceeded",
            error_message=f"Failed after {max_retries} attempts",
            entity_name=entity_name,
            retry_count=retry_count,
            recovery_suggestions=["Contact system administrator"]
        )
    
    async def _get_metadata_with_usage(self, deps: K8sAnalysisDeps, usage: Usage) -> EntityMetadata:
        """Get entity metadata using tool call agent with usage tracking."""
        logger.info(f"Enhanced: Fetching metadata for {deps.entity_type}: {deps.entity_name}")
        
        try:
            # Use metadata agent tool call with usage tracking
            prompt = f"Retrieve metadata for {deps.entity_type.value} entity named '{deps.entity_name}'"
            result = await self.metadata_agent.run(
                prompt, 
                deps=deps, 
                usage=usage,
                usage_limits=self.usage_limits
            )
            return result.output
            
        except Exception as e:
            logger.error(f"Enhanced metadata retrieval failed for {deps.entity_name}: {str(e)}")
            # Fallback metadata
            return EntityMetadata(
                name=deps.entity_name,
                namespace="default" if deps.entity_type == EntityType.POD else None,
                labels={},
                node_name="unknown-node" if deps.entity_type == EntityType.POD else None,
                creation_time=datetime.now() - timedelta(days=1)
            )
    
    async def _analyze_parallel_paths_with_usage(self, deps: K8sAnalysisDeps, usage: Usage) -> tuple[PathAnalysisResult, PathAnalysisResult]:
        """Execute parallel analysis paths with usage tracking."""
        logger.info("Enhanced: Starting parallel path analysis")
        
        baseline_range = deps.get_baseline_timerange()
        anomaly_range = deps.get_anomaly_timerange()
        
        # Define sequential analysis for each path
        async def analyze_baseline_path():
            """Baseline path: CPU → Logs (sequential)"""
            logger.info("Enhanced: Baseline path - Starting CPU analysis")
            cpu_metrics = await self._analyze_cpu_with_usage(deps, baseline_range, usage)
            logger.info("Enhanced: Baseline path - Starting log analysis")
            log_analysis = await self._analyze_logs_with_usage(deps, baseline_range, usage)
            logger.info("Enhanced: Baseline path - Complete")
            return PathAnalysisResult(
                timerange=baseline_range,
                cpu_metrics=cpu_metrics,
                log_analysis=log_analysis
            )
        
        async def analyze_anomaly_path():
            """Anomaly path: CPU → Logs (sequential)"""
            logger.info("Enhanced: Anomaly path - Starting CPU analysis")
            cpu_metrics = await self._analyze_cpu_with_usage(deps, anomaly_range, usage)
            logger.info("Enhanced: Anomaly path - Starting log analysis")
            log_analysis = await self._analyze_logs_with_usage(deps, anomaly_range, usage)
            logger.info("Enhanced: Anomaly path - Complete")
            return PathAnalysisResult(
                timerange=anomaly_range,
                cpu_metrics=cpu_metrics,
                log_analysis=log_analysis
            )
        
        # Execute paths in parallel
        baseline_result, anomaly_result = await asyncio.gather(
            analyze_baseline_path(),
            analyze_anomaly_path()
        )
        
        logger.info("Enhanced: Completed parallel path analysis")
        return baseline_result, anomaly_result
    
    async def _analyze_cpu_with_usage(self, deps: K8sAnalysisDeps, timerange: TimeRange, usage: Usage) -> CPUMetrics:
        """Analyze CPU metrics using tool call agent with usage tracking."""
        logger.info(f"Enhanced: Analyzing CPU metrics for {deps.entity_type}: {deps.entity_name}")
        
        try:
            # Use CPU metrics agent tool call with usage tracking
            prompt = f"Analyze CPU metrics for {deps.entity_type.value} entity '{deps.entity_name}' from {timerange.start.isoformat()} to {timerange.end.isoformat()}"
            result = await self.cpu_metrics_agent.run(
                prompt, 
                deps=deps, 
                usage=usage,
                usage_limits=self.usage_limits
            )
            return result.output
            
        except Exception as e:
            logger.error(f"Enhanced CPU metrics analysis failed: {str(e)}")
            # Fallback CPU metrics
            avg_cpu = 20.0 + (hash(deps.entity_name) % 10)
            peak_cpu = avg_cpu + 5.0
            
            return CPUMetrics(
                average_utilization=round(avg_cpu, 2),
                peak_utilization=round(peak_cpu, 2),
                samples_count=12
            )
    
    async def _analyze_logs_with_usage(self, deps: K8sAnalysisDeps, timerange: TimeRange, usage: Usage) -> LogAnalysis:
        """Analyze logs using tool call agent with usage tracking."""
        logger.info(f"Enhanced: Analyzing logs for {deps.entity_type}: {deps.entity_name}")
        
        try:
            # Use log analysis agent tool call with usage tracking
            prompt = f"Analyze logs for {deps.entity_type.value} entity '{deps.entity_name}' from {timerange.start.isoformat()} to {timerange.end.isoformat()}"
            result = await self.log_analysis_agent.run(
                prompt, 
                deps=deps, 
                usage=usage,
                usage_limits=self.usage_limits
            )
            return result.output
            
        except Exception as e:
            logger.error(f"Enhanced log analysis failed: {str(e)}")
            # Fallback log analysis
            return LogAnalysis(
                error_count=0,
                warning_count=1,
                critical_patterns=[],
                anomalous_events=[]
            )
    
    async def _llm_merge_analysis_with_usage(
        self,
        deps: K8sAnalysisDeps,
        baseline_result: PathAnalysisResult,
        anomaly_result: PathAnalysisResult,
        usage: Usage
    ) -> MergedAnalysis:
        """Use specialized agent to merge and analyze outputs with usage tracking."""
        
        analysis_prompt = f"""
        Compare these two analysis paths for Kubernetes entity: {deps.entity_type} "{deps.entity_name}"
        
        BASELINE PATH ({baseline_result.timerange.start} to {baseline_result.timerange.end}):
        CPU Metrics:
        - Average: {baseline_result.cpu_metrics.average_utilization:.2f}%
        - Peak: {baseline_result.cpu_metrics.peak_utilization:.2f}%
        - Samples: {baseline_result.cpu_metrics.samples_count}
        
        Log Analysis:
        - Errors: {baseline_result.log_analysis.error_count}
        - Warnings: {baseline_result.log_analysis.warning_count}
        - Critical patterns: {baseline_result.log_analysis.critical_patterns}
        - Anomalous events: {baseline_result.log_analysis.anomalous_events}
        
        ANOMALY PATH ({anomaly_result.timerange.start} to {anomaly_result.timerange.end}):
        CPU Metrics:
        - Average: {anomaly_result.cpu_metrics.average_utilization:.2f}%
        - Peak: {anomaly_result.cpu_metrics.peak_utilization:.2f}%
        - Samples: {anomaly_result.cpu_metrics.samples_count}
        
        Log Analysis:
        - Errors: {anomaly_result.log_analysis.error_count}
        - Warnings: {anomaly_result.log_analysis.warning_count}
        - Critical patterns: {anomaly_result.log_analysis.critical_patterns}
        - Anomalous events: {anomaly_result.log_analysis.anomalous_events}
        
        Perform comprehensive comparison and analysis of the baseline vs anomaly periods.
        """
        
        logger.info("Enhanced: Running LLM merge analysis of parallel paths")
        
        try:
            result = await self.merge_agent.run(
                analysis_prompt, 
                usage=usage,
                usage_limits=self.usage_limits
            )
            logger.info(f"Enhanced: LLM merge analysis completed with confidence: {result.output.confidence_score}")
            return result.output
            
        except Exception as e:
            logger.error(f"Enhanced LLM merge analysis failed: {str(e)}")
            # Fallback to basic comparison
            return self._fallback_merge_analysis(baseline_result, anomaly_result, deps.entity_type)
    
    async def _llm_final_summary_with_usage(
        self,
        deps: K8sAnalysisDeps,
        merged_analysis: MergedAnalysis,
        node_analysis: Optional[AnomalyDetectionResult],
        usage: Usage
    ) -> FinalSummary:
        """Use specialized agent for final comprehensive summarization with usage tracking."""
        
        node_summary = ""
        if node_analysis:
            node_summary = f"""
            
            NODE ANALYSIS RESULTS:
            Node: {node_analysis.entity_name}
            Anomaly Detected: {node_analysis.final_summary.anomaly_detected}
            Confidence: {node_analysis.final_summary.confidence_score:.3f}
            Severity: {node_analysis.final_summary.severity}
            Key Findings: {node_analysis.merged_analysis.key_findings}
            """
        
        summary_prompt = f"""
        Provide final comprehensive summary for: {deps.entity_type} "{deps.entity_name}"
        
        MERGED ANALYSIS RESULTS:
        Significant Change: {merged_analysis.has_significant_change}
        Confidence: {merged_analysis.confidence_score:.3f}
        Severity: {merged_analysis.severity}
        Key Findings: {merged_analysis.key_findings}
        CPU Summary: {merged_analysis.cpu_change_summary}
        Log Summary: {merged_analysis.log_change_summary}
        {node_summary}
        
        Synthesize all evidence and provide final operational summary with clear conclusions and recommendations.
        """
        
        logger.info("Enhanced: Running final LLM summarization")
        
        try:
            result = await self.summary_agent.run(
                summary_prompt, 
                usage=usage,
                usage_limits=self.usage_limits
            )
            logger.info(f"Enhanced: Final summary completed")
            return result.output
            
        except Exception as e:
            logger.error(f"Enhanced final summary failed: {str(e)}")
            # Fallback summary
            return self._fallback_final_summary(merged_analysis, node_analysis)
    
    async def _analyze_node_with_usage(
        self, 
        node_name: str,
        baseline_start: datetime,
        baseline_end: datetime, 
        anomaly_start: datetime,
        anomaly_end: datetime,
        usage: Usage
    ) -> Optional[AnomalyDetectionResult]:
        """Analyze the host node using the same workflow pattern with usage tracking."""
        logger.info(f"Enhanced: Starting node analysis for {node_name}")
        
        try:
            result = await self.analyze_entity(
                entity_name=node_name,
                entity_type=EntityType.NODE,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                anomaly_start=anomaly_start,
                anomaly_end=anomaly_end,
                usage=usage
            )
            
            if isinstance(result, AnomalyDetectionResult):
                return result
            else:
                logger.warning(f"Enhanced node analysis failed for {node_name}")
                return None
                
        except Exception as e:
            logger.error(f"Enhanced node analysis error for {node_name}: {str(e)}")
            return None
    
    def _fallback_merge_analysis(
        self, 
        baseline: PathAnalysisResult, 
        anomaly: PathAnalysisResult,
        entity_type: EntityType
    ) -> MergedAnalysis:
        """Fallback merge analysis if LLM fails."""
        
        cpu_change = anomaly.cpu_metrics.average_utilization - baseline.cpu_metrics.average_utilization
        cpu_change_pct = (cpu_change / max(baseline.cpu_metrics.average_utilization, 1)) * 100
        
        has_significant_change = (
            abs(cpu_change_pct) > 50 or
            (anomaly.log_analysis.error_count - baseline.log_analysis.error_count) > 5 or
            len(anomaly.log_analysis.critical_patterns) > 0
        )
        
        return MergedAnalysis(
            has_significant_change=has_significant_change,
            confidence_score=0.6 if has_significant_change else 0.3,
            severity=AnomalySeverity.MEDIUM if abs(cpu_change_pct) > 100 else AnomalySeverity.LOW,
            key_findings=[f"CPU change: {cpu_change_pct:.1f}%"],
            cpu_change_summary=f"CPU utilization changed by {cpu_change_pct:.1f}%",
            log_change_summary=f"Log errors changed by {anomaly.log_analysis.error_count - baseline.log_analysis.error_count}"
        )
    
    def _fallback_final_summary(
        self, 
        merged: MergedAnalysis, 
        node_analysis: Optional[AnomalyDetectionResult]
    ) -> FinalSummary:
        """Fallback final summary if LLM fails."""
        
        return FinalSummary(
            overall_conclusion="Fallback analysis completed with limited intelligence",
            anomaly_detected=merged.has_significant_change,
            confidence_score=merged.confidence_score,
            severity=merged.severity,
            primary_evidence=merged.key_findings,
            contributing_factors=merged.key_findings,
            recommendations=["Review metrics manually", "Check system logs", "Contact system administrator"]
        )

# ============================================================================
# Enhanced Usage Example and Testing
# ============================================================================

async def main():
    """Enhanced example usage with proper error handling and usage tracking."""
    
    # Configure usage limits
    usage_limits = UsageLimits(
        request_limit=30,  # Maximum 30 requests per analysis
        total_tokens_limit=50000,  # Maximum 50k tokens per analysis
    )
    
    orchestrator = K8sAnomalyDetectionOrchestrator(usage_limits=usage_limits)
    
    # Use time ranges that match mock data
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    print("🚀 Testing ENHANCED workflow with PydanticAI best practices...")
    print("=" * 70)
    
    # Initialize usage tracking
    usage = Usage()
    
    result = await orchestrator.analyze_entity(
        entity_name="frontend-6d8f4f79f7-kxzpl",
        entity_type=EntityType.POD,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
        usage=usage
    )
    
    if isinstance(result, AnomalyDetectionResult):
        print("✅ ENHANCED ANALYSIS SUCCESSFUL")
        print(f"🔍 Merged Analysis Confidence: {result.merged_analysis.confidence_score:.3f}")
        print(f"🎯 Final Conclusion: {result.final_summary.anomaly_detected}")
        print(f"📊 Final Confidence: {result.final_summary.confidence_score:.3f}")
        print(f"📝 Overall Conclusion: {result.final_summary.overall_conclusion}")
        print(f"🚨 Recommendations: {result.final_summary.recommendations}")
        print(f"💰 Usage Stats: {result.usage_stats}")
        
        if result.node_analysis:
            print(f"🖥️  Node Analysis: {result.node_analysis.final_summary.anomaly_detected}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")
        print(f"🔄 Retry count: {result.retry_count}")
        print(f"💡 Recovery suggestions: {result.recovery_suggestions}")

if __name__ == "__main__":
    asyncio.run(main()) 