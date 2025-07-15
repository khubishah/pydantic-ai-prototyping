"""
Kubernetes Anomaly Detection with PydanticGraph Architecture

This implementation maintains the exact same workflow as k8s_anomaly_detection_refined.py
but uses PydanticGraph for structured node execution and state management:

1. Parallel tool calls for baseline and anomaly analysis
2. LLM merge and analyze outputs from 2 paths
3. Conditional node analysis (if pod)
4. Final LLM summarization

Uses PydanticGraph for:
- Structured workflow definition with nodes and transitions
- State management across the workflow
- Enhanced observability and debugging
- Proper error handling and retry mechanisms
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
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

# Import mock data functions
from mock_data import get_entities, get_cpu_utilization, get_logs

# Configure logging and observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure()

# ============================================================================
# Data Models and Enums (Same as refined version)
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

# ============================================================================
# Graph State Management
# ============================================================================

class K8sAnalysisState(BaseModel):
    """State maintained throughout the PydanticGraph workflow."""
    # Input parameters
    entity_name: str
    entity_type: EntityType
    baseline_start: datetime
    baseline_end: datetime
    anomaly_start: datetime
    anomaly_end: datetime
    
    # Workflow state
    entity_metadata: Optional[EntityMetadata] = None
    baseline_analysis: Optional[PathAnalysisResult] = None
    anomaly_analysis: Optional[PathAnalysisResult] = None
    merged_analysis: Optional[MergedAnalysis] = None
    node_analysis: Optional[AnomalyDetectionResult] = None
    final_summary: Optional[FinalSummary] = None
    
    # Usage tracking
    usage: Usage = Field(default_factory=Usage)
    
    # Status tracking
    status: AnalysisStatus = AnalysisStatus.SUCCESS
    error_message: Optional[str] = None
    
    def get_baseline_timerange(self) -> TimeRange:
        """Get baseline time range."""
        return TimeRange(start=self.baseline_start, end=self.baseline_end)
    
    def get_anomaly_timerange(self) -> TimeRange:
        """Get anomaly time range."""
        return TimeRange(start=self.anomaly_start, end=self.anomaly_end)

class K8sAnalysisDeps(BaseModel):
    """Dependencies for the K8s analysis workflow."""
    usage_limits: UsageLimits = Field(default_factory=lambda: UsageLimits(
        request_limit=50,
        total_tokens_limit=100000,
    ))

# ============================================================================
# Specialized Agents (Same as refined version)
# ============================================================================

# Agent for merging and analyzing the two paths
merge_analysis_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=MergedAnalysis,
    retries=3,
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
    retries=3,
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

# Tool call agents (same as refined version)
metadata_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=EntityMetadata,
    retries=2,
    system_prompt="""
    You are a Kubernetes metadata retrieval tool. Your job is to call the appropriate tool 
    to retrieve entity metadata based on the entity type and name provided.
    
    Always use the get_k8s_entity_metadata tool to retrieve metadata information.
    If the tool fails, analyze the error and provide helpful context for retry.
    """,
)

cpu_metrics_agent = Agent(
    'openai:gpt-4o-mini', 
    output_type=CPUMetrics,
    retries=3,
    system_prompt="""
    You are a Kubernetes CPU metrics analysis tool. Your job is to call the appropriate tool
    to retrieve and analyze CPU utilization data for the specified entity and time range.
    
    Always use the get_cpu_metrics tool to retrieve CPU data and return structured metrics.
    If the tool fails due to data issues, provide clear guidance on what went wrong.
    """,
)

log_analysis_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=LogAnalysis, 
    retries=3,
    system_prompt="""
    You are a Kubernetes log analysis tool. Your job is to call the appropriate tool
    to retrieve and analyze log data for the specified entity and time range.
    
    Always use the get_entity_logs tool to retrieve log data and return structured analysis.
    If the tool fails, analyze the error and suggest alternative approaches.
    """,
)

# ============================================================================
# Enhanced Tool Functions (Same as refined version)
# ============================================================================

@metadata_agent.tool(retries=2)
async def get_k8s_entity_metadata(ctx: RunContext, entity_type: str, entity_name: str) -> dict:
    """Tool to retrieve Kubernetes entity metadata with enhanced error handling."""
    logger.info(f"Tool call (attempt {ctx.retry + 1}): Retrieving metadata for {entity_type}: {entity_name}")
    
    try:
        entity_data = get_entities(entity_type, entity_name)
        metadata = entity_data["metadata"]
        
        if not metadata.get("start_time"):
            raise ModelRetry(
                f"Metadata for {entity_name} is missing start_time. "
                f"Please ensure the entity exists and has valid metadata."
            )
        
        return metadata
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tool call failed for metadata retrieval: {error_msg}")
        
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
            logger.warning(f"Using fallback metadata for {entity_name}")
            return {
                "labels": {},
                "node_name": "unknown-node" if entity_type == "k8s:pod" else None,
                "start_time": "2024-06-26T09:58:12Z"
            }

@cpu_metrics_agent.tool(retries=3)
async def get_cpu_metrics(ctx: RunContext, entity_type: str, entity_name: str, start_time: str, end_time: str) -> dict:
    """Tool to retrieve and analyze CPU metrics with enhanced error handling."""
    logger.info(f"Tool call (attempt {ctx.retry + 1}): Analyzing CPU metrics for {entity_type}: {entity_name} from {start_time} to {end_time}")
    
    try:
        await asyncio.sleep(1.0)
        cpu_data = get_cpu_utilization(entity_type, entity_name, start_time, end_time)
        
        if not cpu_data:
            raise ModelRetry(
                f"No CPU data available for {entity_name} in the specified time range. "
                f"Please verify the entity exists and the time range is valid."
            )
        
        if len(cpu_data) < 3:
            raise ModelRetry(
                f"Insufficient CPU data points ({len(cpu_data)}) for {entity_name}. "
                f"Need at least 3 data points for reliable analysis."
            )
        
        cpu_values = [point["cpu_percent"] for point in cpu_data]
        
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
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tool call failed for CPU metrics: {error_msg}")
        
        if "Unsupported entity_type" in error_msg:
            raise ModelRetry(
                f"Entity type '{entity_type}' is not supported for CPU metrics. "
                f"Please use 'k8s:pod' or 'k8s:node'."
            )
        elif ctx.retry < 2:
            raise ModelRetry(
                f"CPU metrics retrieval failed: {error_msg}. "
                f"This might be a transient issue. Please try again."
            )
        else:
            logger.warning(f"Using fallback CPU metrics for {entity_name}")
            avg_cpu = 20.0 + (hash(entity_name) % 10)
            peak_cpu = avg_cpu + 5.0
            
            return {
                "average_utilization": round(avg_cpu, 2),
                "peak_utilization": round(peak_cpu, 2),
                "samples_count": 12
            }

@log_analysis_agent.tool(retries=3)
async def get_entity_logs(ctx: RunContext, entity_type: str, entity_name: str, start_time: str, end_time: str) -> dict:
    """Tool to retrieve and analyze entity logs with enhanced error handling."""
    logger.info(f"Tool call (attempt {ctx.retry + 1}): Analyzing logs for {entity_type}: {entity_name} from {start_time} to {end_time}")
    
    try:
        await asyncio.sleep(0.8)
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
            
            if not message:
                continue
            
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
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Tool call failed for log analysis: {error_msg}")
        
        if "Unsupported entity_type" in error_msg:
            raise ModelRetry(
                f"Entity type '{entity_type}' is not supported for log analysis. "
                f"Please use 'k8s:pod' or 'k8s:node'."
            )
        elif ctx.retry < 2:
            raise ModelRetry(
                f"Log analysis failed: {error_msg}. "
                f"This might be a transient issue. Please try again."
            )
        else:
            logger.warning(f"Using fallback log analysis for {entity_name}")
            return {
                "error_count": 0,
                "warning_count": 1,
                "critical_patterns": [],
                "anomalous_events": []
            }

# ============================================================================
# PydanticGraph Workflow Nodes
# ============================================================================

class StartAnalysis(BaseNode[K8sAnalysisState, K8sAnalysisDeps, AnomalyDetectionResult]):
    """Entry point for the analysis workflow."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> 'GetMetadata':
        """Start the analysis workflow."""
        logger.info(f"🚀 Starting PydanticGraph workflow for {ctx.state.entity_type}: {ctx.state.entity_name}")
        return GetMetadata()

class GetMetadata(BaseNode[K8sAnalysisState, K8sAnalysisDeps]):
    """Retrieve entity metadata."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> 'ParallelAnalysis':
        """Get entity metadata using tool call agent."""
        logger.info(f"📋 Fetching metadata for {ctx.state.entity_type}: {ctx.state.entity_name}")
        
        try:
            prompt = f"Retrieve metadata for {ctx.state.entity_type.value} entity named '{ctx.state.entity_name}'"
            result = await metadata_agent.run(
                prompt, 
                usage=ctx.state.usage,
                usage_limits=ctx.deps.usage_limits
            )
            ctx.state.entity_metadata = result.output
            logger.info(f"✅ Metadata retrieved successfully")
            return ParallelAnalysis()
            
        except Exception as e:
            logger.error(f"❌ Metadata retrieval failed: {str(e)}")
            # Fallback metadata
            ctx.state.entity_metadata = EntityMetadata(
                name=ctx.state.entity_name,
                namespace="default" if ctx.state.entity_type == EntityType.POD else None,
                labels={},
                node_name="unknown-node" if ctx.state.entity_type == EntityType.POD else None,
                creation_time=datetime.now() - timedelta(days=1)
            )
            return ParallelAnalysis()

class ParallelAnalysis(BaseNode[K8sAnalysisState, K8sAnalysisDeps]):
    """Execute parallel baseline and anomaly analysis paths."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> 'MergeAnalysis':
        """Execute parallel analysis paths with sequential CPU → Logs within each path."""
        logger.info("🔄 Starting parallel path analysis")
        
        baseline_range = ctx.state.get_baseline_timerange()
        anomaly_range = ctx.state.get_anomaly_timerange()
        
        # Define sequential analysis for each path
        async def analyze_baseline_path():
            """Baseline path: CPU → Logs (sequential)"""
            logger.info("📊 Baseline path - Starting CPU analysis")
            cpu_metrics = await self._analyze_cpu(ctx, baseline_range)
            logger.info("📝 Baseline path - Starting log analysis")
            log_analysis = await self._analyze_logs(ctx, baseline_range)
            logger.info("✅ Baseline path - Complete")
            return PathAnalysisResult(
                timerange=baseline_range,
                cpu_metrics=cpu_metrics,
                log_analysis=log_analysis
            )
        
        async def analyze_anomaly_path():
            """Anomaly path: CPU → Logs (sequential)"""
            logger.info("📊 Anomaly path - Starting CPU analysis")
            cpu_metrics = await self._analyze_cpu(ctx, anomaly_range)
            logger.info("📝 Anomaly path - Starting log analysis")
            log_analysis = await self._analyze_logs(ctx, anomaly_range)
            logger.info("✅ Anomaly path - Complete")
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
        
        ctx.state.baseline_analysis = baseline_result
        ctx.state.anomaly_analysis = anomaly_result
        
        logger.info("🎯 Completed parallel path analysis")
        return MergeAnalysis()
    
    async def _analyze_cpu(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps], timerange: TimeRange) -> CPUMetrics:
        """Analyze CPU metrics using tool call agent."""
        try:
            prompt = f"Analyze CPU metrics for {ctx.state.entity_type.value} entity '{ctx.state.entity_name}' from {timerange.start.isoformat()} to {timerange.end.isoformat()}"
            result = await cpu_metrics_agent.run(
                prompt, 
                usage=ctx.state.usage,
                usage_limits=ctx.deps.usage_limits
            )
            return result.output
            
        except Exception as e:
            logger.error(f"CPU metrics analysis failed: {str(e)}")
            # Fallback CPU metrics
            avg_cpu = 20.0 + (hash(ctx.state.entity_name) % 10)
            peak_cpu = avg_cpu + 5.0
            
            return CPUMetrics(
                average_utilization=round(avg_cpu, 2),
                peak_utilization=round(peak_cpu, 2),
                samples_count=12
            )
    
    async def _analyze_logs(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps], timerange: TimeRange) -> LogAnalysis:
        """Analyze logs using tool call agent."""
        try:
            prompt = f"Analyze logs for {ctx.state.entity_type.value} entity '{ctx.state.entity_name}' from {timerange.start.isoformat()} to {timerange.end.isoformat()}"
            result = await log_analysis_agent.run(
                prompt, 
                usage=ctx.state.usage,
                usage_limits=ctx.deps.usage_limits
            )
            return result.output
            
        except Exception as e:
            logger.error(f"Log analysis failed: {str(e)}")
            # Fallback log analysis
            return LogAnalysis(
                error_count=0,
                warning_count=1,
                critical_patterns=[],
                anomalous_events=[]
            )

class MergeAnalysis(BaseNode[K8sAnalysisState, K8sAnalysisDeps]):
    """Use LLM to merge and analyze the parallel analysis results."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> 'CheckNodeAnalysis':
        """Use specialized agent to merge and analyze outputs."""
        logger.info("🧠 Running LLM merge analysis of parallel paths")
        
        baseline = ctx.state.baseline_analysis
        anomaly = ctx.state.anomaly_analysis
        
        analysis_prompt = f"""
        Compare these two analysis paths for Kubernetes entity: {ctx.state.entity_type} "{ctx.state.entity_name}"
        
        BASELINE PATH ({baseline.timerange.start} to {baseline.timerange.end}):
        CPU Metrics:
        - Average: {baseline.cpu_metrics.average_utilization:.2f}%
        - Peak: {baseline.cpu_metrics.peak_utilization:.2f}%
        - Samples: {baseline.cpu_metrics.samples_count}
        
        Log Analysis:
        - Errors: {baseline.log_analysis.error_count}
        - Warnings: {baseline.log_analysis.warning_count}
        - Critical patterns: {baseline.log_analysis.critical_patterns}
        - Anomalous events: {baseline.log_analysis.anomalous_events}
        
        ANOMALY PATH ({anomaly.timerange.start} to {anomaly.timerange.end}):
        CPU Metrics:
        - Average: {anomaly.cpu_metrics.average_utilization:.2f}%
        - Peak: {anomaly.cpu_metrics.peak_utilization:.2f}%
        - Samples: {anomaly.cpu_metrics.samples_count}
        
        Log Analysis:
        - Errors: {anomaly.log_analysis.error_count}
        - Warnings: {anomaly.log_analysis.warning_count}
        - Critical patterns: {anomaly.log_analysis.critical_patterns}
        - Anomalous events: {anomaly.log_analysis.anomalous_events}
        
        Perform comprehensive comparison and analysis of the baseline vs anomaly periods.
        """
        
        try:
            result = await merge_analysis_agent.run(
                analysis_prompt, 
                usage=ctx.state.usage,
                usage_limits=ctx.deps.usage_limits
            )
            ctx.state.merged_analysis = result.output
            logger.info(f"✅ LLM merge analysis completed with confidence: {result.output.confidence_score}")
            return CheckNodeAnalysis()
            
        except Exception as e:
            logger.error(f"LLM merge analysis failed: {str(e)}")
            # Fallback to basic comparison
            ctx.state.merged_analysis = self._fallback_merge_analysis(baseline, anomaly)
            return CheckNodeAnalysis()
    
    def _fallback_merge_analysis(self, baseline: PathAnalysisResult, anomaly: PathAnalysisResult) -> MergedAnalysis:
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

class CheckNodeAnalysis(BaseNode[K8sAnalysisState, K8sAnalysisDeps]):
    """Check if node analysis is needed (conditional node analysis)."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> 'AnalyzeNode | FinalSummary':
        """Check if we need to analyze the host node."""
        if (ctx.state.entity_type == EntityType.POD and 
            ctx.state.entity_metadata and 
            ctx.state.entity_metadata.node_name):
            
            logger.info(f"🖥️  Pod detected, analyzing host node: {ctx.state.entity_metadata.node_name}")
            return AnalyzeNode()
        else:
            logger.info("⏭️  Skipping node analysis (not a pod or no node name)")
            return FinalSummary()

class AnalyzeNode(BaseNode[K8sAnalysisState, K8sAnalysisDeps]):
    """Analyze the host node using recursive workflow."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> 'FinalSummary':
        """Analyze the host node using the same workflow pattern."""
        node_name = ctx.state.entity_metadata.node_name
        logger.info(f"🔍 Starting node analysis for {node_name}")
        
        try:
            # Create a new state for node analysis
            node_state = K8sAnalysisState(
                entity_name=node_name,
                entity_type=EntityType.NODE,
                baseline_start=ctx.state.baseline_start,
                baseline_end=ctx.state.baseline_end,
                anomaly_start=ctx.state.anomaly_start,
                anomaly_end=ctx.state.anomaly_end,
                usage=ctx.state.usage  # Share usage tracking
            )
            
            # Run the same workflow for the node
            result = await k8s_anomaly_graph.run(
                StartAnalysis(),
                state=node_state,
                deps=ctx.deps
            )
            
            if result.output:
                ctx.state.node_analysis = result.output
                logger.info(f"✅ Node analysis completed successfully")
            else:
                logger.warning(f"⚠️  Node analysis failed for {node_name}")
                
        except Exception as e:
            logger.error(f"❌ Node analysis error for {node_name}: {str(e)}")
        
        return FinalSummary()

class FinalSummary(BaseNode[K8sAnalysisState, K8sAnalysisDeps, AnomalyDetectionResult]):
    """Generate final comprehensive summary."""
    
    async def run(self, ctx: GraphRunContext[K8sAnalysisState, K8sAnalysisDeps]) -> End[AnomalyDetectionResult]:
        """Use specialized agent for final comprehensive summarization."""
        logger.info("📝 Running final LLM summarization")
        
        merged_analysis = ctx.state.merged_analysis
        node_analysis = ctx.state.node_analysis
        
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
        Provide final comprehensive summary for: {ctx.state.entity_type} "{ctx.state.entity_name}"
        
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
        
        try:
            result = await final_summary_agent.run(
                summary_prompt, 
                usage=ctx.state.usage,
                usage_limits=ctx.deps.usage_limits
            )
            ctx.state.final_summary = result.output
            logger.info(f"✅ Final summary completed")
            
        except Exception as e:
            logger.error(f"Final summary failed: {str(e)}")
            # Fallback summary
            ctx.state.final_summary = self._fallback_final_summary(merged_analysis, node_analysis)
        
        # Build final result
        final_result = AnomalyDetectionResult(
            entity_type=ctx.state.entity_type,
            entity_name=ctx.state.entity_name,
            entity_metadata=ctx.state.entity_metadata,
            baseline_analysis=ctx.state.baseline_analysis,
            anomaly_analysis=ctx.state.anomaly_analysis,
            merged_analysis=ctx.state.merged_analysis,
            node_analysis=ctx.state.node_analysis,
            final_summary=ctx.state.final_summary,
            status=ctx.state.status,
            usage_stats=ctx.state.usage
        )
        
        logger.info(f"🎉 PydanticGraph analysis completed successfully for {ctx.state.entity_name}")
        logger.info(f"💰 Total usage: {ctx.state.usage}")
        
        return End(final_result)
    
    def _fallback_final_summary(self, merged: MergedAnalysis, node_analysis: Optional[AnomalyDetectionResult]) -> FinalSummary:
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
# PydanticGraph Definition
# ============================================================================

# Define the workflow graph
k8s_anomaly_graph = Graph(
    nodes=[
        StartAnalysis,
        GetMetadata,
        ParallelAnalysis,
        MergeAnalysis,
        CheckNodeAnalysis,
        AnalyzeNode,
        FinalSummary
    ],
    state_type=K8sAnalysisState
)

# ============================================================================
# Main Orchestrator Class
# ============================================================================

class K8sAnomalyDetectionGraphOrchestrator:
    """Orchestrates the anomaly detection workflow using PydanticGraph."""
    
    def __init__(self, usage_limits: Optional[UsageLimits] = None):
        self.graph = k8s_anomaly_graph
        self.usage_limits = usage_limits or UsageLimits(
            request_limit=50,
            total_tokens_limit=100000,
        )
    
    async def analyze_entity(
        self,
        entity_name: str,
        entity_type: EntityType,
        baseline_start: datetime,
        baseline_end: datetime,
        anomaly_start: datetime,
        anomaly_end: datetime
    ) -> AnomalyDetectionResult:
        """
        Perform anomaly detection analysis using PydanticGraph workflow.
        """
        # Create initial state
        state = K8sAnalysisState(
            entity_name=entity_name,
            entity_type=entity_type,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            anomaly_start=anomaly_start,
            anomaly_end=anomaly_end
        )
        
        # Create dependencies
        deps = K8sAnalysisDeps(usage_limits=self.usage_limits)
        
        # Run the graph
        result = await self.graph.run(
            StartAnalysis(),
            state=state,
            deps=deps
        )
        
        return result.output

# ============================================================================
# Usage Example and Testing
# ============================================================================

async def main():
    """Example usage with PydanticGraph architecture."""
    
    # Configure usage limits
    usage_limits = UsageLimits(
        request_limit=30,
        total_tokens_limit=50000,
    )
    
    orchestrator = K8sAnomalyDetectionGraphOrchestrator(usage_limits=usage_limits)
    
    # Use time ranges that match mock data
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    print("🚀 Testing PydanticGraph workflow architecture...")
    print("=" * 70)
    
    try:
        result = await orchestrator.analyze_entity(
            entity_name="frontend-6d8f4f79f7-kxzpl",
            entity_type=EntityType.POD,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            anomaly_start=anomaly_start,
            anomaly_end=anomaly_end
        )
        
        print("✅ PYDANTIC GRAPH ANALYSIS SUCCESSFUL")
        print(f"🔍 Merged Analysis Confidence: {result.merged_analysis.confidence_score:.3f}")
        print(f"🎯 Final Conclusion: {result.final_summary.anomaly_detected}")
        print(f"📊 Final Confidence: {result.final_summary.confidence_score:.3f}")
        print(f"📝 Overall Conclusion: {result.final_summary.overall_conclusion}")
        print(f"🚨 Recommendations: {result.final_summary.recommendations}")
        print(f"💰 Usage Stats: {result.usage_stats}")
        
        if result.node_analysis:
            print(f"🖥️  Node Analysis: {result.node_analysis.final_summary.anomaly_detected}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

async def demonstrate_graph_visualization():
    """Demonstrate PydanticGraph visualization capabilities."""
    print("\n🎨 Generating workflow visualization...")
    
    # Generate Mermaid diagram
    mermaid_code = k8s_anomaly_graph.mermaid_code(
        start_node=StartAnalysis,
        direction='TB'
    )
    
    print("📊 Workflow Diagram (Mermaid):")
    print(mermaid_code)
    
    # You can also save as image if needed
    # k8s_anomaly_graph.mermaid_save('k8s_anomaly_workflow.png', start_node=StartAnalysis)

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(demonstrate_graph_visualization()) 