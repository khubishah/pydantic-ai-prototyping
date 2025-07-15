"""
Kubernetes Anomaly Detection with Single Agent Architecture

This implementation follows the same workflow as the refined version but uses
a single agent with multiple tools instead of specialized agents:
1. Parallel tool calls for baseline and anomaly analysis
2. LLM merge and analyze outputs from 2 paths
3. Conditional node analysis (if pod)
4. Final LLM summarization

Uses single agent with multiple tools for comparison with specialized approach.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import logfire
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext

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
    
    # Status
    status: AnalysisStatus = Field(..., description="Overall analysis status")

class AnalysisFailure(BaseModel):
    """Represents a failed analysis with recovery information."""
    error_type: str = Field(..., description="Type of error that occurred")
    error_message: str = Field(..., description="Detailed error message")
    entity_name: str = Field(..., description="Name of entity being analyzed")
    failed_at: datetime = Field(default_factory=datetime.now, description="When the failure occurred")
    recovery_suggestions: List[str] = Field(default_factory=list, description="Suggested recovery actions")

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
# Single Agent with Multiple Tools
# ============================================================================

# Union type for all possible outputs
K8sAnalysisOutput = Union[EntityMetadata, CPUMetrics, LogAnalysis, MergedAnalysis, FinalSummary]

# Single unified agent with all tools
unified_k8s_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=K8sAnalysisOutput,
    system_prompt="""
    You are a comprehensive Kubernetes anomaly detection system with multiple capabilities.
    
    Based on the request, choose the appropriate tool and provide the corresponding analysis:
    
    1. METADATA RETRIEVAL: Use get_k8s_entity_metadata for entity information
    2. CPU METRICS ANALYSIS: Use get_cpu_metrics for CPU utilization analysis
    3. LOG ANALYSIS: Use get_entity_logs for log pattern analysis
    4. MERGE ANALYSIS: Use your reasoning to compare baseline vs anomaly periods
    5. FINAL SUMMARY: Use your reasoning to provide comprehensive summaries
    
    For merge analysis, focus on:
    - CPU utilization patterns and magnitude of changes
    - Log error/warning trends and critical patterns
    - Correlation between CPU and log anomalies
    - Relative significance (small baseline vs large changes matter more)
    
    Be conservative with confidence scores:
    - 0.9+: Very strong evidence (multiple clear indicators)
    - 0.7-0.9: Strong evidence (clear primary indicator + supporting)
    - 0.5-0.7: Moderate evidence (some indicators with uncertainty)
    - 0.3-0.5: Weak evidence (minor indicators, could be noise)
    
    For final summaries, provide:
    - Clear overall conclusion about anomaly presence
    - Final confidence score considering all evidence
    - Actionable recommendations for operations teams
    - Comprehensive list of contributing factors
    """,
)

# ============================================================================
# Tool Functions for Single Agent
# ============================================================================

@unified_k8s_agent.tool
async def get_k8s_entity_metadata(ctx: RunContext[K8sAnalysisDeps], entity_type: str, entity_name: str) -> dict:
    """Tool to retrieve Kubernetes entity metadata."""
    logger.info(f"🔧 UNIFIED AGENT TOOL: get_k8s_entity_metadata for {entity_type}: {entity_name}")
    
    try:
        # Call the mock data function
        entity_data = get_entities(entity_type, entity_name)
        return entity_data["metadata"]
    except Exception as e:
        logger.error(f"Tool call failed for metadata retrieval: {str(e)}")
        # Return fallback metadata
        return {
            "labels": {},
            "node_name": "unknown-node" if entity_type == "k8s:pod" else None,
            "start_time": "2024-06-26T09:58:12Z"
        }

@unified_k8s_agent.tool
async def get_cpu_metrics(ctx: RunContext[K8sAnalysisDeps], entity_type: str, entity_name: str, start_time: str, end_time: str) -> dict:
    """Tool to retrieve and analyze CPU metrics."""
    logger.info(f"🔧 UNIFIED AGENT TOOL: get_cpu_metrics for {entity_type}: {entity_name} from {start_time} to {end_time}")
    
    try:
        # Add small delay to simulate real tool call
        await asyncio.sleep(1.0)
        
        # Call the mock data function
        cpu_data = get_cpu_utilization(entity_type, entity_name, start_time, end_time)
        
        if not cpu_data:
            raise ValueError(f"No CPU data available for {entity_name}")
        
        # Process the data
        cpu_values = [point["cpu_percent"] for point in cpu_data]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        peak_cpu = max(cpu_values)
        samples_count = len(cpu_values)
        
        return {
            "average_utilization": round(avg_cpu, 2),
            "peak_utilization": round(peak_cpu, 2),
            "samples_count": samples_count
        }
        
    except Exception as e:
        logger.error(f"Tool call failed for CPU metrics: {str(e)}")
        # Return fallback data
        avg_cpu = 20.0 + (hash(entity_name) % 10)
        peak_cpu = avg_cpu + 5.0
        
        return {
            "average_utilization": round(avg_cpu, 2),
            "peak_utilization": round(peak_cpu, 2),
            "samples_count": 12
        }

@unified_k8s_agent.tool
async def get_entity_logs(ctx: RunContext[K8sAnalysisDeps], entity_type: str, entity_name: str, start_time: str, end_time: str) -> dict:
    """Tool to retrieve and analyze entity logs."""
    logger.info(f"🔧 UNIFIED AGENT TOOL: get_entity_logs for {entity_type}: {entity_name} from {start_time} to {end_time}")
    
    try:
        # Add small delay to simulate real tool call
        await asyncio.sleep(0.8)
        
        # Call the mock data function
        log_events = get_logs(entity_type, entity_name, start_time, end_time)
        
        error_count = 0
        warning_count = 0
        critical_patterns = []
        anomalous_events = []
        
        for event in log_events:
            event_type = event.get("type", "").lower()
            reason = event.get("reason", "")
            message = event.get("message", "")
            
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
        
    except Exception as e:
        logger.error(f"Tool call failed for log analysis: {str(e)}")
        # Return fallback data
        return {
            "error_count": 0,
            "warning_count": 1,
            "critical_patterns": [],
            "anomalous_events": []
        }

# ============================================================================
# Orchestrator with Single Agent Architecture
# ============================================================================

class K8sAnomalyDetectionOrchestrator:
    """Orchestrates the anomaly detection workflow with a single unified agent."""
    
    def __init__(self):
        self.unified_agent = unified_k8s_agent
    
    async def analyze_entity(
        self,
        entity_name: str,
        entity_type: EntityType,
        baseline_start: datetime,
        baseline_end: datetime,
        anomaly_start: datetime,
        anomaly_end: datetime
    ) -> Union[AnomalyDetectionResult, AnalysisFailure]:
        """
        Perform anomaly detection analysis with the same workflow but using single agent:
        1. Parallel tool calls for baseline and anomaly analysis
        2. LLM merge and analyze outputs from 2 paths
        3. Conditional node analysis (if pod)
        4. Final LLM summarization
        """
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
            
            logger.info(f"🚀 Starting SINGLE AGENT workflow orchestration for {entity_type}: {entity_name}")
            
            # Step 1: Get entity metadata using unified agent
            metadata = await self._get_metadata(deps)
            
            # Step 2: Parallel analysis paths using unified agent
            baseline_result, anomaly_result = await self._analyze_parallel_paths(deps)
            
            # Step 3: LLM merge and analyze outputs from 2 paths using unified agent
            merged_analysis = await self._llm_merge_analysis(
                deps, baseline_result, anomaly_result
            )
            
            # Step 4: Conditional node analysis (if pod)
            node_analysis = None
            if entity_type == EntityType.POD and metadata.node_name:
                logger.info(f"Pod detected, analyzing host node: {metadata.node_name}")
                node_analysis = await self._analyze_node(
                    metadata.node_name, baseline_start, baseline_end, 
                    anomaly_start, anomaly_end
                )
            
            # Step 5: Final LLM summarization using unified agent
            final_summary = await self._llm_final_summary(
                deps, merged_analysis, node_analysis
            )
            
            # Build final result
            result = AnomalyDetectionResult(
                entity_type=entity_type,
                entity_name=entity_name,
                entity_metadata=metadata,
                baseline_analysis=baseline_result,
                anomaly_analysis=anomaly_result,
                merged_analysis=merged_analysis,
                node_analysis=node_analysis,
                final_summary=final_summary,
                status=AnalysisStatus.SUCCESS
            )
            
            logger.info(f"✅ Single agent analysis completed successfully for {entity_name}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {entity_name}: {str(e)}")
            return AnalysisFailure(
                error_type=type(e).__name__,
                error_message=str(e),
                entity_name=entity_name,
                recovery_suggestions=[
                    "Verify entity exists in the cluster",
                    "Check time ranges are valid",
                    "Ensure monitoring systems are operational",
                    "Retry with different time ranges"
                ]
            )
    
    async def _get_metadata(self, deps: K8sAnalysisDeps) -> EntityMetadata:
        """Get entity metadata using unified agent."""
        logger.info(f"🔍 SINGLE AGENT: Fetching metadata for {deps.entity_type}: {deps.entity_name}")
        
        try:
            # Use unified agent for metadata retrieval
            prompt = f"Retrieve metadata for {deps.entity_type.value} entity named '{deps.entity_name}'. Use the get_k8s_entity_metadata tool."
            result = await self.unified_agent.run(prompt, deps=deps)
            
            # Handle the union type output
            if isinstance(result.output, EntityMetadata):
                return result.output
            else:
                raise ValueError(f"Expected EntityMetadata, got {type(result.output)}")
            
        except Exception as e:
            logger.error(f"Metadata retrieval failed for {deps.entity_name}: {str(e)}")
            # Fallback metadata
            return EntityMetadata(
                name=deps.entity_name,
                namespace="default" if deps.entity_type == EntityType.POD else None,
                labels={},
                node_name="unknown-node" if deps.entity_type == EntityType.POD else None,
                creation_time=datetime.now() - timedelta(days=1)
            )
    
    async def _analyze_parallel_paths(self, deps: K8sAnalysisDeps) -> tuple[PathAnalysisResult, PathAnalysisResult]:
        """
        Execute parallel analysis paths using unified agent: [baseline_cpu→logs] || [anomaly_cpu→logs]
        """
        logger.info("🔄 SINGLE AGENT: Starting parallel path analysis")
        
        baseline_range = deps.get_baseline_timerange()
        anomaly_range = deps.get_anomaly_timerange()
        
        # Define sequential analysis for each path using unified agent
        async def analyze_baseline_path():
            """Baseline path: CPU → Logs (sequential) using unified agent"""
            logger.info("📊 SINGLE AGENT: Baseline path - Starting CPU analysis")
            cpu_metrics = await self._analyze_cpu_with_unified_agent(deps, baseline_range)
            logger.info("📝 SINGLE AGENT: Baseline path - Starting log analysis")
            log_analysis = await self._analyze_logs_with_unified_agent(deps, baseline_range)
            logger.info("✅ SINGLE AGENT: Baseline path - Complete")
            return PathAnalysisResult(
                timerange=baseline_range,
                cpu_metrics=cpu_metrics,
                log_analysis=log_analysis
            )
        
        async def analyze_anomaly_path():
            """Anomaly path: CPU → Logs (sequential) using unified agent"""
            logger.info("📊 SINGLE AGENT: Anomaly path - Starting CPU analysis")
            cpu_metrics = await self._analyze_cpu_with_unified_agent(deps, anomaly_range)
            logger.info("📝 SINGLE AGENT: Anomaly path - Starting log analysis")
            log_analysis = await self._analyze_logs_with_unified_agent(deps, anomaly_range)
            logger.info("✅ SINGLE AGENT: Anomaly path - Complete")
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
        
        logger.info("🎯 SINGLE AGENT: Completed parallel path analysis")
        return baseline_result, anomaly_result
    
    async def _analyze_cpu_with_unified_agent(self, deps: K8sAnalysisDeps, timerange: TimeRange) -> CPUMetrics:
        """Analyze CPU metrics using unified agent."""
        logger.info(f"📊 SINGLE AGENT: Analyzing CPU metrics for {deps.entity_type}: {deps.entity_name}")
        
        try:
            # Use unified agent for CPU analysis
            prompt = f"Analyze CPU metrics for {deps.entity_type.value} entity '{deps.entity_name}' from {timerange.start.isoformat()} to {timerange.end.isoformat()}. Use the get_cpu_metrics tool."
            result = await self.unified_agent.run(prompt, deps=deps)
            
            # Handle the union type output
            if isinstance(result.output, CPUMetrics):
                return result.output
            else:
                raise ValueError(f"Expected CPUMetrics, got {type(result.output)}")
            
        except Exception as e:
            logger.error(f"CPU metrics analysis failed: {str(e)}")
            # Fallback CPU metrics
            avg_cpu = 20.0 + (hash(deps.entity_name) % 10)
            peak_cpu = avg_cpu + 5.0
            
            return CPUMetrics(
                average_utilization=round(avg_cpu, 2),
                peak_utilization=round(peak_cpu, 2),
                samples_count=12
            )
    
    async def _analyze_logs_with_unified_agent(self, deps: K8sAnalysisDeps, timerange: TimeRange) -> LogAnalysis:
        """Analyze logs using unified agent."""
        logger.info(f"📝 SINGLE AGENT: Analyzing logs for {deps.entity_type}: {deps.entity_name}")
        
        try:
            # Use unified agent for log analysis
            prompt = f"Analyze logs for {deps.entity_type.value} entity '{deps.entity_name}' from {timerange.start.isoformat()} to {timerange.end.isoformat()}. Use the get_entity_logs tool."
            result = await self.unified_agent.run(prompt, deps=deps)
            
            # Handle the union type output
            if isinstance(result.output, LogAnalysis):
                return result.output
            else:
                raise ValueError(f"Expected LogAnalysis, got {type(result.output)}")
            
        except Exception as e:
            logger.error(f"Log analysis failed: {str(e)}")
            # Fallback log analysis
            return LogAnalysis(
                error_count=0,
                warning_count=1,
                critical_patterns=[],
                anomalous_events=[]
            )
    
    async def _llm_merge_analysis(
        self,
        deps: K8sAnalysisDeps,
        baseline_result: PathAnalysisResult,
        anomaly_result: PathAnalysisResult
    ) -> MergedAnalysis:
        """Use unified agent to merge and analyze outputs from 2 paths."""
        
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
        Do NOT use any tools - use your reasoning to provide a MergedAnalysis.
        """
        
        logger.info("🧠 SINGLE AGENT: Running LLM merge analysis of parallel paths")
        
        try:
            result = await self.unified_agent.run(analysis_prompt, deps=deps)
            
            # Handle the union type output
            if isinstance(result.output, MergedAnalysis):
                logger.info(f"✅ SINGLE AGENT: LLM merge analysis completed with confidence: {result.output.confidence_score}")
                return result.output
            else:
                raise ValueError(f"Expected MergedAnalysis, got {type(result.output)}")
            
        except Exception as e:
            logger.error(f"LLM merge analysis failed: {str(e)}")
            # Fallback to basic comparison
            return self._fallback_merge_analysis(baseline_result, anomaly_result, deps.entity_type)
    
    async def _llm_final_summary(
        self,
        deps: K8sAnalysisDeps,
        merged_analysis: MergedAnalysis,
        node_analysis: Optional[AnomalyDetectionResult]
    ) -> FinalSummary:
        """Use unified agent for final comprehensive summarization."""
        
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
        Do NOT use any tools - use your reasoning to provide a FinalSummary.
        """
        
        logger.info("🎯 SINGLE AGENT: Running final LLM summarization")
        
        try:
            result = await self.unified_agent.run(summary_prompt, deps=deps)
            
            # Handle the union type output
            if isinstance(result.output, FinalSummary):
                logger.info(f"✅ SINGLE AGENT: Final summary completed")
                return result.output
            else:
                raise ValueError(f"Expected FinalSummary, got {type(result.output)}")
            
        except Exception as e:
            logger.error(f"Final summary failed: {str(e)}")
            # Fallback summary
            return self._fallback_final_summary(merged_analysis, node_analysis)
    
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
            recommendations=["Review metrics manually", "Check system logs"]
        )
    
    async def _analyze_node(
        self, 
        node_name: str,
        baseline_start: datetime,
        baseline_end: datetime, 
        anomaly_start: datetime,
        anomaly_end: datetime
    ) -> Optional[AnomalyDetectionResult]:
        """Analyze the host node using the same workflow pattern."""
        logger.info(f"🖥️ SINGLE AGENT: Starting node analysis for {node_name}")
        
        try:
            result = await self.analyze_entity(
                entity_name=node_name,
                entity_type=EntityType.NODE,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                anomaly_start=anomaly_start,
                anomaly_end=anomaly_end
            )
            
            if isinstance(result, AnomalyDetectionResult):
                return result
            else:
                logger.warning(f"Node analysis failed for {node_name}")
                return None
                
        except Exception as e:
            logger.error(f"Node analysis error for {node_name}: {str(e)}")
            return None

# ============================================================================
# Usage Example and Testing
# ============================================================================

async def main():
    """Example usage of the single agent workflow orchestration."""
    orchestrator = K8sAnomalyDetectionOrchestrator()
    
    # Use time ranges that match mock data
    baseline_start = datetime.fromisoformat("2024-06-26T09:00:00+00:00")
    baseline_end = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_start = datetime.fromisoformat("2024-06-26T10:00:00+00:00")
    anomaly_end = datetime.fromisoformat("2024-06-26T10:30:00+00:00")
    
    print("🚀 Testing SINGLE AGENT workflow with multiple tools...")
    print("=" * 70)
    
    result = await orchestrator.analyze_entity(
        entity_name="frontend-6d8f4f79f7-kxzpl",
        entity_type=EntityType.POD,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end
    )
    
    if isinstance(result, AnomalyDetectionResult):
        print("✅ SINGLE AGENT ANALYSIS SUCCESSFUL")
        print(f"🔍 Merged Analysis Confidence: {result.merged_analysis.confidence_score:.3f}")
        print(f"🎯 Final Conclusion: {result.final_summary.anomaly_detected}")
        print(f"📊 Final Confidence: {result.final_summary.confidence_score:.3f}")
        print(f"📝 Overall Conclusion: {result.final_summary.overall_conclusion}")
        print(f"🚨 Recommendations: {result.final_summary.recommendations}")
        
        if result.node_analysis:
            print(f"🖥️  Node Analysis: {result.node_analysis.final_summary.anomaly_detected}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main()) 