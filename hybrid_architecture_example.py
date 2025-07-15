"""
Hybrid Architecture: Deterministic Workflow Control + Adaptive Agent Intelligence

This example demonstrates how to combine:
1. Deterministic workflow orchestration (using state machines/graphs)
2. Adaptive agent intelligence within workflow nodes
3. Structured data validation throughout
4. Configurable execution strategies
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Dict, Any, List
from abc import ABC, abstractmethod

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Configure minimal logging
logfire.configure(send_to_logfire=False, console=logfire.ConsoleOptions())

# ============================================================================
# 1. STRUCTURED DATA MODELS (Pydantic for type safety)
# ============================================================================

class WorkflowStage(str, Enum):
    INTAKE = "intake"
    ANALYSIS = "analysis" 
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    COMPLETE = "complete"

class ExecutionStrategy(str, Enum):
    DETERMINISTIC = "deterministic"  # Fixed workflow steps
    ADAPTIVE = "adaptive"           # Agent-driven decisions
    HYBRID = "hybrid"              # Combination of both

class TaskRequest(BaseModel):
    """Structured input for any workflow task"""
    id: str = Field(description="Unique task identifier")
    description: str = Field(description="Task description")
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    strategy: ExecutionStrategy = ExecutionStrategy.HYBRID
    context: Dict[str, Any] = Field(default_factory=dict)

class WorkflowState(BaseModel):
    """Complete workflow state with type safety"""
    task: TaskRequest
    current_stage: WorkflowStage = WorkflowStage.INTAKE
    stage_results: Dict[WorkflowStage, Any] = Field(default_factory=dict)
    execution_log: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def log_action(self, action: str):
        """Add timestamped action to execution log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.execution_log.append(f"{timestamp}: {action}")

# ============================================================================
# 2. WORKFLOW ORCHESTRATOR (Deterministic Control)
# ============================================================================

class WorkflowNode(ABC):
    """Abstract base for workflow nodes"""
    
    @abstractmethod
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute this workflow node"""
        pass
    
    @abstractmethod
    def can_skip(self, state: WorkflowState) -> bool:
        """Determine if this node can be skipped based on state"""
        pass

class WorkflowOrchestrator:
    """Deterministic workflow orchestrator with adaptive nodes"""
    
    def __init__(self):
        self.nodes: Dict[WorkflowStage, WorkflowNode] = {}
        self.transitions: Dict[WorkflowStage, List[WorkflowStage]] = {
            WorkflowStage.INTAKE: [WorkflowStage.ANALYSIS],
            WorkflowStage.ANALYSIS: [WorkflowStage.PLANNING],
            WorkflowStage.PLANNING: [WorkflowStage.EXECUTION],
            WorkflowStage.EXECUTION: [WorkflowStage.REVIEW],
            WorkflowStage.REVIEW: [WorkflowStage.COMPLETE, WorkflowStage.PLANNING],  # Can loop back
            WorkflowStage.COMPLETE: []
        }
    
    def register_node(self, stage: WorkflowStage, node: WorkflowNode):
        """Register a node for a specific workflow stage"""
        self.nodes[stage] = node
    
    async def execute_workflow(self, task: TaskRequest) -> WorkflowState:
        """Execute the complete workflow with deterministic control"""
        state = WorkflowState(task=task)
        state.log_action(f"Started workflow for task: {task.id}")
        
        with logfire.span('workflow_execution', task_id=task.id, strategy=task.strategy):
            while state.current_stage != WorkflowStage.COMPLETE:
                current_node = self.nodes.get(state.current_stage)
                
                if not current_node:
                    raise ValueError(f"No node registered for stage: {state.current_stage}")
                
                # Execute current node
                state.log_action(f"Executing stage: {state.current_stage}")
                
                if current_node.can_skip(state):
                    state.log_action(f"Skipping stage: {state.current_stage}")
                else:
                    state = await current_node.execute(state)
                
                # Determine next stage (deterministic transitions)
                next_stage = await self._determine_next_stage(state)
                state.current_stage = next_stage
                state.log_action(f"Transitioning to: {next_stage}")
            
            state.completed_at = datetime.now()
            state.log_action("Workflow completed")
            
        return state
    
    async def _determine_next_stage(self, state: WorkflowState) -> WorkflowStage:
        """Determine next stage based on current state and results"""
        current_stage = state.current_stage
        possible_transitions = self.transitions.get(current_stage, [])
        
        if not possible_transitions:
            return WorkflowStage.COMPLETE
        
        # For hybrid strategy, use agent to make transition decisions
        if (state.task.strategy == ExecutionStrategy.HYBRID and 
            len(possible_transitions) > 1):
            return await self._agent_decide_transition(state, possible_transitions)
        
        # Default: take first available transition
        return possible_transitions[0]
    
    async def _agent_decide_transition(self, state: WorkflowState, 
                                     options: List[WorkflowStage]) -> WorkflowStage:
        """Use agent intelligence to decide workflow transitions"""
        # This is where agent intelligence enhances deterministic workflow
        decision_agent = Agent(
            'openai:gpt-4o-mini',
            system_prompt=f"""You are a workflow transition decision agent.
            
            Current stage: {state.current_stage}
            Available next stages: {[s.value for s in options]}
            Stage results so far: {state.stage_results}
            
            Choose the most appropriate next stage based on the results."""
        )
        
        try:
            result = await decision_agent.run(
                f"Based on the workflow state, which stage should we transition to next? "
                f"Options: {[s.value for s in options]}. "
                f"Respond with just the stage name."
            )
            
            # Parse agent response back to enum
            for option in options:
                if option.value.lower() in result.data.lower():
                    return option
                    
        except Exception as e:
            logfire.warning('Agent transition decision failed', error=str(e))
        
        # Fallback to first option
        return options[0]

# ============================================================================
# 3. ADAPTIVE WORKFLOW NODES (Agent Intelligence)
# ============================================================================

@dataclass
class WorkflowDeps:
    """Dependencies available to workflow nodes"""
    user_context: Dict[str, Any]
    execution_strategy: ExecutionStrategy
    
class AnalysisNode(WorkflowNode):
    """Analysis stage with adaptive agent intelligence"""
    
    def __init__(self):
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDeps,
            system_prompt="""You are an analysis specialist.
            
            Analyze the given task and provide structured insights:
            1. Task complexity assessment
            2. Required resources and skills
            3. Potential risks and challenges
            4. Recommended approach
            
            Adapt your analysis depth based on the execution strategy."""
        )
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute analysis with agent intelligence"""
        with logfire.span('analysis_node', task_id=state.task.id):
            deps = WorkflowDeps(
                user_context=state.task.context,
                execution_strategy=state.task.strategy
            )
            
            prompt = f"""Analyze this task:
            Description: {state.task.description}
            Priority: {state.task.priority}
            Strategy: {state.task.strategy}
            
            Provide a structured analysis."""
            
            result = await self.agent.run(prompt, deps=deps)
            
            # Store structured result
            state.stage_results[WorkflowStage.ANALYSIS] = {
                "analysis": result.data,
                "complexity": "medium",  # Could be extracted from agent response
                "estimated_duration": "30 minutes"
            }
            
            state.log_action("Analysis completed")
            return state
    
    def can_skip(self, state: WorkflowState) -> bool:
        """Skip analysis for low-priority deterministic tasks"""
        return (state.task.priority == "low" and 
                state.task.strategy == ExecutionStrategy.DETERMINISTIC)

class PlanningNode(WorkflowNode):
    """Planning stage with conditional agent involvement"""
    
    def __init__(self):
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDeps,
            system_prompt="""You are a planning specialist.
            
            Create detailed execution plans based on analysis results.
            Consider the execution strategy when determining plan detail level."""
        )
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute planning with strategy-dependent intelligence"""
        with logfire.span('planning_node', task_id=state.task.id):
            if state.task.strategy == ExecutionStrategy.DETERMINISTIC:
                # Simple, predefined planning
                plan = self._create_deterministic_plan(state)
            else:
                # Agent-driven adaptive planning
                plan = await self._create_adaptive_plan(state)
            
            state.stage_results[WorkflowStage.PLANNING] = plan
            state.log_action("Planning completed")
            return state
    
    def _create_deterministic_plan(self, state: WorkflowState) -> Dict[str, Any]:
        """Create a simple, predefined plan"""
        return {
            "type": "deterministic",
            "steps": [
                "Step 1: Initialize",
                "Step 2: Process",
                "Step 3: Finalize"
            ],
            "estimated_time": "15 minutes"
        }
    
    async def _create_adaptive_plan(self, state: WorkflowState) -> Dict[str, Any]:
        """Create an intelligent, context-aware plan"""
        deps = WorkflowDeps(
            user_context=state.task.context,
            execution_strategy=state.task.strategy
        )
        
        analysis_result = state.stage_results.get(WorkflowStage.ANALYSIS, {})
        
        prompt = f"""Create a detailed execution plan based on:
        Task: {state.task.description}
        Analysis: {analysis_result.get('analysis', 'No analysis available')}
        
        Provide specific, actionable steps."""
        
        result = await self.agent.run(prompt, deps=deps)
        
        return {
            "type": "adaptive",
            "plan": result.data,
            "estimated_time": "30 minutes"
        }
    
    def can_skip(self, state: WorkflowState) -> bool:
        """Never skip planning"""
        return False

class ExecutionNode(WorkflowNode):
    """Execution stage with tool coordination"""
    
    def __init__(self):
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDeps,
            system_prompt="""You are an execution coordinator.
            
            Execute plans by coordinating available tools and resources.
            Adapt execution based on real-time feedback."""
        )
    
    @self.agent.tool
    async def simulate_task_execution(self, ctx: RunContext[WorkflowDeps], 
                                    task_step: str) -> str:
        """Simulate executing a task step"""
        # In real implementation, this would call actual tools/APIs
        return f"Executed: {task_step} - Success"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the planned tasks"""
        with logfire.span('execution_node', task_id=state.task.id):
            plan = state.stage_results.get(WorkflowStage.PLANNING, {})
            
            deps = WorkflowDeps(
                user_context=state.task.context,
                execution_strategy=state.task.strategy
            )
            
            prompt = f"""Execute this plan: {plan}
            
            Use the simulate_task_execution tool for each step.
            Provide a summary of execution results."""
            
            result = await self.agent.run(prompt, deps=deps)
            
            state.stage_results[WorkflowStage.EXECUTION] = {
                "results": result.data,
                "status": "completed",
                "tools_used": ["simulate_task_execution"]
            }
            
            state.log_action("Execution completed")
            return state
    
    def can_skip(self, state: WorkflowState) -> bool:
        """Never skip execution"""
        return False

class ReviewNode(WorkflowNode):
    """Review stage with quality assessment"""
    
    def __init__(self):
        self.agent = Agent(
            'openai:gpt-4o-mini',
            deps_type=WorkflowDeps,
            system_prompt="""You are a quality review specialist.
            
            Review execution results and determine if they meet requirements.
            Decide if the task is complete or needs rework."""
        )
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Review execution results"""
        with logfire.span('review_node', task_id=state.task.id):
            execution_results = state.stage_results.get(WorkflowStage.EXECUTION, {})
            
            deps = WorkflowDeps(
                user_context=state.task.context,
                execution_strategy=state.task.strategy
            )
            
            prompt = f"""Review these execution results:
            {execution_results}
            
            Determine if the task is complete or needs rework.
            Respond with 'APPROVED' or 'NEEDS_REWORK' and provide reasoning."""
            
            result = await self.agent.run(prompt, deps=deps)
            
            # Parse agent decision
            approved = "APPROVED" in result.data.upper()
            
            state.stage_results[WorkflowStage.REVIEW] = {
                "review": result.data,
                "approved": approved,
                "reviewer": "AI Agent"
            }
            
            state.log_action(f"Review completed - {'Approved' if approved else 'Needs rework'}")
            return state
    
    def can_skip(self, state: WorkflowState) -> bool:
        """Skip review for low-priority deterministic tasks"""
        return (state.task.priority == "low" and 
                state.task.strategy == ExecutionStrategy.DETERMINISTIC)

# ============================================================================
# 4. CONFIGURATION AND EXECUTION
# ============================================================================

async def create_configured_workflow() -> WorkflowOrchestrator:
    """Create a fully configured workflow orchestrator"""
    orchestrator = WorkflowOrchestrator()
    
    # Register nodes for each stage
    orchestrator.register_node(WorkflowStage.ANALYSIS, AnalysisNode())
    orchestrator.register_node(WorkflowStage.PLANNING, PlanningNode())
    orchestrator.register_node(WorkflowStage.EXECUTION, ExecutionNode())
    orchestrator.register_node(WorkflowStage.REVIEW, ReviewNode())
    
    return orchestrator

async def demonstrate_hybrid_architecture():
    """Demonstrate the hybrid architecture with different execution strategies"""
    
    orchestrator = await create_configured_workflow()
    
    # Test different execution strategies
    test_tasks = [
        TaskRequest(
            id="task-001",
            description="Process customer support ticket about billing inquiry",
            priority="medium",
            strategy=ExecutionStrategy.DETERMINISTIC,
            context={"customer_tier": "premium"}
        ),
        TaskRequest(
            id="task-002", 
            description="Analyze complex security incident and recommend response",
            priority="critical",
            strategy=ExecutionStrategy.ADAPTIVE,
            context={"incident_type": "data_breach", "severity": "high"}
        ),
        TaskRequest(
            id="task-003",
            description="Create marketing campaign for new product launch",
            priority="high",
            strategy=ExecutionStrategy.HYBRID,
            context={"product": "AI Assistant", "target_audience": "developers"}
        )
    ]
    
    for task in test_tasks:
        print(f"\n{'='*60}")
        print(f"🚀 Executing Task: {task.id}")
        print(f"📝 Description: {task.description}")
        print(f"⚡ Strategy: {task.strategy}")
        print(f"🎯 Priority: {task.priority}")
        print(f"{'='*60}")
        
        try:
            result = await orchestrator.execute_workflow(task)
            
            print(f"\n✅ Workflow completed in {(result.completed_at - result.started_at).total_seconds():.1f}s")
            print(f"📊 Stages executed: {list(result.stage_results.keys())}")
            
            # Show execution log
            print(f"\n📋 Execution Log:")
            for log_entry in result.execution_log[-5:]:  # Show last 5 entries
                print(f"   {log_entry}")
            
            # Show final results
            if WorkflowStage.REVIEW in result.stage_results:
                review = result.stage_results[WorkflowStage.REVIEW]
                status = "✅ APPROVED" if review.get("approved") else "❌ NEEDS REWORK"
                print(f"\n🔍 Final Status: {status}")
        
        except Exception as e:
            print(f"❌ Workflow failed: {e}")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🏗️  Hybrid Architecture Demo: Deterministic + Adaptive")
    print("=" * 60)
    
    asyncio.run(demonstrate_hybrid_architecture()) 