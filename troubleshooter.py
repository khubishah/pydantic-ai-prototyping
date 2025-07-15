"""
PydanticAI Technical Support Demo with Comprehensive Logging

This comprehensive demo showcases the core constructs of PydanticAI:
• Multiple specialized agents with different roles
• Structured inputs/outputs using Pydantic models
• Dependency injection with dataclasses
• Tool calling with context access
• Conditional logic and authorization
• Agent coordination and delegation
• Proper async/await patterns
• Error handling and retries
• Comprehensive logging and tracing with Logfire
"""

import asyncio
import os
import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Union

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Logfire for comprehensive logging
logfire.configure(
    send_to_logfire='if-token-present',  # Only send if token is configured
    console=logfire.ConsoleOptions(
        min_log_level='info',  # Only show info level and above
        include_timestamps=True,
        verbose=False,  # Reduce verbosity
    ),
)

# Instrument PydanticAI for automatic tracing (skip low-level Pydantic validation)
logfire.instrument_pydantic_ai()
# Note: Not instrumenting Pydantic directly to reduce noise from validation calls

# 1. Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable in your .env file.")

# 2. Define structured models for inputs and outputs
class IssueType(str, Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    PERFORMANCE = "performance"
    SECURITY = "security"

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IssueAnalysis(BaseModel):
    """Structured analysis of a technical issue."""
    issue_type: IssueType = Field(description="Category of the technical issue")
    severity: Severity = Field(description="Severity level of the issue")
    summary: str = Field(description="Brief summary of the issue")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the analysis (0-1)")

class TroubleshootingStep(BaseModel):
    """A single troubleshooting step."""
    step_number: int = Field(description="Sequential step number")
    action: str = Field(description="Action to take")
    expected_outcome: str = Field(description="What should happen if successful")
    risk_level: Literal["safe", "caution", "risky"] = Field(description="Risk level of this step")

class TroubleshootingPlan(BaseModel):
    """Complete troubleshooting plan with multiple steps."""
    issue_summary: str = Field(description="Summary of the issue being addressed")
    estimated_time: str = Field(description="Estimated time to complete all steps")
    steps: list[TroubleshootingStep] = Field(description="List of troubleshooting steps")
    emergency_contact: bool = Field(description="Whether to contact IT support if steps fail")

class SystemInfo(BaseModel):
    """System information for context."""
    os_type: str = Field(description="Operating system type")
    last_update: str = Field(description="Last system update date")
    uptime_hours: int = Field(description="System uptime in hours")
    available_memory_gb: float = Field(description="Available memory in GB")

# 3. Define dependencies for agents using dataclass (best practice)
@dataclass
class TechSupportDeps:
    """Dependencies for tech support agents."""
    user_name: str
    user_role: str
    support_level: Literal["basic", "advanced", "expert"]
    session_start: datetime
    
    def is_authorized_for_advanced(self) -> bool:
        """Check if user is authorized for advanced troubleshooting."""
        with logfire.span('checking_authorization', user=self.user_name, level=self.support_level):
            authorized = self.support_level in ["advanced", "expert"]
            logfire.info('Authorization check', user=self.user_name, authorized=authorized, level=self.support_level)
            return authorized
    
    def get_session_duration(self) -> str:
        """Get current session duration."""
        duration = datetime.now() - self.session_start
        return f"{duration.total_seconds():.1f} seconds"

# 4. Create multiple specialized agents with proper configuration

# Analysis Agent - Analyzes and categorizes issues
logfire.info('Initializing Analysis Agent')
analysis_agent = Agent(
    'openai:gpt-4o',
    output_type=IssueAnalysis,
    deps_type=TechSupportDeps,
    retries=2,  # Best practice: configure retries
)

@analysis_agent.system_prompt
def analysis_system_prompt(ctx: RunContext[TechSupportDeps]) -> str:
    """Dynamic system prompt based on user context."""
    with logfire.span('generating_analysis_system_prompt', user=ctx.deps.user_name):
        prompt = f"""You are a technical issue analysis specialist for {ctx.deps.user_name} ({ctx.deps.user_role}). 
        
        Analyze user-reported technical issues and categorize them with confidence scores.
        Consider the user's role and technical level when assessing severity.
        Be thorough but concise in your analysis.
        
        User's support level: {ctx.deps.support_level}
        Session duration: {ctx.deps.get_session_duration()}"""
        
        logfire.info('Analysis system prompt generated', 
                    user=ctx.deps.user_name, 
                    role=ctx.deps.user_role,
                    support_level=ctx.deps.support_level)
        return prompt

# Planning Agent - Creates detailed troubleshooting plans
logfire.info('Initializing Planning Agent')
planning_agent = Agent(
    'openai:gpt-4o', 
    output_type=TroubleshootingPlan,
    deps_type=TechSupportDeps,
    retries=2,
)

@planning_agent.system_prompt
def planning_system_prompt(ctx: RunContext[TechSupportDeps]) -> str:
    """Dynamic system prompt for planning agent."""
    with logfire.span('generating_planning_system_prompt', user=ctx.deps.user_name):
        complexity_level = "advanced" if ctx.deps.is_authorized_for_advanced() else "basic"
        prompt = f"""You are a troubleshooting plan specialist.
        
        Create detailed, step-by-step troubleshooting plans based on issue analysis.
        Always prioritize safe steps first, and tailor complexity to user's level.
        
        User: {ctx.deps.user_name} ({ctx.deps.user_role})
        Technical Level: {complexity_level}
        Support Level: {ctx.deps.support_level}
        
        Guidelines:
        - Start with safest steps (risk_level: "safe")
        - Escalate to "caution" only if authorized
        - Never suggest "risky" steps for basic users
        - Include clear expected outcomes"""
        
        logfire.info('Planning system prompt generated',
                    user=ctx.deps.user_name,
                    complexity_level=complexity_level,
                    support_level=ctx.deps.support_level)
        return prompt

# Coordinator Agent - Orchestrates the entire process
logfire.info('Initializing Coordinator Agent')
coordinator_agent = Agent(
    'openai:gpt-4o',
    deps_type=TechSupportDeps,
    retries=3,  # Higher retries for coordinator
)

@coordinator_agent.system_prompt
def coordinator_system_prompt(ctx: RunContext[TechSupportDeps]) -> str:
    """Dynamic system prompt for coordinator."""
    with logfire.span('generating_coordinator_system_prompt', user=ctx.deps.user_name):
        prompt = f"""You are a technical support coordinator for {ctx.deps.user_name}.
        
        Use your tools to analyze issues, gather system info, and create comprehensive solutions.
        Always adapt your approach based on user authorization level and issue severity.
        
        User Details:
        - Name: {ctx.deps.user_name}
        - Role: {ctx.deps.user_role}
        - Support Level: {ctx.deps.support_level}
        - Session Duration: {ctx.deps.get_session_duration()}
        
        Process:
        1. Analyze the issue first using analyze_issue tool
        2. Gather system information with simulate_system_scan
        3. Check authorization if advanced troubleshooting is needed
        4. Create appropriate troubleshooting plan
        5. Provide emergency contacts for critical issues
        
        Be professional, helpful, and security-conscious."""
        
        logfire.info('Coordinator system prompt generated',
                    user=ctx.deps.user_name,
                    role=ctx.deps.user_role,
                    support_level=ctx.deps.support_level)
        return prompt

# 5. Define tools for the coordinator agent with proper error handling and logging

@coordinator_agent.tool_plain
def simulate_system_scan() -> SystemInfo:
    """Simulate a system information scan.
    
    Returns:
        Current system information including OS, memory, and uptime
    """
    with logfire.span('simulate_system_scan') as span:
        try:
            logfire.info('Starting system scan simulation')
            
            # Simulate system info (in real app, this would query actual system)
            os_types = ["Windows 11", "macOS Sonoma", "Ubuntu 22.04", "Windows 10"]
            system_info = SystemInfo(
                os_type=random.choice(os_types),
                last_update=f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                uptime_hours=random.randint(1, 168),  # 1 hour to 1 week
                available_memory_gb=round(random.uniform(2.0, 16.0), 1)
            )
            
            span.set_attributes({
                'os_type': system_info.os_type,
                'memory_gb': system_info.available_memory_gb,
                'uptime_hours': system_info.uptime_hours,
                'last_update': system_info.last_update
            })
            
            logfire.info('System scan completed', 
                        os_type=system_info.os_type,
                        memory_gb=system_info.available_memory_gb,
                        uptime_hours=system_info.uptime_hours)
            
            return system_info
            
        except Exception as e:
            logfire.error('System scan failed', error=str(e))
            raise ModelRetry(f"Failed to scan system: {str(e)}")

@coordinator_agent.tool
async def analyze_issue(ctx: RunContext[TechSupportDeps], issue_description: str) -> IssueAnalysis:
    """Analyze a technical issue using the specialized analysis agent.
    
    Args:
        ctx: Runtime context with dependencies
        issue_description: Description of the technical issue
        
    Returns:
        Structured analysis of the issue
    """
    with logfire.span('analyze_issue_tool', user=ctx.deps.user_name, issue=issue_description[:100]) as span:
        try:
            logfire.info('Delegating to analysis agent', 
                        user=ctx.deps.user_name,
                        issue_preview=issue_description[:100])
            
            result = await analysis_agent.run(
                f"Analyze this technical issue: {issue_description}",
                deps=ctx.deps,
                usage=ctx.usage,  # Best practice: pass usage for tracking
            )
            
            analysis = result.output
            span.set_attributes({
                'issue_type': analysis.issue_type,
                'severity': analysis.severity,
                'confidence': analysis.confidence
            })
            
            logfire.info('Issue analysis completed',
                        user=ctx.deps.user_name,
                        issue_type=analysis.issue_type,
                        severity=analysis.severity,
                        confidence=analysis.confidence,
                        summary=analysis.summary[:100])
            
            return analysis
            
        except Exception as e:
            logfire.error('Issue analysis failed', user=ctx.deps.user_name, error=str(e))
            raise ModelRetry(f"Issue analysis failed: {str(e)}")

@coordinator_agent.tool
async def create_troubleshooting_plan(
    ctx: RunContext[TechSupportDeps], 
    issue_analysis: IssueAnalysis, 
    system_info: SystemInfo
) -> TroubleshootingPlan:
    """Create a detailed troubleshooting plan.
    
    Args:
        ctx: Runtime context with dependencies
        issue_analysis: Analysis of the issue
        system_info: Current system information
        
    Returns:
        Detailed troubleshooting plan
    """
    with logfire.span('create_troubleshooting_plan_tool', 
                      user=ctx.deps.user_name,
                      issue_type=issue_analysis.issue_type,
                      severity=issue_analysis.severity) as span:
        try:
            logfire.info('Creating troubleshooting plan',
                        user=ctx.deps.user_name,
                        issue_type=issue_analysis.issue_type,
                        severity=issue_analysis.severity,
                        os_type=system_info.os_type)
            
            # Convert Pydantic models to strings for the prompt
            analysis_str = f"Type: {issue_analysis.issue_type}, Severity: {issue_analysis.severity}, Summary: {issue_analysis.summary}"
            system_str = f"OS: {system_info.os_type}, Memory: {system_info.available_memory_gb}GB, Uptime: {system_info.uptime_hours}h"
            
            prompt = f"""Create a troubleshooting plan for:
            Issue Analysis: {analysis_str}
            System Info: {system_str}
            
            Consider the user's technical level and authorization."""
            
            logfire.info('Delegating to planning agent', user=ctx.deps.user_name)
            
            result = await planning_agent.run(
                prompt, 
                deps=ctx.deps,
                usage=ctx.usage,
            )
            
            plan = result.output
            span.set_attributes({
                'step_count': len(plan.steps),
                'estimated_time': plan.estimated_time,
                'emergency_contact': plan.emergency_contact
            })
            
            logfire.info('Troubleshooting plan created',
                        user=ctx.deps.user_name,
                        step_count=len(plan.steps),
                        estimated_time=plan.estimated_time,
                        emergency_contact=plan.emergency_contact)
            
            return plan
            
        except Exception as e:
            logfire.error('Plan creation failed', user=ctx.deps.user_name, error=str(e))
            raise ModelRetry(f"Plan creation failed: {str(e)}")

@coordinator_agent.tool
def check_authorization(ctx: RunContext[TechSupportDeps], required_level: str) -> dict[str, Union[bool, str]]:
    """Check if user is authorized for a specific support level.
    
    Args:
        ctx: Runtime context with dependencies
        required_level: Required authorization level
        
    Returns:
        Authorization status and message
    """
    with logfire.span('check_authorization_tool', 
                      user=ctx.deps.user_name,
                      required_level=required_level) as span:
        try:
            logfire.info('Checking user authorization',
                        user=ctx.deps.user_name,
                        current_level=ctx.deps.support_level,
                        required_level=required_level)
            
            is_authorized = (
                ctx.deps.support_level in ["advanced", "expert"] 
                if required_level == "advanced" 
                else True
            )
            
            result = {
                "authorized": is_authorized,
                "user_level": ctx.deps.support_level,
                "message": f"User {ctx.deps.user_name} {'is' if is_authorized else 'is not'} authorized for {required_level} support",
                "session_duration": ctx.deps.get_session_duration()
            }
            
            span.set_attributes({
                'authorized': is_authorized,
                'user_level': ctx.deps.support_level,
                'required_level': required_level
            })
            
            logfire.info('Authorization check completed',
                        user=ctx.deps.user_name,
                        authorized=is_authorized,
                        current_level=ctx.deps.support_level,
                        required_level=required_level)
            
            return result
            
        except Exception as e:
            logfire.error('Authorization check failed', user=ctx.deps.user_name, error=str(e))
            return {
                "authorized": False,
                "user_level": "unknown",
                "message": f"Authorization check failed: {str(e)}",
                "session_duration": "unknown"
            }

@coordinator_agent.tool_plain
def get_emergency_contacts() -> dict[str, str]:
    """Get emergency IT support contacts.
    
    Returns:
        Dictionary of emergency contact information
    """
    with logfire.span('get_emergency_contacts_tool'):
        logfire.info('Retrieving emergency contacts')
        
        contacts = {
            "helpdesk": "1-800-HELP-DESK",
            "emergency": "1-800-EMERGENCY", 
            "email": "support@company.com",
            "hours": "24/7 for critical issues",
            "escalation": "For security issues, contact security@company.com immediately"
        }
        
        logfire.info('Emergency contacts retrieved', contact_count=len(contacts))
        return contacts

# 6. Main application logic with proper async patterns and error handling

async def handle_support_request(issue_description: str, user_name: str, user_role: str, support_level: str) -> None:
    """Handle a complete support request with conditional logic and error handling."""
    
    with logfire.span('handle_support_request', 
                      user=user_name, 
                      role=user_role, 
                      level=support_level,
                      issue=issue_description[:100]) as main_span:
        
        # Create dependencies
        deps = TechSupportDeps(
            user_name=user_name,
            user_role=user_role,
            support_level=support_level,
            session_start=datetime.now()
        )
        
        logfire.info('Starting support session',
                    user=user_name,
                    role=user_role,
                    support_level=support_level,
                    issue_preview=issue_description[:100])
        
        print(f"\n🔧 Starting support session for {user_name} ({user_role})")
        print(f"📊 Support Level: {support_level}")
        print(f"❓ Issue: {issue_description}")
        print("-" * 60)
        
        # Use coordinator agent to handle the request
        prompt = f"""Help resolve this technical issue: {issue_description}
        
        Please follow this process:
        1. Analyze the issue first to understand its nature and severity
        2. Gather system information to provide context
        3. Check if advanced troubleshooting is needed based on the analysis
        4. Create an appropriate troubleshooting plan tailored to the user's level
        5. Provide emergency contacts if the issue is critical or security-related
        
        Adapt your response based on the user's authorization level and provide clear, actionable guidance."""
        
        try:
            logfire.info('Delegating to coordinator agent', user=user_name)
            
            result = await coordinator_agent.run(prompt, deps=deps)
            
            print(f"\n✅ Coordinator Response:\n{result.output}")
            
            # Show usage statistics (best practice)
            usage = result.usage()
            main_span.set_attributes({
                'total_requests': usage.requests,
                'total_tokens': usage.total_tokens,
                'request_tokens': usage.request_tokens,
                'response_tokens': usage.response_tokens,
                'session_duration': deps.get_session_duration()
            })
            
            print(f"\n📈 Session Stats:")
            print(f"   • Total requests: {usage.requests}")
            print(f"   • Total tokens: {usage.total_tokens}")
            print(f"   • Request tokens: {usage.request_tokens}")
            print(f"   • Response tokens: {usage.response_tokens}")
            print(f"   • Session duration: {deps.get_session_duration()}")
            
            logfire.info('Support session completed successfully',
                        user=user_name,
                        total_requests=usage.requests,
                        total_tokens=usage.total_tokens,
                        session_duration=deps.get_session_duration())
            
        except Exception as e:
            logfire.error('Support session failed', 
                         user=user_name, 
                         error=str(e),
                         issue=issue_description[:100])
            print(f"\n❌ Error during support session: {e}")
            print("🆘 Please contact emergency support if this is a critical issue.")

# 7. Interactive CLI with comprehensive scenarios

async def run_demo_scenario(scenario_num: int, scenarios: list[dict]) -> None:
    """Run a specific demo scenario."""
    scenario = scenarios[scenario_num - 1]
    
    with logfire.span('run_demo_scenario', scenario_num=scenario_num, description=scenario['description']):
        logfire.info('Starting demo scenario',
                    scenario_num=scenario_num,
                    description=scenario['description'],
                    user=scenario['user'])
        
        print(f"\n🎯 Running scenario {scenario_num}: {scenario['description']}")
        
        await handle_support_request(
            scenario["issue"],
            scenario["user"], 
            scenario["role"],
            scenario["level"]
        )
        
        logfire.info('Demo scenario completed', scenario_num=scenario_num)

async def run_custom_scenario() -> None:
    """Run a custom user-defined scenario."""
    with logfire.span('run_custom_scenario'):
        logfire.info('Starting custom scenario setup')
        
        print("\n📝 Custom Scenario Setup")
        issue = input("Describe the technical issue: ")
        user = input("Your name: ")
        role = input("Your role: ")
        level = input("Support level (basic/advanced/expert): ")
        
        if level not in ["basic", "advanced", "expert"]:
            print("❌ Invalid support level. Using 'basic'.")
            level = "basic"
        
        logfire.info('Custom scenario configured',
                    user=user,
                    role=role,
                    level=level,
                    issue=issue[:100])
        
        await handle_support_request(issue, user, role, level)

def main() -> None:
    """Main interactive CLI with demonstration scenarios."""
    
    with logfire.span('main_application'):
        logfire.info('Starting PydanticAI Technical Support Demo')
        
        print("🚀 PydanticAI Technical Support Demo")
        print("=" * 50)
        print("This demo showcases PydanticAI best practices:")
        print("• Multiple specialized agents with role-based prompts")
        print("• Structured inputs/outputs with Pydantic models")
        print("• Dependency injection using dataclasses")
        print("• Conditional logic and authorization checks")
        print("• Tool calling with proper context access")
        print("• Agent coordination and delegation")
        print("• Async/await patterns and error handling")
        print("• Usage tracking and session management")
        print("• Comprehensive logging and tracing with Logfire")
        print("=" * 50)
        
        # Pre-defined scenarios for comprehensive testing
        scenarios = [
            {
                "description": "Performance Issue - Basic User",
                "issue": "My computer is running very slowly and keeps freezing",
                "user": "Alice Johnson",
                "role": "Marketing Manager", 
                "level": "basic"
            },
            {
                "description": "Network Issue - Expert User",
                "issue": "Network connectivity issues, can't access internal servers",
                "user": "Bob Smith",
                "role": "IT Administrator",
                "level": "expert"
            },
            {
                "description": "Security Issue - Advanced User",
                "issue": "Suspicious email attachments, potential security breach",
                "user": "Carol Wilson", 
                "role": "Security Analyst",
                "level": "advanced"
            },
            {
                "description": "Hardware Issue - Basic User",
                "issue": "My laptop won't turn on and the power LED is blinking",
                "user": "David Brown",
                "role": "Sales Representative",
                "level": "basic"
            }
        ]
        
        while True:
            print("\n📋 Choose an option:")
            for i, scenario in enumerate(scenarios, 1):
                print(f"{i}. {scenario['description']}")
            print(f"{len(scenarios) + 1}. Enter custom issue")
            print(f"{len(scenarios) + 2}. Exit")
            
            try:
                choice = input(f"\nEnter your choice (1-{len(scenarios) + 2}): ").strip()
                
                with logfire.span('user_choice', choice=choice):
                    if choice == str(len(scenarios) + 2):
                        logfire.info('User chose to exit')
                        print("👋 Goodbye!")
                        break
                    elif choice in [str(i) for i in range(1, len(scenarios) + 1)]:
                        logfire.info('User chose demo scenario', choice=choice)
                        asyncio.run(run_demo_scenario(int(choice), scenarios))
                    elif choice == str(len(scenarios) + 1):
                        logfire.info('User chose custom scenario')
                        asyncio.run(run_custom_scenario())
                    else:
                        logfire.warn('Invalid user choice', choice=choice)
                        print("❌ Invalid choice. Please try again.")
                        
            except KeyboardInterrupt:
                logfire.info('User interrupted with Ctrl+C')
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logfire.error('Unexpected error in main loop', error=str(e))
                print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 