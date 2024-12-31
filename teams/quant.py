import os
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
from autogen_core import DefaultTopicId, TopicId, MessageContext, RoutedAgent, default_subscription, message_handler, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage, LLMMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from tools.quant_toolbox import (
    fetch_market_data,
    calculate_technical_indicators,
    generate_plot,
    calculate_performance_metrics,
    calculate_metrics
)
from prompts.agent_prompts import (
    REQUIREMENTS_GATHERER_PROMPT,
    STRATEGIST_PROMPT,
    QUANT_PROMPT,
    PROGRAMMER_PROMPT
)
import re
import traceback
import tiktoken

# Update logging configuration to use a file handler instead of console
# First, disable existing root logger handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)

# Configure file logging only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_analysis.log'),  # Log to file instead of console
    ]
)

# Set console output to ERROR level only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Initialize tokenizer for GPT-4
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

class State(Enum):
    TASK_RECEIVED = "task_received"
    REQUIREMENTS_GATHERING = "requirements_gathering"
    STRATEGY_DESIGN = "strategy_design"
    QUANT_ANALYSIS = "quant_analysis"
    CODE_WRITING = "code_writing"
    REVIEWING = "reviewing"
    EXECUTING = "executing"
    DEBUGGING = "debugging"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class StateTransition:
    from_state: State
    to_state: State
    timestamp: datetime
    metadata: dict

class StateManager:
    def __init__(self):
        self.current_state: State = State.TASK_RECEIVED
        self.history: List[StateTransition] = []
        self.session_id: Optional[str] = None
        self.error_context: Optional[dict] = None

    def transition_to(self, new_state: State, metadata: dict = None) -> bool:
        """Attempt to transition to a new state with validation."""
        if self._is_valid_transition(new_state):
            transition = StateTransition(
                from_state=self.current_state,
                to_state=new_state,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.history.append(transition)
            self.current_state = new_state
            logger.info(f"State transition: {transition.from_state.value} -> {transition.to_state.value}")
            return True
        return False

    def _is_valid_transition(self, new_state: State) -> bool:
        """Validate state transitions based on the defined workflow."""
        valid_transitions = {
            State.TASK_RECEIVED: [State.CODE_WRITING],
            State.CODE_WRITING: [State.REVIEWING],
            State.REVIEWING: [State.EXECUTING, State.CODE_WRITING],
            State.EXECUTING: [State.REPORTING, State.DEBUGGING],
            State.DEBUGGING: [State.EXECUTING],
            State.REPORTING: [State.COMPLETED],
            State.COMPLETED: [State.TASK_RECEIVED],  # Allow starting a new task
            State.ERROR: [State.TASK_RECEIVED, State.CODE_WRITING, State.DEBUGGING]  # Recovery paths
        }
        return new_state in valid_transitions.get(self.current_state, [])

@dataclass
class CodeWritingTask:
    task: str

@dataclass
class CodeWritingResult:
    task: str
    code: str
    review: str

@dataclass
class CodeReviewTask:
    session_id: str
    code_writing_task: str
    code_writing_scratchpad: str
    code: str

@dataclass
class CodeReviewResult:
    review: str
    session_id: str
    approved: bool

@dataclass
class DebugTask:
    session_id: str
    code: str
    error_context: dict
    execution_output: str

@dataclass
class DebugResult:
    session_id: str
    fixed_code: str
    debug_report: str
    success: bool

@dataclass
class ReportTask:
    session_id: str
    task: str
    code: str
    execution_output: str
    state_history: List[StateTransition]

@dataclass
class ConsultantTask:
    """Initial task or request from the user."""
    request: str
    is_predefined: bool = False

@dataclass
class ConsultantResult:
    """Final requirements after consultation."""
    requirements_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsultantQuestion:
    """Question from consultant to user."""
    question: str
    context: str = ""

@dataclass
class UserResponse:
    """User's response to consultant's question."""
    response: str

@default_subscription
class UserAgent(RoutedAgent):
    """Agent representing the human user in the conversation."""
    
    def __init__(self) -> None:
        super().__init__("A human user providing requirements and feedback.")
        self._pending_questions: List[ConsultantQuestion] = []
    
    @message_handler(match=lambda msg, ctx: isinstance(msg, ConsultantQuestion))
    async def handle_question(self, message: ConsultantQuestion, ctx: MessageContext) -> None:
        """Handle questions from the consultant."""
        self._pending_questions.append(message)
        print(f"\nConsultant: {message.question}")
        if message.context:
            print(f"Context: {message.context}")
        
        response = input("Your response: ").strip()
        await self.publish_message(
            UserResponse(response=response),
            TopicId(type="consulting", source=self.id.key)
        )

@default_subscription
class ConsultantAgent(RoutedAgent):
    """Agent that gathers and clarifies requirements from the user."""
    
    def __init__(self, model_client: OpenAIChatCompletionClient, state_manager: StateManager) -> None:
        super().__init__("A consulting specialist that gathers and clarifies requirements.")
        self._model_client = model_client
        self._state_manager = state_manager
        self._conversation_history: Dict[str, List[Union[ConsultantQuestion, UserResponse]]] = {}
        self._max_questions = 3  # Reduced from 5 to 3 for more efficiency
        self._active_session: Optional[str] = None
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are a consultant that has a natural conversation with users about their investment goals.

Be concise and conversational. Adapt your style based on the user's responses:
- If the user is brief, ask focused follow-up questions
- If the user is detailed, summarize and confirm understanding
- If the initial request is clear, ask fewer questions

Focus on understanding:
- Investment goals and risk tolerance
- Trading timeframes and preferences
- Performance expectations

DO NOT:
- Ask about technical details
- List multiple questions at once (unless the context requires it)
- Drag the conversation longer than necessary

Keep the interaction efficient:
1. Start with the most important missing information
2. Use the user's language and terminology
3. Move to summary when you have sufficient information
4. Recognize when initial request contains enough detail

When you have enough information or the user indicates completion, provide a concise summary of their objectives and preferences.

Remember: User fatigue is a concern - gather information efficiently and move forward."""
            )
        ]
        
    def _get_active_session(self) -> str:
        """Get the active session ID or raise an error if none exists."""
        if not self._active_session or self._active_session not in self._conversation_history:
            raise RuntimeError("No active consultation session found")
        return self._active_session
    
    def _cleanup_old_sessions(self, max_sessions: int = 10) -> None:
        """Clean up old sessions if we have too many."""
        if len(self._conversation_history) > max_sessions:
            # Keep only the most recent sessions
            sorted_sessions = sorted(self._conversation_history.keys())
            for old_session in sorted_sessions[:-max_sessions]:
                if old_session != self._active_session:
                    del self._conversation_history[old_session]
    
    @message_handler(match=lambda msg, ctx: isinstance(msg, ConsultantTask))
    async def handle_task(self, message: ConsultantTask, ctx: MessageContext) -> None:
        """Handle initial task from user."""
        session_id = str(uuid.uuid4())
        self._active_session = session_id
        self._conversation_history[session_id] = []
        self._cleanup_old_sessions()
        
        if message.is_predefined:
            # For predefined tasks, we can skip the consultation
            await self.publish_message(
                ConsultantResult(
                    requirements_summary=message.request,
                    metadata={"session_id": session_id, "is_predefined": True, "status": "REQUIREMENTS GATHERED"}
                ),
                TopicId(type="code_writing", source=self.id.key)
            )
            return
        
        # First, analyze if the initial request has enough information
        analysis = await self._model_client.create(
            self._system_messages + [
                UserMessage(content=f"Analyze if this request contains enough information to proceed: {message.request}", source="user"),
                UserMessage(content="If the request contains clear investment goals, risk tolerance, and timeframe, respond with 'REQUIREMENTS GATHERED'. Otherwise, identify what's missing.", source="consultant")
            ],
            cancellation_token=ctx.cancellation_token
        )
        
        if "REQUIREMENTS GATHERED" in analysis.content:
            # If we have enough information, generate summary and proceed
            summary = await self._model_client.create(
                self._system_messages + [
                    UserMessage(content=f"User request: {message.request}", source="user"),
                    UserMessage(content="REQUIREMENTS GATHERED. Please provide a concise summary of the requirements.", source="consultant")
                ],
                cancellation_token=ctx.cancellation_token
            )
            
            # Transition to CODE_WRITING state
            self._state_manager.transition_to(State.CODE_WRITING, {
                "session_id": session_id,
                "requirements": summary.content,
                "status": "REQUIREMENTS GATHERED"
            })
            
            await self.publish_message(
                ConsultantResult(
                    requirements_summary=summary.content,
                    metadata={"session_id": session_id, "is_predefined": False, "status": "REQUIREMENTS GATHERED"}
                ),
                TopicId(type="code_writing", source=self.id.key)
            )
            return
        
        # If we need more information, start consultation
        response = await self._model_client.create(
            self._system_messages + [
                UserMessage(content=f"User request: {message.request}", source="user"),
                UserMessage(content="Ask the most important missing information first.", source="consultant")
            ],
            cancellation_token=ctx.cancellation_token
        )
        
        first_question = ConsultantQuestion(
            question=response.content,
            context=f"Initial request: {message.request}"
        )
        self._conversation_history[session_id].append(first_question)
        
        await self.publish_message(
            first_question,
            TopicId(type="user_interaction", source=self.id.key)
        )
    
    @message_handler(match=lambda msg, ctx: isinstance(msg, UserResponse))
    async def handle_response(self, message: UserResponse, ctx: MessageContext) -> None:
        """Handle user's response and decide next action."""
        session_id = list(self._conversation_history.keys())[-1]  # Get most recent session
        history = self._conversation_history[session_id]
        history.append(message)
        
        # Build conversation context
        conversation = []
        for item in history:
            if isinstance(item, ConsultantQuestion):
                conversation.append(UserMessage(content=f"Consultant: {item.question}", source="consultant"))
            else:
                conversation.append(UserMessage(content=f"User: {message.response}", source="user"))
        
        # Get the last question asked
        last_question = next((item for item in reversed(history) if isinstance(item, ConsultantQuestion)), None)
        
        # Check if we're in the confirmation phase
        if last_question and "Is this correct?" in last_question.question:
            if message.response.lower() in ["yes", "correct", "right", "that's right", "exactly"]:
                # Generate final requirements summary
                response = await self._model_client.create(
                    self._system_messages + conversation + [
                        UserMessage(content="REQUIREMENTS GATHERED. Please provide a final requirements summary for the technical team.", source="consultant")
                    ],
                    cancellation_token=ctx.cancellation_token
                )
                
                # Transition to CODE_WRITING state with REQUIREMENTS GATHERED
                self._state_manager.transition_to(State.CODE_WRITING, {
                    "session_id": session_id,
                    "requirements": response.content,
                    "status": "REQUIREMENTS GATHERED"
                })
                
                await self.publish_message(
                    ConsultantResult(
                        requirements_summary=response.content,
                        metadata={"session_id": session_id, "is_predefined": False, "status": "REQUIREMENTS GATHERED"}
                    ),
                    TopicId(type="code_writing", source=self.id.key)
                )
                return
            elif message.response.lower() in ["no", "incorrect", "wrong", "not quite", "not exactly"]:
                response = await self._model_client.create(
                    self._system_messages + conversation + [
                        UserMessage(content="What aspects did I misunderstand? Please help me correct my understanding.", source="consultant")
                    ],
                    cancellation_token=ctx.cancellation_token
                )
                
                next_question = ConsultantQuestion(
                    question=response.content,
                    context="Clarifying understanding"
                )
                history.append(next_question)
                await self.publish_message(
                    next_question,
                    TopicId(type="user_interaction", source=self.id.key)
                )
                return
        
        # Check for completion indicators in regular conversation
        completion_indicators = ["done", "complete", "finished", "this is all", "that's all", "requirements gathered"]
        if (len([h for h in history if isinstance(h, ConsultantQuestion)]) >= self._max_questions or 
            any(indicator in message.response.lower() for indicator in completion_indicators)):
            # Generate and show understanding
            understanding = await self._model_client.create(
                self._system_messages + conversation + [
                    UserMessage(content="REQUIREMENTS GATHERED. Please provide a concise summary of what you understand about the user's requirements. Start with 'Here's my understanding:'", source="consultant")
                ],
                cancellation_token=ctx.cancellation_token
            )
            
            # Send understanding to user
            understanding_question = ConsultantQuestion(
                question=f"{understanding.content}\n\nIs this correct? (yes/no)",
                context="Confirming understanding"
            )
            self._conversation_history[session_id].append(understanding_question)
            await self.publish_message(
                understanding_question,
                TopicId(type="user_interaction", source=self.id.key)
            )
            return
        
        # Generate next question for ongoing conversation
        response = await self._model_client.create(
            self._system_messages + conversation + [
                UserMessage(content="Based on the conversation so far, what's the most important information we still need to understand? Ask a focused question.", source="consultant")
            ],
            cancellation_token=ctx.cancellation_token
        )
        
        next_question = ConsultantQuestion(
            question=response.content,
            context="Based on our discussion so far"
        )
        history.append(next_question)
        await self.publish_message(
            next_question,
            TopicId(type="user_interaction", source=self.id.key)
        )

@default_subscription
class Programmer(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__("A Python programmer specializing in quantitative finance.")
        self._model_client = model_client
        self._session_memory: Dict[str, List[Union[ConsultantResult, CodeReviewTask, CodeReviewResult]]] = {}
        self._system_messages = [
            SystemMessage(
                content="""You are an expert Python programmer specializing in quantitative finance.

OUTPUT FORMAT:
- ONLY output executable Python code
- NO explanations, comments, or markdown
- NO text before or after the code
- Code must be complete and runnable

DATA SOURCES:
- Use yfinance for fetching both stock and options data

BACKTESTING REQUIREMENTS:
1. For stock strategy backtesting: Use vectorbt EXCLUSIVELY
2. For options strategy backtesting: Use optopsy EXCLUSIVELY
3. Return raw results as data structures
4. Include proper error handling
5. Use type hints

CODE MUST:
- Be clean and efficient
- Handle all edge cases
- Use proper logging
- Return data for Reporter to format

DO NOT:
- Add any text explanations
- Include markdown formatting
- Generate plots
- Format output
- Include comments (except for complex calculations)"""
            )
        ]

    @message_handler(match=lambda msg, ctx: isinstance(msg, ConsultantResult) and ctx.topic_id.type == "code_writing")
    async def handle_requirements(self, message: ConsultantResult, ctx: MessageContext) -> None:
        """Handle requirements from the consultant."""
        # Create new session
        session_id = message.metadata.get("session_id", str(uuid.uuid4()))
        self._session_memory.setdefault(session_id, []).append(message)
        
        # Generate code based on requirements
        response = await self._model_client.create(
            self._system_messages + [UserMessage(content=message.requirements_summary, source="consultant")],
            cancellation_token=ctx.cancellation_token,
        )
        
        # Use the response directly as code since we're expecting code-only output
        code = response.content.strip()
        logger.info(f"Programmer generated code:\n{code}")
        
        # Create review task
        code_review_task = CodeReviewTask(
            session_id=session_id,
            code_writing_task=message.requirements_summary,
            code_writing_scratchpad="",  # No scratchpad needed for code-only output
            code=code,
        )
        self._session_memory[session_id].append(code_review_task)
        await self.publish_message(
            code_review_task,
            TopicId(type="code_review", source=self.id.key)
        )

    @message_handler(match=lambda msg, ctx: isinstance(msg, CodeReviewResult) and ctx.topic_id.type == "code_writing")
    async def handle_review(self, message: CodeReviewResult, ctx: MessageContext) -> None:
        # Store review result
        self._session_memory[message.session_id].append(message)
        
        # Get original task and count review rounds
        review_request = next(
            m for m in reversed(self._session_memory[message.session_id]) 
            if isinstance(m, CodeReviewTask)
        )
        
        review_count = sum(1 for m in self._session_memory[message.session_id] if isinstance(m, CodeReviewResult))
        logger.info(f"Review round {review_count} completed")
        
        if message.approved or review_count >= 5:
            # If approved or reached max rounds, publish final result
            await self.publish_message(
                CodeWritingResult(
                    task=review_request.code_writing_task,
                    code=review_request.code,
                    review=message.review
                ),
                TopicId(type="execution", source=self.id.key)
            )
        else:
            # Continue with another round of review
            response = await self._model_client.create(
                self._system_messages + [
                    UserMessage(content=f"Original task: {review_request.code_writing_task}", source="programmer"),
                    UserMessage(content=f"Previous code:\n{review_request.code}", source="programmer"),
                    UserMessage(content=f"Review feedback:\n{message.review}", source="reviewer")
                ],
                cancellation_token=ctx.cancellation_token,
            )
            
            # Use response directly as code
            code = response.content.strip()
            
            new_review_task = CodeReviewTask(
                session_id=message.session_id,
                code_writing_task=review_request.code_writing_task,
                code_writing_scratchpad="",  # No scratchpad needed for code-only output
                code=code,
            )
            self._session_memory[message.session_id].append(new_review_task)
            await self.publish_message(
                new_review_task,
                TopicId(type="code_review", source=self.id.key)
            )

@default_subscription
class CodeReviewer(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__("A code review specialist.")
        self._model_client = model_client
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are an expert code reviewer specializing in Python and quantitative finance.
The review process should be limited to 3-5 rounds maximum to maintain efficiency.

Your task is to review code for:
1. Correctness - Check for logical errors and bugs
2. Performance - Identify potential performance issues
3. Security - Spot security vulnerabilities
4. Best Practices - Ensure code follows Python best practices
5. Documentation - Verify adequate documentation and type hints
6. Output Format - Ensure results are properly formatted and meaningful

Review Format:
1. Analysis: Provide detailed analysis of the code
2. Issues: List any issues found (prioritize by severity)
3. Suggestions: Provide specific improvement suggestions
4. Output Review: Evaluate if the output format is user-friendly and informative
5. Decision: State "APPROVED" or "NEEDS_REVISION"
6. Review Round: Track which review round this is (1-5)

Review Round Guidelines:
- Round 1-2: Focus on major issues (architecture, logic, security)
- Round 3-4: Focus on optimizations and best practices
- Round 5: Only critical issues should prevent approval
- After Round 5: Must approve unless critical security/correctness issues exist

If you approve the code, respond with "APPROVED" in the decision.
If you find issues that need to be fixed, respond with "NEEDS_REVISION" in the decision.

Remember: After 5 rounds, the code should be approved unless there are critical security or correctness issues.
"""
            )
        ]
        self._review_history: Dict[str, List[str]] = {}

    @message_handler(match=lambda msg, ctx: isinstance(msg, CodeReviewTask) and ctx.topic_id.type == "code_review")
    async def handle_review(self, message: CodeReviewTask, ctx: MessageContext) -> None:
        try:
            # Store review history
            if message.session_id not in self._review_history:
                self._review_history[message.session_id] = []
            
            review_prompt = f"""Review this code for the task: {message.code_writing_task}

Code:
{message.code}

Previous Reviews:
{chr(10).join(self._review_history[message.session_id]) if self._review_history[message.session_id] else "No previous reviews"}

Please provide a comprehensive review following the specified format."""

            review_result = await self._model_client.create(
                self._system_messages + [UserMessage(content=review_prompt, source="user")],
                cancellation_token=ctx.cancellation_token,
            )
            
            # Store the review
            self._review_history[message.session_id].append(review_result.content)
            
            # Determine if code is approved
            is_approved = "APPROVED" in review_result.content.upper()
            
            logger.info(f"Code review result:\n{review_result.content}")
            
            await self.publish_message(
                CodeReviewResult(
                    review=review_result.content,
                    session_id=message.session_id,
                    approved=is_approved
                ),
                TopicId(type="code_writing", source=self.id.key)
            )
        except Exception as e:
            logger.error(f"Error during code review: {str(e)}", exc_info=True)
            await self.publish_message(
                CodeReviewResult(
                    review=f"Error during code review: {str(e)}",
                    session_id=message.session_id,
                    approved=False
                ),
                TopicId(type="code_writing", source=self.id.key)
            )

@default_subscription
class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor, state_manager: StateManager) -> None:
        super().__init__("A code execution specialist.")
        self._model_client = None  # Will be set in handle_message if needed
        self._code_executor = code_executor
        self._state_manager = state_manager
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are a code execution specialist who formats and presents results.
Your task is to:
1. Execute the code and capture all outputs
2. Format the results in a clear, professional report
3. Include any generated plots or visualizations
4. Highlight key findings and insights
5. Provide actionable conclusions

Report Format:
1. Task Overview
2. Analysis Results
3. Key Findings
4. Visualizations (if any)
5. Conclusions and Recommendations
"""
            )
        ]

    @message_handler(match=lambda msg, ctx: isinstance(msg, CodeWritingResult) and ctx.topic_id.type == "execution")
    async def handle_execution(self, message: CodeWritingResult, ctx: MessageContext) -> None:
        try:
            # Execute the code
            code_block = CodeBlock(code=message.code, language="python")
            logger.info("Executing approved code")
            print("\nExecuting analysis...")
            
            # Log the code being executed
            print("\nExecuting code:")
            print("=" * 80)
            print(message.code)
            print("=" * 80)
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self._code_executor.execute_code_blocks(
                        [code_block], cancellation_token=ctx.cancellation_token
                    ),
                    timeout=60  # 60 second timeout
                )
                print("\nExecution completed.")
            except asyncio.TimeoutError:
                raise Exception("Code execution timed out after 60 seconds")
            except Exception as e:
                raise Exception(f"Code execution failed: {str(e)}")
            
            # Check for execution errors
            if hasattr(result, 'exit_code') and result.exit_code != 0:
                print(f"\nExecution failed with exit code {result.exit_code}")
                print("\nError output:")
                print(result.output)
                raise Exception(f"Code execution failed with exit code {result.exit_code}: {result.output}")
            
            # Print execution output
            print("\nExecution Output:")
            print("=" * 80)
            print(result.output if hasattr(result, 'output') else "No output generated")
            print("=" * 80)
            
            # Save successful code with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("outputs", f"analysis_{timestamp}.py")
            
            # Add imports and save path to the code
            code_with_imports = f"""# Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Task: {message.task}

import os
import pandas as pd
import numpy as np
import yfinance as yf
import vectorbt as vbt
import optopsy as op
from datetime import datetime, timedelta
import quantstats as qs
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Create output directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

{message.code}

# Save results
if 'results_df' in locals():
    results_df.to_csv(f'outputs/results_{timestamp}.csv')
if 'stats_df' in locals():
    stats_df.to_csv(f'outputs/stats_{timestamp}.csv')
if 'performance_df' in locals():
    performance_df.to_csv(f'outputs/performance_{timestamp}.csv')
"""
            
            # Save the code
            os.makedirs("outputs", exist_ok=True)
            with open(save_path, "w") as f:
                f.write(code_with_imports)
            logger.info(f"Saved analysis code to {save_path}")
            print(f"\nAnalysis code saved to: {save_path}")
            
            # If execution successful, proceed with report generation
            if not self._model_client:
                self._model_client = OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            
            # Generate formatted report
            report_prompt = f"""Task: {message.task}

Code Review: {message.review}

Execution Output: {result.output}

Please format this into a clear, professional report. Focus on:
1. Actual numerical results and metrics
2. Statistical significance of findings
3. Key performance indicators
4. Risk metrics
5. Actionable insights based on the data

Include specific numbers and avoid generic statements."""

            report_response = await self._model_client.create(
                self._system_messages + [UserMessage(content=report_prompt, source="user")],
                cancellation_token=ctx.cancellation_token,
            )
            
            formatted_report = f"""
=================================================================
                    QUANTITATIVE ANALYSIS REPORT
=================================================================

{report_response.content}

Analysis Code: {save_path}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=================================================================
"""
            print("\nAnalysis Report:")
            print(formatted_report)
            logger.info(f"Execution result:\n{formatted_report}")
            
            # Transition to reporting state
            self._state_manager.transition_to(State.REPORTING)
            
            # Publish the formatted report
            await self.publish_message(
                ReportTask(
                    session_id=str(uuid.uuid4()),
                    task=message.task,
                    code=message.code,
                    execution_output=result.output,
                    state_history=self._state_manager.history
                ),
                TopicId(type="reporting", source=self.id.key)
            )
            
        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            execution_output = getattr(result, 'output', error_msg) if 'result' in locals() else error_msg
            
            print(f"\nError during execution:")
            print("=" * 80)
            print(error_msg)
            print(error_traceback)
            print("=" * 80)
            
            logger.error(f"Error in handle_execution: {error_msg}", exc_info=True)
            
            # Transition to debugging state
            self._state_manager.transition_to(State.DEBUGGING, {
                "error": error_msg,
                "traceback": error_traceback
            })
            
            # Create debug task
            debug_task = DebugTask(
                session_id=str(uuid.uuid4()),
                code=message.code,
                error_context={
                    "error": error_msg,
                    "traceback": error_traceback,
                    "task": message.task
                },
                execution_output=execution_output
            )
            
            await self.publish_message(
                debug_task,
                TopicId(type="debugging", source=self.id.key)
            )

@default_subscription
class ReporterAgent(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__("A reporting specialist that formats and presents analysis results.")
        self._model_client = model_client
        self._system_messages = [
            SystemMessage(
                content="""You are a reporting specialist that formats and presents quantitative analysis results.

Key Responsibilities:
1. Extract and present numerical results
2. Calculate and show key performance metrics
3. Present statistical analysis with confidence levels
4. Highlight risk-adjusted returns
5. Provide data-driven recommendations

Report Structure:
1. Performance Metrics
   - Returns (absolute and %)
   - Sharpe/Sortino ratios
   - Maximum drawdown
   - Win rate

2. Risk Analysis
   - Value at Risk (VaR)
   - Standard deviation
   - Beta and correlation metrics
   - Options-specific metrics (if applicable)

3. Statistical Analysis
   - Confidence intervals
   - P-values where applicable
   - Distribution characteristics
   - Statistical significance of results

4. Trading Metrics (if applicable)
   - Number of trades
   - Average profit/loss
   - Profit factor
   - Maximum consecutive wins/losses

5. Actionable Insights
   - Data-driven recommendations
   - Specific entry/exit points
   - Risk management parameters
   - Position sizing suggestions

Format Requirements:
- Always include actual numbers
- Use proper decimal places
- Include units (%, $, etc.)
- Show time periods
- Compare to benchmarks

DO NOT:
- Make generic statements without numbers
- Omit statistical significance
- Skip risk metrics
- Present conclusions without data"""
            )
        ]

    @message_handler(match=lambda msg, ctx: isinstance(msg, ReportTask) and ctx.topic_id.type == "reporting")
    async def handle_report(self, message: ReportTask, ctx: MessageContext) -> None:
        # Create timeline from state history
        timeline = "\n".join([
            f"- {t.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {t.from_state.value} -> {t.to_state.value}"
            for t in message.state_history
        ])

        report_prompt = f"""Generate a comprehensive quantitative analysis report:

Task:
{message.task}

Code:
{message.code}

Execution Output:
{message.execution_output}

Process Timeline:
{timeline}

Requirements:
1. Extract all numerical results and metrics
2. Present statistical analysis with confidence levels
3. Include risk-adjusted performance metrics
4. Provide specific, data-driven recommendations
5. Show all relevant trading metrics

Format all numbers appropriately (e.g., 10.5%, $1,234.56, 2.5σ)."""

        report_response = await self._model_client.create(
            self._system_messages + [UserMessage(content=report_prompt, source="user")],
            cancellation_token=ctx.cancellation_token,
        )
        
        formatted_report = f"""
=================================================================
                    FINAL QUANTITATIVE ANALYSIS REPORT
=================================================================

{report_response.content}

Process Timeline:
{timeline}

=================================================================
"""
        print("\nFinal Report:")
        print(formatted_report)
        logger.info(f"Generated report:\n{formatted_report}")
        
        # Signal task completion
        await self.publish_message(
            CodeWritingResult(
                task=message.task,
                code=message.code,
                review=formatted_report
            ),
            TopicId(type="code_writing", source=self.id.key)
        )

def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks

@default_subscription
class DebuggerAgent(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient) -> None:
        super().__init__("A debugging specialist.")
        self._model_client = model_client
        self._system_messages = [
            SystemMessage(
                content="""You are an expert debugging specialist focusing on Python and quantitative finance code.

Your task is to:
1. Analyze execution errors and their context
2. Identify root causes of failures
3. Propose and implement fixes
4. Verify the fixes address the core issues
5. Provide detailed debug reports

Debug Report Format:
1. Error Analysis
2. Root Cause
3. Fix Implementation
4. Verification Steps
5. Prevention Recommendations"""
            )
        ]

    @message_handler(match=lambda msg, ctx: isinstance(msg, DebugTask) and ctx.topic_id.type == "debugging")
    async def handle_debug(self, message: DebugTask, ctx: MessageContext) -> None:
        debug_prompt = f"""Debug this code execution failure:

Code:
{message.code}

Error Context:
{message.error_context}

Execution Output:
{message.execution_output}

Please analyze the error and provide fixes."""

        debug_response = await self._model_client.create(
            self._system_messages + [UserMessage(content=debug_prompt, source="user")],
            cancellation_token=ctx.cancellation_token,
        )
        
        print("\nDebug Analysis:")
        print("=" * 80)
        print(debug_response.content)
        print("=" * 80)
        
        # Extract fixed code from the response
        code_blocks = extract_markdown_code_blocks(debug_response.content)
        fixed_code = code_blocks[0].code if code_blocks else None
        
        if fixed_code:
            print("\nFixed code generated. Attempting to run...")
            # Publish debug result back to debugging topic
            await self.publish_message(
                DebugResult(
                    session_id=message.session_id,
                    fixed_code=fixed_code,
                    debug_report=debug_response.content,
                    success=True
                ),
                TopicId(type="debugging", source=self.id.key)
            )
            
            # Also publish the fixed code for execution
            await self.publish_message(
                CodeWritingResult(
                    task=message.error_context.get("task", "Debug fix"),
                    code=fixed_code,
                    review=debug_response.content
                ),
                TopicId(type="execution", source=self.id.key)
            )
        else:
            print("\nNo fixed code could be extracted from the debug response.")
            await self.publish_message(
                DebugResult(
                    session_id=message.session_id,
                    fixed_code=message.code,
                    debug_report=debug_response.content,
                    success=False
                ),
                TopicId(type="debugging", source=self.id.key)
            )

async def main():
    try:
        # Suppress all library logging to console
        for log_name in ['asyncio', 'docker', 'urllib3', 'matplotlib']:
            logging.getLogger(log_name).setLevel(logging.ERROR)
            
        # Initialize code executor
        logger.info("Initializing Docker code executor")
        code_executor = DockerCommandLineCodeExecutor(
            image="aifund-quant",
            work_dir="workspace",
            auto_remove=True,
            stop_container=True
        )
        await code_executor.start()
        logger.info("Docker code executor started successfully")

        # Initialize state manager
        state_manager = StateManager()
        logger.info("State manager initialized")

        # Create runtime with termination condition
        runtime = SingleThreadedAgentRuntime()
        logger.info("Created SingleThreadedAgentRuntime")

        # Initialize OpenAI client
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        logger.info("Registering agents")

        # Register user agent
        await UserAgent.register(
            runtime,
            "user",
            lambda: UserAgent()
        )
        await runtime.add_subscription(TypeSubscription(topic_type="user_interaction", agent_type="user"))

        # Register consultant agent
        await ConsultantAgent.register(
            runtime,
            "consultant",
            lambda: ConsultantAgent(model_client, state_manager)
        )
        await runtime.add_subscription(TypeSubscription(topic_type="consulting", agent_type="consultant"))

        # Register other agents
        await Programmer.register(
            runtime,
            "programmer",
            lambda: Programmer(model_client)
        )
        await runtime.add_subscription(TypeSubscription(topic_type="code_writing", agent_type="programmer"))
        
        await CodeReviewer.register(
            runtime,
            "reviewer",
            lambda: CodeReviewer(model_client)
        )
        await runtime.add_subscription(TypeSubscription(topic_type="code_review", agent_type="reviewer"))

        await DebuggerAgent.register(
            runtime,
            "debugger",
            lambda: DebuggerAgent(model_client)
        )
        await runtime.add_subscription(TypeSubscription(topic_type="debugging", agent_type="debugger"))

        await ReporterAgent.register(
            runtime,
            "reporter",
            lambda: ReporterAgent(model_client)
        )
        await runtime.add_subscription(TypeSubscription(topic_type="reporting", agent_type="reporter"))

        await Executor.register(
            runtime,
            "executor",
            lambda: Executor(code_executor, state_manager)
        )
        await runtime.add_subscription(TypeSubscription(topic_type="execution", agent_type="executor"))
        
        logger.info("Agents registered successfully")

        # Start the runtime
        runtime.start()
        logger.info("Runtime started")

        # Define predefined tasks
        tasks = [
            "Analyze AAPL stock data using basic statistical measures and generate a simple trading signal based on moving averages.",
            "Perform statistical analysis on SPY daily returns including mean, std, skew, and kurtosis.",
            "Calculate technical indicators (RSI, MACD) for TSLA and identify potential trading signals.",
            "Backtest a momentum strategy on SPY over 2 years with daily rebalancing and performance metrics."
        ]

        while True:
            print("\nOptions:")
            print("1. Choose from predefined tasks")
            print("2. Enter custom analysis request")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "3":
                break
                
            if choice == "1":
                print("\nAvailable Analysis Tasks:")
                for i, task in enumerate(tasks, 1):
                    print(f"{i}. {task}")
                    
                task_input = input("\nEnter task number (1-4): ").strip()
                try:
                    task_num = int(task_input)
                    if 1 <= task_num <= len(tasks):
                        task = tasks[task_num - 1]
                        is_predefined = True
                    else:
                        print(f"Please enter a number between 1 and {len(tasks)}")
                        continue
                except ValueError:
                    print("Please enter a valid task number")
                    continue
            
            elif choice == "2":
                print("\nEnter your custom analysis request.")
                print("Describe what you want to analyze and what insights you're looking for.")
                task = input("\nYour request: ").strip()
                is_predefined = False
            
            else:
                print("Please enter a valid choice (1-3)")
                continue
            
            try:
                # Initialize new session
                state_manager.session_id = str(uuid.uuid4())
                state_manager.transition_to(State.TASK_RECEIVED, {"task": task})
                
                logger.info(f"Processing task: {task}")
                print("\nProcessing analysis request...")
                
                # Start the task processing
                state_manager.transition_to(State.CODE_WRITING)
                
                # Send task to consultant
                await runtime.publish_message(
                    ConsultantTask(request=task, is_predefined=is_predefined),
                    TopicId(type="consulting", source="user")
                )
                
                await runtime.stop_when_idle()

                print("\nTask Processing Timeline:")
                for transition in state_manager.history:
                    print(f"{transition.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                          f"{transition.from_state.value} -> {transition.to_state.value}")
                          
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}", exc_info=True)
                print(f"\nError processing task: {str(e)}")
                state_manager.transition_to(State.ERROR, {"error": str(e)})

    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}", exc_info=True)
        print(f"\nError during initialization: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Clean up
        try:
            logger.info("Stopping code executor")
            await code_executor.stop()
            logger.info("Code executor stopped")
        except Exception as e:
            logger.error(f"Error stopping code executor: {str(e)}")
            print(f"\nError stopping code executor: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 