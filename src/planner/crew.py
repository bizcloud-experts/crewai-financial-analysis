import os
import sys
import yaml
from typing import Dict, Any, List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

# Configure CrewAI for Lambda environment
os.environ["CREWAI_STORAGE_DIR"] = "/tmp"
os.environ["HOME"] = "/tmp"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure custom storage in /tmp
storage_path = "/tmp/crewai_storage"
os.makedirs(storage_path, exist_ok=True)

@CrewBase
class MovingFinancialCrew():
    """Moving and Storage Financial Analysis Crew"""

    def __init__(self) -> None:
        # Load configuration files
        self.agents_config = self._load_config('config/agents.yaml')
        self.tasks_config = self._load_config('config/tasks.yaml')

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found")
            return {}
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            return {}

    @agent
    def task_planner(self) -> Agent:
        """Financial Analysis Task Planner Agent"""
        return Agent(
            config=self.agents_config.get('task_planner', {}),
            verbose=True,
            allow_delegation=False
        )
    
    @agent
    def metadata_agent(self) -> Agent:
        """Financial Data Metadata Specialist Agent"""
        return Agent(
            config=self.agents_config.get('metadata_agent', {}),
            verbose=True,
            allow_delegation=False
            # TODO: Add Weaviate search tool here
            # tools=[WeaviateSearchTool()]
        )
    
    @agent
    def query_builder_agent(self) -> Agent:
        """Financial Database Query Specialist Agent"""
        return Agent(
            config=self.agents_config.get('query_builder_agent', {}),
            verbose=True,
            allow_delegation=False
            # TODO: Add database query tools here
            # tools=[SQLDatabaseTool(), AthenaQueryTool()]
        )

    @task
    def financial_planning_task(self) -> Task:
        """Create execution plan for financial analysis"""
        return Task(
            config=self.tasks_config.get('financial_planning_task', {}),
            agent=self.task_planner()
            # output_file='/tmp/financial_plan.md'
        )
    
    @task
    def metadata_retrieval_task(self) -> Task:
        """Retrieve metadata for financial data schemas"""
        return Task(
            config=self.tasks_config.get('metadata_retrieval_task', {}),
            agent=self.metadata_agent()
            # output_file='/tmp/metadata_analysis.md'
        )
    
    @task
    def financial_query_task(self) -> Task:
        """Execute financial data queries"""
        return Task(
            config=self.tasks_config.get('financial_query_task', {}),
            agent=self.query_builder_agent(),
            context=[self.metadata_retrieval_task()]  # Depends on metadata
            # output_file='/tmp/query_results.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Moving Financial Analysis crew"""
        return Crew(
            agents=[
                self.task_planner(),
                self.metadata_agent(),
                self.query_builder_agent()
            ],
            tasks=[
                self.financial_planning_task(),
                self.metadata_retrieval_task(),
                self.financial_query_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=False  # Disable memory to avoid SQLite issues in Lambda
        )
    
    def analyze_financial_question(self, question: str, context: Dict[str, Any] = None) -> str:
        """
        Analyze a financial question using the crew
        
        Args:
            question: The financial question to analyze
            context: Additional context like time periods, filters, etc.
        
        Returns:
            Analysis results as a string
        """
        # Prepare inputs for the crew
        inputs = {
            'question': question,
            **(context or {})
        }
        
        # Execute the crew
        result = self.crew().kickoff(inputs=inputs)
        return str(result)

    # Individual agent methods for Step Function use
    def create_planning_task(self, question: str, context: Dict[str, Any] = None) -> Task:
        """Create a standalone planning task"""
        inputs = {'question': question, **(context or {})}
        
        return Task(
            description=self.tasks_config.get('financial_planning_task', {}).get('description', '').format(**inputs),
            agent=self.task_planner(),
            expected_output=self.tasks_config.get('financial_planning_task', {}).get('expected_output', '')
        )
    
    def create_metadata_task(self, question: str, execution_plan: str = '') -> Task:
        """Create a standalone metadata task"""
        inputs = {'question': question, 'execution_plan': execution_plan}
        
        return Task(
            description=self.tasks_config.get('metadata_retrieval_task', {}).get('description', '').format(**inputs),
            agent=self.metadata_agent(),
            expected_output=self.tasks_config.get('metadata_retrieval_task', {}).get('expected_output', '')
        )
    
    def create_query_task(self, question: str, metadata_analysis: str = '', execution_plan: str = '') -> Task:
        """Create a standalone query task"""
        inputs = {
            'question': question, 
            'metadata_analysis': metadata_analysis,
            'execution_plan': execution_plan
        }
        
        return Task(
            description=self.tasks_config.get('financial_query_task', {}).get('description', '').format(**inputs),
            agent=self.query_builder_agent(),
            expected_output=self.tasks_config.get('financial_query_task', {}).get('expected_output', '')
        )

    def get_financial_metrics(self, metric_type: str, time_period: str = None) -> str:
        """
        Helper method for common financial metric requests
        
        Args:
            metric_type: Type of financial metric (revenue, costs, profitability, etc.)
            time_period: Time period for analysis (monthly, quarterly, yearly)
        
        Returns:
            Financial metrics analysis
        """
        question = f"What are the {metric_type} metrics"
        if time_period:
            question += f" for the {time_period} time period"
        question += "?"
        
        return self.analyze_financial_question(question)

    def compare_financial_performance(self, 
                                    comparison_type: str, 
                                    periods: List[str] = None,
                                    segments: List[str] = None) -> str:
        """
        Helper method for financial performance comparisons
        
        Args:
            comparison_type: Type of comparison (YoY, MoM, regional, service-line)
            periods: Specific time periods to compare
            segments: Business segments to compare
        
        Returns:
            Comparative financial analysis
        """
        question = f"Compare financial performance {comparison_type}"
        
        context = {}
        if periods:
            context['periods'] = periods
            question += f" for periods: {', '.join(periods)}"
        if segments:
            context['segments'] = segments
            question += f" across segments: {', '.join(segments)}"
        
        return self.analyze_financial_question(question, context)