"""
Enhanced Research Agent v2
This script creates an advanced research assistant that helps gather, analyze, and synthesize information
on any given topic. It uses multiple AI models (DeepSeek and Perplexity) to provide comprehensive research results.

How it works:
1. Takes a research question from the user
2. Asks follow-up questions to better understand the research needs
3. Creates a detailed research plan
4. Executes multiple research queries in parallel
5. Synthesizes all findings into a well-organized report
"""

import os
import asyncio  # Handles running multiple tasks at once
from typing import List, Dict, Any, Optional  # Type hints for better code understanding
from pydantic import BaseModel, Field  # Helps validate and structure our data
from openai import OpenAI  # Library to interact with AI models
from dotenv import load_dotenv
from datetime import datetime
from rich import print as rprint  # Makes console output pretty and colorful
from rich.panel import Panel  # Creates nice boxes around our output
from rich.markdown import Markdown  # Formats text nicely
from rich.progress import Progress  # Shows progress bars
from termcolor import colored  # Adds colors to terminal text
import json  # Handles JSON data
import aiohttp  # Makes HTTP requests in parallel

# ==============================================
# API KEYS - Replace these with your own keys
# ==============================================
# Get your DeepSeek API key from: https://platform.deepseek.com/
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"

# Get your Perplexity API key from: https://www.perplexity.ai/
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"

# ==============================================
# Data Models - These define the structure of our data
# ==============================================

class FollowUpQuestion(BaseModel):
    """Structure for follow-up questions to better understand user needs"""
    question: str = Field(..., description="The actual question to ask the user")
    reason: str = Field(..., description="Why we're asking this question")
    priority: int = Field(..., ge=1, le=5, description="How important this question is (1-5)")

class ResearchQuery(BaseModel):
    """Structure for individual research queries"""
    query: str = Field(..., description="What we'll ask the AI")
    aspect: str = Field(..., description="What part of the research this covers")
    expected_insight: str = Field(..., description="What we hope to learn")

class ResearchPlan(BaseModel):
    """Overall structure for our research plan"""
    main_goal: str = Field(..., description="The main thing we're researching")
    key_aspects: List[str] = Field(..., description="Important areas to investigate")
    queries: List[ResearchQuery] = Field(..., description="List of specific things to research")
    success_criteria: List[str] = Field(..., description="How we'll know if the research was successful")

class ResearchResult(BaseModel):
    """Structure for storing research results"""
    query: ResearchQuery  # The query that was executed
    content: str  # The answer we got
    citations: List[str] = []  # Sources/references
    error: Optional[str] = None  # Any errors that occurred

class EnhancedResearchAgent:
    """
    Main research agent class that handles the entire research process.
    Think of this as your personal research assistant that:
    - Asks clarifying questions
    - Plans the research
    - Gathers information
    - Organizes everything into a nice report
    """

    def __init__(self):
        """Sets up the research agent with necessary tools and connections"""
        
        # Initialize connections to AI models
        self.deepseek_chat = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.deepseek_r1 = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.perplexity_client = OpenAI(
            api_key=PERPLEXITY_API_KEY,
            base_url="https://api.perplexity.ai"
        )
        
        # Create a folder to save research results
        self.output_dir = "research_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    async def get_follow_up_questions(self, research_goal: str) -> List[FollowUpQuestion]:
        """
        Generates smart follow-up questions to better understand what the user needs.
        For example, if researching cars, it might ask about budget, preferred features, etc.
        """
        # Tell the AI how to format its response
        system_prompt = """
        You are a research assistant that helps formulate follow-up questions to better understand research needs.
        You will receive a research goal and should generate up to 5 follow-up questions.
        Output must be in valid JSON format following this structure:

        EXAMPLE OUTPUT:
        {
            "questions": [
                {
                    "question": "What specific features are most important to you?",
                    "reason": "Helps prioritize which aspects to focus on in the research",
                    "priority": 1
                }
            ]
        }
        """
        
        # Ask the AI to generate questions based on the research goal
        user_prompt = f"""
        Given this research goal: '{research_goal}'
        
        Generate up to 5 follow-up questions that would help better understand the user's needs.
        Focus on questions that would significantly impact the research direction or scope.
        Assign priority levels (1-5) based on importance, with 1 being most important.
        """
        
        try:
            # Get questions from DeepSeek Chat
            response = self.deepseek_chat.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Controls randomness (0.0 = very focused, 1.0 = more creative)
                response_format={'type': 'json_object'}  # Ensures we get properly formatted JSON
            )
            
            # Convert the AI's response into our question format
            response_data = json.loads(response.choices[0].message.content)
            questions = response_data.get('questions', [])
            return [FollowUpQuestion(**q) for q in questions]
            
        except Exception as e:
            rprint(f"[red]Error getting follow-up questions: {str(e)}[/red]")
            return []

    async def generate_research_plan(self, research_goal: str, follow_up_responses: Dict[str, str]) -> ResearchPlan:
        """
        Creates a detailed plan for the research based on the user's goal and their answers to follow-up questions.
        This is like creating a roadmap for the research journey.
        """
        # First get a natural language research plan from DeepSeek R1
        r1_prompt = f"""
        Research Goal: '{research_goal}'
        
        Follow-up Questions and Answers:
        {json.dumps(follow_up_responses, indent=2)}
        
        Create a comprehensive research plan that includes:
        1. The main research goal
        2. Key aspects to investigate
        3. Specific search queries to execute
        4. Success criteria for the research
        
        Be specific and thorough in your plan.
        """
        
        try:
            # Get the initial plan in natural language
            r1_response = self.deepseek_r1.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": r1_prompt}],
                temperature=0.7
            )
            r1_plan = r1_response.choices[0].message.content

            # Convert the natural language plan into a structured JSON format
            chat_prompt = f"""
            Convert this research plan into a properly formatted JSON object:

            {r1_plan}

            The JSON should follow this exact structure:
            {{
                "main_goal": "string",
                "key_aspects": ["string"],
                "queries": [
                    {{
                        "query": "string",
                        "aspect": "string",
                        "expected_insight": "string"
                    }}
                ],
                "success_criteria": ["string"]
            }}

            Ensure the output is valid JSON and includes all components from the research plan.
            """

            # Get the structured JSON version
            chat_response = self.deepseek_chat.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": chat_prompt}],
                temperature=0.7,
                response_format={'type': 'json_object'}
            )
            
            plan_json = chat_response.choices[0].message.content
            return ResearchPlan(**json.loads(plan_json))
            
        except Exception as e:
            rprint(f"[red]Error generating research plan: {str(e)}[/red]")
            raise

    async def execute_query(self, query: ResearchQuery) -> ResearchResult:
        """
        Executes a single research query using Perplexity AI.
        This is like asking a very knowledgeable expert about a specific topic.
        """
        try:
            # Send the query to Perplexity
            response = self.perplexity_client.chat.completions.create(
                model="sonar-pro",  # Perplexity's most capable model
                messages=[{
                    "role": "user",
                    "content": query.query
                }],
                max_tokens=8000,  # Maximum length of the response
                stream=False  # Get the full response at once
            )
            
            # Package the response with any citations/sources
            return ResearchResult(
                query=query,
                content=response.choices[0].message.content,
                citations=response.citations if hasattr(response, 'citations') else []
            )
        except Exception as e:
            # If something goes wrong, return an error result
            return ResearchResult(
                query=query,
                content="",
                error=str(e)
            )

    async def execute_research_plan(self, plan: ResearchPlan) -> List[ResearchResult]:
        """
        Executes all research queries in parallel for faster results.
        This is like having multiple researchers working on different aspects simultaneously.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.execute_query(query) for query in plan.queries]
            return await asyncio.gather(*tasks)

    def generate_final_synthesis(self, research_goal: str, plan: ResearchPlan, results: List[ResearchResult]) -> str:
        """
        Takes all the research results and combines them into a clear, organized final report.
        This is like having an expert analyze all the findings and write a comprehensive summary.
        """
        try:
            # Create a prompt that tells the AI how to organize all our findings
            synthesis_prompt = f"""
            Research Goal: {research_goal}

            Key Aspects Investigated:
            {json.dumps(plan.key_aspects, indent=2)}

            Research Results:
            {json.dumps([{
                'aspect': r.query.aspect,
                'query': r.query.query,
                'findings': r.content,
                'citations': r.citations
            } for r in results], indent=2)}

            Please provide a comprehensive synthesis of the research findings that:
            1. Directly addresses the research goal
            2. Highlights key findings for each aspect
            3. Makes clear recommendations based on the findings
            4. Notes any limitations or areas needing further research
            5. Includes relevant citations where available

            Format the response in markdown with clear sections and bullet points.
            """

            # Get the AI to synthesize everything into a final report
            response = self.deepseek_chat.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            rprint(f"[red]Error generating synthesis: {str(e)}[/red]")
            return f"Error generating synthesis: {str(e)}"

    async def run_research_session(self):
        """
        The main function that runs the entire research process from start to finish.
        This orchestrates all the steps: asking questions, planning, researching, and synthesizing.
        """
        # Get the user's research question
        research_goal = input(colored("\nWhat would you like to research? ", "cyan"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/research_{timestamp}.txt"
        
        # Step 1: Get follow-up questions
        rprint("\n[cyan]Generating follow-up questions...[/cyan]")
        questions = await self.get_follow_up_questions(research_goal)
        
        if not questions:
            rprint("[red]Failed to generate follow-up questions. Proceeding with basic research.[/red]")
            follow_up_responses = {}
        else:
            # Ask each follow-up question and collect answers
            follow_up_responses = {}
            for q in sorted(questions, key=lambda x: x.priority):
                rprint(f"\n[yellow]Question {q.priority}:[/yellow] {q.question}")
                rprint(f"[dim]Reason: {q.reason}[/dim]")
                while True:
                    answer = input(colored("Your answer (or press Enter to skip): ", "cyan")).strip()
                    if answer or input(colored("Skip this question? (y/n): ", "cyan")).lower() == 'y':
                        break
                if answer:
                    follow_up_responses[q.question] = answer
                    rprint("[green]Answer recorded![/green]")
                else:
                    rprint("[yellow]Question skipped[/yellow]")
        
        # Step 2: Generate and review research plan
        rprint("\n[cyan]Generating research plan...[/cyan]")
        try:
            plan = await self.generate_research_plan(research_goal, follow_up_responses)
            
            # Show the plan to the user
            rprint("\n[yellow]Research Plan Summary:[/yellow]")
            rprint(f"Main Goal: {plan.main_goal}")
            rprint("\nKey Aspects:")
            for aspect in plan.key_aspects:
                rprint(f"• {aspect}")
            rprint("\nPlanned Queries:")
            for query in plan.queries:
                rprint(f"• {query.aspect}: {query.query}")
            
            # Let the user decide whether to proceed
            if input(colored("\nProceed with this plan? (y/n): ", "cyan")).lower() != 'y':
                rprint("[yellow]Research session cancelled[/yellow]")
                return
            
            # Step 3: Execute the research plan
            with Progress() as progress:
                task = progress.add_task("[cyan]Executing queries...", total=len(plan.queries))
                results = await self.execute_research_plan(plan)
                progress.update(task, advance=len(plan.queries))
            
            # Step 4: Generate final synthesis
            rprint("\n[cyan]Generating final synthesis...[/cyan]")
            final_synthesis = self.generate_final_synthesis(research_goal, plan, results)
            
            if final_synthesis:
                # Display the results in a nice format
                rprint(Panel(
                    Markdown(final_synthesis),
                    title="Final Research Synthesis",
                    border_style="green"
                ))
                
                # Save everything to a file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Research Goal: {research_goal}\n\n")
                    f.write("Follow-up Questions and Answers:\n")
                    for q, a in follow_up_responses.items():
                        f.write(f"Q: {q}\nA: {a}\n\n")
                    f.write("\nResearch Plan:\n")
                    f.write(json.dumps(plan.model_dump(), indent=2))  # Updated to use model_dump()
                    f.write("\n\nResearch Results:\n")
                    for result in results:
                        f.write(f"\nQuery: {result.query.query}\n")
                        f.write(f"Content: {result.content}\n")
                        if result.citations:
                            f.write("Citations:\n")
                            for citation in result.citations:
                                f.write(f"- {citation}\n")
                    f.write("\nFinal Synthesis:\n")
                    f.write(final_synthesis)
            
            rprint("\n[green]Research completed! Check the research_results directory for outputs.[/green]")
            
        except Exception as e:
            rprint(f"[red]Error during research: {str(e)}[/red]")

# This is the entry point of the script
async def main():
    """
    The starting point of our research tool.
    Creates a research agent and keeps running research sessions until the user wants to stop.
    """
    agent = EnhancedResearchAgent()
    while True:
        await agent.run_research_session()
        if input(colored("\nWould you like to start a new research session? (y/n): ", "cyan")).lower() != 'y':
            break

# This makes sure our script only runs when executed directly (not when imported)
if __name__ == "__main__":
    asyncio.run(main()) 
