import asyncio
from datetime import datetime
from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools.get_fundamentals import (
    income_statement_tool,
    balance_sheet_tool,
    cash_flow_tool,
    company_overview_tool
)

current_date = datetime.now().strftime("%Y-%m-%d")

class FundamentalAnalyst:
    def __init__(self):
        self.analyst = AssistantAgent(
            name="Fundamental_Analyst",
            system_message=f"""You are an expert Financial Analyst specializing in fundamental analysis. Today is {current_date}.

PRIMARY OBJECTIVE:
Conduct comprehensive fundamental analysis of companies using financial statements and metrics.

ANALYSIS METHODOLOGY:
1. Financial Statement Analysis:
   - Income Statement Analysis
     * Revenue trends and growth
     * Margin analysis (Gross, Operating, Net)
     * Expense management
     * Earnings quality

   - Balance Sheet Analysis
     * Asset composition and quality
     * Liability structure
     * Working capital management
     * Capital structure

   - Cash Flow Analysis
     * Operating cash flow trends
     * Free cash flow generation
     * Investment activities
     * Financing decisions

2. Key Metrics and Ratios:
   - Profitability Ratios
     * ROE, ROA, ROIC
     * Profit margins
     * Asset turnover

   - Liquidity Ratios
     * Current ratio
     * Quick ratio
     * Working capital

   - Solvency Ratios
     * Debt/Equity
     * Interest coverage
     * Debt service capability

   - Valuation Metrics
     * P/E, P/B, P/S ratios
     * EV/EBITDA
     * DCF implications

3. Comparative Analysis:
   - Historical trends
   - Industry benchmarking
   - Peer comparison
   - Sector positioning

OUTPUT STRUCTURE:
1. Company Overview
   - Business model
   - Market position
   - Key metrics summary

2. Financial Analysis
   - Statement analysis
   - Key ratios
   - Trend analysis

3. Strengths & Weaknesses
   - Financial position
   - Operational efficiency
   - Growth prospects

4. Risk Assessment
   - Financial risks
   - Business risks
   - Market risks

5. Valuation Analysis
   - Current valuation
   - Fair value estimate
   - Investment thesis

End your analysis with a "FUNDAMENTAL SCORECARD":
- Financial Health (1-5)
- Growth Prospects (1-5)
- Competitive Position (1-5)
- Management Efficiency (1-5)
- Investment Appeal (1-5)

Conclude with "ANALYSIS COMPLETE" to signal completion.""",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-mini"
            ),
            tools=[
                income_statement_tool,
                balance_sheet_tool,
                cash_flow_tool,
                company_overview_tool
            ]
        )

    async def analyze_company(self, ticker: str) -> Dict[str, Any]:
        """
        Conduct fundamental analysis for a given company.
        
        Args:
            ticker (str): The stock ticker symbol to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Create a simple team with just the analyst
            termination = TextMentionTermination("ANALYSIS COMPLETE")
            team = RoundRobinGroupChat(
                [self.analyst],
                termination_condition=termination
            )
            
            # Run the analysis
            messages = []
            async for msg in team.run_stream(
                task=f"""Conduct a comprehensive fundamental analysis for {ticker}. 
                Focus on:
                1. Financial health and stability
                2. Growth trends and prospects
                3. Operational efficiency
                4. Competitive position
                5. Valuation and investment potential
                
                Please use all available tools to gather the necessary financial data before starting your analysis."""
            ):
                if hasattr(msg, 'messages'):
                    messages.extend(msg.messages)
                else:
                    messages.append(msg)
            
            # Extract the final analysis
            final_analysis = next(
                (msg.content for msg in reversed(messages) 
                 if hasattr(msg, 'source') and msg.source == "Fundamental_Analyst" 
                 and "ANALYSIS COMPLETE" in msg.content),
                "No complete analysis found"
            )
            
            return {
                "analysis": final_analysis,
                "messages": messages
            }
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

async def main():
    try:
        print(f"Initializing Fundamental Analysis for date: {current_date}")
        analyst = FundamentalAnalyst()
        print("Running fundamental analysis...")
        result = await analyst.analyze_company(input("Enter a ticker: "))  # Example with Apple
        print("\nFundamental Analysis Report:")
        print("-" * 80)
        print(result["analysis"])
        print("-" * 80)
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())