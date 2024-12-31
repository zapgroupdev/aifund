import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools.get_sent_av import sentiment_tool

# Get current date and time information
current_date = datetime.now()
current_date_str = current_date.strftime("%Y-%m-%d")
one_day_ago = (current_date - timedelta(days=1)).strftime("%Y%m%dT%H%M")
one_week_ago = (current_date - timedelta(days=7)).strftime("%Y%m%dT0000")
one_month_ago = (current_date - timedelta(days=30)).strftime("%Y%m%dT0000")
current_time = current_date.strftime("%Y%m%dT%H%M")

class MarketResearchTeam:
    def __init__(self):
        # Market Impact Researcher
        self.researcher = AssistantAgent(
            name="Market_Impact_Researcher",
            system_message=f"""You are an expert Market Impact Researcher specializing in analyzing global market dynamics. Today is {current_date_str}.

PRIMARY OBJECTIVE:
Conduct comprehensive market analysis focusing on geopolitical events, supply chain impacts, and broader market sentiment across multiple time horizons.

ANALYSIS METHODOLOGY:
1. Temporal Analysis Framework:
   - Immediate Impact (Last 24h): {one_day_ago} to {current_time}
   - Recent Trends (Last Week): {one_week_ago} to {current_time}
   - Extended Patterns (Last Month): {one_month_ago} to {current_time}

2. Key Analysis Areas:
   - Geopolitical Events & Market Response
   - Supply Chain Disruptions & Adaptations
   - Cross-Market Correlations
   - Sector-Specific Vulnerabilities
   - Regional Market Dynamics

3. Impact Assessment Categories:
   - Direct Market Impacts
   - Secondary Effects
   - Potential Future Implications
   - Risk Factors
   - Opportunity Zones

RESEARCH PRIORITIES:
1. Geopolitical Analysis:
   - Political events affecting markets
   - Trade policy changes
   - International relations shifts
   - Regulatory developments
   - Regional conflicts/resolutions

2. Supply Chain Focus:
   - Disruption points
   - Alternative supply routes
   - Inventory levels
   - Transportation costs
   - Manufacturing impacts

3. Market Sentiment Indicators:
   - News sentiment trends
   - Institutional responses
   - Market participant behavior
   - Volume and volatility patterns
   - Cross-asset correlations

OUTPUT STRUCTURE:
1. Executive Summary
   - Key findings across all time horizons
   - Critical alerts and developments

2. Temporal Analysis
   - Last 24 Hours:
     * Breaking developments
     * Immediate market reactions
     * Emerging situations
   
   - Past Week:
     * Trend development
     * Pattern confirmation
     * Initial impact assessment
   
   - Past Month:
     * Long-term pattern analysis
     * Structural changes
     * Baseline shifts

3. Impact Analysis
   - Geopolitical Developments
   - Supply Chain Status
   - Market Sentiment Shifts
   - Risk Assessment
   - Opportunity Analysis

4. Forward-Looking Insights
   - Short-term Outlook (24-48h)
   - Medium-term Projections (1-2 weeks)
   - Long-term Considerations (1+ months)

5. Action Items
   - Critical Watch Points
   - Risk Mitigation Suggestions
   - Strategic Opportunities

End your analysis with a "CONFIDENCE MATRIX":
- Data Quality Score (1-5)
- Geopolitical Impact Confidence (1-5)
- Supply Chain Disruption Level (1-5)
- Market Sentiment Reliability (1-5)
- Forward Projection Confidence (1-5)

Conclude with "ANALYSIS COMPLETE" to signal completion.""",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-mini"
            ),
            tools=[sentiment_tool]
        )

        # Initialize team with termination condition
        self.termination = TextMentionTermination("ANALYSIS COMPLETE")
        self.team = RoundRobinGroupChat(
            [self.researcher],
            termination_condition=self.termination
        )

    async def analyze_market_impact(self, request: str) -> Dict[str, Any]:
        """
        Run a comprehensive market impact analysis based on the given request.
        
        Args:
            request (str): The analysis request (e.g., "Analyze global supply chain impacts on tech sector")
            
        Returns:
            Dict[str, Any]: Structured analysis results
        """
        try:
            messages = []
            async for msg in self.team.run_stream(task=request):
                if hasattr(msg, 'messages'):
                    messages.extend(msg.messages)
                else:
                    messages.append(msg)
                
            # Extract final report
            final_report = next(
                (msg.content for msg in reversed(messages) 
                 if hasattr(msg, 'source') and msg.source == "Market_Impact_Researcher" 
                 and "ANALYSIS COMPLETE" in msg.content),
                "No complete analysis found"
            )
            
            return {
                "report": final_report,
                "messages": messages
            }
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

async def main():
    try:
        print(f"Initializing Market Impact Analysis for: {current_date_str}")
        team = MarketResearchTeam()
        print("Running comprehensive market impact analysis...")
        result = await team.analyze_market_impact(
            """Analyze current market conditions with focus on:
            1. Major geopolitical events affecting global markets
            2. Supply chain disruptions and their cascading effects
            3. Cross-sector impacts and vulnerabilities
            4. Emerging risks and opportunities"""
        )
        print("\nMarket Impact Analysis Report:")
        print("-" * 80)
        print(result["report"])
        print("-" * 80)
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 