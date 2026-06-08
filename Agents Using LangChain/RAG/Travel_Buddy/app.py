import os

from dotenv import load_dotenv

from serpapi import GoogleSearch
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    tavily_api_key=TAVILY_API_KEY
)

@tool
def search_flights(
    origin: str,
    destination: str,
    date: str
) -> str:
    """
    Search flights using SerpAPI Google Flights.
    """

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "currency": "INR",
        "type": "2",
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)

    results = search.get_dict()

    return str(results)


tools = [
    tavily_search,
    search_flights
]

system_prompt = """
You are TravelBuddy Agent.

Research destinations using Tavily Search.
Search flights using Google Flights.
Recommend attractions, travel tips, and flights.
Always use available tools when necessary.
"""

# Memory Checkpointer
checkpointer = InMemorySaver()

# Conversation Configuration
config = {
    "configurable": {
        "thread_id": "travelbuddy-thread"
    }
}

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer
)

# First Query
response1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
Plan a trip from Bengaluru to Singapore.
Find destination information and available flights
for 2026-09-15.
"""
            }
        ]
    },
    config=config
)

print(response1["messages"][-1].content)

# Follow-up Query Demonstrating Memory
response2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
Based on the trip we discussed earlier,
what are the top attractions I should visit there?
"""
            }
        ]
    },
    config=config
)

print(response2["messages"][-1].content)