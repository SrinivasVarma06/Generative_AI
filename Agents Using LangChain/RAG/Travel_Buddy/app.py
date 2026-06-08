import os
import requests

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize Gemini model
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

# Tavily Search Tool
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    tavily_api_key=TAVILY_API_KEY
)

# Flight Search Tool using SerpAPI Google Flights
@tool
def search_flights(origin: str,destination: str,date: str) -> str:
    """
    Search available flights between two locations.
    """

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "currency": "INR",
        "type": "2",
        "api_key": SERPAPI_API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    return response.text


tools = [
    tavily_search,
    search_flights
]

system_prompt = """
You are TravelBuddy Agent.

Responsibilities:
1. Research travel destinations using Tavily Search.
2. Find flight options using the Google Flights API.
3. Suggest attractions and travel tips.
4. Compare flight choices when available.
5. Help users plan complete trips.
6. Use available tools whenever needed.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

user_query = """
Plan a trip from Bengaluru to Singapore.
Find destination information and available flights
for 2026-09-15.
"""

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": user_query
            }
        ]
    }
)

print(response)