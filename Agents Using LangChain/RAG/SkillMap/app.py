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
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

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

# Custom Job Search Tool
@tool
def search_jobs(query: str) -> str:
    """
    Search for jobs using JSearch API.
    """

    url = "https://jsearch.p.rapidapi.com/search"

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }

    params = {
        "query": query,
        "page": "1",
        "num_pages": "1"
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    return response.text


tools = [
    tavily_search,
    search_jobs
]

system_prompt = """
You are SkillMap Agent.

Your responsibilities:
1. Research skill demand using Tavily Search.
2. Find matching jobs using the JSearch API.
3. Explain which skills are currently in demand.
4. Recommend learning paths.
5. Suggest relevant job opportunities.
6. Use available tools whenever necessary.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

user_query = """
I know Python, SQL, and Machine Learning.
What skills are currently in demand and
what jobs match my profile?
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