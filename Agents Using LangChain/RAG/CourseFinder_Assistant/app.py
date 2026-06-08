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
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize Gemini model
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

# Tavily Search Tool
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced"
)

# YouTube Search Tool
@tool
def search_courses(skill: str) -> str:
    """
    Search YouTube for complete course tutorials.
    """

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": f"{skill} complete course tutorial",
        "type": "video",
        "videoDuration": "long",
        "maxResults": 5,
        "order": "relevance",
        "key": YOUTUBE_API_KEY
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    results = []

    for item in data.get("items", []):
        title = item["snippet"]["title"]
        channel = item["snippet"]["channelTitle"]
        published_date = item["snippet"]["publishedAt"]

        description = item["snippet"]["description"][:150]

        video_id = item["id"]["videoId"]

        link = f"https://www.youtube.com/watch?v={video_id}"

        results.append(
            {
                "title": title,
                "channel": channel,
                "published_date": published_date,
                "description": description,
                "link": link,
            }
        )

    return str(results)


tools = [
    tavily_search,
    search_courses
]

system_prompt = """
You are CourseFinder Agent.

Responsibilities:
1. Research learning resources using Tavily Search.
2. Find high-quality YouTube courses.
3. Recommend learning paths.
4. Suggest beginner, intermediate, and advanced resources.
5. Use the available tools whenever needed.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

user_query = """
I want to learn Machine Learning from scratch.
Recommend the best courses and YouTube tutorials.
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

print(response["messages"][-1].content)