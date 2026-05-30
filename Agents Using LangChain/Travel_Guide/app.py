import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

human_msg=HumanMessage("Suggest top 3 places to visit in Japan.")
system_msg=SystemMessage("You are a travel guide who gives brief recommendations for tourist destinations.")

model=init_chat_model("google_genai:gemini-2.5-flash",api_key=GEMINI_API_KEY)
messages=[human_msg,system_msg]

response=model.invoke(messages)
print(response.content)