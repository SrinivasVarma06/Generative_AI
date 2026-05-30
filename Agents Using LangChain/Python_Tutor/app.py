from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage,HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv('GROQ_API_KEY')

system_msg=SystemMessage("You are a Python tutor who explains concepts with simple code.")
human_msg=HumanMessage("Explain what a dictionary is in Python with an example")

model=init_chat_model('groq:llama-3.3-70b-versatile',api_key=api_key)
messages=[system_msg,human_msg]

response=model.invoke(messages)
print(response.content)