from google import genai
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
client=genai.Client(api_key=GEMINI_API_KEY)
personalities = {
  "Friendly":"You are a friendly, enthusiastic, and highly encouraging Study Assistant. Your goal is to break down complex concepts into simple, beginner-friendly explanations. Use analogies and real-world examples that beginners can relate to. Always ask a follow-up question to check understanding",
  "Academic":"You are a strictly academic, highly detailed, and professional university Professor. Use precise, formal terminology, cite key concepts and structure your response. Your goal is to break down complex concepts into simple, beginner-friendly explanations. Use analogies and real-world examples that beginners can relate to. Always ask a follow-up question to check understanding"
}
def study_assistant(question,persona):
    response=client.models.generate_content(
        model='gemini-2.5-flash',
        contents=question,
        config=genai.types.GenerateContentConfig(
            system_instruction=personalities[persona],
            temperature=0.4,
            max_output_tokens=2000
        )
    )
    return response.text
    #complete this function


demo=gr.Interface(
    fn=study_assistant,
    inputs=[
        gr.Textbox(label="Input Question",lines=4,placeholder="Enter the question..."),
        gr.Radio(choices=list(personalities.keys()),label="Personality",value="Friendly")
    ],
    outputs=gr.Textbox(label="Generated Output",lines=12),
    title="Study Assistant",
    description="Helps study the required topic."
)

demo.launch(share=True)