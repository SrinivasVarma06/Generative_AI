from google import genai
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
client=genai.Client(api_key=GEMINI_API_KEY)
question_types={
    "MCQs":"Generate MCQ type questions from the given content.",
    "Short Answer":"Generate short answer type questions from the given content.",
    "Interview":"Generate interview based questions from the given content."
}

def question_generator(content,q_type):
    response=client.models.generate_content(
        model='gemini-2.5-flash',
        contents=content,
        config=genai.types.GenerateContentConfig(
            system_instruction=question_types[q_type],
            temperature=0.4,
            max_output_tokens=2000
        )
    )
    return response.text
    #complete this function
    
demo=gr.Interface(
    fn=question_generator,
    inputs=[
        gr.Textbox(lines=6,label="Input Content",placeholder="Paste study material or content here..."),
        gr.Radio(choices=list(question_types.keys()),label="Question Type",value="MCQs")
    ],
    outputs=gr.Textbox(label="Generated Questions",lines=12),
    title="Questions Generator",
    description="Generates questions on the topic given."
)

demo.launch(share=True)