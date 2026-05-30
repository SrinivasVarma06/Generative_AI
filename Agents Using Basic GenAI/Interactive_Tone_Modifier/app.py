from google import genai
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
tones = {
    "Formal": "Rewrite the given sentence in a formal tone.",
    "Casual": "Rewrite the given sentence in a casual tone."
}

def tone_translator(sentence, tone):
    response=client.models.generate_content(
        model="gemini-2.5-flash",
        contents=sentence,
        config=genai.types.GenerateContentConfig(
            system_instruction=tones[tone],
            temperature=0.4,
            max_output_tokens=2000
        )
    )
    return response.text

demo=gr.Interface(
    fn=tone_translator,
    inputs=[
        gr.Textbox(label="Input Sentence",lines=3,placeholder="Enter a sentence to rewrite..."),
        gr.Radio(choices=list(tones.keys()),label="Tone",value="Formal")
    ],
    outputs=gr.Textbox(label="Rewritten Text",lines=5),
    title="Tone_Modifier",
    description="Modifies the tone based on requirements."
)

demo.launch(share=True)