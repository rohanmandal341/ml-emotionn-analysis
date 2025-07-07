from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv('.env.local')
api_key = os.getenv("api_key")

app = FastAPI()

# Define input schema
class UserInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_entry(data: UserInput):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions")
    result = classifier(data.text)
    emotion = result[0]['label']

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(f"""
    You are a caring and supportive friend. Based on the following input and detected emotion, write a short, friendly mental health tip. Keep it warm, understanding, and natural.

    User Input: {data.text}
    Detected Emotion: {emotion}

    Tip:
    """)

    return {
        "emotion": emotion,
        "confidence": result[0]['score'],
        "tip": response.text
    }

