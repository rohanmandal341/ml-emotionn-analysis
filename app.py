from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')
api_key = os.getenv("api_key")

# ✅ Load Hugging Face model ONCE (outside the route)
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions")

# ✅ Configure Gemini once
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Initialize FastAPI app
app = FastAPI()

# Request model
class UserInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_entry(data: UserInput):
    result = classifier(data.text)
    emotion = result[0]['label']
    confidence = result[0]['score']

    # Generate AI tip
    response = model.generate_content(f"""
    You are a caring and supportive friend. Based on the following input and detected emotion, write a short, friendly mental health tip. Keep it warm, understanding, and natural.

    User Input: {data.text}
    Detected Emotion: {emotion}

    Tip:
    """)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "tip": response.text
    }
