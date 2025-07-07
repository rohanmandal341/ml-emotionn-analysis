from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the API key from .env.local file
load_dotenv(dotenv_path='.env.local')
api_key = os.getenv("api_key")

# User input
user_input = "10 java question"

def analyze(user_input):
    # Use a predefined HuggingFace model for emotion detection
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions")
    result = classifier(user_input)
    emotion = result[0]['label']

    # Configure Gemini with the API key
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    # Generate a friendly tip using Gemini
    response = model.generate_content(f"""
    You are a caring and supportive friend. Based on the following input and detected emotion, write a short, friendly mental health tip. Keep it warm, understanding, and natural. Avoid sounding robotic or overly formal.

    User Input: {user_input}
    Detected Emotion: {emotion}

    Tip:
    """)

    # Print results
    print("\n--- Analysis Result ---")
    print(f"User Input       : {user_input}")
    print(f"Detected Emotion : {emotion}")
    print("\n--- Gemini's Tip ---")
    print(response.text)

# Call the function
analyze(user_input)