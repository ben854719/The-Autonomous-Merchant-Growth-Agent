from fastapi import FastAPI
import os
import google.generativeai as genai

app = FastAPI()

# Load Gemini key from environment variable
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@app.get("/")
def home():
    return {"message": "Autonomous Merchant Growth Agent is running!"}
