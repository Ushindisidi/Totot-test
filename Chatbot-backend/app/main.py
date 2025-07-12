from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import dotenv_values

# Load environment variables at the very beginning of the application startup
env_values = dotenv_values()
for key, value in env_values.items():
    if key not in os.environ:
        os.environ[key] = value

print(
    f"DEBUG (main.py): PINECONE_API_KEY from os.getenv: '{os.getenv('PINECONE_API_KEY')}'"
)

from app.core.config import settings
from app.api.endpoints import chat

app = FastAPI(
    title="Totot Restaurant Chatbot API",
    description="AI chatbot for restaurant-related questions using LangChain and Llama 2.",
    version="1.0.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods, including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# Include the chat router
app.include_router(chat.router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Totot Restaurant Chatbot API!"}
