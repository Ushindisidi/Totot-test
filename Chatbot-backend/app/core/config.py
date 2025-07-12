import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Configuration settings for the application, loaded from environment variables.
    """


    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "") # Keep this

    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

settings = Settings()
