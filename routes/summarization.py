from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
router = APIRouter()

# Load all API keys
API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
]
if not all(API_KEYS):
    logger.error("One or more GOOGLE_API_KEYs not set")
    raise RuntimeError("GOOGLE_API_KEYs missing in environment variables")

# Global state for key rotation
current_key_index = 0

def configure_genai():
    global current_key_index
    genai.configure(api_key=API_KEYS[current_key_index])
    logger.info(f"Using API key at index {current_key_index}")

configure_genai()  # Initial config

class TranscriptionItem(BaseModel):
    timestamp: str
    text: str

class TranscriptionRequest(BaseModel):
    transcription: list[TranscriptionItem]

@router.post("/summarize")
async def summarize_text(request: TranscriptionRequest, duration: float = None):
    if not request.transcription:
        raise HTTPException(status_code=400, detail="Transcription list is empty")

    num_points = "7-10" if not duration else (
        "5-7" if duration / 60 <= 5 else
        "7-10" if duration / 60 <= 15 else
        "10-12" if duration / 60 <= 30 else "12-15"
    )

    formatted_text = "\n".join(f"{entry.timestamp} {entry.text}" for entry in request.transcription)
    prompt = f"""
    Extract the most important {num_points} key points from the following transcription.
    Keep timestamps in HH:MM:SS format and remove unnecessary details.
    
    Format:
    HH:MM:SS Key point

    Transcription:
    {formatted_text}
    """

    global current_key_index
    max_attempts = len(API_KEYS)
    for attempt in range(max_attempts):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            if not response or not response.text:
                raise HTTPException(status_code=500, detail="Empty summary response")
            
            key_points = response.text.strip().split("\n")
            cleaned_key_points = [
                {"timestamp": match.group(1), "text": match.group(2)}
                for line in key_points if (match := re.match(r"(\d{2}:\d{2}:\d{2})\s+(.+)", line))
            ]
            logger.info(f"Generated {len(cleaned_key_points)} key points with key {current_key_index}")
            return {"key_points": cleaned_key_points}
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                logger.warning(f"Rate limit hit for key {current_key_index}: {str(e)}")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                configure_genai()
                logger.info(f"Switched to API key {current_key_index}")
                if attempt == max_attempts - 1:
                    raise HTTPException(status_code=429, detail="All API keys exhausted")
            else:
                logger.error(f"Summarization failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")