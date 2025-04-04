from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
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

@router.post("/plaintext_summary")
async def plaintext_summarize_text(request: TranscriptionRequest):
    if not request.transcription:
        raise HTTPException(status_code=400, detail="Transcription list is empty")

    full_text = " ".join(entry.text for entry in request.transcription)
    prompt = f"""
    Summarize the following text into a concise paragraph of 7-10 sentences.
    Focus on the main points and exclude timestamps or unnecessary details.
    
    Text:
    {full_text}
    """

    global current_key_index
    max_attempts = len(API_KEYS)
    for attempt in range(max_attempts):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            if not response or not response.text:
                raise HTTPException(status_code=500, detail="Empty summary response")
            
            summary = response.text.strip()
            logger.info(f"Plaintext summary generated with key {current_key_index}")
            return {"summary": summary}
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                logger.warning(f"Rate limit hit for key {current_key_index}: {str(e)}")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                configure_genai()
                logger.info(f"Switched to API key {current_key_index}")
                if attempt == max_attempts - 1:
                    raise HTTPException(status_code=429, detail="All API keys exhausted")
            else:
                logger.error(f"Plaintext summarization failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Plaintext summarization failed: {str(e)}")