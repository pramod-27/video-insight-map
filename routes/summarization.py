from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
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
    logger.info(f"Starting summarization for {len(request.transcription)} segments")
    if not request.transcription:
        logger.error("Transcription list is empty")
        raise HTTPException(status_code=400, detail="Transcription list is empty")

    formatted_text = "\n".join(f"{entry.timestamp} {entry.text}" for entry in request.transcription)
    prompt = f"""
    Extract the most impactful and meaningful key insights from the following transcription, focusing on ideas that capture the core essence and primary value of the content.
    Each insight must be a complete, concise sentence (maximum one sentence per point) that carries significant weight and resonates with the video’s main narrative.
    Be selective—include only insights that are critical to the content’s purpose, excluding minor details or redundant ideas, but ensure enough points to comprehensively represent the video’s key messages.
    Distribute insights across the entire duration of the video to capture key moments from beginning to end, ensuring a balanced representation of the content.
    The number of insights should reflect the content’s richness and duration (e.g., roughly 3-4 insights for sparse content per 10 minutes, up to 5-7 for dense content per 10 minutes), prioritizing quality to ensure a curated, trustworthy output.
    Include the corresponding timestamp in HH:MM:SS format.
    
    Format:
    HH:MM:SS <Complete sentence describing a high-impact, meaningful insight.>

    Transcription:
    {formatted_text}
    """

    global current_key_index
    max_attempts = len(API_KEYS)
    for attempt in range(max_attempts):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if not response or not response.text:
                logger.error("Empty summary response from Gemini")
                raise HTTPException(status_code=500, detail="Empty summary response")
            
            key_points = response.text.strip().split("\n")
            cleaned_key_points = [
                {"timestamp": match.group(1), "text": match.group(2)}
                for line in key_points if (match := re.match(r"(\d{2}:\d{2}:\d{2})\s+(.+\.)", line))
            ]
            logger.info(f"Generated {len(cleaned_key_points)} key points with key {current_key_index}")
            return {"key_points": cleaned_key_points}
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                logger.warning(f"Rate limit hit for key {current_key_index}: {str(e)}")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                configure_genai()
                if attempt == max_attempts - 1:
                    logger.error("All API keys exhausted")
                    raise HTTPException(status_code=429, detail="All API keys exhausted")
            else:
                logger.error(f"Summarization failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")