from fastapi import APIRouter, HTTPException
from faster_whisper import WhisperModel
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

def seconds_to_hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@router.post("/transcription")
async def transcribe(video_path: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    try:
        segments, _ = model.transcribe(video_path, language="en")
        transcription = [{"timestamp": seconds_to_hhmmss(segment.start), "text": segment.text.strip()} for segment in segments]
        logger.info(f"Transcription completed with {len(transcription)} segments")
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")