from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from fuzzywuzzy import fuzz
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

class TranscriptionItem(BaseModel):
    timestamp: str
    text: str

class KeyPointItem(BaseModel):
    timestamp: str
    text: str

class MappingRequest(BaseModel):
    transcription: List[TranscriptionItem]
    key_points: List[KeyPointItem]

def hhmmss_to_seconds(hhmmss: str) -> float:
    try:
        time_obj = datetime.strptime(hhmmss, "%H:%M:%S")
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    except ValueError:
        logger.warning(f"Invalid timestamp: {hhmmss}")
        return 0

@router.post("/mapping")
async def map_timestamps(request: MappingRequest):
    if not request.transcription or not request.key_points:
        raise HTTPException(status_code=400, detail="Transcription or key points empty")

    try:
        transcription_with_seconds = [
            {"timestamp": entry.timestamp, "text": entry.text.strip(), "seconds": hhmmss_to_seconds(entry.timestamp)}
            for entry in request.transcription
        ]

        mapped_data = []
        for key_point in request.key_points:
            key_point_seconds = hhmmss_to_seconds(key_point.timestamp)
            best_match, best_score = None, -1

            for trans_entry in transcription_with_seconds:
                time_diff = abs(trans_entry["seconds"] - key_point_seconds)
                time_similarity = max(0, 1 - (time_diff / 60.0)) * 100
                text_similarity = fuzz.partial_ratio(key_point.text.lower(), trans_entry["text"].lower())
                combined_score = 0.6 * text_similarity + 0.4 * time_similarity

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = trans_entry

            if best_score > 50 and best_match:
                mapped_data.append({"timestamp": best_match["timestamp"], "text": key_point.text})
            else:
                mapped_data.append({"timestamp": key_point.timestamp, "text": key_point.text, "note": "No close match"})

        logger.info(f"Mapped {len(mapped_data)} key points")
        return {"mapped_data": mapped_data}
    except Exception as e:
        logger.error(f"Mapping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mapping failed: {str(e)}")