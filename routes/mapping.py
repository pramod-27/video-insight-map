from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

router = APIRouter()
sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

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
    logger.info(f"Starting mapping for {len(request.transcription)} transcription items and {len(request.key_points)} key points")
    if not request.transcription or not request.key_points:
        logger.error("Transcription or key points empty")
        raise HTTPException(status_code=400, detail="Transcription or key points empty")

    try:
        transcription_with_seconds = sorted(
            [
                {"timestamp": entry.timestamp, "text": entry.text.strip(), "seconds": hhmmss_to_seconds(entry.timestamp)}
                for entry in request.transcription
            ],
            key=lambda x: x["seconds"]
        )

        trans_texts = [entry["text"] for entry in transcription_with_seconds]
        trans_embeddings = sentence_model.encode(trans_texts, convert_to_tensor=True)

        mapped_data = []
        for key_point in request.key_points:
            key_point_seconds = hhmmss_to_seconds(key_point.timestamp)
            key_embedding = sentence_model.encode([key_point.text], convert_to_tensor=True)

            similarities = util.cos_sim(key_embedding, trans_embeddings)[0].numpy()
            time_diffs = np.array([abs(entry["seconds"] - key_point_seconds) for entry in transcription_with_seconds])
            time_similarities = np.maximum(0, 1 - (time_diffs / 60.0)) * 100
            combined_scores = 0.6 * similarities * 100 + 0.4 * time_similarities

            best_idx = np.argmax(combined_scores)
            best_score = combined_scores[best_idx]

            if best_score > 50:
                best_match = transcription_with_seconds[best_idx]
                mapped_data.append({"timestamp": best_match["timestamp"], "text": key_point.text})
            else:
                mapped_data.append({"timestamp": key_point.timestamp, "text": key_point.text, "note": "No close match"})

        logger.info(f"Mapped {len(mapped_data)} key points")
        return {"mapped_data": mapped_data}
    except Exception as e:
        logger.error(f"Mapping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Mapping failed: {str(e)}")