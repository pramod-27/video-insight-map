from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
import os
import tempfile
import logging
import yt_dlp
from faster_whisper import WhisperModel
from routes.summarization import summarize_text, TranscriptionRequest
from routes.mapping import map_timestamps, MappingRequest
import ffmpeg
import shutil
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

# Minimal Whisper model, single thread
model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=1)
logger.info("Whisper tiny.en model loaded successfully")

def seconds_to_hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(None),
    youtube_url: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    if not file and not youtube_url:
        raise HTTPException(status_code=400, detail="Provide either a file or a YouTube URL")
    
    logger.info(f"Starting upload: file={file.filename if file else None}, url={youtube_url}")
    try:
        if file:
            result = await process_local_video(file, background_tasks)
        else:
            result = await process_youtube_video(youtube_url, background_tasks)
        
        transcription_data = result["transcription"]
        duration = result["duration"]
        logger.info(f"Transcription done: {len(transcription_data)} segments, duration={duration}s")
        
        transcription_request = TranscriptionRequest(transcription=transcription_data)
        summary_result = await summarize_text(transcription_request, duration=duration)
        key_points = summary_result["key_points"]
        logger.info(f"Generated {len(key_points)} key points")

        mapping_request = MappingRequest(transcription=transcription_data, key_points=key_points)
        mapping_result = await map_timestamps(mapping_request)
        mapped_data = mapping_result["mapped_data"]
        logger.info(f"Mapped {len(mapped_data)} key points")

        return {
            "message": result["message"],
            "source": file.filename if file else youtube_url,
            "mapped_data": mapped_data
        }
    except Exception as e:
        error_detail = f"Upload failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

async def process_local_video(file: UploadFile, background_tasks: BackgroundTasks):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            if len(content) > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(status_code=400, detail="File too large, max 50MB")
            temp_file.write(content)
            logger.info(f"Local file saved to {temp_path}")
            
            probe = ffmpeg.probe(temp_path)
            duration = float(probe['format']['duration'])
            logger.info(f"Local video duration: {duration} seconds")
            
            transcription_data = transcribe_video(temp_path)
            background_tasks.add_task(cleanup_file, temp_path)
            return {
                "message": "Local video processed successfully",
                "transcription": transcription_data,
                "duration": duration
            }
        except Exception as e:
            background_tasks.add_task(cleanup_file, temp_path)
            raise HTTPException(status_code=500, detail=f"Local video processing failed: {str(e)}")

async def process_youtube_video(youtube_url: str, background_tasks: BackgroundTasks):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_path[:-4],  # Strip .mp3 for yt-dlp
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",  # Low quality to save memory
            }],
            "quiet": False,
            "no_warnings": False,
            "retries": 3,
            "fragment_retries": 3,
            "buffersize": "16k",  # Smaller buffer
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                duration = info.get('duration', 0)
                logger.info(f"YouTube video duration: {duration} seconds")

            if not os.path.exists(temp_path):
                raise HTTPException(status_code=500, detail="YouTube download failed")
            
            # Chunked transcription for memory efficiency
            transcription_data = transcribe_video_chunks(temp_path, duration)
            background_tasks.add_task(cleanup_file, temp_path)
            return {
                "message": "YouTube video processed successfully",
                "transcription": transcription_data,
                "duration": duration
            }
        except Exception as e:
            background_tasks.add_task(cleanup_file, temp_path)
            logger.error(f"YouTube processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"YouTube processing failed: {str(e)}")

def transcribe_video_chunks(audio_path: str, duration: float, chunk_size: int = 300):  # 5min chunks
    """Transcribe audio in chunks to manage memory."""
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail=f"Audio file not found at {audio_path}")
    
    transcription = []
    for start in range(0, int(duration), chunk_size):
        end = min(start + chunk_size, int(duration))
        logger.info(f"Processing chunk: {start}s to {end}s")
        try:
            stream = ffmpeg.input(audio_path, ss=start, t=chunk_size)
            chunk_path = f"/tmp/chunk_{start}_{end}.mp3"
            stream.output(chunk_path, format="mp3", acodec="mp3", loglevel="quiet").run(overwrite_output=True)
            
            segments, _ = model.transcribe(chunk_path, language="en")
            for segment in segments:
                adjusted_start = segment.start + start
                transcription.append({
                    "timestamp": seconds_to_hhmmss(adjusted_start),
                    "text": segment.text.strip()
                })
            cleanup_file(chunk_path)
        except Exception as e:
            logger.error(f"Chunk transcription failed: {str(e)}")
            raise
    logger.info(f"Transcription completed with {len(transcription)} segments")
    return transcription

def transcribe_video(video_path: str):
    return transcribe_video_chunks(video_path, ffmpeg.probe(video_path)['format']['duration'])

def cleanup_file(temp_path: str):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            break
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(0.5)
            else:
                logger.error(f"Failed to delete {temp_path}: {str(e)}")