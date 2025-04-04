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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

# Load model with minimal settings
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
            transcription_result = await process_local_video(file, background_tasks)
        else:
            transcription_result = await process_youtube_video(youtube_url)
        
        transcription_data = transcription_result["transcription"]
        duration = transcription_result["duration"]
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
            "message": transcription_result["message"],
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

async def process_youtube_video(youtube_url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_base = os.path.join(temp_dir, "audio")
        temp_path = f"{temp_base}.mp3"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": temp_base,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": False,
            "no_warnings": False,
            "retries": 3,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                duration = info.get('duration', 0)
                logger.info(f"YouTube video duration: {duration} seconds")

            if not os.path.exists(temp_path):
                raise HTTPException(status_code=500, detail="YouTube download failed")
            transcription_data = transcribe_video(temp_path)
            return {
                "message": "YouTube video processed successfully",
                "transcription": transcription_data,
                "duration": duration
            }
        except Exception as e:
            logger.error(f"YouTube processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"YouTube processing failed: {str(e)}")

def transcribe_video(video_path: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=500, detail=f"Video file not found at {video_path}")
    segments, _ = model.transcribe(video_path, language="en")
    transcription = [{"timestamp": seconds_to_hhmmss(segment.start), "text": segment.text.strip()} for segment in segments]
    logger.info(f"Transcription completed with {len(transcription)} segments")
    return transcription

def cleanup_file(temp_path: str):
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True) if os.path.isdir(temp_path) else os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            break
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(1)
            else:
                logger.error(f"Failed to delete {temp_path}: {str(e)}")