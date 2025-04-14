from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
import os
import tempfile
import logging
import yt_dlp
from faster_whisper import WhisperModel
from routes.summarization import summarize_text, TranscriptionRequest
from routes.mapping import map_timestamps, MappingRequest
import ffmpeg
import time
from datetime import datetime
import torch
import asyncio
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
# Suppress DEBUG logs from other libraries
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

router = APIRouter()

# Optimized Whisper setup
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel("tiny.en", device=device, compute_type=compute_type, cpu_threads=os.cpu_count() // 2)
logger.info(f"Whisper tiny.en model loaded on {device.upper()} using {compute_type}")

# Process pool for CPU-bound tasks
executor = ProcessPoolExecutor(max_workers=2)

def seconds_to_hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def hhmmss_to_seconds(hhmmss: str) -> int:
    try:
        t = datetime.strptime(hhmmss, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        logger.warning(f"Invalid timestamp format: {hhmmss}")
        return 0

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(None),
    youtube_url: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    if not file and not youtube_url:
        raise HTTPException(status_code=400, detail="Provide either a file or a YouTube URL")

    logger.info(f"Starting upload processing: file={file.filename if file else None}, youtube_url={youtube_url}")
    total_start = time.time()

    try:
        start = time.time()
        transcription_result = await (
            process_local_video(file, background_tasks) if file else process_youtube_video(youtube_url)
        )
        logger.info(f"Transcription stage completed in {time.time() - start:.2f} seconds")

        transcription_data = transcription_result["transcription"]
        duration = transcription_result["duration"]

        start = time.time()
        summary_result = await summarize_text(TranscriptionRequest(transcription=transcription_data), duration=duration)
        logger.info(f"Summarization completed in {time.time() - start:.2f} seconds")
        key_points = summary_result["key_points"]

        start = time.time()
        mapping_result = await map_timestamps(MappingRequest(transcription=transcription_data, key_points=key_points))
        logger.info(f"Mapping completed in {time.time() - start:.2f} seconds")
        mapped_data = mapping_result["mapped_data"]

        filtered_data = [
            point for point in mapped_data if hhmmss_to_seconds(point["timestamp"]) <= duration
        ]
        sorted_data = sorted(filtered_data, key=lambda x: hhmmss_to_seconds(x["timestamp"]))
        total_time = time.time() - total_start
        logger.info(f"Total processing time: {total_time:.2f} seconds")

        return {
            "message": transcription_result["message"],
            "source": file.filename if file else youtube_url,
            "mapped_data": sorted_data
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def process_local_video(file: UploadFile, background_tasks: BackgroundTasks):
    logger.info(f"Processing local video: {file.filename}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
        try:
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)
            logger.info(f"Local file saved to {temp_path}")

            duration = float(ffmpeg.probe(temp_path)['format']['duration'])
            logger.info(f"Local video duration: {duration:.2f} seconds")

            start = time.time()
            transcription_data = await asyncio.get_event_loop().run_in_executor(
                executor, transcribe_video, temp_path
            )
            logger.info(f"Transcription completed in {time.time() - start:.2f} seconds")

            background_tasks.add_task(cleanup_file, temp_path)

            return {
                "message": "Local video processed successfully",
                "transcription": transcription_data,
                "duration": duration
            }
        except Exception as e:
            background_tasks.add_task(cleanup_file, temp_path)
            logger.error(f"Error processing local video: {e}")
            raise HTTPException(status_code=500, detail=f"Local video processing failed: {str(e)}")

async def process_youtube_video(youtube_url: str):
    logger.info(f"Processing YouTube URL: {youtube_url}")
    # Relaxed validation for youtube.com and youtu.be
    if not youtube_url.startswith(("https://www.youtube.com/", "https://youtu.be/", "https://youtube.com/")):
        logger.error(f"Invalid YouTube URL: {youtube_url}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.webm")
        ydl_opts = {
            "format": "bestaudio[ext=webm]",
            "outtmpl": audio_path,
            "quiet": True,
            "noplaylist": True,
            "retries": 3,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
            }
        }

        try:
            start = time.time()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                duration = info.get("duration", 0)
            logger.info(f"YouTube download completed in {time.time() - start:.2f} seconds. Duration: {duration:.2f}s")

            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found at {audio_path}")
                raise HTTPException(status_code=500, detail="Failed to download YouTube audio")

            start = time.time()
            transcription_data = await asyncio.get_event_loop().run_in_executor(
                executor, transcribe_video, audio_path
            )
            logger.info(f"Transcription completed in {time.time() - start:.2f} seconds")

            return {
                "message": "YouTube video processed successfully",
                "transcription": transcription_data,
                "duration": duration
            }

        except Exception as e:
            logger.error(f"YouTube processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"YouTube processing failed: {str(e)}")

def transcribe_video(video_path: str):
    logger.info(f"Starting transcription for {video_path}")
    if not os.path.exists(video_path):
        logger.error(f"Video file not found at {video_path}")
        raise HTTPException(status_code=500, detail=f"Video file not found at {video_path}")

    segments, _ = model.transcribe(video_path, language="en", vad_filter=False, beam_size=5)
    transcription = [
        {"timestamp": seconds_to_hhmmss(segment.start), "text": segment.text.strip()}
        for segment in segments
    ]
    logger.info(f"Transcription generated {len(transcription)} segments")
    return transcription

def cleanup_file(temp_path: str):
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")
    except Exception as e:
        logger.warning(f"Failed to delete {temp_path}: {str(e)}")