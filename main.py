from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Insight Mapping API",
    description="API for uploading, transcribing, summarizing, and mapping video content",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

from routes.transcription import router as transcription_router
from routes.summarization import router as summarization_router
from routes.mapping import router as mapping_router
from routes.upload import router as upload_router
from routes.plaintext_summarization import router as plaintext_summarization_router

app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(transcription_router, prefix="/api", tags=["Transcription"])
app.include_router(summarization_router, prefix="/api", tags=["Summarization"])
app.include_router(mapping_router, prefix="/api", tags=["Mapping"])
app.include_router(plaintext_summarization_router, prefix="/api", tags=["Plaintext Summarization"])

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to Video Insight Mapping API!"}

@app.get("/ping", tags=["Health"])
async def ping():
    return {"message": "Backend is running!", "date": "April 04, 2025"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")