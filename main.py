from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
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
except Exception as e:
    logger.error(f"Failed to load routes: {str(e)}")

@app.get("/")
async def root():
    return FileResponse("static/index.html")  # Serve index.html at root

@app.get("/health")
async def health():
    logger.info("Health check hit")
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", workers=1, timeout_keep_alive=600)