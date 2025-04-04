from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    logger.info("Root endpoint hit")
    return {"message": "Hello World!"}

@app.get("/health")
async def health():
    logger.info("Health check hit")
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", workers=1, timeout_keep_alive=60)  # Longer keep-alive