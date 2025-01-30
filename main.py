from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from gpt import *

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the uploads directory
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """Handles video uploads and saves them to the uploads folder."""
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "location": file.filename}

@app.post("/analyze/")
async def analyze_video(request: Request):
    """Processes the video and detects events."""
    data = await request.json()
    video_path = data.get("video_path")
    if not video_path:
        raise HTTPException(status_code=422, detail="Video path is required.")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    global events
    events = process_video(video_path)
    save_results(events)
    return {"status": "completed", "events": events}

@app.get("/events/")
async def get_events():
    """Returns the detected events."""
    return events

@app.post("/seek/")
async def seek_video(timestamp: float):
    """Seeks the video to a specific timestamp."""
    return {"status": "success", "timestamp": timestamp}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
