from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks
import os
import asyncio
from gpt import *

events = []
analyzing = False

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
async def analyze_video(request: Request, background_tasks: BackgroundTasks):
    """Processes the video in the background and detects events."""
    global analyzing, events
    if analyzing:
        raise HTTPException(status_code=429, detail="Analysis is already in progress.") 
    data = await request.json()
    video_path = data.get("video_path")
    if not video_path:
        raise HTTPException(status_code=422, detail="Video path is required.")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    analyzing = True
    def background_task():
        """Runs video processing in the background."""
        global events, analyzing
        events = process_video(video_path)
        save_results(events)
        analyzing = False
    background_tasks.add_task(background_task)
    return {"status": "started", "message": "Video analysis is running in the background."}

@app.get("/events/")
async def get_events():
    """Returns the detected events."""
    return events

@app.post("/seek/")
async def seek_video(timestamp: float):
    """Seeks the video to a specific timestamp."""
    return {"status": "success", "timestamp": timestamp}

@app.get("/status/")
async def get_analysis_status():
    """Returns whether the analysis is still running or completed."""
    global analyzing
    return {"analyzing": analyzing}

@app.post("/chatbot/")
async def chatbot_endpoint(request: Request):
    """Handles chatbot interactions using detected video events."""
    data = await request.json()
    user_message = data.get("message")

    if not user_message:
        raise HTTPException(status_code=422, detail="User message is required.")

    global events, analyzing
    if not events and not analyzing:
        return {"response": "No events have been analyzed yet. Please upload and analyze a video first."}

    if analyzing:
        return {"response": "I'm currently analyzing the video, please wait..."}

    response_text = get_response_with_events(events, user_message)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
