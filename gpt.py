import os
import json
import cv2
from openai import AzureOpenAI
import base64
from dotenv import load_dotenv
import asyncio

load_dotenv()

api_key = os.getenv("AZURE_GPT_4O_KEY")
if not api_key:
    raise ValueError("API key not found. Set the AZURE_OPENAI_API_KEY environment variable.")
azure_endpoint = os.getenv("AZURE_GPT_4O_EP")
api_version = "2024-08-01-preview"

analyzer = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)
assistant = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

def encode_image_to_base64(image):
    """Encodes an image to base64."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def extract_frames(video_path, frame_interval=5):
    """Extracts frames from the video at specified intervals."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frames = []
    timestamps = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
            timestamps.append(frame_count / fps)
        frame_count += 1

    cap.release()
    return frames, timestamps

def divide_video_into_chunks(video_path, parallel_calls):
    frames, timestamps = extract_frames(video_path)
    total_frames = len(frames)
    chunk_size = total_frames // parallel_calls

    print(f"Total frames: {total_frames}, Chunk size: {chunk_size}")
    
    
    chunks = [frames[i*chunk_size:(i+1)*chunk_size] for i in range(parallel_calls)]
    chunked_timestamps = [timestamps[i*chunk_size:(i+1)*chunk_size] for i in range(parallel_calls)]
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} has {len(chunk)} frames starting at {chunked_timestamps[i]} seconds")

    return chunks, chunked_timestamps

async def detect_events_with_gpt_parallel(frame_chunk, chunked_timestamp):
    events = []
    for i in range(0, len(frame_chunk) - 2, 3):
        encoded_frame1 = encode_image_to_base64(frame_chunk[i])
        encoded_frame2 = encode_image_to_base64(frame_chunk[i+1])
        encoded_frame3 = encode_image_to_base64(frame_chunk[i+2])

        timestamp = chunked_timestamp[i]
        
        response = analyzer.chat.completions.create(
            model="gpt-4o-video-understand",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI that detects basketball actions (shoot, pass) from images."
                        "You MUST use the given timestamp as the 'time' value in the output."
                        "Respond only in JSON format: "
                        "{\"actions\": [{\"time\": <provided_timestamp>, \"event\": \"<shoot_or_pass>\"}]}"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze these frames and determine if any basketball player performs a 'shoot' or 'pass' action. Use {timestamp} as the 'time' value. Return the result strictly in JSON format."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame1}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame2}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame3}"}}
                    ]
                }
            ],
            temperature=1,
            max_tokens=300,
            top_p=1
        )

        result_text = response.choices[0].message.content.strip()
        print(f"API response: {result_text}")
        try:
            result_json = json.loads(result_text.replace("```json", "").replace("```", ""))
            for action in result_json.get("actions", []):
                events.append({"time": action["time"], "event": action["event"]})
        except json.JSONDecodeError:
            continue

    return events

async def process_video_parallel(video_path, parallel_calls):
    chunks, chunked_timestamps = divide_video_into_chunks(video_path, parallel_calls)

    tasks = [
        detect_events_with_gpt_parallel(chunk, chunked_timestamp) 
        for chunk, chunked_timestamp in zip(chunks, chunked_timestamps)
    ]
    all_events = await asyncio.gather(*tasks)
    return [event for chunk_events in all_events for event in chunk_events]

def save_results(events, output_file="events.json"):
    """Saves the detected events to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(events, f, indent=4)

def get_response_with_events(events, user_message):
    """Generates a chatbot response based on user input and detected events."""
    if not events:
        return "No events detected yet. Please analyze a video first."

    # Format events into a readable summary
    events_summary = "\n".join([f"At {event['time']}s: {event['event']}" for event in events]) or "No significant events detected."

    # Send user message + detected events to Azure OpenAI
    response = assistant.chat.completions.create(
        model="gpt-4o-video-understand",
        messages=[
            {"role": "system", "content": "You are a basketball analyst assistant. Use provided game events to enhance responses."},
            {"role": "user", "content": f"Here are the detected basketball events:\n{events_summary}\n\nUser question: {user_message}"}
        ],
        temperature=1,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
