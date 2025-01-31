import os
import json
import cv2
from openai import OpenAI
import base64
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Set the OPENAI_API_KEY environment variable.")
analyzer = OpenAI(api_key=api_key)
assistant = OpenAI(api_key=api_key)

def encode_image_to_base64(image):
    """Encodes an image to base64."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def extract_frames(video_path, frame_interval=15):
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

def detect_events_with_gpt(frame1, frame2, frame3, timestamp):
    """Sends three consecutive frames to GPT-4V for structured action detection."""
    encoded_frame1 = encode_image_to_base64(frame1)
    encoded_frame2 = encode_image_to_base64(frame2)
    encoded_frame3 = encode_image_to_base64(frame3)

    response = analyzer.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI that detects basketball actions (shoot, pass) from images. "
                    "You MUST use the given timestamp as the 'time' value in the output. "
                    "Respond only in the following JSON format: "
                    "{\"actions\": [{\"time\": <provided_timestamp>, \"event\": \"<shoot_or_pass>\"}]}. "
                    "DO NOT make up timestamps; use ONLY the provided one."
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
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    result_text = response.choices[0].message.content.strip()
    print(result_text)

    try:
        result_json = json.loads(result_text.replace("```json", "").replace("```", ""))
        return [(timestamp, action["event"]) for action in result_json.get("actions", [])]
    except json.JSONDecodeError:
        return []

def process_video(video_path):
    """Processes video and detects key basketball events."""
    frames, timestamps = extract_frames(video_path)
    detected_events = []
    last_event_time = {}

    for i in range(len(frames) - 2):
        events = detect_events_with_gpt(frames[i], frames[i + 1], frames[i + 2], timestamps[i])
        for timestamp, event in events:
            if event not in last_event_time or (timestamp - last_event_time[event]) >= 1.5:
                detected_events.append({"time": timestamp, "event": event})
                last_event_time[event] = timestamp

    return detected_events

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

    # Send user message + detected events to OpenAI
    response = assistant.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a basketball analyst assistant. Use provided game events to enhance responses."},
            {"role": "user", "content": f"Here are the detected basketball events:\n{events_summary}\n\nUser question: {user_message}"}
        ],
        temperature=1,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
