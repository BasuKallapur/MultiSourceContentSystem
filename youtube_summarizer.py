import os
import re
import time
import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Extract Video ID from YouTube URL
def extract_video_id(url):
    match = re.search(r"(?:v=|\/|embed\/|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Fetch Transcript with Timestamps (Every 30s)
def extract_transcript_details(youtube_video_url, target_language='en'):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            return [("Error", "Invalid YouTube URL.")]

        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[target_language])
        except NoTranscriptFound:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                if transcript.is_generated:
                    transcript_data = transcript.translate(target_language).fetch()
                    break
            else:
                return [("Error", "No suitable transcript found.")]
        except TranscriptsDisabled:
            return [("Error", "Transcripts are disabled for this video.")]
        
        transcript = []
        last_timestamp = -30
        temp_text = []

        for item in transcript_data:
            minutes = int(item["start"] // 60)
            seconds = int(item["start"] % 60)
            timestamp = f"{minutes:02}:{seconds:02}"

            if item["start"] - last_timestamp >= 30:
                if temp_text:
                    transcript.append((last_timestamp_text, " ".join(temp_text)))
                    temp_text = []
                last_timestamp = item["start"]
                last_timestamp_text = timestamp

            temp_text.append(item["text"])

        if temp_text:
            transcript.append((last_timestamp_text, " ".join(temp_text)))

        return transcript
    except Exception as e:
        return [("Error", str(e))]

# Call Groq API with Retry & Token Management
def call_groq_api(api_key, prompt, content):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    for attempt in range(3):
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from model.").strip()
        elif response.status_code == 429:
            wait_time = min(60, 2 ** attempt * 10)  # Exponential backoff, max wait 60s
            print(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    return "Error: Too many retries due to rate limit."

# Chunk Transcript for API Efficiency
def chunk_transcript(transcript, chunk_size=20):
    return ["\n".join([f"[{ts}] {text}" for ts, text in transcript[i:i + chunk_size]]) for i in range(0, len(transcript), chunk_size)]
