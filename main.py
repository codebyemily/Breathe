from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time
from pathlib import Path
from datetime import datetime
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

router = APIRouter()

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_dir / "mentor.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

class WebhookRequest(BaseModel):
    session_id: str
    segments: List[Dict[str, Any]] = []

class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 600  # 10 minutes
        self.last_cleanup = time.time()
        self.silence_threshold = 120  # 2 minutes silence threshold
        self.min_words_after_silence = 5  # minimum words needed after silence

    def get_buffer(self, session_id):
        current_time = time.time()

        # Cleanup old sessions periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_sessions()

        with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {
                    'messages': [],
                    'last_analysis_time': time.time(),
                    'last_activity': current_time,
                    'words_after_silence': 0,
                    'silence_detected': False
                }
            else:
                # Check for silence period
                time_since_activity = current_time - self.buffers[session_id]['last_activity']
                if time_since_activity > self.silence_threshold:
                    self.buffers[session_id]['silence_detected'] = True
                    self.buffers[session_id]['words_after_silence'] = 0
                    self.buffers[session_id]['messages'] = []  # Clear old messages after silence

                self.buffers[session_id]['last_activity'] = current_time

        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600  # Remove sessions older than 1 hour
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time

# Initialize message buffer
message_buffer = MessageBuffer()

ANALYSIS_INTERVAL = 300  # 120 seconds between analyses

def create_notification_prompt(messages: list) -> dict:
    """Create notification with prompt template"""

    # Format the discussion with speaker labels
    formatted_discussion = []
    for msg in messages:
        speaker = "{{{{user_name}}}}" if msg.get('is_user') else "other"
        formatted_discussion.append(f"{msg['text']} ({speaker})")

    discussion_text = "\n".join(formatted_discussion)

# To do: update system prompt
    system_prompt = system_prompt = """You are a calm grounding companion designed to gently interrupt moments of anxiety.

Your role is NOT to analyze, diagnose, explain, or fix anything.
Your role is to help the user regulate in the moment.

STEP 1 — Evaluate SILENTLY whether an interruption is appropriate.

Only interrupt if ALL of the following are true:
1. {{{{user_name}}}} is actively speaking (messages marked with '({{{{user_name}}}})' are present)
2. The speech shows signs of anxiety, rumination, fear spiraling, or mental looping
3. A brief grounding prompt would help more than staying silent
4. The moment is time-sensitive (waiting would make it worse)

If ANY condition is not met:
→ Respond with an empty string and nothing else.

STEP 2 — If ALL conditions are met, generate ONE grounding interruption.

Rules for the response:
- Speak directly and gently to {{{{user_name}}}}
- Do NOT analyze thoughts or explain emotions
- Do NOT ask “why”
- Do NOT give advice or solutions
- Keep it under 300 characters
- Use warm, simple, everyday language
- Focus on breath, body, senses, or present-moment anchoring
- Normalize the feeling without validating fear
- End with ONE soft grounding question (sensations, breath, or immediate next step)

Allowed tones:
- calm
- reassuring
- steady
- non-judgmental

Current conversation:
{text}

User context (if helpful): {{{{user_context}}}}

Remember:
This is not therapy.
This is a moment of regulation.
Output either:
• an empty string
• OR a single short grounding message
""".format(text=discussion_text)

    return {
        "notification": {
            "prompt": system_prompt,
            "params": ["user_name", "user_facts", "user_context", "user_chat"],
            "context": {
                "filters": {
                    "people": [],
                    "entities": [],
                    "topics": discussion_text
                }
            }
        }
    }

@router.post('/notification/mentor/webhook')
async def webhook(request: WebhookRequest):
    session_id = request.session_id
    segments = request.segments

    current_time = time.time()
    buffer_data = message_buffer.get_buffer(session_id)

    # Process new messages
    for segment in segments:
        if not segment.get('text'):
            continue

        text = segment['text'].strip()
        if text:
            timestamp = segment.get('start', 0) or current_time
            is_user = segment.get('is_user', False)

            # Count words after silence
            if buffer_data['silence_detected']:
                words_in_segment = len(text.split())
                buffer_data['words_after_silence'] += words_in_segment

                # If we have enough words, start fresh conversation
                if buffer_data['words_after_silence'] >= message_buffer.min_words_after_silence:
                    buffer_data['silence_detected'] = False
                    buffer_data['last_analysis_time'] = current_time  # Reset analysis timer
                    logger.info(f"Silence period ended for session {session_id}, starting fresh conversation")

            can_append = (
                buffer_data['messages'] and 
                abs(buffer_data['messages'][-1]['timestamp'] - timestamp) < 2.0 and
                buffer_data['messages'][-1].get('is_user') == is_user
            )

            if can_append:
                buffer_data['messages'][-1]['text'] += ' ' + text
            else:
                buffer_data['messages'].append({
                    'text': text,
                    'timestamp': timestamp,
                    'is_user': is_user
                })

    # Check if it's time to analyze
    time_since_last_analysis = current_time - buffer_data['last_analysis_time']

    if (time_since_last_analysis >= ANALYSIS_INTERVAL and 
        buffer_data['messages'] and 
        not buffer_data['silence_detected']):  # Only analyze if not in silence period

        # Sort messages by timestamp
        sorted_messages = sorted(buffer_data['messages'], key=lambda x: x['timestamp'])

        # Create notification with formatted discussion
        notification = create_notification_prompt(sorted_messages)

        buffer_data['last_analysis_time'] = current_time
        buffer_data['messages'] = []  # Clear buffer after analysis

        logger.info(f"Sending notification with prompt template for session {session_id}")
        logger.info(notification)

        return JSONResponse(content=notification, status_code=200)

    return JSONResponse(content={}, status_code=202)

@router.get('/notification/mentor/webhook/setup-status')
async def setup_status():
    return {"is_setup_completed": True}

@router.get('/notification/mentor/status')
async def status():
    return {
        "active_sessions": len(message_buffer.buffers),
        "uptime": time.time() - start_time
    }

# Add start time tracking
start_time = time.time()