from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time
from pathlib import Path
import threading
from dotenv import load_dotenv

# ----------------------------
# Logging setup
# ----------------------------
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "breathe.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Message buffer for sessions
# ----------------------------
class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 600  # 10 min
        self.last_cleanup = time.time()
        self.silence_threshold = 30  # 30 secs
        self.min_words_after_silence = 10

    def get_buffer(self, session_id):
        current_time = time.time()

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
                    self.buffers[session_id]['messages'] = []

                self.buffers[session_id]['last_activity'] = current_time

        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time

# Initialize buffer
message_buffer = MessageBuffer()
ANALYSIS_INTERVAL = 2  

# ----------------------------
# Request model
# ----------------------------
class WebhookRequest(BaseModel):
    session_id: str
    segments: List[Dict[str, Any]] = []

# ----------------------------
# Notification logic
# ----------------------------
def create_notification_prompt(messages: list) -> dict:
    formatted_discussion = []
    for msg in messages:
        speaker = "{{{{user_name}}}}" if msg.get('is_user') else "other"
        formatted_discussion.append(f"{msg['text']} ({speaker})")
    discussion_text = "\n".join(formatted_discussion)

    system_prompt = f"""You are a gentle grounding companion inside a calming app called Breathe.

STEP 1 — Evaluate SILENTLY if ALL are true:
1. The user is speaking (messages marked with '({{{{user_name}}}})' are present)
2. The user sounds mentally stuck, overwhelmed, or caught in repetitive thinking
3. A brief grounding interruption would likely help regulate the moment
4. The moment feels time-sensitive

If ANY are not met, respond with an empty string and nothing else.

STEP 2 — If ALL are met, produce ONE short grounding message.

Rules:
- Speak directly to {{{{user_name}}}} in a warm, human tone
- You can give advice, explanations, or interpretations
- You can ask for further reasoning on the user's source of anxiety
- You can assist the user on their problems, for example providing a new perspective or reasoning
- You can mention mental health, anxiety, or techniques
- You can Focus on breath, body sensations, or gentle reassurance
- Keep it 1-2 sentences max, under 50 characters.
- The message should feel optional, never urgent or commanding
- When appropriate, you can crack a joke to calm the situation 

Use this information within your response:
- Known user preferences or patterns: {{{{user_facts}}}}
- Recent background or ongoing context: {{{{user_context}}}}
- The overall conversational tone across time: {{{{user_chat}}}}

Current conversation:
{discussion_text}

Remember:
If interruption is not clearly helpful, respond with an empty string.
""".format(text=discussion_text)
    return {
        "notification": {
            "prompt": system_prompt,
            "params": ["user_name", "user_facts", "user_context", "user_chat"],
            "context": {
                "filters": {
                    "people": [],
                    "entities": [],
                }
            }
        }
    }

# ----------------------------
# FastAPI router endpoints
# ----------------------------
router = APIRouter()

@router.post('/notification/webhook')
async def webhook(request: WebhookRequest):
    session_id = request.session_id
    segments = request.segments
    current_time = time.time()
    buffer_data = message_buffer.get_buffer(session_id)

    for segment in segments:
        text = segment.get('text', '').strip()
        if not text:
            continue
        timestamp = segment.get('start', 0) or current_time
        is_user = segment.get('is_user', False)

        if buffer_data['silence_detected']:
            buffer_data['words_after_silence'] += len(text.split())
            if buffer_data['words_after_silence'] >= message_buffer.min_words_after_silence:
                buffer_data['silence_detected'] = False
                buffer_data['last_analysis_time'] = current_time
                logger.info(f"Silence ended for session {session_id}, starting fresh conversation")

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

    time_since_last_analysis = current_time - buffer_data['last_analysis_time']
    if (time_since_last_analysis >= ANALYSIS_INTERVAL and
        buffer_data['messages'] and
        not buffer_data['silence_detected']):
        sorted_messages = sorted(buffer_data['messages'], key=lambda x: x['timestamp'])
        notification = create_notification_prompt(sorted_messages)
        buffer_data['last_analysis_time'] = current_time
        buffer_data['messages'] = []
        logger.info(f"Sending notification for session {session_id}")
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

# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(title="Breathe Mentor Webhook API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

# Track start time
start_time = time.time()

# ----------------------------
# Run app via: uvicorn main:app --reload
# ----------------------------
