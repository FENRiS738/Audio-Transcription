from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
from pydub import AudioSegment
import tempfile
import io
import os
import warnings
import logging
from logging.handlers import RotatingFileHandler
from pydub.utils import which

# ========== Setup Logging ==========
os.makedirs("logs", exist_ok=True)

file_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
)
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Capture both app and FastAPI logs
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_logger.addHandler(file_handler)

# ========== Suppress Whisper Warnings ==========
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ========== Configure FFmpeg ==========
AudioSegment.converter = which("ffmpeg") or "C:\\ffmpeg\\bin\\ffmpeg.exe"

# ========== Load Whisper Model ==========
WHISPER_MODEL_NAME = "base"
WHISPER_DEVICE = "cpu"
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)

# ========== Create FastAPI App ==========
app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        return JSONResponse(content={"error": "Uploaded file must be an audio file"}, status_code=400)

    try:
        app_logger.info(f"Received file: {file.filename}")
        
        # Handle large file uploads by reading in chunks
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_input_path = os.path.join(tmpdir, "input_audio")
            with open(tmp_input_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    f.write(chunk)

            # Load audio file and convert to WAV format
            audio = AudioSegment.from_file(tmp_input_path)
            tmp_output_path = os.path.join(tmpdir, "converted.wav")
            audio.export(tmp_output_path, format="wav")
            
            # Transcribe audio using Whisper model
            transcription = whisper_model.transcribe(tmp_output_path)

        # Return successful response
        app_logger.info(f"Transcription successful for file: {file.filename}")
        return JSONResponse(content={
            "speaker": "Speaker_1",
            "text": transcription["text"].strip()
        })

    except Exception as e:
        app_logger.exception("Unhandled transcription error")
        return JSONResponse(content={"error": str(e)}, status_code=500)
