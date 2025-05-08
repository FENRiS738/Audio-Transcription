from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
from pydub import AudioSegment
import tempfile
import io
import os

app = FastAPI()

whisper_model = whisper.load_model("base", device="cpu")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
        try:
            print(file.filename)
            file_bytes = await file.read()
            audio_buffer = io.BytesIO(file_bytes)

            try:
                audio = AudioSegment.from_file(audio_buffer)
            except Exception as decode_err:
                return JSONResponse(content={"error": f"Failed to decode audio: {decode_err}"}, status_code=400)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                transcription = whisper_model.transcribe(tmp.name)

            os.remove(tmp.name)

            return JSONResponse(content={
                "speaker": "Speaker_1",
                "text": transcription["text"].strip()
            })

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
