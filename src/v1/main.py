import os
import torch

# Monkey patch torch.load to fix weights_only=True issue in PyTorch 2.6+
# This prevents initialization errors when loading checkpoints saved with older pickle logic
_original_load = torch.load
def _patched_load(*args, **kwargs):
    # Force weights_only=False regardless of what was passed
    # This is critical because some libraries might explicitly pass weights_only=True
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Attempt to register safe globals if patching is not enough
try:
    import typing
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, typing.Any])
except ImportError:
    pass
except Exception:
    pass

import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import torch
import uvicorn
import whisperx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "src/resources/models")
WHISPER_ARCH = os.getenv("WHISPER_ARCH", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Global State
model_pipeline = None
align_models: Dict[str, tuple] = {} # {lang: (model, metadata)}

def get_audio_duration(file_path: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        audio = whisperx.load_audio(file_path)
        return len(audio) / whisperx.audio.SAMPLE_RATE
    except Exception as e:
        logger.error(f"Error reading audio duration: {e}")
        return 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle model loading and unloading.
    Models are loaded once at startup and kept in memory.
    """
    global model_pipeline, align_models
    
    logger.info(f"Loading WhisperX models from {MODEL_DIR}...")
    logger.info(f"Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}")

    try:
        # 1. Load Whisper Model
        whisper_dir = os.path.join(MODEL_DIR, "whisper")
        if not os.path.exists(whisper_dir):
            logger.warning(f"Whisper model directory not found at {whisper_dir}. Attempting standard load or download.")
        
        logger.info(f"Loading Whisper model: {WHISPER_ARCH}...")
        model_pipeline = whisperx.load_model(
            WHISPER_ARCH, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE, 
            download_root=whisper_dir
        )
        logger.info("Whisper model loaded successfully.")

        # 2. Load Alignment Models (Pre-load common languages)
        # We pre-load 'en' and 'ko' as they are likely targets. 
        # Others can be loaded on demand if needed, but for "always loaded" req, we load here.
        alignment_dir = os.path.join(MODEL_DIR, "alignment")
        target_langs = ["en", "ko"]
        
        for lang in target_langs:
            logger.info(f"Loading Alignment model for: {lang}...")
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=lang, 
                    device=DEVICE, 
                    model_dir=alignment_dir
                )
                align_models[lang] = (align_model, align_metadata)
                logger.info(f"Alignment model for {lang} loaded.")
            except Exception as e:
                logger.error(f"Failed to load alignment model for {lang}: {e}")

    except Exception as e:
        logger.error(f"Critical error during startup model loading: {e}")
        raise e
    
    yield
    
    # Cleanup if necessary
    logger.info("Shutting down. Cleaning up resources...")
    if model_pipeline:
        del model_pipeline
    align_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="WhisperX On-Premise API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Health check endpoint to verify models are loaded."""
    status = {
        "status": "ok",
        "device": DEVICE,
        "whisper_model": WHISPER_ARCH,
        "loaded": model_pipeline is not None,
        "alignment_models": list(align_models.keys())
    }
    return status

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    batch_size: int = Form(BATCH_SIZE),
    align: bool = Form(True)
):
    """
    Transcribe audio file using WhisperX.
    Optionally performs forced alignment.
    """
    if not model_pipeline:
        logger.error("Model not initialized.")
        raise HTTPException(status_code=503, detail="Model service not initialized")

    temp_file_path = None
    try:
        # Save UploadFile to a temporary file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
        
        logger.info(f"Received file: {file.filename}, saved to {temp_file_path}")
        start_time = time.time()

        # 1. Transcribe
        logger.info("Starting transcription...")
        audio = whisperx.load_audio(temp_file_path)
        
        # If language is not provided, it will be detected
        result = model_pipeline.transcribe(audio, batch_size=batch_size, language=language)
        
        # Log detected language
        detected_lang = result.get("language", language)
        logger.info(f"Transcription complete. Language: {detected_lang}")

        # 2. Align (if requested)
        if align:
            if detected_lang in align_models:
                logger.info(f"Aligning with {detected_lang} model...")
                model_a, metadata = align_models[detected_lang]
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    DEVICE, 
                    return_char_alignments=False
                )
                logger.info("Alignment complete.")
            else:
                logger.warning(f"Alignment requested but model for '{detected_lang}' is not loaded. Skipping alignment.")
                # We could attempt to load it dynamically here, but keeping it simple for now.

        process_time = time.time() - start_time
        logger.info(f"Processing finished in {process_time:.2f}s")

        return {
            "filename": file.filename,
            "language": detected_lang,
            "segments": result["segments"],
            # "word_segments": result.get("word_segments", []), # Include if aligned
            "processing_time": process_time
        }

    except Exception as e:
        logger.error(f"Error processing transcription request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file {temp_file_path}: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8012, reload=False)

