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
    file: UploadFile = File(..., description="변환할 오디오 파일입니다. (지원 형식: mp3, wav, m4a 등)"),
    language: Optional[str] = Form("ko", description="언어 코드입니다 (예: 'en', 'ko'). 기본값은 'ko'입니다. None으로 설정하면 언어를 자동으로 감지합니다."),
    batch_size: int = Form(16, description="인퍼런스 배치 크기입니다. 값이 클수록 처리 속도는 빨라지지만 VRAM 사용량이 증가합니다."),
    beam_size: int = Form(5, description="빔 탐색(Beam Search)의 크기입니다. 빔 탐색은 여러 가능성을 동시에 고려하여 가장 확률이 높은 문장을 찾는 알고리즘입니다. 값이 클수록 정확도는 높아지지만 속도는 느려집니다. (기본값: 5)"),
    patience: float = Form(1.0, description="빔 탐색의 인내심 계수(Patience Factor)입니다. 1.0이면 최적의 결과를 찾았다고 판단되면 즉시 멈추고, 값이 클수록 더 오래 탐색합니다."),
    length_penalty: float = Form(1.0, description="생성된 문장의 길이에 대한 페널티입니다. 1.0보다 크면 긴 문장을 선호하고, 작으면 짧은 문장을 선호합니다."),
    temperature: float = Form(0.0, description="샘플링 온도(Temperature)입니다. 0.0은 가장 확률이 높은 단어만 선택(Greedy Decoding)하며, 값이 클수록 더 다양하고 창의적인(하지만 덜 정확할 수 있는) 결과를 생성합니다."),
    compression_ratio_threshold: float = Form(2.4, description="압축률 임계값입니다. 생성된 텍스트의 gzip 압축률이 이 값보다 높으면(너무 반복적인 텍스트 등), 디코딩 실패로 간주하고 다른 온도값으로 재시도합니다."),
    log_prob_threshold: float = Form(-1.0, description="평균 로그 확률 임계값입니다. 생성된 토큰들의 평균 확률이 이 값보다 낮으면(확신이 없으면), 디코딩 실패로 간주합니다."),
    no_speech_threshold: float = Form(0.6, description="묵음 감지 임계값입니다. <|nospeech|> 토큰의 확률이 이 값보다 높고, 평균 로그 확률이 `log_prob_threshold`보다 낮으면 해당 구간을 묵음으로 처리합니다."),
    condition_on_previous_text: bool = Form(False, description="이전 텍스트 문맥 사용 여부입니다. True로 설정하면 모델이 이전 윈도우의 텍스트를 프롬프트로 사용하여 문맥을 파악합니다. (환각 현상이 발생할 수 있어 기본값은 False입니다.)"),
    initial_prompt: Optional[str] = Form(None, description="초기 프롬프트입니다. 모델에게 문맥 정보나 스타일, 고유명사 등을 미리 알려주어 인식 정확도를 높일 수 있습니다."),
    suppress_tokens: str = Form("-1", description="생성을 억제할 토큰 ID 목록입니다(쉼표로 구분). '-1'은 기본 억제 토큰들을 사용합니다."),
    align: bool = Form(True, description="강제 정렬(Forced Alignment) 수행 여부입니다. True일 경우, 인식된 텍스트와 오디오를 정렬하여 정확한 단어 단위 타임스탬프를 생성합니다."),
    # VAD Parameters
    vad_onset: float = Form(0.500, description="VAD(음성 활동 감지) 시작 임계값입니다. 이 값보다 확률이 높아야 음성 구간으로 인식하기 시작합니다. (기본값: 0.500)"),
    vad_offset: float = Form(0.363, description="VAD 종료 임계값입니다. 음성 구간 중 확률이 이 값보다 낮아지면 묵음으로 간주하고 구간을 종료합니다. (기본값: 0.363)"),
    chunk_size: int = Form(30, description="VAD 처리를 위한 청크 크기(초 단위)입니다. (기본값: 30)")
):
    """
    Transcribe audio file using WhisperX with configurable parameters.
    
    - **file**: Audio file to transcribe
    - **language**: Language code (default: "ko")
    - **batch_size**: Batch size for inference (default: 16)
    - **beam_size**: Beam size for beam search (default: 5)
    - **align**: Whether to perform forced alignment (default: True)
    - **vad_onset**: VAD onset threshold (default: 0.500)
    - **vad_offset**: VAD offset threshold (default: 0.363)
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
        
        # Parse suppress_tokens
        suppress_tokens_list = [int(x) for x in suppress_tokens.split(",")] if suppress_tokens else [-1]

        # Use updated VAD parameters if needed (WhisperX pipeline manages VAD internally based on load options,
        # but we can override some via transcription options if supported, or we might need to reload VAD.
        # However, WhisperX pipeline structure fixes VAD parameters at initialization.
        # To support dynamic VAD params, we would need to manually run VAD here or modify the pipeline.
        # FasterWhisperPipeline in whisperX allows passing vad_params to transcribe() method indirectly?
        # Checking source: transcribe() calls merge_chunks using self._vad_params. 
        # But we can monkey-patch or pass them if the method allows.
        # Looking at FasterWhisperPipeline.transcribe source: 
        # It uses self._vad_params["vad_onset"] etc. 
        # We can temporarily update these parameters or pass them if transcribe supported it.
        # The transcribe method in asr.py does NOT take vad_onset/offset as arguments.
        # It uses self._vad_params.
        # So we will update the pipeline's vad_params temporarily.
        
        original_vad_params = model_pipeline._vad_params.copy()
        model_pipeline._vad_params["vad_onset"] = vad_onset
        model_pipeline._vad_params["vad_offset"] = vad_offset
        # chunk_size is used in merge_chunks
        
        # Prepare ASR options
        # Note: FasterWhisperPipeline.transcribe takes specific kwargs, but deeper configuration 
        # is stored in model_pipeline.options (TranscriptionOptions).
        # We need to update model_pipeline.options for beam_size etc.
        
        original_options = model_pipeline.options
        # Create a new options object or update existing (dataclass replacement is cleaner)
        from dataclasses import replace
        new_options = replace(
            original_options,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            temperatures=[temperature] if isinstance(temperature, float) else temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            suppress_tokens=suppress_tokens_list
        )
        model_pipeline.options = new_options

        try:
            # If language is not provided, it will be detected
            result = model_pipeline.transcribe(
                audio, 
                batch_size=batch_size, 
                language=language,
                chunk_size=chunk_size
            )
        finally:
            # Revert options to defaults (thread safety issue if concurrent requests, but this is a simple PoC)
            model_pipeline.options = original_options
            model_pipeline._vad_params = original_vad_params
        
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

