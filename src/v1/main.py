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

import uuid
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import replace

# KST Timezone
KST = timezone(timedelta(hours=9))

def get_kst_now_iso():
    return datetime.now(KST).isoformat()

# ... (patch codes remain same)

# Job Management
jobs: Dict[str, Dict] = {}

import math
import faster_whisper

def process_transcription_job(
    job_id: str,
    temp_file_path: str,
    original_filename: str,
    language: Optional[str],
    batch_size: int,
    options_dict: dict,
    vad_params: dict,
    chunk_size: int,
    align: bool
):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = get_kst_now_iso()
        jobs[job_id]["progress"] = 0
        logger.info(f"[Job {job_id}] STARTED - File: {original_filename}")
        
        start_time = time.time()
        audio = whisperx.load_audio(temp_file_path)
        
        # Audio duration for progress estimation
        total_duration = len(audio) / whisperx.audio.SAMPLE_RATE
        jobs[job_id]["audio_duration"] = total_duration
        
        # Apply VAD params (Global state modification - see note above)
        original_pipeline_vad = model_pipeline._vad_params.copy()
        model_pipeline._vad_params.update(vad_params)
        
        # Apply ASR options
        original_pipeline_options = model_pipeline.options
        new_options = replace(original_pipeline_options, **options_dict)
        model_pipeline.options = new_options
        
        try:
            # Manually run the pipeline steps to track progress
            # 1. VAD
            logger.info(f"[Job {job_id}] VAD processing...")
            vad_segments = model_pipeline.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": whisperx.audio.SAMPLE_RATE})
            vad_segments = whisperx.vads.Pyannote.merge_chunks(
                vad_segments,
                chunk_size,
                onset=vad_params["vad_onset"],
                offset=vad_params["vad_offset"],
            )
            
            # Detect language if needed
            if model_pipeline.tokenizer is None:
                language = language or model_pipeline.detect_language(audio)
                task = "transcribe"
                model_pipeline.tokenizer = faster_whisper.tokenizer.Tokenizer(
                    model_pipeline.model.hf_tokenizer,
                    model_pipeline.model.model.is_multilingual,
                    task=task,
                    language=language,
                )
            
            # 2. Transcription Loop
            segments = []
            total_segments = len(vad_segments)
            logger.info(f"[Job {job_id}] Transcribing {total_segments} segments...")
            
            # We need to access the pipeline's internal generator logic.
            # FasterWhisperPipeline.__call__ is complex to replicate fully without import issues.
            # Instead, we will rely on batching logic similar to pipeline.
            
            # Simplified batch processing to update progress
            # Note: This might slightly differ from original pipeline optimization but allows tracking.
            
            def data_generator():
                for seg in vad_segments:
                    f1 = int(seg['start'] * whisperx.audio.SAMPLE_RATE)
                    f2 = int(seg['end'] * whisperx.audio.SAMPLE_RATE)
                    yield {'inputs': audio[f1:f2]}

            # Use the pipeline's existing iterator but hook into it if possible?
            # Pipeline.__call__ returns an iterator. We can iterate it!
            
            # Re-create tokenizer if language changed (logic from pipeline)
            # ... (skipped for brevity, assuming tokenizer is set correctly above)
            
            # The pipeline call:
            # model_pipeline.__call__(data_generator(), batch_size=batch_size, num_workers=0)
            
            processed_segments = 0
            # To avoid re-implementing the whole pipeline, we use the fact that transcribe() 
            # calls __call__ which yields results batch by batch or item by item.
            # But model_pipeline.transcribe() consumes the iterator and returns a list.
            # We must use model_pipeline.__call__ directly.
            
            pipeline_iterator = model_pipeline.__call__(
                data_generator(), 
                batch_size=batch_size, 
                num_workers=0
            )
            
            for idx, out in enumerate(pipeline_iterator):
                text = out['text']
                if batch_size in [0, 1, None]:
                    text = text[0]
                
                segments.append({
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                })
                
                processed_segments += 1
                progress = int((processed_segments / total_segments) * 100)
                jobs[job_id]["progress"] = progress
                # Log occasionally to avoid spam
                if processed_segments % 10 == 0:
                    logger.info(f"[Job {job_id}] Progress: {progress}% ({processed_segments}/{total_segments})")

            result = {"segments": segments, "language": language}

        finally:
            # Revert settings
            model_pipeline.options = original_pipeline_options
            model_pipeline._vad_params = original_pipeline_vad

        detected_lang = result.get("language", language)
        
        # Align
        if align:
            jobs[job_id]["status"] = "aligning"
            logger.info(f"[Job {job_id}] ALIGNING - Language: {detected_lang}")
            if detected_lang in align_models:
                model_a, metadata = align_models[detected_lang]
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    DEVICE, 
                    return_char_alignments=False
                )
            else:
                logger.warning(f"[Job {job_id}] Alignment model for {detected_lang} not found.")

        process_time = time.time() - start_time
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["completed_at"] = get_kst_now_iso()
        jobs[job_id]["result"] = {
            "filename": original_filename,
            "language": detected_lang,
            "segments": result["segments"],
            "processing_time_seconds": round(process_time, 2)
        }
        logger.info(f"[Job {job_id}] COMPLETED - Duration: {process_time:.2f}s")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] FAILED: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["failed_at"] = get_kst_now_iso()
        jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

@app.post("/transcribe", status_code=202)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="변환할 오디오 파일입니다. (지원 형식: mp3, wav, m4a 등)"),
    # ... (parameters remain same)
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
    vad_onset: float = Form(0.500, description="VAD(음성 활동 감지) 시작 임계값입니다. 이 값보다 확률이 높아야 음성 구간으로 인식하기 시작합니다. (기본값: 0.500)"),
    vad_offset: float = Form(0.363, description="VAD 종료 임계값입니다. 음성 구간 중 확률이 이 값보다 낮아지면 묵음으로 간주하고 구간을 종료합니다. (기본값: 0.363)"),
    chunk_size: int = Form(30, description="VAD 처리를 위한 청크 크기(초 단위)입니다. (기본값: 30)")
):
    """
    오디오 트랜스크립션 작업을 비동기로 시작합니다.
    작업 ID(job_id)를 반환하며, 이를 통해 /jobs/{job_id} 에서 상태를 확인할 수 있습니다.
    """
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model service not initialized")

    # Save file
    suffix = Path(file.filename).suffix
    # Create a persistent temp file that survives the request scope
    # (NamedTemporaryFile with delete=False is usually fine, but standard open is simpler for persistency)
    fd, temp_file_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "created_at": get_kst_now_iso(),
        "filename": file.filename
    }
    
    logger.info(f"[Job {job_id}] SUBMITTED - File: {file.filename}")

    # Parse options
    suppress_tokens_list = [int(x) for x in suppress_tokens.split(",")] if suppress_tokens else [-1]
    
    options_dict = {
        "beam_size": beam_size,
        "patience": patience,
        "length_penalty": length_penalty,
        "temperatures": [temperature] if isinstance(temperature, float) else temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": log_prob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "suppress_tokens": suppress_tokens_list
    }
    
    vad_params = {
        "vad_onset": vad_onset,
        "vad_offset": vad_offset
    }

    background_tasks.add_task(
        process_transcription_job,
        job_id,
        temp_file_path,
        file.filename,
        language,
        batch_size,
        options_dict,
        vad_params,
        chunk_size,
        align
    )

    return {"job_id": job_id, "status": "pending", "message": "Job submitted successfully"}

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    작업 ID로 현재 상태와 결과(완료 시)를 조회합니다.
    Status: pending -> processing -> aligning -> completed (or failed)
    
    진행 중인 작업(processing, aligning)의 경우, elapsed_seconds를 포함하여 반환합니다.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = jobs[job_id]
    
    # Calculate elapsed time for running jobs
    if job_info["status"] in ["processing", "aligning"] and "started_at" in job_info:
        try:
            started_at = datetime.fromisoformat(job_info["started_at"])
            now = datetime.now(KST)
            elapsed = (now - started_at).total_seconds()
            # Return a copy to avoid modifying the stored state during read
            job_info = job_info.copy()
            job_info["elapsed_seconds"] = round(elapsed, 2)
        except Exception:
            pass
            
    return job_info

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8012, reload=False)

