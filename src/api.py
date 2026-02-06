import os
import logging
import shutil
import tempfile
import time
import uuid
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Import Service and Utils
from src.service import WhisperXService
from src.v1.utils.time_utils import get_kst_now_iso
from src.v1.utils.torch_utils import patch_torch_load

# Apply Patch
patch_torch_load()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global Service Instance
service = WhisperXService()
jobs: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    try:
        service.load_models()
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
    yield

app = FastAPI(title="WhisperX API with Diarization", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def background_process(
    job_id: str,
    temp_file_path: str,
    filename: str,
    language: Optional[str],
    batch_size: int,
    align: bool,
    diarize: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    vad_params: dict,
    options_dict: dict
):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = get_kst_now_iso()
        start_time = time.time()
        
        logger.info(f"[Job {job_id}] Processing {filename}...")
        
        result_data = service.transcribe_audio(
            audio_path=temp_file_path,
            language=language,
            batch_size=batch_size,
            align=align,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            vad_params=vad_params,
            options_dict=options_dict
        )
        
        result = result_data["result"]
        meta = result_data.get("meta", {})
        
        process_time = time.time() - start_time
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = get_kst_now_iso()
        jobs[job_id]["result"] = {
            "segments": result["segments"],
            "language": result.get("language"),
            "processing_time": round(process_time, 2),
            "meta": meta
        }
        logger.info(f"[Job {job_id}] Completed in {process_time:.2f}s")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Failed: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

@app.get("/health")
def health():
    return {"status": "ok", "models": service.get_status()}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/transcribe", status_code=202)
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="변환할 오디오 파일입니다."),
    language: Optional[str] = Form("ko", description="언어 코드 (기본값: 'ko')"),
    batch_size: int = Form(16, description="배치 크기"),
    beam_size: int = Form(5, description="빔 탐색 크기"),
    patience: float = Form(1.0, description="빔 탐색 인내심 계수"),
    length_penalty: float = Form(1.0, description="길이 페널티"),
    temperature: float = Form(0.0, description="샘플링 온도"),
    compression_ratio_threshold: float = Form(2.4, description="압축률 임계값"),
    log_prob_threshold: float = Form(-1.0, description="평균 로그 확률 임계값"),
    no_speech_threshold: float = Form(0.6, description="묵음 감지 임계값"),
    condition_on_previous_text: bool = Form(False, description="이전 텍스트 문맥 사용 여부"),
    initial_prompt: Optional[str] = Form(None, description="초기 프롬프트"),
    suppress_tokens: str = Form("-1", description="생성 억제 토큰 ID 목록 (쉼표 구분)"),
    align: bool = Form(True, description="강제 정렬 수행 여부"),
    diarize: bool = Form(True, description="화자 분리(Speaker Diarization) 수행 여부"),
    min_speakers: Optional[int] = Form(None, description="최소 화자 수 (Diarization 힌트)"),
    max_speakers: Optional[int] = Form(None, description="최대 화자 수 (Diarization 힌트)"),
    vad_onset: float = Form(0.500, description="VAD 시작 임계값"),
    vad_offset: float = Form(0.363, description="VAD 종료 임계값")
):
    if not service.model_pipeline:
         raise HTTPException(status_code=503, detail="Models not loaded")

    suffix = Path(file.filename).suffix
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending", 
        "created_at": get_kst_now_iso(), 
        "filename": file.filename
    }
    
    suppress_tokens_list = [int(x) for x in suppress_tokens.split(",")] if suppress_tokens else [-1]
    vad_params = {"vad_onset": vad_onset, "vad_offset": vad_offset}
    
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

    # Handle 0 values for speaker counts (treat as None)
    if min_speakers == 0:
        min_speakers = None
    if max_speakers == 0:
        max_speakers = None

    background_tasks.add_task(
        background_process,
        job_id,
        temp_path,
        file.filename,
        language,
        batch_size,
        align,
        diarize,
        min_speakers,
        max_speakers,
        vad_params,
        options_dict
    )

    return {"job_id": job_id, "message": "Job submitted"}
