import os
import gc
import logging
import torch
import whisperx
import yaml
import traceback
from typing import Dict, Optional, Any, Callable
from dataclasses import replace
import pandas as pd

ProgressCallback = Callable[[str, int], None]

# Configure Logging
logger = logging.getLogger(__name__)

class WhisperXService:
    def __init__(self):
        self.model_pipeline = None
        self.diarize_model = None
        self.diarization_error = None  # 에러 메시지 저장용
        self.align_models: Dict[str, tuple] = {}  # {lang: (model, metadata)}
        
        # Configuration
        self.model_dir = os.getenv("WHISPER_MODEL_DIR", "src/resources/models")
        self.whisper_arch = os.getenv("WHISPER_ARCH", "large-v3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.hf_token = os.getenv("HF_TOKEN") # Hugging Face Token for Pyannote

    def _log_directory_contents(self, path: str, name: str):
        """Helper to log contents of a model directory."""
        if os.path.exists(path):
            files = os.listdir(path)
            display_files = files[:5]
            remaining = len(files) - 5
            suffix = f"... and {remaining} more" if remaining > 0 else ""
            logger.info(f"[{name}] Directory found at {path}. Files: {display_files} {suffix}")
        else:
            logger.warning(f"[{name}] Directory NOT found at {path}")

    def load_models(self):
        """Load all necessary models (Whisper, Alignment, Diarization)."""
        logger.info(f"Loading WhisperX models on {self.device} ({self.compute_type})...")
        logger.info(f"HF_TOKEN detected: {'Yes' if self.hf_token else 'No'}")
        
        # 1. Load Whisper Model
        whisper_dir = os.path.join(self.model_dir, "whisper")
        self._log_directory_contents(whisper_dir, "Whisper")
        os.makedirs(whisper_dir, exist_ok=True)
        
        try:
            self.model_pipeline = whisperx.load_model(
                whisper_dir,
                device=self.device, 
                compute_type=self.compute_type, 
            )
            logger.info("Whisper model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

        # 2. Load Alignment Models (Pre-load en/ko)
        alignment_dir = os.path.join(self.model_dir, "alignment")
        self._log_directory_contents(alignment_dir, "Alignment")
        os.makedirs(alignment_dir, exist_ok=True)
        for lang in ["en", "ko"]:
            try:
                self._load_align_model(lang, alignment_dir)
            except Exception as e:
                logger.warning(f"Could not pre-load alignment for {lang}: {e}")

        # 3. Load Diarization Model (Offline Mode)
        diar_dir = os.path.join(self.model_dir, "diarization")
        vad_dir = os.path.join(self.model_dir, "vad")
        emb_dir = os.path.join(self.model_dir, "embedding")
        
        self._log_directory_contents(diar_dir, "Diarization")
        self._log_directory_contents(vad_dir, "VAD")
        self._log_directory_contents(emb_dir, "Embedding")
        
        logger.info("Loading Diarization model (Offline Mode)...")
        
        try:
            config_path = os.path.join(diar_dir, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Diarization config not found at {config_path}")
                
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            vad_model_path = os.path.abspath(os.path.join(vad_dir, "pytorch_model.bin"))
            emb_model_path = os.path.abspath(os.path.join(emb_dir, "pytorch_model.bin"))

            logger.info(f"Segmentation model: {vad_model_path} (exists: {os.path.exists(vad_model_path)})")
            logger.info(f"Embedding model: {emb_model_path} (exists: {os.path.exists(emb_model_path)})")

            from pyannote.audio.pipelines import SpeakerDiarization

            pipeline_params = config.get("pipeline", {}).get("params", {})

            self.diarize_model = SpeakerDiarization(
                segmentation=vad_model_path,
                embedding=emb_model_path,
                clustering=pipeline_params.get("clustering", "AgglomerativeClustering"),
                embedding_batch_size=pipeline_params.get("embedding_batch_size", 32),
                embedding_exclude_overlap=pipeline_params.get("embedding_exclude_overlap", True),
                segmentation_batch_size=pipeline_params.get("segmentation_batch_size", 32),
            )

            self.diarize_model.instantiate(config.get("params", {}))

            if self.device == "cuda":
                self.diarize_model.to(torch.device("cuda"))

            logger.info("Diarization model loaded successfully (Offline).")
            self.diarization_error = None

        except Exception as e:
            logger.error(f"Failed to load Diarization model: {e}")
            self.diarization_error = str(e)
            self.diarize_model = None

    def _load_align_model(self, language_code: str, model_dir: str):
        if language_code not in self.align_models:
            lang_dir = os.path.join(model_dir, language_code)
            model_name = lang_dir if os.path.isdir(lang_dir) else None

            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
                model_name=model_name,
                model_dir=model_dir,
            )
            self.align_models[language_code] = (model, metadata)
    
    def _calculate_step_weights(self, align: bool, diarize: bool):
        # 기본 가중치
        weights = {"transcribe": 0.0, "align": 0.0, "diarize": 0.0}
        
        if align and diarize:
            weights = {"transcribe": 60, "align": 10, "diarize": 30}
        elif align and not diarize:
            weights = {"transcribe": 80, "align": 20, "diarize": 0}
        elif not align and diarize:
            weights = {"transcribe": 70, "align": 0, "diarize": 30}
        else:
            weights = {"transcribe": 100, "align": 0, "diarize": 0}
            
        return weights

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        batch_size: int = 16,
        align: bool = True,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        vad_params: Optional[dict] = None,
        options_dict: Optional[dict] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        
        if not self.model_pipeline:
            raise RuntimeError("Models not initialized.")

        weights = self._calculate_step_weights(align, diarize)
        base_progress = 0

        def update_progress(step_name, current_percent):
            nonlocal base_progress
            # 현재 단계의 시작점 + (현재 단계 진행률 * 현재 단계 가중치 / 100)
            step_weight = weights[step_name]
            total_percent = base_progress + (current_percent * step_weight / 100)
            if on_progress:
                on_progress(step_name, int(total_percent))

        audio = whisperx.load_audio(audio_path)
        
        original_vad = self.model_pipeline._vad_params.copy()
        if vad_params:
            self.model_pipeline._vad_params.update(vad_params)
            
        original_options = self.model_pipeline.options
        if options_dict:
            new_options = replace(original_options, **options_dict)
            self.model_pipeline.options = new_options

        diarization_status = "disabled"

        try:
            # 1. Transcribe
            update_progress("transcribe", 0)
            
            result = self.model_pipeline.transcribe(
                audio, 
                batch_size=batch_size, 
                language=language,
            )
            base_progress += weights["transcribe"]
            update_progress("transcribe", 100)
            
            detected_lang = result["language"]
            
            # 2. Align
            if align:
                update_progress("align", 0)
                if detected_lang not in self.align_models:
                    try:
                        self._load_align_model(detected_lang, os.path.join(self.model_dir, "alignment"))
                    except Exception:
                        logger.warning(f"Alignment model for {detected_lang} unavailable.")
                
                if detected_lang in self.align_models:
                    model_a, metadata = self.align_models[detected_lang]
                    
                    result = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=False,
                    )
                base_progress += weights["align"]
                update_progress("align", 100)

            # 3. Diarize
            if diarize:
                update_progress("diarize", 0)
                if self.diarize_model:
                    try:
                        logger.info("Running Diarization...")
                        
                        waveform = torch.from_numpy(audio).unsqueeze(0)
                        if self.device == "cuda":
                            waveform = waveform.to("cuda")
                            
                        diarize_input = {"waveform": waveform, "sample_rate": 16000}
                        
                        diarize_segments = self.diarize_model(
                            diarize_input,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                        
                        diarize_df = pd.DataFrame(
                            diarize_segments.itertracks(yield_label=True), 
                            columns=['segment', 'label', 'speaker']
                        )
                        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
                        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
                        
                        result = whisperx.assign_word_speakers(diarize_df, result)
                        diarization_status = "success"
                    except Exception as e:
                        logger.error(f"Diarization runtime error: {e}")
                        logger.error(traceback.format_exc())
                        diarization_status = f"failed_runtime: {repr(e)}"
                else:
                    logger.warning("Diarization requested but model not available. Skipping.")
                    error_msg = self.diarization_error if self.diarization_error else "unknown_load_failure"
                    diarization_status = f"skipped_model_not_loaded: {error_msg}"
                
                base_progress += weights["diarize"]
                update_progress("diarize", 100)

            return {
                "result": result,
                "meta": {
                    "diarization_status": diarization_status
                }
            }

        finally:
            self.model_pipeline._vad_params = original_vad
            self.model_pipeline.options = original_options
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    def get_status(self):
        return {
            "device": self.device,
            "whisper_model": self.whisper_arch,
            "diarization_enabled": self.diarize_model is not None,
            "diarization_error": self.diarization_error,
            "alignment_langs": list(self.align_models.keys())
        }
