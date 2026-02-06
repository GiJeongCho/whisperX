import os
import argparse
from pathlib import Path
import logging
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models(target_dir: str):
    """
    Downloads WhisperX models (Whisper, Alignment, VAD, Diarization) 
    to a local directory for fully offline on-premise usage.
    """
    # Create target directory
    models_dir = Path(target_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Target directory for models: {models_dir}")
    
    # HF Token (Required for Pyannote models)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Pyannote models (VAD, Diarization) might fail to download if not public.")

    # 1. Download Whisper Model (large-v3)
    whisper_dir = models_dir / "whisper"
    logger.info("Downloading Whisper model (large-v3)...")
    try:
        # Use snapshot_download for robust downloading
        snapshot_download(
            repo_id="systran/faster-whisper-large-v3",
            local_dir=whisper_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Whisper model downloaded to {whisper_dir}")
    except Exception as e:
        logger.error(f"Error downloading Whisper model: {e}")

    # 2. Download Alignment Models (Wav2Vec2)
    # Default alignment models used by whisperx
    alignment_repos = {
        "en": "facebook/wav2vec2-large-960h-lv60-self",
        "ko": "kresnik/wav2vec2-large-xlsr-korean" # WhisperX default for Korean
    }
    
    alignment_dir = models_dir / "alignment"
    for lang, repo_id in alignment_repos.items():
        logger.info(f"Downloading Alignment model for {lang} ({repo_id})...")
        try:
            lang_dir = alignment_dir / lang
            snapshot_download(
                repo_id=repo_id,
                local_dir=lang_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"Alignment model for {lang} downloaded.")
        except Exception as e:
             logger.error(f"Error downloading alignment model for {lang}: {e}")

    # 3. Download VAD Model (Pyannote Segmentation)
    vad_dir = models_dir / "vad"
    logger.info("Downloading VAD model (pyannote/segmentation-3.0)...")
    try:
        snapshot_download(
            repo_id="pyannote/segmentation-3.0",
            local_dir=vad_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        logger.info(f"VAD model downloaded to {vad_dir}")
    except Exception as e:
        logger.error(f"Error downloading VAD model: {e}")

    # 4. Download Diarization Model (Pyannote Speaker Diarization)
    diar_dir = models_dir / "diarization"
    logger.info("Downloading Diarization model (pyannote/speaker-diarization-3.1)...")
    try:
        snapshot_download(
            repo_id="pyannote/speaker-diarization-3.1",
            local_dir=diar_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        logger.info(f"Diarization model downloaded to {diar_dir}")
        
        # Also need embedding model for diarization usually
        emb_dir = models_dir / "embedding"
        logger.info("Downloading Speaker Embedding model (pyannote/wespeaker-voxceleb-resnet34-LM)...")
        snapshot_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            local_dir=emb_dir,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        logger.info(f"Embedding model downloaded to {emb_dir}")

    except Exception as e:
        logger.error(f"Error downloading Diarization models: {e}")

    logger.info("Download process finished.")
    logger.info(f"ALL MODELS SAVED IN: {models_dir}")
    logger.info("Please set WHISPER_MODEL_DIR environment variable to this path when running the server.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WhisperX models for offline use.")
    parser.add_argument("--output", type=str, default="src/resources/models", help="Directory to save models")
    args = parser.parse_args()
    
    download_models(args.output)
