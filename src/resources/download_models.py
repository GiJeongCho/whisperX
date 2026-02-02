import os
import torch
import whisperx
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models(target_dir: str):
    """
    Downloads WhisperX models to a local directory for on-premise usage.
    """
    # Create target directory
    models_dir = Path(target_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    whisper_dir = models_dir / "whisper"
    alignment_dir = models_dir / "alignment"
    
    logger.info(f"Downloading models to {models_dir}...")

    # 1. Download Whisper Model
    logger.info("Downloading Whisper model (large-v3)...")
    try:
        # load_model attempts to load the model after downloading.
        # Even if loading fails (e.g. due to torch.load restrictions),
        # we check if the file was downloaded.
        whisperx.load_model(
            "large-v3", 
            device="cpu", 
            compute_type="int8", 
            download_root=str(whisper_dir)
        )
        logger.info("Whisper model downloaded and loaded successfully.")
    except Exception as e:
        # Check if files exist
        if list(whisper_dir.glob("*")):
            logger.info(f"Whisper model download seems successful (found files in {whisper_dir}). Load error ignored: {e}")
        else:
            logger.error(f"Error downloading Whisper model: {e}")

    # 2. Download Alignment Models
    languages = ["en", "ko"]
    
    for lang in languages:
        logger.info(f"Downloading Alignment model for language: {lang}...")
        try:
            whisperx.load_align_model(
                language_code=lang, 
                device="cpu", 
                model_dir=str(alignment_dir)
            )
            logger.info(f"Alignment model for {lang} downloaded.")
        except Exception as e:
             if list(alignment_dir.glob(f"*{lang}*")) or list(alignment_dir.glob("wav2vec2*")):
                logger.info(f"Alignment model for {lang} seems successful. Load error ignored: {e}")
             else:
                logger.error(f"Error downloading alignment model for {lang}: {e}")

    # 3. VAD Model
    logger.info("Checking VAD model...")
    try:
        from whisperx.vads.pyannote import load_vad_model
        load_vad_model(device="cpu")
        logger.info("VAD model checked/downloaded.")
    except Exception as e:
        # VAD model is cached to TORCH_HOME usually.
        logger.warning(f"VAD model load failed (likely due to PyTorch security). This is expected during download phase if files are cached. Error: {e}")

    logger.info("Download process finished. Please verify ./models contains the necessary files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WhisperX models for offline use.")
    parser.add_argument("--output", type=str, default="./models", help="Directory to save models")
    args = parser.parse_args()
    
    download_models(args.output)
