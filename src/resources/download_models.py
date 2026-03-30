import os
import argparse
from pathlib import Path
import logging
import urllib.request

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TORCHAUDIO_WAV2VEC2_URL = (
    "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
)

ALIGNMENT_MODELS_HF = {
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
}


def download_file(url: str, dest: Path):
    if dest.exists():
        logger.info(f"Already exists, skipping: {dest}")
        return
    logger.info(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Done: {dest}")


def download_models(target_dir: str):
    models_dir = Path(target_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory: {models_dir}")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Pyannote models might fail to download.")

    # 1. Whisper (faster-whisper large-v3)
    whisper_dir = models_dir / "whisper"
    logger.info("Downloading Whisper model (large-v3)...")
    try:
        snapshot_download(
            repo_id="systran/faster-whisper-large-v3",
            local_dir=whisper_dir,
            local_dir_use_symlinks=False,
        )
        logger.info(f"Whisper -> {whisper_dir}")
    except Exception as e:
        logger.error(f"Whisper download failed: {e}")

    # 2. Alignment — English (torchaudio wav2vec2)
    alignment_dir = models_dir / "alignment"
    alignment_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading English alignment model (torchaudio wav2vec2)...")
    try:
        download_file(
            TORCHAUDIO_WAV2VEC2_URL,
            alignment_dir / "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
        )
    except Exception as e:
        logger.error(f"English alignment download failed: {e}")

    # 2-1. Alignment — HuggingFace models (ko, etc.)
    for lang, repo_id in ALIGNMENT_MODELS_HF.items():
        logger.info(f"Downloading alignment model for {lang} ({repo_id})...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=alignment_dir / lang,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Alignment [{lang}] -> {alignment_dir / lang}")
        except Exception as e:
            logger.error(f"Alignment [{lang}] download failed: {e}")

    # 3. VAD (pyannote segmentation-3.0)
    vad_dir = models_dir / "vad"
    logger.info("Downloading VAD model (pyannote/segmentation-3.0)...")
    try:
        snapshot_download(
            repo_id="pyannote/segmentation-3.0",
            local_dir=vad_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        logger.info(f"VAD -> {vad_dir}")
    except Exception as e:
        logger.error(f"VAD download failed: {e}")

    # 4. Diarization (pyannote speaker-diarization-3.1)
    diar_dir = models_dir / "diarization"
    logger.info("Downloading Diarization model (pyannote/speaker-diarization-3.1)...")
    try:
        snapshot_download(
            repo_id="pyannote/speaker-diarization-3.1",
            local_dir=diar_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        logger.info(f"Diarization -> {diar_dir}")
    except Exception as e:
        logger.error(f"Diarization download failed: {e}")

    # 5. Speaker Embedding (wespeaker)
    emb_dir = models_dir / "embedding"
    logger.info("Downloading Embedding model (pyannote/wespeaker-voxceleb-resnet34-LM)...")
    try:
        snapshot_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            local_dir=emb_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        logger.info(f"Embedding -> {emb_dir}")
    except Exception as e:
        logger.error(f"Embedding download failed: {e}")

    logger.info("=" * 50)
    logger.info(f"ALL MODELS SAVED IN: {models_dir}")
    logger.info("Set WHISPER_MODEL_DIR to this path when running the server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WhisperX models for offline use.")
    parser.add_argument("--output", type=str, default="src/resources/models", help="Directory to save models")
    args = parser.parse_args()

    download_models(args.output)
