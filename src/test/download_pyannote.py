"""
Pyannote 모델 오프라인 다운로드 스크립트

사전 조건:
  1. https://hf.co/pyannote/segmentation-3.0 사용 약관 동의
  2. https://hf.co/pyannote/speaker-diarization-3.1 사용 약관 동의
  3. https://hf.co/pyannote/speaker-diarization-community-1 사용 약관 동의 (PLDA 파일 포함)
  4. https://hf.co/settings/tokens 에서 토큰 발급

사용법:
  HF_TOKEN=hf_xxxx python src/test/download_pyannote.py
  HF_TOKEN=hf_xxxx python src/test/download_pyannote.py --output ./models
"""

import os
import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PYANNOTE_REPOS = {
    "vad": "pyannote/segmentation-3.0",
    "diarization": "pyannote/speaker-diarization-3.1",
    "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
}

COMMUNITY_REPO = "pyannote/speaker-diarization-community-1"
COMMUNITY_FILES = [
    "plda/xvec_transform.npz",
]


def download_pyannote_models(target_dir: str, token: str):
    models_dir = Path(target_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, repo_id in PYANNOTE_REPOS.items():
        out_dir = models_dir / name
        logger.info(f"[{name}] Downloading {repo_id} -> {out_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
                token=token,
            )
            logger.info(f"[{name}] Done.")
        except Exception as e:
            logger.error(f"[{name}] Failed: {e}")

    diar_dir = models_dir / "diarization"
    logger.info(f"[community] Downloading PLDA files from {COMMUNITY_REPO}...")
    for filename in COMMUNITY_FILES:
        try:
            hf_hub_download(
                repo_id=COMMUNITY_REPO,
                filename=filename,
                local_dir=str(diar_dir),
                local_dir_use_symlinks=False,
                token=token,
            )
            logger.info(f"  -> {filename} OK")
        except Exception as e:
            logger.error(f"  -> {filename} Failed: {e}")

    logger.info(f"Complete. Models saved in: {models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Pyannote models for offline diarization.")
    parser.add_argument("--output", type=str, default="models", help="Directory to save models (default: ./models)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN 환경변수가 설정되지 않았습니다.")
        logger.error("사용법: HF_TOKEN=hf_xxxx python src/test/download_pyannote.py")
        exit(1)

    download_pyannote_models(args.output, token)
