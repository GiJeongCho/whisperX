import whisperx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_whisperx")

logger.info(f"WhisperX version: {getattr(whisperx, '__version__', 'unknown')}")
logger.info(f"Attributes in whisperx: {dir(whisperx)}")

try:
    from whisperx.diarize import DiarizationPipeline
    logger.info("Found DiarizationPipeline in whisperx.diarize")
except ImportError as e:
    logger.error(f"Could not import DiarizationPipeline from whisperx.diarize: {e}")

try:
    model = whisperx.DiarizationPipeline
    logger.info("Found whisperx.DiarizationPipeline")
except AttributeError:
    logger.error("whisperx.DiarizationPipeline not found")
