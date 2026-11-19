import io
import logging
from typing import Optional, Dict, List
from gtts import gTTS

logger = logging.getLogger(__name__)


def validate_audio_bytes(audio_bytes: Optional[bytes]) -> bool:
    """
    Robust audio validation for decodable MP3 format:
    - Handles None, empty bytes, non-bytes types, and truncated buffers safely (< 100 bytes).
    - Recognizes valid ID3 header (ID3v2 tags: 'ID3' at offset 0).
    - Recognizes valid MPEG audio frame sync (11 consecutive sync bits: 0xFF followed by 0b111xxxxx).
      Scans up to the first 1024 bytes (or buffer length) to tolerate metadata or container padding.
    - Rejects clearly invalid, corrupt, or non-audio data.
    - Never raises an exception; always returns a predictable boolean.
    - Avoids rejecting valid MP3 files that lack an ID3 header (via MPEG frame sync).
    """
    if not isinstance(audio_bytes, (bytes, bytearray)):
        return False
    if len(audio_bytes) < 100:
        return False

    try:
        # Check for ID3 header (ID3v2 standard tag)
        if audio_bytes.startswith(b"ID3"):
            return True

        # Check for MPEG frame synchronization (0xFF followed by 3 set bits 0xE0)
        # Search within the first 1024 bytes (or buffer length)
        limit = min(len(audio_bytes) - 1, 1024)
        for i in range(limit):
            if audio_bytes[i] == 0xFF and (audio_bytes[i + 1] & 0xE0) == 0xE0:
                return True

        return False
    except Exception as e:
        logger.warning(f"Audio validation failed with error: {e}")
        return False


class TextToSpeech:
    """
    High-performance Text-to-Speech engine using gTTS with:
    - In-memory memoization/caching for repeated coaching phrases
    - Audio bytes validation (MPEG frame sync / ID3 check & length verification)
    - Exception handling and graceful degradation
    - Batch pre-caching for instant sub-millisecond coaching cues
    """

    # Expose validate_audio_bytes on TextToSpeech class/instances pointing to canonical function
    validate_audio_bytes = staticmethod(validate_audio_bytes)

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._cache: Dict[str, bytes] = {}

    def speak(self, text: str, lang: Optional[str] = None) -> Optional[bytes]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        target_lang = lang or self.lang
        cache_key = f"{target_lang}:{cleaned.lower()}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            buffer = io.BytesIO()
            tts = gTTS(text=cleaned, lang=target_lang)
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()

            if validate_audio_bytes(audio_bytes):
                self._cache[cache_key] = audio_bytes
                return audio_bytes
            else:
                logger.warning(f"Generated audio failed validation for text: {cleaned}")
                return None
        except Exception as e:
            logger.error(f"TTS generation error for '{cleaned}': {e}")
            return None

    def precache_phrases(self, phrases: List[str]) -> int:
        """Pre-warms the cache with essential phrases."""
        count = 0
        for phrase in phrases:
            cleaned = (phrase or "").strip()
            if not cleaned:
                continue
            cache_key = f"{self.lang}:{cleaned.lower()}"
            if cache_key not in self._cache:
                data = self.speak(cleaned)
                if data:
                    count += 1
        return count

    def get_cached_count(self) -> int:
        return len(self._cache)
