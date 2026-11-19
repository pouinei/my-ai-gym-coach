import io
import logging
from typing import Optional, Dict, List
from gtts import gTTS

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    High-performance Text-to-Speech engine using gTTS with:
    - In-memory memoization/caching for repeated coaching phrases
    - Audio bytes validation (MPEG frame sync / ID3 check & length verification)
    - Exception handling and graceful degradation
    - Batch pre-caching for instant sub-millisecond coaching cues
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._cache: Dict[str, bytes] = {}

    @staticmethod
    def validate_audio_bytes(audio_bytes: Optional[bytes]) -> bool:
        """
        Validates that audio bytes are decodable MP3 format:
        - Non-empty and minimum threshold (> 100 bytes)
        - Valid MPEG frame sync bits (0xFF and upper 3 bits of byte 1) or ID3 container tag
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return False
        has_id3 = audio_bytes.startswith(b"ID3")
        has_sync = (audio_bytes[0] == 0xFF) and ((audio_bytes[1] & 0xE0) == 0xE0)
        return has_id3 or has_sync

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

            if self.validate_audio_bytes(audio_bytes):
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
