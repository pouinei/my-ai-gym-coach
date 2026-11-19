import io
import logging
from typing import Optional, Dict, List
from gtts import gTTS

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    High-performance Text-to-Speech engine using gTTS with:
    - In-memory memoization/caching for repeated coaching phrases
    - Audio bytes validation (> 100 bytes)
    - Exception handling and graceful degradation
    - Batch pre-caching for instant sub-millisecond coaching cues
    """

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

            if len(audio_bytes) > 100:
                self._cache[cache_key] = audio_bytes
                return audio_bytes
            else:
                logger.warning(f"Generated audio too small for text: {cleaned}")
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
