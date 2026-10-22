"""Скачивание YAMNet (TFLite) с проверкой хэша."""

from __future__ import annotations

import hashlib
import sys
import urllib.request

from .yamnet import DEFAULT_MODEL

URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
# хэш файла, проверенного 23.10.2026 (4 126 810 байт)
SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"


def main() -> int:
    if DEFAULT_MODEL.exists():
        print(f"Уже есть: {DEFAULT_MODEL}")
        return 0
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Качаю {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:  # noqa: S310 - адрес зашит выше
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"Хэш не совпал: {digest}. Файл не сохранён.")
        return 1
    DEFAULT_MODEL.write_bytes(data)
    print(f"Сохранено: {DEFAULT_MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
