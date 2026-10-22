"""Обёртка над YAMNet (TFLite из MediaPipe): 521 класс AudioSet, вход 15600 отсчётов 16 кГц."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from .audio import YAMNET_WIN

# индексы классов AudioSet в yamnet_class_map.csv
CLASS_SPEECH = 0
CLASS_BREATHING = 36
CLASS_SNORING = 38
CLASS_SNORT = 41

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "yamnet" / "yamnet.tflite"


class Yamnet:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Нет модели {path}. Скачайте: python -m hrapometr_proto.fetch_model"
            )
        self._interpreter = Interpreter(model_path=str(path))
        self._interpreter.allocate_tensors()
        self._input_index = self._interpreter.get_input_details()[0]["index"]
        self._output_index = self._interpreter.get_output_details()[0]["index"]

    def scores(self, frame: np.ndarray) -> np.ndarray:
        """Оценки 521 класса (0..1) для одного кадра ровно в YAMNET_WIN отсчётов."""
        if frame.shape != (YAMNET_WIN,):
            raise ValueError(f"Кадр должен быть ({YAMNET_WIN},), получили {frame.shape}")
        self._interpreter.set_tensor(self._input_index, frame.astype(np.float32, copy=False))
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output_index)[0].copy()
