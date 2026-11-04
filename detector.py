from __future__ import annotations

import hashlib
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n.onnx"
MODEL_URL = "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx?download=true"
MODEL_SHA256 = "65158dad735be799c2466fa15e260c09558080bd530b42a8d0c3d1b419afd8b5"
INPUT_SIZE = 640
PERSON_CLASS_ID = 0
NMS_IOU_THRESHOLD = 0.45
MINIMUM_MODEL_SIZE = 1_000_000

ProgressCallback = Callable[[int, int], None]


class DetectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    elapsed: float

    @property
    def count(self) -> int:
        return len(self.detections)


def download_model(progress: ProgressCallback | None = None, destination: Path = MODEL_PATH) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".onnx.part")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "PeopleCount/1.0"})
    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            if progress:
                progress(downloaded, total)

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

        if digest.hexdigest() != MODEL_SHA256:
            raise DetectorError("The downloaded detector model failed its integrity check.")

        temp_path.replace(destination)
        return destination
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        if isinstance(exc, DetectorError):
            raise
        raise DetectorError(f"Could not download the detector model: {exc}") from exc


def draw_detections(image: np.ndarray, detections: tuple[Detection, ...]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    line_thickness = max(2, round(min(height, width) / 500))
    font_scale = max(0.5, min(1.0, width / 1600))
    text_thickness = max(1, round(font_scale * 2))

    for index, detection in enumerate(detections, start=1):
        x, y = detection.x, detection.y
        x2, y2 = x + detection.width, y + detection.height
        color = (94, 197, 34)
        cv2.rectangle(annotated, (x, y), (x2, y2), color, line_thickness)

        label = f"Person {index}  {detection.confidence:.0%}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        padding = max(5, round(8 * font_scale))
        label_top = y - text_height - (2 * padding)
        if label_top < 0:
            label_top = y + padding
        label_top = max(0, min(label_top, height - text_height - (2 * padding) - 1))
        label_left = max(0, min(x, width - text_width - (2 * padding) - 1))
        label_right = min(width - 1, label_left + text_width + (2 * padding))
        label_bottom = min(height - 1, label_top + text_height + (2 * padding))

        cv2.rectangle(annotated, (label_left, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            (label_left + padding, label_top + text_height + padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return annotated


def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, scale, float(pad_x), float(pad_y)


class PeopleDetector:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._net = None
        self._lock = threading.RLock()

    def is_model_available(self) -> bool:
        return self.model_path.is_file() and self.model_path.stat().st_size >= MINIMUM_MODEL_SIZE

    def ensure_model(self, progress: ProgressCallback | None = None) -> None:
        if not self.is_model_available():
            download_model(progress, self.model_path)

    def load(self):
        with self._lock:
            if self._net is None:
                if not self.is_model_available():
                    raise DetectorError("The detector model is missing.")
                try:
                    self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
                except cv2.error as exc:
                    raise DetectorError("The detector model could not be loaded.") from exc
            return self._net

    def detect(self, image: np.ndarray, confidence: float = 0.35) -> DetectionResult:
        if image is None or image.size == 0:
            raise DetectorError("The selected image is empty.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        started = time.perf_counter()
        canvas, scale, pad_x, pad_y = _letterbox(image)
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
        )

        net = self.load()
        with self._lock:
            net.setInput(blob)
            output = net.forward()

        predictions = np.asarray(output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            raise DetectorError("The detector returned an unexpected result.")
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        person_mask = (class_ids == PERSON_CLASS_ID) & (confidences >= confidence)

        image_height, image_width = image.shape[:2]
        boxes: list[list[float]] = []
        box_confidences: list[float] = []

        for row, score in zip(predictions[person_mask], confidences[person_mask]):
            center_x, center_y, box_width, box_height = row[:4]
            x1 = (center_x - box_width / 2.0 - pad_x) / scale
            y1 = (center_y - box_height / 2.0 - pad_y) / scale
            x2 = (center_x + box_width / 2.0 - pad_x) / scale
            y2 = (center_y + box_height / 2.0 - pad_y) / scale

            x1 = max(0.0, min(float(x1), image_width - 1.0))
            y1 = max(0.0, min(float(y1), image_height - 1.0))
            x2 = max(0.0, min(float(x2), image_width - 1.0))
            y2 = max(0.0, min(float(y2), image_height - 1.0))

            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            box_confidences.append(float(score))

        detections: list[Detection] = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, box_confidences, confidence, NMS_IOU_THRESHOLD
            )
            index_array = np.asarray(indices).reshape(-1) if indices is not None else np.array([])
            for index in index_array:
                box = boxes[int(index)]
                detections.append(
                    Detection(
                        x=round(box[0]),
                        y=round(box[1]),
                        width=round(box[2]),
                        height=round(box[3]),
                        confidence=box_confidences[int(index)],
                    )
                )

        detections.sort(key=lambda item: (item.x, item.y))
        return DetectionResult(
            detections=tuple(detections),
            elapsed=time.perf_counter() - started,
        )
