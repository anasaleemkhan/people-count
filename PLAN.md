# Real-Time People Counting with Webcam and Negative-Count CSV Logging

## Current codebase fit

`people-count` is a Tkinter desktop app that currently counts people in **static images**. It already has:

- `detector.py:164` - `PeopleDetector.detect()` using YOLOv8n ONNX through OpenCV DNN, person class `0`.
- `detector.py:84` - `draw_detections()` for annotated frames.
- `app.py:532` - an event-queue pattern that safely moves worker results onto the Tk main thread.
- `app.py:67` - `ImagePreview.set_image()` for displaying `numpy` frames.

The plan below reuses those pieces and adds webcam, tracking, discrepancy counting, and CSV export. **No new detection framework is required.**

Verified locally: the bundled ONNX model has a **fixed 640x640 input** and takes roughly **0.5 s/frame on CPU** after warm-up. The design should therefore run detection slower than the camera and use tracking between detections.

---

## Recommended module layout

```text
people-count/
|-- app.py            # extend: Live Camera mode, start/stop, export button
|-- camera.py         # new: webcam capture thread
|-- tracker.py        # new: IoU multi-object tracker
|-- counting.py       # new: occupancy + negative-count logic
|-- export.py         # new: streaming CSV writer + filtered export
|-- detector.py       # existing: reuse detect() / draw_detections()
|-- main.py           # mostly unchanged
`-- logs/             # generated session logs
```

Recommended dependencies:

- **OpenCV** - camera capture, preprocessing, inference, drawing. Already installed.
- **NumPy** - frame/array handling. Already installed transitively.
- **Pillow** - Tk display. Already installed.
- **csv / datetime** - real-time streaming log. Standard library.
- **Pandas** - optional; use only for filtering/export and session summaries.
- **ONNX Runtime DirectML / OpenVINO / Ultralytics** - optional, only if the measured 2 FPS detection rate is insufficient.

---

# Step 1 - Webcam feed and permissions

## Underlying logic

The camera must run in its own thread because `cv2.VideoCapture.read()` blocks. The capture thread should:

1. Open the camera with a Windows-friendly backend.
2. Configure resolution/FPS.
3. Warm up the sensor, since many webcams return black or empty frames initially.
4. Publish frames into a **bounded, latest-frame queue**. If the detection worker is slower than the camera, discard stale frames rather than building latency.
5. Timestamp every frame at capture time.
6. Release the camera cleanly when the session stops.

On Windows, camera access can fail because of OS privacy settings or because another application owns the camera. Show an actionable error rather than a generic "camera unavailable".

**Recommended backend order on Windows:** `CAP_DSHOW` first, then `CAP_MSMF`, then `CAP_ANY`.

## `camera.py`

```python
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    index: int
    image: np.ndarray          # BGR
    captured_at: float         # time.time()
    captured_monotonic: float


class CameraStream:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        queue_size: int = 2,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.frames: queue.Queue[Frame] = queue.Queue(maxsize=queue_size)
        self.error: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _open(self) -> cv2.VideoCapture:
        if sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()

        raise CameraError(
            "Could not open the camera. On Windows, open "
            "Settings > Privacy & security > Camera, enable camera access, "
            "and allow desktop apps. Also close other apps using the camera."
        )

    @staticmethod
    def open_privacy_settings() -> None:
        if sys.platform == "win32":
            os.startfile("ms-settings:privacy-webcam")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._cap = self._open()

        # Warm-up: discard initial frames from slow-starting webcams.
        for _ in range(10):
            self._cap.read()

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="camera-capture", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        assert self._cap is not None
        started = time.monotonic()
        index = 0

        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                self.error = "The camera stopped returning frames."
                break

            frame = Frame(
                index=index,
                image=image,
                captured_at=time.time(),
                captured_monotonic=time.monotonic() - started,
            )
            index += 1

            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                # Keep only the newest frame to avoid latency buildup.
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    pass

    def read_latest(self, timeout: float = 0.2) -> Frame | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
```

## Permission handling

- **Windows 10/11:** Settings > Privacy & security > Camera. Enable "Camera access" and "Let desktop apps access your camera".
- If opening fails, display the `CameraError` message and optionally a button calling `CameraStream.open_privacy_settings()`.
- If the camera opens but immediately returns empty frames, distinguish that from permission failure. Try a different index or backend.
- On macOS, packaged apps need `NSCameraUsageDescription`; on Linux, the user usually needs membership in the `video` group.
- Always call `stream.stop()` from the Tk `WM_DELETE_WINDOW` handler.

---

# Step 2 - Real-time object detection

## Underlying logic

The existing `PeopleDetector` is already suitable:

- YOLOv8n ONNX.
- OpenCV DNN preprocessing and inference.
- Person class filtering.
- NMS already implemented.
- Thread-safe `load()`/`detect()`.

The main problem is throughput. On this machine, 640x640 inference is about 0.5 seconds. That is acceptable only if tracking bridges the gaps, because people move relatively little between detection ticks.

### Detection strategy

1. Capture at 30 FPS.
2. Detection worker always takes the **newest** available frame.
3. Run detection at its natural speed, or rate-limit it to a target such as 5-10 FPS.
4. Feed detections into the tracker every detection tick.
5. Display the latest annotated frame and count. Do not try to display 30 FPS from the detection worker.

### Performance options, in order

1. **Use tracking between detections** - no model change, lowest effort.
2. **Re-export YOLOv8n at 320x320 or 416x416.** The current ONNX is fixed at 640; passing 416 directly will fail. Example export command:

   ```python
   from ultralytics import YOLO

   model = YOLO("yolov8n.pt")
   model.export(format="onnx", imgsz=416, simplify=True, dynamic=False)
   ```

   Then change `INPUT_SIZE` in `detector.py`.

3. **Switch runtime to ONNX Runtime DirectML** on Windows for GPU acceleration.
4. **Use OpenVINO** on Intel hardware.
5. **Use a region of interest / smaller camera resolution** if the counting area is fixed.
6. **Quantize to INT8** if accuracy permits.

Do not add Ultralytics just for inference; the existing OpenCV DNN path is lighter.

## Detection worker

```python
def detection_worker(
    camera: CameraStream,
    detector: PeopleDetector,
    tracker: IouTracker,
    counter: DiscrepancyCounter,
    logger: CountLogWriter,
    ui_events: queue.Queue,
    stop_event: threading.Event,
    confidence: float = 0.35,
    min_interval: float = 0.1,       # cap detection to ~10 FPS
) -> None:
    last_detection = 0.0
    frame_times: list[float] = []

    while not stop_event.is_set():
        now = time.monotonic()
        if now - last_detection < min_interval:
            time.sleep(0.005)
            continue

        frame = camera.read_latest(timeout=0.2)
        if frame is None:
            continue

        last_detection = time.monotonic()
        started = time.perf_counter()

        result = detector.detect(frame.image, confidence)
        boxes = [(d.x, d.y, d.width, d.height) for d in result.detections]
        update = tracker.update(boxes)

        inference_ms = (time.perf_counter() - started) * 1000.0
        frame_times.append(frame.captured_monotonic)
        fps = 0.0
        if len(frame_times) > 1:
            elapsed = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0
        if len(frame_times) > 60:
            frame_times.pop(0)

        snapshot = counter.update(
            frame_index=frame.index,
            captured_at=frame.captured_at,
            raw_count=result.count,
            tracker_update=update,
            inference_ms=inference_ms,
            fps=fps,
        )

        annotated = draw_detections(frame.image, result.detections)
        annotated = draw_count_overlay(annotated, snapshot)

        ui_events.put(("live_frame", annotated, snapshot))
        logger.write(snapshot, confidence=confidence)

    ui_events.put(("live_stopped",))
```

`draw_count_overlay()` can draw a compact HUD:

```python
def draw_count_overlay(image, snapshot) -> np.ndarray:
    out = image.copy()
    color = (94, 197, 34)
    if snapshot.negative_count < 0:
        color = (60, 60, 240)

    cv2.rectangle(out, (12, 12), (360, 100), (11, 17, 32), -1)
    cv2.putText(
        out,
        f"Count: {snapshot.display_count}",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Delta: {snapshot.negative_count}   FPS: {snapshot.fps:.1f}",
        (28, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 210, 225),
        1,
        cv2.LINE_AA,
    )
    return out
```

---

# Step 3 - Counting logic and negative-count definition

## Why tracking is required

Raw detector counts flicker:

- A person is missed for one frame, so the count drops.
- A detection is duplicated, so the count rises.
- Occlusion temporarily removes a person.
- Boxes shift and are detected inconsistently.

An IoU tracker assigns stable IDs and lifecycle events:

- **Tentative track:** detected once, not yet counted.
- **Confirmed track:** detected at least `min_hits` times. This is part of the live count.
- **Retired track:** unmatched for more than `max_age` frames. This is an exit.
- **Entry:** a track becomes confirmed for the first time.

A simple greedy IoU tracker is enough for webcam people counting and avoids adding heavy tracking dependencies. ByteTrack/DeepSORT can be added later if ID switches become a problem.

## Tracker implementation

```python
from dataclasses import dataclass

Box = tuple[float, float, float, float]   # x, y, width, height


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


@dataclass(frozen=True)
class TrackerUpdate:
    tracks: tuple[Track, ...]
    entries: int
    exits: int
    unmatched_confirmed: int


class IouTracker:
    def __init__(
        self,
        iou_threshold: float = 0.30,
        min_hits: int = 3,
        max_age: int = 15,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[Box]) -> TrackerUpdate:
        used = [False] * len(boxes)
        new_entries = 0
        new_exits = 0
        unmatched_confirmed = 0

        for track in self._tracks:
            best_index = -1
            best_score = self.iou_threshold

            for index, box in enumerate(boxes):
                if used[index]:
                    continue
                score = iou(track.box, box)
                if score >= best_score:
                    best_index = index
                    best_score = score

            if best_index < 0:
                track.misses += 1
                if track.confirmed:
                    unmatched_confirmed += 1
                continue

            used[best_index] = True
            track.box = boxes[best_index]
            track.hits += 1
            track.misses = 0

            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True
                new_entries += 1

        for index, box in enumerate(boxes):
            if not used[index]:
                self._tracks.append(Track(self._next_id, box))
                self._next_id += 1

        survivors: list[Track] = []
        for track in self._tracks:
            if track.misses <= self.max_age:
                survivors.append(track)
            elif track.confirmed:
                new_exits += 1

        self._tracks = survivors
        confirmed = tuple(t for t in self._tracks if t.confirmed)

        return TrackerUpdate(
            tracks=confirmed,
            entries=new_entries,
            exits=new_exits,
            unmatched_confirmed=unmatched_confirmed,
        )
```

## Define "negative count" explicitly

There are two reasonable meanings. Pick one as the primary mode and store it in the CSV so there is no ambiguity.

**Recommended mode: detection discrepancy**

For each detection tick:

- `R_t` = raw person detections after NMS.
- `T_t` = confirmed, currently active tracks.
- `delta_t = R_t - T_t`.
- `negative_count_t = min(0, delta_t)`.

A negative value means: **the detector saw fewer people than the tracker believes are currently present.** It indicates occlusion, missed detection, a fast exit, or a model failure.

To avoid noisy one-frame blips:

- Require `negative_count <= -1` for `min_consecutive = 2` detection ticks.
- Record the event with a reason:
  - `occlusion_or_exit` - the missing quantity is covered by unmatched confirmed tracks.
  - `unexplained_drop` - more detections disappeared than the tracker can explain.

The displayed live count is **not** allowed to go negative. It is a stable occupancy count. "Negative" is a signed discrepancy stored in the log.

**Optional mode: signed directional flow**

If you add line-crossing logic later:

- `net_flow = entries - exits`.
- A negative `net_flow` means an exit occurred without a matching entry. This is common when the session starts with people already in frame, or after an ID switch.

Store it as a separate column and export it only if the user selects "include negative net flow".

## Counter implementation

```python
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class CountSnapshot:
    timestamp_iso: str
    frame_index: int
    observed_count: int       # raw detections
    tracked_count: int        # confirmed active tracks
    display_count: int        # smoothed count for the UI
    expected_count: int       # equals tracked_count in this model
    delta: int
    negative_count: int       # signed; <= 0
    entries_total: int
    exits_total: int
    net_flow: int
    active_track_ids: tuple[int, ...]
    reason: str
    negative_event: bool
    inference_ms: float
    fps: float


class DiscrepancyCounter:
    def __init__(
        self,
        smoothing_window: int = 5,
        min_consecutive: int = 2,
    ) -> None:
        self._display_history: deque[int] = deque(maxlen=smoothing_window)
        self._entries_total = 0
        self._exits_total = 0
        self._negative_streak = 0
        self.min_consecutive = min_consecutive

    def update(
        self,
        *,
        frame_index: int,
        captured_at: float,
        raw_count: int,
        tracker_update: TrackerUpdate,
        inference_ms: float,
        fps: float,
    ) -> CountSnapshot:
        self._entries_total += tracker_update.entries
        self._exits_total += tracker_update.exits

        tracked_count = len(tracker_update.tracks)
        self._display_history.append(tracked_count)
        display_count = int(statistics.median(self._display_history))

        expected_count = tracked_count
        delta = raw_count - expected_count
        negative_count = min(0, delta)

        if negative_count < 0:
            self._negative_streak += 1
            if tracker_update.unmatched_confirmed >= -negative_count:
                reason = "occlusion_or_exit"
            else:
                reason = "unexplained_drop"
        else:
            self._negative_streak = 0
            reason = ""

        timestamp = datetime.fromtimestamp(
            captured_at, tz=timezone.utc
        ).isoformat(timespec="milliseconds")

        return CountSnapshot(
            timestamp_iso=timestamp,
            frame_index=frame_index,
            observed_count=raw_count,
            tracked_count=tracked_count,
            display_count=display_count,
            expected_count=expected_count,
            delta=delta,
            negative_count=negative_count,
            entries_total=self._entries_total,
            exits_total=self._exits_total,
            net_flow=self._entries_total - self._exits_total,
            active_track_ids=tuple(t.track_id for t in tracker_update.tracks),
            reason=reason,
            negative_event=self._negative_streak >= self.min_consecutive,
            inference_ms=inference_ms,
            fps=fps,
        )
```

---

# Step 4 - CSV export pipeline

## Design

Use the standard `csv` module for **streaming writes** because it is fast, dependency-free, and safe for continuous append. Use **Pandas only for export/filtering**, where it is convenient.

Log every detection tick with a `negative_count` and `negative_event` column. This preserves context around anomalies. The export function then produces a smaller file containing only the requested discrepancy rows.

Recommended file names:

```text
logs/session_20261106_143012_counts.csv
exports/negative_counts_20261106_143012.csv
```

CSV schema:

```text
timestamp_iso, timestamp_unix_ms, session_id, frame_index, camera_index,
observed_count, tracked_count, display_count, expected_count,
delta, negative_count, negative_event, entries_total, exits_total,
net_flow, active_track_ids, reason, confidence, inference_ms, fps, mode
```

Write `timestamp_iso` in UTC with millisecond precision. Also store epoch milliseconds for sorting and interoperability.

## CSV writer

```python
import csv
import queue
import threading
import time
from pathlib import Path


CSV_FIELDS = [
    "timestamp_iso",
    "timestamp_unix_ms",
    "session_id",
    "frame_index",
    "camera_index",
    "observed_count",
    "tracked_count",
    "display_count",
    "expected_count",
    "delta",
    "negative_count",
    "negative_event",
    "entries_total",
    "exits_total",
    "net_flow",
    "active_track_ids",
    "reason",
    "confidence",
    "inference_ms",
    "fps",
    "mode",
]


class CountLogWriter:
    def __init__(
        self,
        path: Path,
        *,
        session_id: str,
        camera_index: int,
        mode: str = "discrepancy",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.camera_index = camera_index
        self.mode = mode
        self._queue: queue.Queue[dict | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="csv-writer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            handle.flush()

            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)
                handle.flush()

    def write(self, snapshot: CountSnapshot, *, confidence: float) -> None:
        self._queue.put(
            {
                "timestamp_iso": snapshot.timestamp_iso,
                "timestamp_unix_ms": int(time.time() * 1000),
                "session_id": self.session_id,
                "frame_index": snapshot.frame_index,
                "camera_index": self.camera_index,
                "observed_count": snapshot.observed_count,
                "tracked_count": snapshot.tracked_count,
                "display_count": snapshot.display_count,
                "expected_count": snapshot.expected_count,
                "delta": snapshot.delta,
                "negative_count": snapshot.negative_count,
                "negative_event": snapshot.negative_event,
                "entries_total": snapshot.entries_total,
                "exits_total": snapshot.exits_total,
                "net_flow": snapshot.net_flow,
                "active_track_ids": "|".join(map(str, snapshot.active_track_ids)),
                "reason": snapshot.reason,
                "confidence": f"{confidence:.2f}",
                "inference_ms": f"{snapshot.inference_ms:.1f}",
                "fps": f"{snapshot.fps:.1f}",
                "mode": self.mode,
            }
        )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
```

The writer thread prevents disk I/O from blocking the detection loop. `flush()` makes the file immediately readable by an export operation. For a stronger durability guarantee on critical logs, call `os.fsync()` after each negative event, not after every row.

## Filtered export with Pandas

```python
import pandas as pd
from pathlib import Path


def export_negative_counts(
    source_csv: Path,
    destination_csv: Path,
    *,
    include_net_flow: bool = False,
) -> int:
    frame = pd.read_csv(source_csv)

    mask = frame["negative_event"].astype(bool)
    if include_net_flow:
        mask = mask | (frame["net_flow"] < 0)

    events = frame.loc[mask].sort_values("timestamp_iso").copy()
    events.to_csv(destination_csv, index=False, encoding="utf-8")
    return len(events)
```

If the user wants **every raw negative row**, use:

```python
mask = frame["negative_count"] < 0
```

## Export behavior

- Export from `app.py` via `filedialog.asksaveasfilename()`.
- Before reading a live log, either:
  - stop the session and close the writer, or
  - flush the writer and export a snapshot copy.
- Recommended UX: **stop live mode before exporting**, then offer "Export Negative Counts".
- Also expose "Open Logs Folder" so users can retrieve the unfiltered session CSV.
- Include the `reason`, `negative_count`, and `active_track_ids` columns; these make the discrepancy logs actionable rather than just a count.

---

# End-to-end data flow

```text
Webcam
  |  BGR frames, 30 FPS
  v
CameraStream thread
  |  Frame(index, captured_at, image)
  |  bounded latest-frame queue; stale frames dropped
  v
Detection worker thread
  |  newest frame only
  v
PeopleDetector.detect(frame, confidence)
  |  letterbox to 640x640 -> ONNX -> NMS -> person class 0
  v
IouTracker.update(boxes)
  |  track IDs, tentative/confirmed/retired lifecycle
  |  entries, exits, unmatched_confirmed
  v
DiscrepancyCounter.update(...)
  |  raw_count vs tracked_count
  |  negative_count = min(0, raw_count - tracked_count)
  |  debounced negative_event with reason
  |-------------------------------|
  |                               |
  v                               v
draw_detections + HUD        CountLogWriter queue
  |                               | writer thread
  v                               v
Tk event queue                logs/session_<id>_counts.csv
  |                               |
  v                               v
ImagePreview + labels         pandas filter/export
(display count, delta, FPS)   exports/negative_counts_<id>.csv
```

Timestamps should travel with the frame from capture through detection and into the CSV. Do not timestamp only at write time, or the log will reflect processing delays rather than video time.

---

# Integrating into the existing Tkinter app

Recommended changes to `app.py`:

1. Add a **"Live Camera"** mode alongside the current image mode.
2. Add **Start Camera** / **Stop Camera** buttons and a camera-index selector.
3. Reuse `ImagePreview` for live frames.
4. Poll a UI event queue from the Tk main thread, as the existing app does at `app.py:532`.
5. Add **Export Negative Counts** and **Open Logs Folder** buttons.
6. Disable image loading/counting controls while live mode is active.
7. Show:
   - large live count
   - signed delta in red when negative
   - FPS and inference time
   - session negative-event count
8. Override `WM_DELETE_WINDOW` to stop the camera, join the detection worker, close the CSV writer, and release the camera.

Threading rules:

- All `cv2.VideoCapture` calls happen on the capture thread.
- All `PeopleDetector.detect()` calls happen on the detection worker.
- All Tk widget updates happen on the main thread.
- CSV writing happens on the writer thread.
- Never call `cv2.imshow()` inside a Tk application; use `ImagePreview`.
- Downscale annotated frames before putting them in the UI queue. Creating a `PhotoImage` from a full 1080p frame every update is expensive.

Suggested teardown order:

```python
stop_event.set()
camera.stop()
detection_thread.join(timeout=5)
log_writer.close()
root.destroy()
```

---

# Testing and acceptance

## Unit tests

- `tests/test_tracker.py`
  - A track becomes confirmed after `min_hits`.
  - Entry count increments exactly once.
  - A confirmed track retires after `max_age` missed frames and increments exits.
  - Tentative tracks that disappear do not count as exits.
  - Greedy IoU association preserves IDs for moving boxes.

- `tests/test_counting.py`
  - `raw_count=3`, `tracked_count=5` -> `negative_count=-2`.
  - Debounce suppresses a single-frame negative blip.
  - Two consecutive negative ticks produce `negative_event=True`.
  - Deficit covered by unmatched tracks -> `occlusion_or_exit`.
  - Larger deficit -> `unexplained_drop`.

- `tests/test_export.py`
  - CSV has the exact header and UTF-8 encoding.
  - Timestamps are valid ISO 8601 UTC.
  - Export keeps only negative events.
  - Export works when there are zero events.
  - No partial line is produced after `close()`.

## Manual verification

- Camera permission denied -> actionable error, no crash.
- Camera unplugged mid-session -> capture thread exits, UI shows a stopped state.
- Start/stop repeatedly -> no leaked threads, camera can be reopened.
- One person leaves the frame -> count eventually drops, no permanent false negative event.
- One person briefly occluded -> discrepancy logged with `occlusion_or_exit`.
- Two people disappear while only one track is unmatched -> `unexplained_drop`.
- Export while live -> either blocked or creates a consistent snapshot.
- Close the app -> CSV writer flushes, camera releases, process exits cleanly.

## Acceptance criteria

- Live preview and count update continuously.
- Count does not flicker excessively because of the median smoothing.
- Negative discrepancies are recorded with timestamps and reasons.
- CSV export opens correctly in Excel/Pandas.
- No UI freezes during inference or export.
- The camera and all worker threads shut down cleanly.

---

# Recommended implementation order

1. **`camera.py`** - get a live preview into the existing `ImagePreview`.
2. **`tracker.py`** - add IDs and lifecycle events; test with synthetic boxes.
3. **`counting.py`** - add stable display count and negative-count logic; test debounce.
4. **`export.py`** - add asynchronous CSV logging and Pandas filtering.
5. **`app.py`** - wire live mode, start/stop, HUD, export controls, teardown.
6. **Performance pass** - measure actual FPS, then decide whether to re-export at 320/416, use ONNX Runtime DirectML, or keep frame skipping.

The key engineering decision is to keep **raw detection**, **tracked occupancy**, and **discrepancy events** separate. That prevents display flicker, gives "negative count" a precise meaning, and produces CSV logs that are useful for debugging rather than just a stream of noisy counts.
