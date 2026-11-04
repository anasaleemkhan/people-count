from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from detector import DetectionResult, PeopleDetector, draw_detections

FONT = "Segoe UI"
BG = "#0B1120"
PANEL = "#111827"
CARD = "#0D1728"
VIEWER = "#070D18"
BORDER = "#243247"
TRACK = "#1E293B"
BUTTON = "#1E293B"
BUTTON_HOVER = "#334155"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
FAINT = "#64748B"
ACCENT = "#22C55E"
ACCENT_HOVER = "#4ADE80"
ACCENT_TEXT = "#052E16"
DANGER = "#F87171"
WARNING = "#FBBF24"


def load_image(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode != "RGB":
                image = image.convert("RGB")
            array = np.asarray(image)
    except Exception as exc:
        raise ValueError(f"Could not open this image: {exc}") from exc
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def shorten_name(name: str, limit: int) -> str:
    if len(name) <= limit:
        return name
    side = max(1, (limit - 1) // 2)
    return f"{name[:side]}…{name[-side:]}"


class ImagePreview(tk.Canvas):
    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(
            master,
            bg=VIEWER,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._image: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._render_job: str | None = None
        self.bind("<Configure>", self._schedule_render)

    def set_image(self, image: np.ndarray | None) -> None:
        self._image = image
        self._schedule_render()

    def _schedule_render(self, _event: tk.Event | None = None) -> None:
        if self._render_job is not None:
            try:
                self.after_cancel(self._render_job)
            except tk.TclError:
                pass
        self._render_job = self.after(60, self._render)

    def _render(self) -> None:
        self._render_job = None
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width < 40 or height < 40:
            return

        if self._image is None:
            self._draw_placeholder(width, height)
            return

        margin = 36
        available_width = max(1, width - (margin * 2))
        available_height = max(1, height - (margin * 2))
        image_height, image_width = self._image.shape[:2]
        scale = min(available_width / image_width, available_height / image_height)
        target_width = max(1, round(image_width * scale))
        target_height = max(1, round(image_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(
            self._image,
            (target_width, target_height),
            interpolation=interpolation,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        x = (width - target_width) // 2
        y = (height - target_height) // 2
        self.create_rectangle(
            x - 3,
            y - 3,
            x + target_width + 3,
            y + target_height + 3,
            fill=PANEL,
            outline=BORDER,
        )
        self.create_image(x, y, anchor="nw", image=self._photo)

    def _draw_placeholder(self, width: int, height: int) -> None:
        center_x = width / 2
        center_y = height / 2
        self.create_oval(
            center_x - 22,
            center_y - 92,
            center_x + 22,
            center_y - 48,
            outline=BORDER,
            width=3,
        )
        self.create_arc(
            center_x - 52,
            center_y - 42,
            center_x + 52,
            center_y + 40,
            start=0,
            extent=180,
            style=tk.ARC,
            outline=BORDER,
            width=3,
        )
        self.create_text(
            center_x,
            center_y + 72,
            text="No image selected",
            fill=TEXT,
            font=(FONT, 14, "bold"),
        )
        self.create_text(
            center_x,
            center_y + 102,
            text="Choose an image to count the people in it.",
            fill=MUTED,
            font=(FONT, 10),
            width=max(120, min(380, width - 80)),
        )


class PeopleCountApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("People Count")
        self.root.geometry("1180x780")
        self.root.minsize(1020, 700)
        self.root.configure(bg=BG)

        self.detector = PeopleDetector()
        self.events: queue.Queue[tuple] = queue.Queue()
        self.current_image: np.ndarray | None = None
        self.current_path: Path | None = None
        self.annotated_image: np.ndarray | None = None
        self.result: DetectionResult | None = None
        self.busy = False
        self.model_ready = False
        self.model_error: str | None = None
        self.pending_detection = False
        self.request_id = 0
        self.last_directory = Path.home()

        self._build_ui()
        self._update_controls()
        self.root.bind("<Control-o>", self._open_shortcut)
        self.root.bind("<Control-s>", self._save_shortcut)
        self._poll_events()
        threading.Thread(
            target=self._prepare_detector,
            name="model-loader",
            daemon=True,
        ).start()

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0, minsize=360)
        self.root.grid_columnconfigure(1, weight=1)

        sidebar = tk.Frame(self.root, bg=PANEL, padx=28, pady=28)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        tk.Label(
            sidebar,
            text="PEOPLE DETECTION",
            font=(FONT, 9, "bold"),
            fg=ACCENT,
            bg=PANEL,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            sidebar,
            text="People Count",
            font=(FONT, 24, "bold"),
            fg=TEXT,
            bg=PANEL,
        ).grid(row=1, column=0, sticky="w", pady=(4, 2))

        tk.Label(
            sidebar,
            text="Choose an image and find every person in it.",
            font=(FONT, 10),
            fg=MUTED,
            bg=PANEL,
            justify="left",
            wraplength=290,
        ).grid(row=2, column=0, sticky="w")

        self.choose_button = self._create_button(
            sidebar,
            "Choose an Image",
            self._choose_image,
            primary=True,
        )
        self.choose_button.grid(row=3, column=0, sticky="ew", pady=(26, 10))

        actions = tk.Frame(sidebar, bg=PANEL)
        actions.grid(row=4, column=0, sticky="ew")
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)

        self.count_button = self._create_button(
            actions,
            "Count People",
            self._count_people,
        )
        self.count_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.save_button = self._create_button(
            actions,
            "Save Result",
            self._save_result,
        )
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        tk.Frame(sidebar, bg=BORDER, height=1).grid(
            row=5, column=0, sticky="ew", pady=(24, 20)
        )

        confidence_header = tk.Frame(sidebar, bg=PANEL)
        confidence_header.grid(row=6, column=0, sticky="ew")
        confidence_header.grid_columnconfigure(0, weight=1)

        tk.Label(
            confidence_header,
            text="CONFIDENCE",
            font=(FONT, 9, "bold"),
            fg=MUTED,
            bg=PANEL,
        ).grid(row=0, column=0, sticky="w")

        self.confidence_value = tk.Label(
            confidence_header,
            text="35%",
            font=(FONT, 10, "bold"),
            fg=ACCENT,
            bg=PANEL,
        )
        self.confidence_value.grid(row=0, column=1, sticky="e")

        self.confidence_scale = tk.Scale(
            sidebar,
            from_=0.10,
            to=0.90,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg=PANEL,
            fg=TEXT,
            troughcolor=TRACK,
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            sliderrelief=tk.FLAT,
            sliderlength=18,
            width=8,
            takefocus=0,
        )
        self.confidence_scale.set(0.35)
        self.confidence_scale.grid(row=7, column=0, sticky="ew", pady=(8, 2))

        tk.Label(
            sidebar,
            text="Lower values find more people, but may add false positives.",
            font=(FONT, 9),
            fg=FAINT,
            bg=PANEL,
            justify="left",
            wraplength=290,
        ).grid(row=8, column=0, sticky="w")

        tk.Frame(sidebar, bg=BORDER, height=1).grid(
            row=9, column=0, sticky="ew", pady=(20, 20)
        )

        result_card = tk.Frame(
            sidebar,
            bg=CARD,
            padx=20,
            pady=16,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        result_card.grid(row=10, column=0, sticky="ew")
        result_card.grid_columnconfigure(0, weight=1)

        tk.Label(
            result_card,
            text="PEOPLE FOUND",
            font=(FONT, 9, "bold"),
            fg=MUTED,
            bg=CARD,
        ).grid(row=0, column=0, sticky="w")

        self.count_label = tk.Label(
            result_card,
            text="—",
            font=(FONT, 48, "bold"),
            fg=ACCENT,
            bg=CARD,
        )
        self.count_label.grid(row=1, column=0, sticky="w")

        self.result_note = tk.Label(
            result_card,
            text="Choose an image to begin.",
            font=(FONT, 9),
            fg=MUTED,
            bg=CARD,
            justify="left",
            wraplength=270,
        )
        self.result_note.grid(row=2, column=0, sticky="w")

        tk.Frame(result_card, bg=BORDER, height=1).grid(
            row=3, column=0, sticky="ew", pady=(14, 12)
        )

        self.file_label = tk.Label(
            result_card,
            text="No file selected",
            font=(FONT, 9, "bold"),
            fg=TEXT,
            bg=CARD,
            anchor="w",
        )
        self.file_label.grid(row=4, column=0, sticky="w")

        self.meta_label = tk.Label(
            result_card,
            text="",
            font=(FONT, 9),
            fg=FAINT,
            bg=CARD,
            anchor="w",
        )
        self.meta_label.grid(row=5, column=0, sticky="w", pady=(3, 0))

        sidebar.grid_rowconfigure(11, weight=1)

        model_status = tk.Frame(sidebar, bg=PANEL)
        model_status.grid(row=12, column=0, sticky="ew", pady=(18, 0))

        self.model_dot_canvas = tk.Canvas(
            model_status,
            width=10,
            height=10,
            bg=PANEL,
            highlightthickness=0,
            bd=0,
        )
        self.model_dot_canvas.pack(side="left")
        self.model_dot = self.model_dot_canvas.create_oval(
            2,
            2,
            8,
            8,
            fill=WARNING,
            outline="",
        )

        self.model_label = tk.Label(
            model_status,
            text="Loading detector…",
            font=(FONT, 9),
            fg=MUTED,
            bg=PANEL,
        )
        self.model_label.pack(side="left", padx=(8, 0))

        tk.Label(
            sidebar,
            text="YOLOv8n · ONNX · runs locally",
            font=(FONT, 8),
            fg=FAINT,
            bg=PANEL,
        ).grid(row=13, column=0, sticky="w", padx=(18, 0), pady=(4, 0))

        main = tk.Frame(self.root, bg=BG, padx=26, pady=26)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        header = tk.Frame(main, bg=BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Preview",
            font=(FONT, 15, "bold"),
            fg=TEXT,
            bg=BG,
        ).grid(row=0, column=0, sticky="w")

        self.filename_label = tk.Label(
            header,
            text="",
            font=(FONT, 9),
            fg=FAINT,
            bg=BG,
            anchor="e",
        )
        self.filename_label.grid(row=0, column=1, sticky="e")

        viewer_border = tk.Frame(main, bg=BORDER, padx=1, pady=1)
        viewer_border.grid(row=1, column=0, sticky="nsew")
        self.preview = ImagePreview(viewer_border)
        self.preview.pack(fill="both", expand=True)

        footer = tk.Frame(main, bg=BG)
        footer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        footer.grid_columnconfigure(0, weight=1)

        self.status_label = tk.Label(
            footer,
            text="Preparing detector…",
            font=(FONT, 9),
            fg=MUTED,
            bg=BG,
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        tk.Label(
            footer,
            text="Ctrl+O  Open    Ctrl+S  Save",
            font=(FONT, 9),
            fg=FAINT,
            bg=BG,
        ).grid(row=0, column=1, sticky="e")

        self.confidence_scale.configure(command=self._on_confidence_changed)
        self.confidence_scale.bind("<ButtonRelease-1>", self._on_confidence_released)

    def _create_button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        primary: bool = False,
    ) -> tk.Button:
        if primary:
            background = ACCENT
            hover = ACCENT_HOVER
            foreground = ACCENT_TEXT
        else:
            background = BUTTON
            hover = BUTTON_HOVER
            foreground = TEXT

        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=hover,
            activeforeground=foreground,
            disabledforeground=FAINT,
            bd=0,
            relief=tk.FLAT,
            cursor="hand2",
            font=(FONT, 10, "bold"),
            padx=14,
            pady=11,
            highlightthickness=0,
        )
        button.bind(
            "<Enter>",
            lambda _event: button.configure(bg=hover)
            if button["state"] == tk.NORMAL
            else None,
        )
        button.bind(
            "<Leave>",
            lambda _event: button.configure(bg=background)
            if button["state"] == tk.NORMAL
            else None,
        )
        return button

    def _prepare_detector(self) -> None:
        try:
            self.detector.ensure_model(
                lambda downloaded, total: self.events.put(
                    ("model_progress", downloaded, total)
                )
            )
            self.detector.load()
            self.events.put(("model_ready",))
        except Exception as exc:
            self.events.put(("model_error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self.events.get_nowait())
        except queue.Empty:
            pass

        try:
            self.root.after(80, self._poll_events)
        except tk.TclError:
            pass

    def _handle_event(self, event: tuple) -> None:
        kind = event[0]

        if kind == "model_progress":
            downloaded, total = event[1], event[2]
            if total > 0:
                percentage = min(100, round(downloaded * 100 / total))
                message = f"Downloading detector… {percentage}%"
            else:
                message = f"Downloading detector… {downloaded / (1024 * 1024):.1f} MB"
            self._set_model_status(message, WARNING)
            self._set_status(message)

        elif kind == "model_ready":
            self.model_ready = True
            self.model_error = None
            self._set_model_status("Detector ready", ACCENT)
            self._update_controls()
            if self.pending_detection and self.current_image is not None:
                self.pending_detection = False
                self._count_people()
            elif self.current_image is None:
                self._set_status("Detector ready. Choose an image.")

        elif kind == "model_error":
            self.model_error = event[1]
            self.model_ready = False
            self.pending_detection = False
            self._set_model_status("Detector unavailable", DANGER)
            self._set_status("Could not load the detector.")
            self.result_note.configure(
                text="The detector could not be loaded. Check your internet connection and restart the app."
            )
            self._update_controls()
            messagebox.showerror(
                "Detector unavailable",
                f"{self.model_error}\n\nThe first launch needs internet access to download the YOLOv8n model (about 13 MB).",
                parent=self.root,
            )

        elif kind == "detection_success":
            request_id, result, annotated = event[1], event[2], event[3]
            if request_id != self.request_id:
                return
            self.busy = False
            self.result = result
            self.annotated_image = annotated
            self.preview.set_image(annotated)
            self._populate_result(result)
            self._update_controls()
            if result.count:
                noun = "person" if result.count == 1 else "people"
                self._set_status(
                    f"Found {result.count} {noun} in {result.elapsed:.2f} seconds."
                )
            else:
                self._set_status(
                    f"No people found in {result.elapsed:.2f} seconds. Try lowering the confidence."
                )

        elif kind == "detection_error":
            request_id, message = event[1], event[2]
            if request_id != self.request_id:
                return
            self.busy = False
            self.result = None
            self.annotated_image = None
            self.preview.set_image(self.current_image)
            self.count_label.configure(text="—", fg=MUTED)
            self.result_note.configure(text="Detection failed. Try another image.")
            self._update_controls()
            self._set_status("Detection failed.")
            messagebox.showerror("Detection failed", message, parent=self.root)

    def _choose_image(self) -> None:
        if self.busy:
            return

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Choose an image",
            initialdir=str(self.last_directory),
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            image = load_image(path)
        except ValueError as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)
            return

        self.current_path = Path(path)
        self.last_directory = self.current_path.parent
        self.current_image = image
        self.annotated_image = None
        self.result = None
        self.request_id += 1
        self.pending_detection = False

        self.preview.set_image(image)
        self.filename_label.configure(text=shorten_name(self.current_path.name, 56))
        self.file_label.configure(text=shorten_name(self.current_path.name, 30))
        self.meta_label.configure(
            text=f"{image.shape[1]:,} × {image.shape[0]:,} px"
        )
        self.count_label.configure(text="—", fg=MUTED)
        self.result_note.configure(text="Preparing image…")
        self._update_controls()

        if self.model_ready:
            self._count_people()
        elif self.model_error:
            self.result_note.configure(
                text="The detector is unavailable. Restart the app to try again."
            )
            self._set_status("Detector unavailable.")
        else:
            self.pending_detection = True
            self.result_note.configure(text="Waiting for the detector…")
            self._set_status("Waiting for the detector to finish loading…")

    def _count_people(self) -> None:
        if self.current_image is None or self.busy:
            return

        if not self.model_ready:
            self.pending_detection = True
            if self.model_error:
                self._set_status("Detector unavailable.")
            else:
                self._set_status("Waiting for the detector to finish loading…")
            return

        self.pending_detection = False
        self.busy = True
        self.request_id += 1
        request_id = self.request_id
        image = self.current_image.copy()
        confidence = float(self.confidence_scale.get())

        self.result = None
        self.annotated_image = None
        self.preview.set_image(image)
        self.count_label.configure(text="…", fg=ACCENT)
        self.result_note.configure(text="Scanning the image…")
        self.meta_label.configure(
            text=f"{image.shape[1]:,} × {image.shape[0]:,} px"
        )
        self._update_controls()
        self._set_status("Detecting people…")

        threading.Thread(
            target=self._detect_worker,
            args=(request_id, image, confidence),
            name="people-detector",
            daemon=True,
        ).start()

    def _detect_worker(
        self,
        request_id: int,
        image: np.ndarray,
        confidence: float,
    ) -> None:
        try:
            result = self.detector.detect(image, confidence)
            annotated = draw_detections(image, result.detections)
            self.events.put(("detection_success", request_id, result, annotated))
        except Exception as exc:
            self.events.put(("detection_error", request_id, str(exc)))

    def _populate_result(self, result: DetectionResult) -> None:
        self.count_label.configure(text=str(result.count), fg=ACCENT)
        if result.count == 0:
            self.result_note.configure(
                text="No people detected. Try lowering the confidence threshold."
            )
        else:
            self.result_note.configure(
                text="Detections are outlined in the preview."
            )
        if self.current_image is not None:
            self.meta_label.configure(
                text=(
                    f"{self.current_image.shape[1]:,} × {self.current_image.shape[0]:,} px"
                    f"   ·   {result.elapsed:.2f} s"
                )
            )

    def _save_result(self) -> None:
        if self.annotated_image is None or self.current_path is None or self.busy:
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save annotated image",
            initialdir=str(self.last_directory),
            initialfile=f"{self.current_path.stem}_counted.jpg",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG image", "*.jpg"),
                ("PNG image", "*.png"),
            ],
        )
        if not path:
            return

        save_path = Path(path)
        extension = save_path.suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png"}:
            save_path = save_path.with_suffix(".jpg")
            extension = ".jpg"

        try:
            success, encoded = cv2.imencode(extension, self.annotated_image)
            if not success:
                raise ValueError("OpenCV could not encode the image.")
            save_path.write_bytes(encoded.tobytes())
        except Exception as exc:
            messagebox.showerror(
                "Could not save image",
                str(exc),
                parent=self.root,
            )
            return

        self._set_status(f"Saved {shorten_name(save_path.name, 48)}.")

    def _update_controls(self) -> None:
        choose_state = tk.DISABLED if self.busy else tk.NORMAL
        count_state = (
            tk.NORMAL
            if self.current_image is not None and self.model_ready and not self.busy
            else tk.DISABLED
        )
        save_state = (
            tk.NORMAL
            if self.annotated_image is not None and not self.busy
            else tk.DISABLED
        )
        scale_state = tk.DISABLED if self.busy else tk.NORMAL

        self.choose_button.configure(state=choose_state)
        self.count_button.configure(state=count_state)
        self.save_button.configure(state=save_state)
        self.confidence_scale.configure(state=scale_state)

    def _set_status(self, message: str, color: str = MUTED) -> None:
        self.status_label.configure(text=message, fg=color)

    def _set_model_status(self, message: str, color: str) -> None:
        self.model_label.configure(text=message, fg=MUTED if color != DANGER else DANGER)
        self.model_dot_canvas.itemconfigure(self.model_dot, fill=color)

    def _on_confidence_changed(self, value: str) -> None:
        self.confidence_value.configure(text=f"{float(value):.0%}")
        if self.result is not None and not self.busy:
            self._set_status("Confidence updated. Press Count People to re-run.")

    def _on_confidence_released(self, _event: tk.Event) -> None:
        if self.current_image is not None and self.model_ready and not self.busy:
            self._count_people()

    def _open_shortcut(self, _event: tk.Event) -> str:
        self._choose_image()
        return "break"

    def _save_shortcut(self, _event: tk.Event) -> str:
        self._save_result()
        return "break"
