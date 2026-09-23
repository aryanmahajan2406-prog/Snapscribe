"""
Minimal always-on-top UI: a slim live-caption overlay + a system tray icon.
This is what makes SnapScribe feel like part of the OS rather than an app
you have to remember to open — it just sits there, quietly captioning and
summarizing, until you glance at it.
"""

import threading
import tkinter as tk

import pystray
from PIL import Image, ImageDraw


class CaptionOverlay:
    """A small, always-on-top, borderless window showing live captions."""

    def __init__(self, max_lines: int = 4):
        self.max_lines = max_lines
        self._lines: list[str] = []
        self._root = None
        self._label = None

    def _build(self):
        self._root = tk.Tk()
        self._root.overrideredirect(True)       # borderless
        self._root.attributes("-topmost", True)  # always on top
        self._root.attributes("-alpha", 0.85)
        self._root.configure(bg="black")

        screen_w = self._root.winfo_screenwidth()
        self._root.geometry(f"600x140+{(screen_w - 600) // 2}+40")

        self._label = tk.Label(
            self._root,
            text="SnapScribe listening…",
            fg="white",
            bg="black",
            font=("Segoe UI", 12),
            justify="left",
            anchor="w",
            wraplength=580,
        )
        self._label.pack(fill="both", expand=True, padx=10, pady=10)

    def push_line(self, speaker: str, text: str):
        self._lines.append(f"{speaker}: {text}")
        self._lines = self._lines[-self.max_lines :]
        if self._label is not None:
            self._label.config(text="\n".join(self._lines))

    def run(self):
        """Blocking — call this on its own thread."""
        self._build()
        self._root.mainloop()


def make_tray_icon(on_quit):
    image = Image.new("RGB", (64, 64), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse((14, 14, 50, 50), fill="lime")  # simple "recording" dot

    menu = pystray.Menu(pystray.MenuItem("Quit SnapScribe", on_quit))
    icon = pystray.Icon("snapscribe", image, "SnapScribe (listening)", menu)
    return icon


def run_ui(overlay_enabled: bool, on_quit) -> CaptionOverlay | None:
    """Starts tray icon (always) and caption overlay (optional) on background threads."""
    tray = make_tray_icon(lambda icon, item: on_quit())
    threading.Thread(target=tray.run, daemon=True).start()

    overlay = None
    if overlay_enabled:
        overlay = CaptionOverlay()
        threading.Thread(target=overlay.run, daemon=True).start()

    return overlay
