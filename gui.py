#!/usr/bin/env python3
"""A small desktop window so the tool can be used without a terminal.

Pick videos, set the camera height, press Run. The heavy work happens in a
separate process, so a crash there prints an error instead of killing the
window.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")
WEIGHTS = os.path.join(HERE, "weights")


def models_present() -> bool:
    return os.path.isfile(os.path.join(WEIGHTS, "autodrive_fp32.onnx"))


class Application:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.running = False
        self.videos: list[str] = []
        self.default_out = os.path.join(HERE, "results")

        root.title("Vision Pilot - dashcam video")
        root.geometry("900x660")
        root.minsize(760, 560)

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Video").grid(row=0, column=0, sticky="w", pady=4)
        self.video_field = ttk.Entry(frame)
        self.video_field.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Choose files...", command=self.choose_videos).grid(
            row=0, column=2
        )

        ttk.Label(frame, text="Save results to").grid(row=1, column=0, sticky="w", pady=4)
        self.out_field = ttk.Entry(frame)
        self.out_field.insert(0, self.default_out)
        self.out_field.grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse...", command=self.choose_output).grid(
            row=1, column=2
        )

        options = ttk.Frame(frame)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(options, text="Camera height (m)").pack(side="left")
        self.height_field = ttk.Spinbox(options, from_=0.8, to=2.5, increment=0.01,
                                        width=7)
        self.height_field.set("1.30")
        self.height_field.pack(side="left", padx=(6, 18))
        ttk.Label(options, text="Frames per second").pack(side="left")
        self.hz_field = ttk.Combobox(options, values=["5", "10"], width=5,
                                     state="readonly")
        self.hz_field.set("10")
        self.hz_field.pack(side="left", padx=(6, 18))
        ttk.Label(options, text="From second").pack(side="left")
        self.from_field = ttk.Entry(options, width=6)
        self.from_field.insert(0, "0")
        self.from_field.pack(side="left", padx=6)
        ttk.Label(options, text="to").pack(side="left")
        self.to_field = ttk.Entry(options, width=6)
        self.to_field.insert(0, "end")
        self.to_field.pack(side="left", padx=6)

        advanced = ttk.Frame(frame)
        advanced.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(advanced, text="Ego speed (km/h)").pack(side="left")
        self.ego_field = ttk.Entry(advanced, width=10)
        self.ego_field.insert(0, "auto")
        self.ego_field.pack(side="left", padx=(6, 4))
        ttk.Label(advanced, text="('auto' reads the GPS log in the video)").pack(
            side="left", padx=(0, 16)
        )
        ttk.Label(advanced, text="Speed limit").pack(side="left")
        self.limit_field = ttk.Entry(advanced, width=6)
        self.limit_field.insert(0, "50")
        self.limit_field.pack(side="left", padx=6)
        self.tracking = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced, text="Temporal tracking",
                        variable=self.tracking).pack(side="left", padx=(16, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        self.models_button = ttk.Button(
            buttons, text="1. Download models (first run only, 114 MB)",
            command=self.download_models
        )
        self.models_button.pack(side="left")
        self.run_button = ttk.Button(buttons, text="2. RUN", command=self.run)
        self.run_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop,
                                      state="disabled")
        self.stop_button.pack(side="left")
        ttk.Button(buttons, text="Open results folder",
                   command=self.open_output).pack(side="right")

        self.log_view = scrolledtext.ScrolledText(frame, height=18, wrap="word")
        self.log_view.grid(row=5, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(5, weight=1)

        self.status = ttk.Label(frame, text="")
        self.status.grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.refresh_models_button()
        self.log("Choose a video and press RUN.")
        if not models_present():
            self.log("First run: press 'Download models' — 114 MB, once only.")
        self.root.after(100, self.drain_events)

    # --- helpers -----------------------------------------------------------
    def log(self, text: str) -> None:
        self.log_view.insert("end", text.rstrip() + "\n")
        self.log_view.see("end")

    def refresh_models_button(self) -> None:
        if models_present():
            self.models_button.config(text="Models: ready", state="disabled")
        else:
            self.models_button.config(
                text="1. Download models (first run only, 114 MB)", state="normal"
            )

    def choose_videos(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Choose one or more videos",
            filetypes=[("Video", "*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv"),
                       ("All files", "*.*")],
        )
        if chosen:
            self.videos = list(chosen)
            names = ", ".join(os.path.basename(v) for v in self.videos[:3])
            suffix = " ..." if len(self.videos) > 3 else ""
            self.video_field.delete(0, "end")
            self.video_field.insert(0, f"{len(self.videos)} file(s): {names}{suffix}")

    def choose_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose an output folder")
        if folder:
            self.out_field.delete(0, "end")
            self.out_field.insert(0, folder)

    def open_output(self) -> None:
        folder = self.out_field.get().strip() or self.default_out
        os.makedirs(folder, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            messagebox.showinfo("Results folder", f"{folder}\n\n(could not open: {exc})")

    # --- actions -----------------------------------------------------------
    def download_models(self) -> None:
        if self.running:
            return
        self.models_button.config(state="disabled")
        self.log("\nDownloading models, please wait...")

        def worker():
            try:
                sys.path.insert(0, HERE)
                import download_models

                download_models.download(
                    WEIGHTS, log=lambda s: self.events.put(("log", s))
                )
            except BaseException as exc:
                self.events.put(("log", f"Download error: {exc}"))
            finally:
                self.events.put(("models_done", None))

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        if self.running:
            return
        if not self.videos:
            messagebox.showwarning("No video", "Press 'Choose files...' first.")
            return
        if not models_present():
            messagebox.showwarning("No models", "Press 'Download models' first.")
            return
        try:
            camera_height = float(self.height_field.get())
        except ValueError:
            messagebox.showwarning("Bad height", "Camera height must be a number, "
                                                 "for example 1.32")
            return
        if not 0.8 <= camera_height <= 2.5:
            messagebox.showwarning("Bad height", "Camera height must be between "
                                                 "0.8 and 2.5 metres.")
            return
        ego = self.ego_field.get().strip().lower()
        if ego and ego not in ("auto", ""):
            try:
                float(ego)
            except ValueError:
                messagebox.showwarning("Bad speed",
                                       "Ego speed must be a number in km/h, or 'auto'.")
                return
        try:
            speed_limit = float(self.limit_field.get())
        except ValueError:
            messagebox.showwarning("Bad limit", "Speed limit must be a number in km/h.")
            return

        out_root = self.out_field.get().strip() or self.default_out
        hz = self.hz_field.get()
        from_s = self.from_field.get().strip() or "0"
        to_s = self.to_field.get().strip()

        self.running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status.config(text="Running...")

        def worker():
            for position, video in enumerate(self.videos, 1):
                if not self.running:
                    break
                stem = os.path.splitext(os.path.basename(video))[0]
                out_dir = os.path.join(out_root, stem)
                self.events.put(
                    ("log", f"\n=== [{position}/{len(self.videos)}] "
                            f"{os.path.basename(video)} ===")
                )
                command = [
                    sys.executable, "-u", MAIN, video,
                    "--out", out_dir,
                    "--hz", hz,
                    "--camera-height", f"{camera_height}",
                    "--from-s", from_s,
                    "--speed-limit-kmh", f"{speed_limit}",
                ]
                if to_s and to_s.lower() not in ("end", ""):
                    command += ["--to-s", to_s]
                if ego and ego not in ("auto", ""):
                    command += ["--ego-speed-kmh", ego]
                if not self.tracking.get():
                    command += ["--no-tracking"]
                try:
                    self.process = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        cwd=HERE, bufsize=1,
                    )
                    for line in self.process.stdout:
                        self.events.put(("log", line))
                    code = self.process.wait()
                    if code != 0 and self.running:
                        self.events.put(("log", f"  (exited with code {code})"))
                except BaseException as exc:
                    self.events.put(("log", f"  ERROR: {exc}"))
                finally:
                    self.process = None
            self.events.put(("finished", out_root))

        threading.Thread(target=worker, daemon=True).start()

    def stop(self) -> None:
        self.running = False
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.log("\nStop requested.")

    # --- pump background messages onto the UI thread -----------------------
    def drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "models_done":
                    self.refresh_models_button()
                elif kind == "finished":
                    self.running = False
                    self.run_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.status.config(text=f"Done. Results in {payload}")
                    self.log(f"\nDONE. Results are in {payload}")
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)


if __name__ == "__main__":
    window = tk.Tk()
    Application(window)
    window.mainloop()
