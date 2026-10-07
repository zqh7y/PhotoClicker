"""Photo Clicker: clicks a picture whenever it shows up on your screen.

1. Take a picture of the thing to click (straight from the screen, or a file).
2. Press START (or P) and switch to your game.
"""

import ctypes
import json
import os
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from autoclicker import Finder, click, grab, load_template, open_screen

APP_NAME = "Photo Clicker"
HOTKEY = "p"
HOTKEY_VK = 0x50  # the P key on every keyboard layout
SIZE_SCALES = (0.9, 1.0, 1.1)
SETTINGS_VERSION = 3
SAME_SPOT_COOLDOWN = 0.1  # don't click the same spot again right away
START_DELAY = 3
THUMB = 88

BG = "#f3f5fa"
CARD = "#ffffff"
BORDER = "#e2e6ef"
TEXT = "#1d2230"
MUTED = "#6b7385"
ACCENT = "#4f6bed"
ACCENT_HOVER = "#3e58d6"
GREEN = "#1fa463"
GREEN_HOVER = "#178a52"
RED = "#e5484d"
RED_HOVER = "#cc3a3f"
LIGHT_BTN = "#eef1fb"
LIGHT_BTN_HOVER = "#e0e5f8"
FONT = "Segoe UI"


def resource(name):
    """A file shipped with the app (works inside the .exe too)."""
    base = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    return Path(base) / name


def data_dir():
    """Where pictures and settings are kept between runs."""
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", Path.home()))
    else:
        root = Path.home() / ".config"
    path = root / "ImageAutoClicker"
    (path / "pictures").mkdir(parents=True, exist_ok=True)
    return path


def make_dpi_aware():
    # Without this, Windows scaling (125%, 150%...) makes screenshots and
    # mouse positions disagree.
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def to_photo(bgr):
    """OpenCV image -> Tk image, without needing Pillow."""
    ok, data = cv2.imencode(".ppm", bgr)
    return tk.PhotoImage(data=data.tobytes(), format="ppm")


def fit(bgr, max_w, max_h):
    """Shrink (never enlarge) an image to fit a box. Returns (image, factor)."""
    h, w = bgr.shape[:2]
    f = min(1.0, max_w / w, max_h / h)
    if f < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * f)), max(1, int(h * f))),
                         interpolation=cv2.INTER_AREA)
    return bgr, f


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        rgb = image[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)  # on white
    return image


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class ClickerThread(threading.Thread):
    """Watches the screen and clicks. Talks to the window through `events`."""

    def __init__(self, images, confidence, interval, click_all, any_size, any_color,
                 move_time, clicks, events):
        super().__init__(daemon=True)
        self.move_time = move_time
        self.clicks = clicks
        self.any_color = any_color
        self.images = images
        self.confidence = confidence
        self.interval = interval
        self.click_all = click_all
        self.scales = SIZE_SCALES if any_size else (1.0,)
        self.events = events
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self._run()
        except Exception as e:  # show the problem instead of dying silently
            self.events.put(("error", str(e)))
        finally:
            self.events.put(("stopped", None))

    def _run(self):
        import pyautogui

        pyautogui.PAUSE = 0.02
        pyautogui.FAILSAFE = True  # mouse into a screen corner = emergency stop

        finders = [Finder(*load_template(path), scales=self.scales, shape=self.any_color)
                   for path in self.images]

        for left in range(START_DELAY, 0, -1):
            self.events.put(("status", (f"Starting in {left}...", "Open your game now")))
            if self.stop_event.wait(1):
                return
        self.events.put(("status", ("Looking for your picture...",
                                    f"Press {HOTKEY.upper()} to stop")))

        clicks = 0
        recent = []  # (x, y, time) of the last clicks
        with open_screen() as sct:
            area = sct.monitors[0]  # every screen together
            while not self.stop_event.is_set():
                screen, factor = grab(sct, area)
                for finder in finders:
                    for cx, cy, _ in finder.find(screen, self.confidence, self.click_all):
                        if self.stop_event.is_set():
                            return
                        x = area["left"] + round(cx * factor)
                        y = area["top"] + round(cy * factor)
                        now = time.monotonic()
                        recent = [r for r in recent if now - r[2] < SAME_SPOT_COOLDOWN]
                        if any(abs(x - rx) < 20 and abs(y - ry) < 20 for rx, ry, _ in recent):
                            continue
                        click(pyautogui, x, y, "left", self.clicks, self.move_time)
                        recent.append((x, y, time.monotonic()))
                        clicks += 1
                        self.events.put(("click", clicks))
                self.stop_event.wait(self.interval)


# ---------------------------------------------------------------------------
# Box picker: drag a rectangle over a screenshot or picture
# ---------------------------------------------------------------------------

class BoxPicker:
    """Shows `image` and lets the user drag a box. Calls on_done(crop or None).

    With `fullscreen` the image covers the screen 1:1 (for "take from
    screen"); otherwise it opens in a window, scaled down to fit.
    """

    def __init__(self, root, image, on_done, fullscreen_geometry=None, allow_whole=False):
        self.image = image
        self.on_done = on_done
        self.allow_whole = allow_whole
        self.start = None
        self.rect = None

        self.win = tk.Toplevel(root)
        self.win.configure(bg="black")
        if fullscreen_geometry:
            self.win.overrideredirect(True)
            self.win.geometry(fullscreen_geometry)
            shown, self.factor = image, 1.0
        else:
            self.win.title("Choose what to click")
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            shown, self.factor = fit(image, sw * 0.85, sh * 0.75)
        self.win.attributes("-topmost", True)

        dim = cv2.addWeighted(shown, 0.55, np.zeros_like(shown), 0.45, 0)
        self.bright = shown
        self.photo = to_photo(dim)
        h, w = shown.shape[:2]
        self.canvas = tk.Canvas(self.win, width=w, height=h, highlightthickness=0,
                                cursor="crosshair", bg="black")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.crop_item = self.canvas.create_image(0, 0, anchor="nw")

        hint = "Drag a box around the thing to click      Esc = cancel"
        if allow_whole:
            hint += "      Enter = use the whole picture"
        x = w // 2
        self.canvas.create_rectangle(x - 330, 14, x + 330, 54, fill="#1d2230", outline="")
        self.canvas.create_text(x, 34, text=hint, fill="white", font=(FONT, 12, "bold"))

        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.win.bind("<Escape>", lambda e: self.finish(None))
        self.win.bind("<Return>", lambda e: self.finish(self.image) if allow_whole else None)
        self.win.protocol("WM_DELETE_WINDOW", lambda: self.finish(None))
        self.win.focus_force()
        self.win.grab_set()

    def press(self, e):
        self.start = (e.x, e.y)

    def drag(self, e):
        if not self.start:
            return
        x0, y0 = self.start
        if self.rect is None:
            self.rect = self.canvas.create_rectangle(x0, y0, e.x, e.y, outline=ACCENT, width=3)
        self.canvas.coords(self.rect, x0, y0, e.x, e.y)
        # show the selected part bright, the rest stays dimmed
        l, t, r, b = min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y)
        h, w = self.bright.shape[:2]
        l, t, r, b = max(0, l), max(0, t), min(w, r), min(h, b)
        if r - l > 1 and b - t > 1:
            self.crop_photo = to_photo(self.bright[t:b, l:r])
            self.canvas.itemconfigure(self.crop_item, image=self.crop_photo)
            self.canvas.coords(self.crop_item, l, t)
            self.canvas.tag_raise(self.rect)

    def release(self, e):
        if not self.start:
            return
        x0, y0 = self.start
        f = self.factor
        l, t = int(min(x0, e.x) / f), int(min(y0, e.y) / f)
        r, b = int(max(x0, e.x) / f), int(max(y0, e.y) / f)
        h, w = self.image.shape[:2]
        l, t, r, b = max(0, l), max(0, t), min(w, r), min(h, b)
        if r - l < 8 or b - t < 8:
            self.start = None
            return  # too small, probably a misclick
        self.finish(self.image[t:b, l:r].copy())

    def finish(self, crop):
        self.win.grab_release()
        self.win.destroy()
        self.on_done(crop)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def flat_button(parent, text, command, bg, hover, fg="white", size=11, bold=True, **kw):
    btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                    activebackground=hover, activeforeground=fg, relief="flat",
                    bd=0, cursor="hand2", font=(FONT, size, "bold" if bold else "normal"),
                    **kw)
    btn.bind("<Enter>", lambda e: btn.configure(bg=btn.hover))
    btn.bind("<Leave>", lambda e: btn.configure(bg=btn.base))
    btn.base, btn.hover = bg, hover
    return btn


def recolor(btn, bg, hover):
    btn.base, btn.hover = bg, hover
    btn.configure(bg=bg, activebackground=hover)


class App:
    def __init__(self, root):
        self.root = root
        self.worker = None
        self.clicks = 0
        self.events = queue.Queue()
        self.data = data_dir()
        self.settings_path = self.data / "settings.json"
        self.pictures = []
        self.thumbs = []

        root.title(APP_NAME)
        root.configure(bg=BG)
        root.resizable(False, False)
        try:
            root.iconphoto(True, tk.PhotoImage(file=str(resource("assets/icon.png"))))
        except tk.TclError:
            pass

        self.confidence = tk.IntVar(value=70)
        self.interval = tk.DoubleVar(value=0.0)
        self.click_all = tk.BooleanVar(value=False)
        self.any_size = tk.BooleanVar(value=True)
        self.any_color = tk.BooleanVar(value=True)
        self.move_time = tk.DoubleVar(value=0.05)
        self.clicks_each = tk.IntVar(value=2)

        self.build()
        self.load_settings()
        self.show_pictures()
        self.start_hotkey()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll_events)

    # --- layout -------------------------------------------------------------

    def card(self, number, title):
        outer = tk.Frame(self.root, bg=BORDER)
        outer.pack(fill="x", padx=20, pady=(0, 14))
        inner = tk.Frame(outer, bg=CARD, padx=18, pady=14)
        inner.pack(fill="both", padx=1, pady=1)
        head = tk.Frame(inner, bg=CARD)
        head.pack(fill="x", pady=(0, 10))
        tk.Label(head, text=str(number), bg=ACCENT, fg="white", width=2,
                 font=(FONT, 11, "bold")).pack(side="left")
        tk.Label(head, text=title, bg=CARD, fg=TEXT,
                 font=(FONT, 13, "bold")).pack(side="left", padx=10)
        return inner

    def build(self):
        root = self.root
        tk.Label(root, text=APP_NAME, bg=BG, fg=TEXT,
                 font=(FONT, 22, "bold")).pack(anchor="w", padx=20, pady=(18, 0))
        tk.Label(root, text="Clicks your picture every time it shows up on the screen.",
                 bg=BG, fg=MUTED, font=(FONT, 10)).pack(anchor="w", padx=20, pady=(0, 14))

        # Step 1: pictures
        step1 = self.card(1, "Choose what to click")
        row = tk.Frame(step1, bg=CARD)
        row.pack(fill="x")
        flat_button(row, "Take from screen", self.take_from_screen, ACCENT, ACCENT_HOVER,
                    padx=14, pady=8).pack(side="left")
        flat_button(row, "Open picture...", self.open_picture, LIGHT_BTN, LIGHT_BTN_HOVER,
                    fg=ACCENT, padx=14, pady=8).pack(side="left", padx=8)
        self.gallery = tk.Frame(step1, bg=CARD)
        self.gallery.pack(fill="x", pady=(12, 0))

        # Step 2: start
        step2 = self.card(2, "Press Start, then open your game")
        self.start_btn = flat_button(step2, "START", self.toggle, GREEN, GREEN_HOVER,
                                     size=18, pady=12)
        self.start_btn.pack(fill="x")
        self.status = tk.Label(step2, text="Ready", bg=CARD, fg=TEXT, font=(FONT, 13, "bold"))
        self.status.pack(pady=(12, 0))
        self.substatus = tk.Label(step2, text=f"Tip: you can press {HOTKEY.upper()} "
                                  "to start and stop, even inside the game",
                                  bg=CARD, fg=MUTED, font=(FONT, 9))
        self.substatus.pack()

        # Settings (hidden until opened)
        self.settings_btn = tk.Label(root, text="Settings  ▸", bg=BG, fg=ACCENT,
                                     cursor="hand2", font=(FONT, 10, "bold"))
        self.settings_btn.pack(anchor="w", padx=20)
        self.settings_btn.bind("<Button-1>", lambda e: self.toggle_settings())
        self.settings = tk.Frame(root, bg=BG, padx=20)
        self.settings_open = False
        self.build_settings(self.settings)
        tk.Frame(root, bg=BG, height=16).pack(side="bottom")

    def build_settings(self, frame):
        def slider_row(label, left, right, var, lo, hi, step, show):
            top = tk.Frame(frame, bg=BG)
            top.pack(fill="x", pady=(10, 0))
            tk.Label(top, text=label, bg=BG, fg=TEXT,
                     font=(FONT, 10, "bold")).pack(side="left")
            value = tk.Label(top, text=show(var.get()), bg=BG, fg=ACCENT,
                             font=(FONT, 10, "bold"))
            value.pack(side="left", padx=6)
            line = tk.Frame(frame, bg=BG)
            line.pack(fill="x")
            tk.Label(line, text=left, bg=BG, fg=MUTED, font=(FONT, 9)).pack(side="left")
            tk.Scale(line, from_=lo, to=hi, resolution=step, orient="horizontal",
                     variable=var, showvalue=False, bg=BG, troughcolor=BORDER,
                     highlightthickness=0, bd=0, sliderrelief="flat",
                     activebackground=ACCENT, length=240, sliderlength=22, width=12,
                     command=lambda _: (value.configure(text=show(var.get())),
                                        self.save_settings())).pack(side="left", padx=6)
            tk.Label(line, text=right, bg=BG, fg=MUTED, font=(FONT, 9)).pack(side="left")

        slider_row("How exact must it match?", "Loose", "Exact", self.confidence, 50, 99, 1,
                   lambda v: f"{int(v)}%")
        tk.Label(frame, text="If it clicks wrong things, go more exact. "
                 "If it misses the picture, go looser.",
                 bg=BG, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        slider_row("Pause between looks", "None", "Long", self.interval, 0.0, 1.0, 0.05,
                   lambda v: "none (fastest)" if float(v) == 0 else f"{float(v):.2f} s")
        slider_row("Mouse travel time", "Instant", "Slow", self.move_time, 0.0, 0.5, 0.01,
                   lambda v: f"{float(v):.2f} s")
        slider_row("Clicks each time", "1", "5", self.clicks_each, 1, 5, 1,
                   lambda v: f"{int(float(v))}x")
        for text, var in (("Find it in any color (matches the shape)", self.any_color),
                          ("Click every copy on the screen, not just one", self.click_all),
                          ("Also find it a bit bigger or smaller", self.any_size)):
            tk.Checkbutton(frame, text=text, variable=var, bg=BG, fg=TEXT,
                           activebackground=BG, selectcolor=CARD, font=(FONT, 10),
                           highlightthickness=0, bd=0,
                           command=self.save_settings).pack(anchor="w", pady=(6, 0))
        tk.Label(frame, text="Emergency stop: push the mouse into any corner of the screen.",
                 bg=BG, fg=MUTED, font=(FONT, 9)).pack(anchor="w", pady=(10, 0))

    def toggle_settings(self):
        self.settings_open = not self.settings_open
        if self.settings_open:
            self.settings.pack(fill="x", after=self.settings_btn)
            self.settings_btn.configure(text="Settings  ▾")
        else:
            self.settings.pack_forget()
            self.settings_btn.configure(text="Settings  ▸")

    # --- pictures -----------------------------------------------------------

    def show_pictures(self):
        for child in self.gallery.winfo_children():
            child.destroy()
        self.thumbs = []
        if not self.pictures:
            tk.Label(self.gallery, text="No picture yet. Press \"Take from screen\" and "
                     "drag a box around the button you want clicked.",
                     bg=CARD, fg=MUTED, font=(FONT, 10), wraplength=380,
                     justify="left").pack(anchor="w")
            return
        for i, path in enumerate(self.pictures):
            image = read_bgr(path)
            if image is None:
                continue
            thumb, _ = fit(image, THUMB, THUMB)
            photo = to_photo(thumb)
            self.thumbs.append(photo)
            cell = tk.Frame(self.gallery, bg=CARD, highlightbackground=BORDER,
                            highlightthickness=1, width=THUMB + 16, height=THUMB + 34)
            cell.grid(row=i // 4, column=i % 4, padx=(0, 8), pady=(0, 8))
            cell.pack_propagate(False)
            tk.Label(cell, image=photo, bg=CARD).pack(expand=True)
            remove = tk.Label(cell, text="Remove", bg=CARD, fg=RED, cursor="hand2",
                              font=(FONT, 9, "bold"))
            remove.pack(pady=(0, 4))
            remove.bind("<Button-1>", lambda e, p=path: self.remove_picture(p))

    def add_picture(self, crop):
        if crop is None:
            return
        path = self.data / "pictures" / f"picture_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(path), crop)
        self.pictures.append(str(path))
        self.save_settings()
        self.show_pictures()

    def remove_picture(self, path):
        if self.worker:
            return
        self.pictures = [p for p in self.pictures if p != path]
        if Path(path).parent == self.data / "pictures":
            Path(path).unlink(missing_ok=True)
        self.save_settings()
        self.show_pictures()

    def take_from_screen(self):
        if self.worker:
            return
        self.root.withdraw()
        self.root.after(350, self._snap)  # let the window disappear first

    def _snap(self):
        with open_screen() as sct:
            area = sct.monitors[0]
            shot, _ = grab(sct, area)
        geometry = f"{area['width']}x{area['height']}+{area['left']}+{area['top']}"

        def done(crop):
            self.root.deiconify()
            self.add_picture(crop)

        BoxPicker(self.root, shot, done, fullscreen_geometry=geometry)

    def open_picture(self):
        if self.worker:
            return
        path = filedialog.askopenfilename(
            title="Pick a picture of the thing to click",
            filetypes=[("Pictures", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path:
            return
        image = read_bgr(path)
        if image is None:
            messagebox.showerror(APP_NAME, "That file is not a picture I can open.")
            return
        BoxPicker(self.root, image, self.add_picture, allow_whole=True)

    # --- start / stop -------------------------------------------------------

    def toggle(self):
        if self.worker:
            self.worker.stop()
            self.set_status("Stopping...", "")
            return
        if not self.pictures:
            messagebox.showinfo(APP_NAME, "First choose what to click (step 1).")
            return
        self.save_settings()
        self.worker = ClickerThread(list(self.pictures), self.confidence.get() / 100,
                                    max(0.0, self.interval.get()), self.click_all.get(),
                                    self.any_size.get(), self.any_color.get(),
                                    self.move_time.get(), self.clicks_each.get(), self.events)
        self.worker.start()
        self.clicks = 0
        self.start_btn.configure(text="STOP")
        recolor(self.start_btn, RED, RED_HOVER)
        # get out of the way, and don't let the clicker see our own preview
        self.root.iconify()

    def set_status(self, title, sub):
        self.status.configure(text=title)
        self.substatus.configure(text=sub)

    def start_hotkey(self):
        try:
            from pynput import keyboard
        except ImportError:
            return

        def on_press(key):
            if getattr(key, "vk", None) == HOTKEY_VK or \
                    (getattr(key, "char", None) or "").lower() == HOTKEY:
                self.root.after(0, self.toggle)

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    def poll_events(self):
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.set_status(*data)
                self.root.title(f"{APP_NAME} - {data[0]}")
            elif kind == "click":
                self.clicks = data
                text = "Clicked 1 time" if data == 1 else f"Clicked {data} times"
                self.set_status(text, f"Press {HOTKEY.upper()} to stop")
                self.root.title(f"{APP_NAME} - {text}")
            elif kind == "error":
                if "FailSafe" in data:
                    self.events.put(("note", "Stopped: the mouse went into a screen corner."))
                else:
                    self.root.deiconify()
                    messagebox.showerror(APP_NAME, data)
            elif kind == "note":
                self.set_status("Stopped", data)
                continue
            elif kind == "stopped":
                self.worker = None
                self.start_btn.configure(text="START")
                recolor(self.start_btn, GREEN, GREEN_HOVER)
                done = {0: "Stopped", 1: "Stopped after 1 click"}.get(
                    self.clicks, f"Stopped after {self.clicks} clicks")
                self.set_status(done, "Press START to go again")
                self.root.title(APP_NAME)
                self.root.deiconify()
        self.root.after(100, self.poll_events)

    # --- settings -----------------------------------------------------------

    def load_settings(self):
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        example = resource("assets/example.png")
        example_copy = self.data / "pictures" / "example.png"
        if data is None:  # first run: start with the example picture
            if example.exists():
                shutil.copyfile(example, example_copy)
                self.pictures = [str(example_copy)]
            return
        self.pictures = [p for p in data.get("pictures", []) if Path(p).exists()]
        if str(example_copy) in self.pictures and example.exists():
            shutil.copyfile(example, example_copy)  # keep the example up to date
        if data.get("version", 1) >= SETTINGS_VERSION:  # older saves keep new defaults
            self.confidence.set(data.get("confidence", 70))
            self.interval.set(data.get("interval", 0.0))
            self.move_time.set(data.get("move_time", 0.05))
        self.click_all.set(data.get("click_all", False))
        self.any_size.set(data.get("any_size", True))
        self.any_color.set(data.get("any_color", True))
        self.clicks_each.set(data.get("clicks", 2))

    def save_settings(self):
        data = {
            "version": SETTINGS_VERSION,
            "pictures": self.pictures,
            "confidence": self.confidence.get(),
            "interval": round(self.interval.get(), 2),
            "click_all": self.click_all.get(),
            "any_size": self.any_size.get(),
            "any_color": self.any_color.get(),
            "move_time": round(self.move_time.get(), 2),
            "clicks": self.clicks_each.get(),
        }
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def close(self):
        if self.worker:
            self.worker.stop()
        self.save_settings()
        self.root.destroy()


def selftest():
    """Used by the build: check the packed app can see the screen, match and
    open its window. Writes OK or the error to the file in SELFTEST_OUT."""
    import traceback
    out = Path(os.environ.get("SELFTEST_OUT", "selftest.txt"))
    try:
        import pyautogui
        from pynput import keyboard  # noqa: F401
        click(pyautogui, 200, 150, clicks=0, move_time=0.2)  # glide only
        mouse_at = tuple(pyautogui.position())
        finder = Finder(*load_template(resource("assets/example.png")),
                        scales=SIZE_SCALES, shape=True)
        with open_screen() as sct:
            start = time.perf_counter()
            for _ in range(10):
                shot, _ = grab(sct, sct.monitors[0])
                finder.find(shot, 0.7)
            per_look = (time.perf_counter() - start) * 100  # ms per look
        root = tk.Tk()
        App(root)
        root.update()
        root.destroy()
        out.write_text(f"OK screen {shot.shape[1]}x{shot.shape[0]}, one look {per_look:.0f} ms, "
                       f"mouse glided to {mouse_at} (asked 200,150)", encoding="utf-8")
        return 0
    except Exception:
        out.write_text(traceback.format_exc(), encoding="utf-8")
        return 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    make_dpi_aware()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
