"""Image-detect auto clicker.

Finds a reference image on screen with OpenCV template matching and clicks
its center, over and over, until you press the stop hotkey.

Run `python autoclicker.py --help` for all options.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MAX_CANDIDATES = 2000
COARSE_MIN_SIDE = 40  # templates this big get the fast half-size search
COARSE_SLACK = 0.15   # the rough pass accepts slightly worse scores
REFINE_MARGIN = 6     # pixels around a rough hit to search again at full size
MAX_REFINE = 50
MIN_EDGE_RATIO = 0.3  # shape mode: skip spots with far fewer edges than the picture


# ---------------------------------------------------------------------------
# Matching (pure functions, no screen or mouse needed)
# ---------------------------------------------------------------------------

def load_template(path, grayscale=False):
    """Load a reference image. Transparent pixels become a match mask."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    mask = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if alpha.min() < 255:
            mask = cv2.merge([alpha, alpha, alpha]) if not grayscale else alpha
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, mask


def shape_map(image):
    """Edge strength of an image. A white, pink or purple version of the same
    icon gives (almost) the same map, so matching it ignores colors."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def find_matches(screen, template, confidence, scales=(1.0,), mask=None,
                 find_all=False, shape=False):
    """Return [(x, y, score), ...] centers of matches in `screen`, best first
    (only the single best one unless `find_all`).

    `screen` and `template` must both be BGR or both grayscale. Each scale
    resizes the template, so the image still matches when the game is shown
    a bit bigger or smaller than when you took the screenshot.

    With `shape`, outlines are compared instead of colors, so the picture is
    still found when the game shows it in another color.

    Big templates are first searched on a half-size copy of the screen (about
    4x faster), and each rough hit is then checked again at full size.
    """
    prep = shape_map if shape else (lambda image: image)
    if shape and mask is not None and mask.ndim == 3:
        mask = mask[:, :, 0]
    screen_f = prep(screen)

    matches = []
    half_screen = None
    for scale in scales:
        tpl, tpl_mask = _resize(template, mask, scale)
        th, tw = tpl.shape[:2]
        if th < 4 or tw < 4 or th > screen.shape[0] or tw > screen.shape[1]:
            continue
        tpl_f = prep(tpl)

        if min(th, tw) < COARSE_MIN_SIDE:
            matches += _match(screen_f, tpl_f, tpl_mask, confidence, find_all, shape)
            continue

        if half_screen is None:
            half_screen = prep(cv2.resize(screen, None, fx=0.5, fy=0.5,
                                          interpolation=cv2.INTER_AREA))
        half_tpl, half_mask = _resize(tpl, tpl_mask, 0.5)
        rough = _suppress_overlaps(_match(half_screen, prep(half_tpl), half_mask,
                                          confidence - COARSE_SLACK, True, shape))
        for cx, cy, _, _, _ in rough[:MAX_REFINE if find_all else 5]:
            x0 = max(0, cx * 2 - tw // 2 - REFINE_MARGIN)
            y0 = max(0, cy * 2 - th // 2 - REFINE_MARGIN)
            window = screen_f[y0:y0 + th + 2 * REFINE_MARGIN, x0:x0 + tw + 2 * REFINE_MARGIN]
            if window.shape[0] < th or window.shape[1] < tw:
                continue
            for x, y, score, _, _ in _match(window, tpl_f, tpl_mask, confidence, False, shape):
                matches.append((x + x0, y + y0, score, tw, th))

    matches.sort(key=lambda m: m[2], reverse=True)
    if not find_all:
        matches = matches[:1]
    return [(cx, cy, score) for cx, cy, score, _, _ in _suppress_overlaps(matches)]


def _resize(template, mask, scale):
    if scale == 1.0:
        return template, mask
    tpl = cv2.resize(template, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, (tpl.shape[1], tpl.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return tpl, mask


def _match(screen, tpl, mask, threshold, find_all, shape=False):
    """Raw matches [(cx, cy, score, tw, th), ...] of one template size."""
    th, tw = tpl.shape[:2]
    result = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result[result > 1.001] = 0  # flat spots can give nonsense scores
    if shape:
        # A plain area has almost no edges; its score means nothing.
        rh, rw = result.shape
        local = cv2.boxFilter(screen, -1, (tw, th), anchor=(0, 0),
                              borderType=cv2.BORDER_CONSTANT)[:rh, :rw]
        result[local < MIN_EDGE_RATIO * float(tpl.mean())] = 0

    if not find_all:
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        return [(x + tw // 2, y + th // 2, float(score), tw, th)] if score >= threshold else []

    ys, xs = np.where(result >= threshold)
    if len(xs) > MAX_CANDIDATES:  # threshold far too low, keep the best
        best = np.argsort(result[ys, xs])[-MAX_CANDIDATES:]
        ys, xs = ys[best], xs[best]
    found = [(int(x) + tw // 2, int(y) + th // 2, float(result[y, x]), tw, th)
             for x, y in zip(xs, ys)]
    found.sort(key=lambda m: m[2], reverse=True)
    return found


def _suppress_overlaps(matches):
    """Keep the best match in each spot, drop the near-duplicates around it.
    `matches` must be sorted best first."""
    kept = []
    for cx, cy, score, tw, th in matches:
        if all(abs(cx - kx) > max(tw, kw) / 2 or abs(cy - ky) > max(th, kh) / 2
               for kx, ky, _, kw, kh in kept):
            kept.append((cx, cy, score, tw, th))
    return kept


class Finder:
    """Fast repeated search for one picture, for the live click loop.

    Everything about the picture is prepared once. Each screen is then
    searched in two steps: a quick pass on a half-size copy finds the few
    places that could be it, and only those spots are checked at full size,
    at every size in `scales`. A full-HD screen takes about 10-20 ms.
    """

    def __init__(self, template, mask=None, scales=(1.0,), shape=False, coarse_shape=None):
        self.shape = shape
        self.prep = shape_map if shape else (lambda image: image)
        if shape and mask is not None and mask.ndim == 3:
            mask = mask[:, :, 0]
        self.levels = []  # (template features, mask, w, h) for each size
        for scale in sorted(set(scales)):
            tpl, tpl_mask = _resize(template, mask, scale)
            if min(tpl.shape[:2]) >= 4:
                self.levels.append((self.prep(tpl), tpl_mask, tpl.shape[1], tpl.shape[0]))
        if not self.levels:
            raise ValueError("picture is too small")
        self.max_w = max(level[2] for level in self.levels)
        self.max_h = max(level[3] for level in self.levels)

        # The quick pass: one size, half resolution, no mask (masks are slow).
        self.half = min(template.shape[:2]) >= 24
        f = 0.5 if self.half else 1.0
        coarse, coarse_mask = _resize(template, mask, f)
        coarse = self.prep(coarse)
        if coarse_mask is not None:  # blank out what the mask hides
            m = coarse_mask if coarse_mask.ndim == coarse.ndim else coarse_mask[..., None]
            coarse = (coarse * (m > 0)).astype(coarse.dtype)
        self.coarse = coarse
        self.factor = f

    def find(self, screen, confidence, find_all=False, max_spots=None):
        """[(x, y, score), ...] like find_matches, best first."""
        f = self.factor
        small = screen if f == 1.0 else cv2.resize(screen, None, fx=f, fy=f,
                                                   interpolation=cv2.INTER_AREA)
        small = self.prep(small)
        ch, cw = self.coarse.shape[:2]
        if ch > small.shape[0] or cw > small.shape[1]:
            return []
        rough = cv2.matchTemplate(small, self.coarse, cv2.TM_CCOEFF_NORMED)
        rough = np.nan_to_num(rough, nan=0.0, posinf=0.0, neginf=0.0)
        rough[rough > 1.001] = 0
        if self.shape:
            rh, rw = rough.shape
            local = cv2.boxFilter(small, -1, (cw, ch), anchor=(0, 0),
                                  borderType=cv2.BORDER_CONSTANT)[:rh, :rw]
            rough[local < MIN_EDGE_RATIO * float(self.coarse.mean())] = 0

        spots = max_spots or (20 if find_all else 4)
        floor = max(0.2, confidence - 0.4)  # the quick pass is only a hint
        matches = []
        for _ in range(spots):
            _, score, _, (x, y) = cv2.minMaxLoc(rough)
            if score < floor:
                break
            # blank this spot so the next round finds the next one
            rough[max(0, y - ch // 2):y + ch // 2 + 1, max(0, x - cw // 2):x + cw // 2 + 1] = 0
            cx, cy = (x + cw / 2) / f, (y + ch / 2) / f
            best = self._refine(screen, cx, cy)
            if best and best[2] >= confidence:
                matches.append(best)
                if not find_all:
                    break
        matches.sort(key=lambda m: m[2], reverse=True)
        return [(cx, cy, score) for cx, cy, score, _, _ in _suppress_overlaps(matches)]

    def _refine(self, screen, cx, cy):
        """Best full-size match near (cx, cy), trying every size."""
        pad = REFINE_MARGIN + int(4 / self.factor)
        x0 = max(0, int(cx - self.max_w / 2) - pad)
        y0 = max(0, int(cy - self.max_h / 2) - pad)
        x1 = min(screen.shape[1], int(cx + self.max_w / 2) + pad + 1)
        y1 = min(screen.shape[0], int(cy + self.max_h / 2) + pad + 1)
        window = self.prep(screen[y0:y1, x0:x1])
        best = None
        for tpl, mask, w, h in self.levels:
            if h > window.shape[0] or w > window.shape[1]:
                continue
            for x, y, score, _, _ in _match(window, tpl, mask, -1.0, False, self.shape):
                if best is None or score > best[2]:
                    best = (x + x0, y + y0, score, w, h)
        return best


def parse_scales(text):
    """'0.8,1,1.25' -> (0.8, 1.0, 1.25)"""
    scales = tuple(float(s) for s in text.split(",") if s.strip())
    if not scales or any(s <= 0 for s in scales):
        raise argparse.ArgumentTypeError("scales must be positive numbers, e.g. 0.9,1,1.1")
    return scales


def parse_region(text):
    """'x,y,width,height' in screen pixels."""
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("region must be x,y,width,height")
    return dict(zip(("left", "top", "width", "height"), parts))


# ---------------------------------------------------------------------------
# Screen, mouse and hotkeys
# ---------------------------------------------------------------------------

class Controls:
    """Stop / pause hotkeys, read from a background keyboard listener."""

    def __init__(self, stop_key, pause_key):
        self.stop = threading.Event()
        self.paused = False
        self._listener = None
        try:
            from pynput import keyboard
        except ImportError:
            print("pynput is not installed: hotkeys are off, use Ctrl+C to stop.")
            return

        def key_name(key):
            return getattr(key, "name", None) or getattr(key, "char", None)

        def on_press(key):
            name = (key_name(key) or "").lower()
            if name == stop_key:
                print(f"\n[{stop_key.upper()}] stopping.")
                self.stop.set()
                return False
            if name == pause_key:
                self.paused = not self.paused
                print(f"\n[{pause_key.upper()}] {'paused' if self.paused else 'resumed'}.")

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def wait(self, seconds):
        """Sleep, but wake up right away when the stop key is pressed."""
        return self.stop.wait(seconds)


def open_screen():
    """Screen grabber (newer mss renamed mss.mss to mss.MSS)."""
    import mss
    return mss.MSS() if hasattr(mss, "MSS") else mss.mss()


def grab(sct, area):
    """Screenshot `area` as BGR, plus the factor from image pixels to mouse
    coordinates (Retina / HiDPI screens capture more pixels than points)."""
    shot = np.asarray(sct.grab(area))
    bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    factor = area["width"] / bgr.shape[1]
    return bgr, factor


CLICK_GAP = 0.03   # seconds between the clicks of one hit
CLICK_HOLD = 0.02  # how long the button stays down per click


def _ease(t):
    return 1 - (1 - t) ** 3  # fast start, gentle stop, like a hand


def click(pyautogui, x, y, button="left", clicks=2, move_time=0.05):
    """Glide the mouse to (x, y) over `move_time` seconds, then click.

    Games such as Roblox ignore a cursor that teleports, so the mouse travels
    there in many small steps first, then each click holds the button for a
    moment.
    """
    check = getattr(pyautogui, "failSafeCheck", None)
    if check and pyautogui.FAILSAFE:
        check()  # mouse pushed into a screen corner = emergency stop

    if sys.platform == "win32":
        mouse = _WinMouse()
    else:
        mouse = _PyAutoGuiMouse(pyautogui)

    sx, sy = mouse.position()
    start = time.perf_counter()
    while move_time > 0:
        t = (time.perf_counter() - start) / move_time
        if t >= 1:
            break
        k = _ease(t)
        mouse.move(round(sx + (x - sx) * k), round(sy + (y - sy) * k))
        time.sleep(0.008)
    mouse.move(x, y)
    time.sleep(0.005)

    for i in range(clicks):
        if i:
            time.sleep(CLICK_GAP)
        mouse.down(button)
        time.sleep(CLICK_HOLD)
        mouse.up(button)


class _PyAutoGuiMouse:
    def __init__(self, pyautogui):
        self.p = pyautogui

    def position(self):
        return tuple(self.p.position())

    def move(self, x, y):
        self.p.moveTo(x, y, _pause=False)

    def down(self, button):
        self.p.mouseDown(button=button, _pause=False)

    def up(self, button):
        self.p.mouseUp(button=button, _pause=False)


class _WinMouse:
    """Real mouse input through SendInput, which games read like a physical
    mouse (SetCursorPos-style jumps are often ignored)."""

    MOVE, ABSOLUTE, VIRTUALDESK = 0x0001, 0x8000, 0x4000
    BUTTONS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010),
               "middle": (0x0020, 0x0040)}

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]

        self.ctypes, self.wintypes = ctypes, wintypes
        self.MOUSEINPUT, self.INPUT = MOUSEINPUT, INPUT
        self.user32 = ctypes.windll.user32
        metric = self.user32.GetSystemMetrics
        self.vx, self.vy = metric(76), metric(77)  # virtual desktop origin
        self.vw, self.vh = max(2, metric(78)), max(2, metric(79))

    def _send(self, flags, dx=0, dy=0):
        inp = self.INPUT(0, self.MOUSEINPUT(dx, dy, 0, flags, 0, 0))
        self.user32.SendInput(1, self.ctypes.byref(inp), self.ctypes.sizeof(inp))

    def position(self):
        pt = self.wintypes.POINT()
        self.user32.GetCursorPos(self.ctypes.byref(pt))
        return pt.x, pt.y

    def move(self, x, y):
        nx = round((x - self.vx) * 65535 / (self.vw - 1))
        ny = round((y - self.vy) * 65535 / (self.vh - 1))
        self._send(self.MOVE | self.ABSOLUTE | self.VIRTUALDESK, nx, ny)

    def down(self, button):
        self._send(self.BUTTONS[button][0])

    def up(self, button):
        self._send(self.BUTTONS[button][1])


def run(args):
    import pyautogui

    pyautogui.PAUSE = 0.02
    pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort

    templates = []
    for path in args.images:
        tpl, mask = load_template(path, grayscale=args.grayscale)
        finder = Finder(tpl, mask, args.scales, shape=args.any_color)
        templates.append((Path(path).name, finder))

    controls = Controls(args.stop_key.lower(), args.pause_key.lower())

    with open_screen() as sct:
        if args.region:
            area = args.region
        else:
            if args.monitor >= len(sct.monitors):
                sys.exit(f"Monitor {args.monitor} does not exist "
                         f"(found {len(sct.monitors) - 1}).")
            area = sct.monitors[args.monitor]

        names = ", ".join(name for name, _ in templates)
        print(f"Looking for {names} (confidence {args.confidence}, "
              f"every {args.interval}s).")
        print(f"Press {args.stop_key.upper()} to stop, "
              f"{args.pause_key.upper()} to pause/resume.")
        if args.dry_run:
            print("Dry run: matches are printed, nothing is clicked.")

        if controls.wait(args.start_delay):
            return

        total = 0
        last_seen = time.monotonic()
        while not controls.stop.is_set():
            if controls.paused:
                controls.wait(0.2)
                continue

            screen, factor = grab(sct, area)
            if args.grayscale:
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

            clicked = False
            for name, finder in templates:
                matches = finder.find(screen, args.confidence, args.all)
                if not args.all:
                    matches = matches[:1]
                for cx, cy, score in matches:
                    x = area["left"] + round(cx * factor)
                    y = area["top"] + round(cy * factor)
                    total += 1
                    action = "found" if args.dry_run else "click"
                    print(f"{action} #{total}: {name} at ({x}, {y}) score {score:.2f}")
                    if not args.dry_run:
                        click(pyautogui, x, y, args.button, args.clicks, args.move_time)
                    clicked = True
                    if args.max_clicks and total >= args.max_clicks:
                        print(f"Reached {args.max_clicks} clicks, stopping.")
                        return
                if clicked and args.first_only:
                    break

            if clicked:
                last_seen = time.monotonic()
            elif args.timeout and time.monotonic() - last_seen > args.timeout:
                print(f"Nothing found for {args.timeout}s, stopping.")
                return

            controls.wait(args.interval)


def build_parser():
    p = argparse.ArgumentParser(
        description="Find a picture on screen and click it automatically.")
    p.add_argument("images", nargs="+",
                   help="reference image(s) to look for (PNG/JPG, cropped tightly)")
    p.add_argument("-c", "--confidence", type=float, default=0.85,
                   help="match threshold 0..1, higher is stricter (default 0.85)")
    p.add_argument("-i", "--interval", type=float, default=0.0,
                   help="pause between screen checks in seconds (default 0 = nonstop)")
    p.add_argument("--stop-key", default="f8",
                   help="hotkey that stops the clicker (default F8)")
    p.add_argument("--pause-key", default="f7",
                   help="hotkey that pauses/resumes (default F7)")
    p.add_argument("--all", action="store_true",
                   help="click every match on screen, not just the best one")
    p.add_argument("--first-only", action="store_true",
                   help="with several images, stop at the first one found each round")
    p.add_argument("--scales", type=parse_scales, default=(1.0,),
                   help="template sizes to try, e.g. 0.8,0.9,1,1.1,1.25 (default 1)")
    p.add_argument("--any-color", action="store_true",
                   help="match the outline, so it is found in any color")
    p.add_argument("--grayscale", action="store_true",
                   help="ignore colors (faster, but can confuse similar shapes)")
    p.add_argument("--region", type=parse_region,
                   help="only search x,y,width,height of the screen (faster)")
    p.add_argument("--monitor", type=int, default=1,
                   help="monitor to search: 1 = main, 2 = second..., 0 = all (default 1)")
    p.add_argument("--button", choices=("left", "right", "middle"), default="left")
    p.add_argument("--clicks", type=int, default=2,
                   help="clicks per hit (default 2)")
    p.add_argument("--move-time", type=float, default=0.05,
                   help="seconds the mouse takes to glide to the target (default 0.05)")
    p.add_argument("--max-clicks", type=int, default=0,
                   help="stop after this many clicks (default 0 = never)")
    p.add_argument("--timeout", type=float, default=0,
                   help="stop when nothing is found for this many seconds (default 0 = never)")
    p.add_argument("--start-delay", type=float, default=3,
                   help="seconds to wait before starting, to switch windows (default 3)")
    p.add_argument("--dry-run", action="store_true",
                   help="print matches without clicking, to tune --confidence")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 < args.confidence <= 1:
        sys.exit("--confidence must be between 0 and 1")
    if args.interval < 0:
        sys.exit("--interval can't be negative")
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
