# PhotoClicker
Simple Windows auto clicker that finds a picture on your screen (in any color) and clicks it.

**[⬇ Download PhotoClicker.exe](https://github.com/zqh7y/PhotoClicker/releases/download/latest/PhotoClicker.exe)** (Windows)

Give it a picture of a button (or anything else), and it keeps looking for
that picture on your screen and clicks its center whenever it shows up.

## Easy way: PhotoClicker.exe (Windows)

1. Download and open **PhotoClicker.exe**. If Windows says "Windows protected your PC",
   click **More info**, then **Run anyway** (the app isn't signed, that's all).
2. **Step 1:** press **Take from screen**, then drag a box around the button
   you want clicked. You can add more than one picture.
3. **Step 2:** press **START** (or **P**) and open your game. The window
   hides itself and watches the screen nonstop. When the picture shows up it
   glides the mouse there (0.05 s) and clicks it twice, about 0.1-0.15 s
   after it appeared. Games like Roblox ignore a mouse that just teleports,
   which is why it glides.
4. Press **P** to stop. Pushing the mouse into a screen corner also stops it.

**Click one spot (regular auto clicker):** press **Click one spot** at the
top, then **Add a spot** and click anywhere on the screen. Add more spots
and it clicks them one after another. Set the speed with "Clicks per
second", then press **START** (or **P**). **Clear** removes the spots.

By default it matches the picture's shape, not its colors, so a button
that changes color (pink, white, on a grey or purple background...) is
still found. Untick "Find it in any color" in **Settings** to match colors
exactly. If it clicks wrong things or misses the picture, move "How exact
must it match?" there too. Pictures and settings are kept for next time
(in `%APPDATA%\ImageAutoClicker`).

The .exe is built and tested on Windows by the "Build PhotoClicker.exe"
GitHub Action on every push to master, then published to the Releases page. To run from
source instead: `pip install -r requirements.txt`, then `python gui.py`.

## Command line

Needs Python 3.9+.

```bash
pip install -r requirements.txt
```

macOS: allow your terminal under System Settings → Privacy & Security →
**Screen Recording** and **Accessibility**, or it can't see the screen or click.

### 1. Make the reference image

Take a screenshot (Windows: `Win+Shift+S`, macOS: `Cmd+Shift+4`) and crop it
tightly around the thing to click, for example `hatch.png`. Take it at the
same window size / zoom you'll play at. A PNG with a transparent background
is fine: transparent pixels are ignored while matching.

### 2. Run it

```bash
python autoclicker.py hatch.png
```

You get 3 seconds to switch to the game, then it checks the screen every
0.5 s and clicks when it finds the image.

- **F8** stops it, **F7** pauses / resumes.
- Emergency stop: move the mouse into any screen corner (or Ctrl+C in the
  terminal).

### Tuning

First check what it sees without clicking:

```bash
python autoclicker.py hatch.png --dry-run
```

Each line shows the match score. If it clicks the wrong things, raise
`--confidence`; if it misses the button, lower it.

| Option | Default | What it does |
|---|---|---|
| `-c`, `--confidence` | 0.85 | Match threshold 0–1, higher is stricter |
| `-i`, `--interval` | 0 | Pause between screen checks (0 = nonstop) |
| `--stop-key` / `--pause-key` | f8 / f7 | Hotkeys (e.g. `f6`, `esc`, `q`) |
| `--all` | off | Click every copy on screen, not just the best one |
| `--scales` | 1 | Template sizes to try, e.g. `0.8,0.9,1,1.1,1.25` if the window size changes |
| `--any-color` | off | Match the outline, so the picture is found in any color |
| `--grayscale` | off | Ignore colors (faster) |
| `--region` | whole screen | Only search `x,y,width,height` (faster) |
| `--monitor` | 1 | 1 = main screen, 2 = second, 0 = all screens |
| `--button` / `--clicks` | left / 2 | Mouse button, and clicks per hit |
| `--move-time` | 0.05 | Seconds the mouse takes to glide to the target |
| `--max-clicks` | 0 (never) | Stop after this many clicks |
| `--timeout` | 0 (never) | Stop when nothing was found for this many seconds |
| `--start-delay` | 3 | Seconds before it starts |
| `--dry-run` | off | Print matches, don't click |

Several images at once (each round it clicks every one it finds; add
`--first-only` to click only the first one found):

```bash
python autoclicker.py hatch.png claim.png ok.png -c 0.8 -i 1
```

## Tests

```bash
pip install pytest
python -m pytest
```

The tests cover the image matching on generated pictures; they don't need a
screen.
