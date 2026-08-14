"""Render an asciicast v2 file to an animated GIF — pure Python (Pillow), no agg/Docker.

Usage:
    python scripts/cast_to_gif.py trustgate.cast docs/trustgate.gif [font.ttf]

A minimal terminal emulator: it processes the cast's output events (handling \\r, \\n, line
wrap, and basic SGR colors), samples the screen on a fixed time grid, and writes the frames
as a palette-optimized GIF. Good enough for a README demo without a Rust toolchain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Tokyo-Night-ish palette.
BG = (26, 27, 38)
DEFAULT_FG = (192, 202, 245)
SGR = {
    30: (86, 95, 137), 31: (247, 118, 142), 32: (158, 206, 106), 33: (224, 175, 104),
    34: (122, 162, 247), 35: (187, 154, 247), 36: (125, 207, 255), 37: (192, 202, 245),
    90: (86, 95, 137), 91: (247, 118, 142), 92: (158, 206, 106), 93: (224, 175, 104),
    94: (122, 162, 247), 95: (187, 154, 247), 96: (125, 207, 255), 97: (255, 255, 255),
}

WIDTH, HEIGHT = 100, 30       # terminal columns / visible rows
FONT_SIZE = 15
SAMPLE_DT = 0.35              # seconds between rendered frames
PAD = 12


class Term:
    def __init__(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self.buf: list[list[tuple[str, tuple[int, int, int]]]] = [[]]
        self.fg = DEFAULT_FG

    def _newline(self) -> None:
        self.buf.append([])

    def _sgr(self, params: str) -> None:
        for p in params.split(";"):
            if p in ("", "0"):
                self.fg = DEFAULT_FG
            elif p == "1":
                continue                       # bold: keep color
            else:
                try:
                    self.fg = SGR.get(int(p), self.fg)
                except ValueError:
                    pass

    def feed(self, text: str) -> None:
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "\x1b" and i + 1 < n and text[i + 1] == "[":
                j = i + 2
                while j < n and not text[j].isalpha():
                    j += 1
                if j < n and text[j] == "m":
                    self._sgr(text[i + 2:j])
                i = j + 1
                continue
            if ch == "\r":
                pass                           # CR: cursor to col 0; \r\n is handled by the \n
            elif ch == "\n":
                self._newline()
            elif ch == "\t":
                for _ in range(4):
                    self.buf[-1].append((" ", self.fg))
            else:
                self.buf[-1].append((ch, self.fg))
                if len(self.buf[-1]) >= self.cols:
                    self._newline()
            i += 1

    def viewport(self) -> list[list[tuple[str, tuple[int, int, int]]]]:
        return self.buf[-self.rows:]


def render_frame(vp, font, cw, chh) -> Image.Image:
    img = Image.new("RGB", (WIDTH * cw + 2 * PAD, HEIGHT * chh + 2 * PAD), BG)
    d = ImageDraw.Draw(img)
    for r, line in enumerate(vp):
        y = PAD + r * chh
        for c, (ch, color) in enumerate(line):
            if ch != " ":
                d.text((PAD + c * cw, y), ch, font=font, fill=color)
    return img


def main() -> None:
    cast = Path(sys.argv[1] if len(sys.argv) > 1 else "trustgate.cast")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/trustgate.gif")
    font_path = (sys.argv[3] if len(sys.argv) > 3
                 else "../agg-1.9.0/fonts/JetBrainsMono-Regular.ttf")
    font = ImageFont.truetype(font_path, FONT_SIZE)
    bbox = font.getbbox("M")
    cw, chh = bbox[2] - bbox[0] + 1, FONT_SIZE + 4

    lines = Path(cast).read_text(encoding="utf-8").splitlines()
    events = [json.loads(l) for l in lines[1:] if l.strip()]

    term = Term(WIDTH, HEIGHT)
    frames: list[Image.Image] = []
    durations: list[int] = []
    next_sample = 0.0
    last_t = 0.0
    for t, kind, text in events:
        if kind == "o":
            term.feed(text)
        while t >= next_sample:
            frames.append(render_frame(term.viewport(), font, cw, chh))
            durations.append(int(SAMPLE_DT * 1000))
            next_sample += SAMPLE_DT
        last_t = t
    frames.append(render_frame(term.viewport(), font, cw, chh))
    durations.append(2000)     # hold the final frame

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print(f"Wrote {out}  ({len(frames)} frames, ~{last_t:.0f}s, "
          f"{out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
