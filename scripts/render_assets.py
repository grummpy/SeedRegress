#!/usr/bin/env python3
"""Redraw the SeedRegress die icon and cover from the project artwork.

The die matches the cube in the lower-right of the cover: cyan edges, a magenta
top, and a teal side. Outputs are committed so a clone does not need this script.
"""

from __future__ import annotations

import math
import random
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DOCS = ROOT / "docs"
FONT = Path("/usr/share/fonts/truetype/macos/Inter-Bold.ttf")
FONT_MED = Path("/usr/share/fonts/truetype/macos/Inter-Medium.ttf")

BG = (8, 6, 18, 255)
LEFT = (62, 34, 122, 255)
RIGHT = (36, 198, 208, 255)
TOP = (188, 48, 188, 255)
EDGE = (186, 250, 255, 255)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def _die_points(cx, cy, w, h):
    top = (cx, cy - 2 * h)
    left = (cx - w, cy - h)
    right = (cx + w, cy - h)
    mid = (cx, cy)
    bottom = (cx, cy + 2 * h)
    lower_left = (cx - w, cy + h)
    lower_right = (cx + w, cy + h)
    return top, left, right, mid, bottom, lower_left, lower_right


def _pip(draw, center, radius, fill):
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def draw_die(base: Image.Image, cx: float, cy: float, scale: float) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    gdraw = ImageDraw.Draw(glow)
    w = scale
    h = scale * 0.56
    top, left, right, mid, bottom, lower_left, lower_right = _die_points(cx, cy, w, h)
    faces = [
        ([top, right, mid, left], TOP),
        ([left, mid, bottom, lower_left], LEFT),
        ([right, lower_right, bottom, mid], RIGHT),
    ]
    for polygon, _color in faces:
        gdraw.polygon(polygon, fill=(80, 220, 240, 90))
    blurred = glow.filter(ImageFilter.GaussianBlur(radius=max(2, int(scale * 0.08))))
    base.alpha_composite(blurred)
    for polygon, color in faces:
        draw.polygon(polygon, fill=color)
    width = max(2, int(scale * 0.035))
    for polygon, _color in faces:
        draw.line(polygon + [polygon[0]], fill=EDGE, width=width)

    def face_point(a, b, c, u, v):
        # u, v in 0..1 across a parallelogram a -> b, a -> c
        return (
            a[0] + (b[0] - a[0]) * u + (c[0] - a[0]) * v,
            a[1] + (b[1] - a[1]) * u + (c[1] - a[1]) * v,
        )

    radius = max(2, int(scale * 0.075))
    cyan = (186, 250, 255, 255)
    ink = (18, 24, 64, 255)
    # Top face, 5 pips.
    for u, v in ((0.28, 0.28), (0.72, 0.28), (0.5, 0.5), (0.28, 0.72), (0.72, 0.72)):
        _pip(draw, face_point(top, right, left, u, v), radius, cyan)
    # Left face, 3 pips.
    for u, v in ((0.32, 0.28), (0.5, 0.5), (0.68, 0.72)):
        _pip(draw, face_point(left, mid, lower_left, u, v), radius, cyan)
    # Right face, 2 pips.
    for u, v in ((0.38, 0.34), (0.66, 0.66)):
        _pip(draw, face_point(right, lower_right, mid, u, v), radius * 0.9, ink)
    base.alpha_composite(overlay)


def render_icon(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = int(size * 0.04)
    radius = int(size * 0.2)
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=radius,
        fill=BG,
        outline=(150, 90, 220, 255),
        width=max(2, size // 48),
    )
    draw_die(image, size * 0.50, size * 0.54, size * 0.30)
    return image


def _landscape(width: int, height: int, seed: int, mood: str) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    if mood == "diff":
        for y in range(height):
            for x in range(width):
                cell = int(18 * math.sin(x * 0.17 + seed) * math.sin(y * 0.21))
                crack = 1 if abs(math.sin(x * 0.35 + y * 0.08)) < 0.08 else 0
                pixels[x, y] = (
                    min(255, 90 + cell + crack * 140),
                    min(255, 10 + crack * 40),
                    min(255, 80 + cell // 2 + crack * 80),
                )
        return image
    top_color = (28, 18, 58) if mood == "dusk" else (18, 28, 52)
    bottom_color = (86, 40, 92) if mood == "dusk" else (24, 36, 64)
    for y in range(height):
        color = _lerp(top_color, bottom_color, y / max(1, height - 1))
        for x in range(width):
            pixels[x, y] = color
    draw = ImageDraw.Draw(image, "RGBA")
    # A pale moon, kept inside the sky.
    if mood == "dusk":
        moon = (int(width * 0.72), int(height * 0.22))
        r = max(4, width // 18)
        draw.ellipse((moon[0] - r, moon[1] - r, moon[0] + r, moon[1] + r), fill=(236, 220, 255, 180))
    layers = (
        ((70, 48, 96, 255), 0.50),
        ((48, 32, 78, 255), 0.62),
        ((28, 18, 48, 255), 0.76),
        ((14, 10, 28, 255), 0.90),
    )
    for index, (color, level) in enumerate(layers):
        points = [(0, height)]
        base = int(height * level)
        step = max(4, width // 28)
        for x in range(0, width + step, step):
            wave = math.sin(x * 0.045 + seed + index) * (height * 0.08)
            jitter = rng.randint(-3, 3)
            points.append((x, int(base + wave + jitter)))
        points.append((width, height))
        draw.polygon(points, fill=color)
    return image.convert("RGB")


def _thumb(image: Image.Image, box, highlight: bool = False) -> None:
    x, y, w, h = box
    radius = 14
    frame = ImageDraw.Draw(image, "RGBA")
    if highlight:
        for grow, alpha in ((8, 40), (5, 80), (2, 140)):
            frame.rounded_rectangle(
                (x - grow, y - grow, x + w + grow, y + h + grow),
                radius=radius + grow,
                outline=(255, 45, 170, alpha),
                width=3,
            )
        frame.rounded_rectangle((x, y, x + w, y + h), radius=radius, outline=(255, 70, 180, 255), width=3)


def _paste_thumb(canvas: Image.Image, picture: Image.Image, box, highlight: bool = False) -> None:
    x, y, w, h = box
    shot = picture.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=14, fill=255)
    canvas.paste(shot, (x, y), mask)
    _thumb(canvas, box, highlight)


def _gradient_text(canvas: Image.Image, text: str, font: ImageFont.FreeTypeFont, origin) -> None:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    width, height = right - left, bottom - top
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text((-left, -top), text, fill=255, font=font)
    gradient = Image.new("RGBA", (width, height))
    pixels = gradient.load()
    start, end = (124, 92, 255, 255), (255, 45, 154, 255)
    for x in range(width):
        color = _lerp(start, end, x / max(1, width - 1))
        for y in range(height):
            pixels[x, y] = color
    canvas.paste(gradient, origin, mask)


def render_cover(icon: Image.Image) -> Image.Image:
    width, height = 1920, 1080
    canvas = Image.new("RGBA", (width, height), BG)
    rng = random.Random(7)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(28):
        x = rng.randint(20, 420)
        y = rng.randint(20, 280)
        draw.ellipse((x, y, x + 4, y + 4), fill=(180, 150, 255, 90))
    nodes = [(rng.randint(30, 380), rng.randint(30, 240)) for _ in range(7)]
    for a, b in zip(nodes, nodes[1:], strict=False):
        draw.line((a, b), fill=(140, 100, 220, 70), width=2)
    nodes = [(rng.randint(1500, 1880), rng.randint(40, 260)) for _ in range(6)]
    for a, b in zip(nodes, nodes[1:], strict=False):
        draw.line((a, b), fill=(140, 100, 220, 60), width=2)
    canvas.alpha_composite(overlay)

    font = ImageFont.truetype(str(FONT), 118)
    label_font = ImageFont.truetype(str(FONT_MED), 36)
    title = "SeedRegress"
    probe = ImageDraw.Draw(canvas)
    bounds = probe.textbbox((0, 0), title, font=font)
    text_w = bounds[2] - bounds[0]
    _gradient_text(canvas, title, font, ((width - text_w) // 2, 48))
    line_y = 190
    ImageDraw.Draw(canvas).line((360, line_y, width - 360, line_y), fill=(180, 140, 255, 120), width=2)
    ImageDraw.Draw(canvas).ellipse((width // 2 - 8, line_y - 4, width // 2 + 8, line_y + 8), fill=(255, 80, 180, 220))

    thumb_w, thumb_h = 210, 140
    gap = 22
    count = 6
    row_w = count * thumb_w + (count - 1) * gap
    start = (width - row_w) // 2
    before_y = 280
    after_y = 620
    moods = ["dusk", "night", "dusk", "night", "dusk", "dusk"]
    for index in range(count):
        box = (start + index * (thumb_w + gap), before_y, thumb_w, thumb_h)
        picture = _landscape(thumb_w * 2, thumb_h * 2, 20 + index * 9, moods[index])
        _paste_thumb(canvas, picture, box, highlight=index == 5)
    # Arrow from the highlighted before frame down toward the diff frame.
    arrow_x = start + 3 * (thumb_w + gap) + thumb_w // 2
    arrow = ImageDraw.Draw(canvas)
    arrow.polygon(
        [
            (arrow_x - 16, 500),
            (arrow_x + 16, 500),
            (arrow_x + 16, 545),
            (arrow_x + 34, 545),
            (arrow_x, 590),
            (arrow_x - 34, 545),
            (arrow_x - 16, 545),
        ],
        fill=(255, 55, 170, 230),
    )
    for index in range(5):
        box = (start + index * (thumb_w + gap), after_y, thumb_w, thumb_h)
        mood = "diff" if index == 3 else ("night" if index % 2 else "dusk")
        picture = _landscape(thumb_w * 2, thumb_h * 2, 80 + index * 5, mood)
        _paste_thumb(canvas, picture, box, highlight=index == 3)
    icon_box = start + 5 * (thumb_w + gap)
    icon_size = 168
    icon_image = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
    canvas.alpha_composite(icon_image, (icon_box + (thumb_w - icon_size) // 2, after_y - 10))

    labels = ImageDraw.Draw(canvas)
    labels.rounded_rectangle((start, 224, start + 36, 260), radius=18, outline=(210, 190, 255, 255), width=3)
    labels.pieslice((start + 6, 230, start + 30, 254), 90, 270, fill=(210, 190, 255, 255))
    labels.text((start + 48, 222), "Before", font=label_font, fill=(220, 210, 255, 255))
    labels.ellipse((start, 564, start + 36, 600), outline=(80, 230, 240, 255), width=3)
    labels.ellipse((start + 12, 576, start + 24, 588), fill=(80, 230, 240, 255))
    labels.text((start + 48, 562), "After", font=label_font, fill=(120, 235, 245, 255))
    return canvas.convert("RGB")


def write_icns(master: Image.Image, dest: Path) -> None:
    chunks = {
        "icp4": 16,
        "icp5": 32,
        "icp6": 64,
        "ic07": 128,
        "ic08": 256,
        "ic09": 512,
        "ic10": 1024,
    }
    parts = []
    for ostype, size in chunks.items():
        buffer = __import__("io").BytesIO()
        master.resize((size, size), Image.Resampling.LANCZOS).save(buffer, format="PNG")
        blob = buffer.getvalue()
        parts.append(ostype.encode("ascii") + struct.pack(">I", 8 + len(blob)) + blob)
    body = b"".join(parts)
    dest.write_bytes(b"icns" + struct.pack(">I", 8 + len(body)) + body)


def write_svg(dest: Path) -> None:
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024"',
        ' role="img" aria-label="SeedRegress">',
        '  <rect x="40" y="40" width="944" height="944" rx="200"',
        ' fill="#080612" stroke="#9658dc" stroke-width="22"/>',
        '  <polygon points="512,250 790,390 512,530 234,390"',
        ' fill="#bc30bc" stroke="#bafaff" stroke-width="14" stroke-linejoin="round"/>',
        '  <polygon points="234,390 512,530 512,810 234,670"',
        ' fill="#3e227a" stroke="#bafaff" stroke-width="14" stroke-linejoin="round"/>',
        '  <polygon points="790,390 790,670 512,810 512,530"',
        ' fill="#24c6d0" stroke="#bafaff" stroke-width="14" stroke-linejoin="round"/>',
        '  <g fill="#bafaff">',
        '    <circle cx="400" cy="360" r="22"/>',
        '    <circle cx="624" cy="360" r="22"/>',
        '    <circle cx="512" cy="410" r="22"/>',
        '    <circle cx="400" cy="470" r="22"/>',
        '    <circle cx="624" cy="470" r="22"/>',
        '    <circle cx="330" cy="500" r="20"/>',
        '    <circle cx="400" cy="590" r="20"/>',
        '    <circle cx="470" cy="680" r="20"/>',
        "  </g>",
        '  <g fill="#121840">',
        '    <circle cx="650" cy="560" r="20"/>',
        '    <circle cx="730" cy="680" r="20"/>',
        "  </g>",
        "</svg>",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    master = render_icon(1024)
    master.save(ASSETS / "icon.png")
    master.resize((512, 512), Image.Resampling.LANCZOS).save(ASSETS / "icon-512.png")
    master.save(
        ASSETS / "icon.ico",
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    write_icns(master, ASSETS / "icon.icns")
    write_svg(ASSETS / "icon.svg")
    cover = render_cover(master)
    cover.save(DOCS / "cover.jpg", quality=90, optimize=True)
    print(f"wrote {ASSETS / 'icon.png'} and {DOCS / 'cover.jpg'}")


if __name__ == "__main__":
    main()
