"""Self-contained HTML report with before, after, and diff side by side."""

from __future__ import annotations

import html
import shutil
from pathlib import Path


def write_report(
    run_dir: Path,
    suite_name: str,
    cases: list,
    *,
    host: str,
    comfyui_version: str | None,
) -> Path:
    report_dir = run_dir / "report"
    image_dir = report_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for case in cases:
        cards.append(_card(case, image_dir))
    version = html.escape(comfyui_version or "unknown")
    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SeedRegress — {html.escape(suite_name)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07060f;
      --card: #141228;
      --line: rgba(168, 85, 247, 0.35);
      --text: #f4f0ff;
      --muted: #b7aecf;
      --pink: #ff3d9a;
      --cyan: #22d3ee;
      --violet: #a78bfa;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", Helvetica, sans-serif;
      background:
        radial-gradient(ellipse at 0% 0%, rgba(124, 58, 237, 0.28), transparent 42%),
        radial-gradient(ellipse at 100% 100%, rgba(255, 61, 154, 0.16), transparent 40%),
        var(--bg);
      color: var(--text);
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
    h1 {{
      margin: 0;
      font-size: 42px;
      background: linear-gradient(90deg, #8b6cff, #ff3d9a);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }}
    .meta {{ color: var(--muted); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-top: 18px;
    }}
    .row {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .pill {{
      border-radius: 999px;
      padding: 4px 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 12px;
    }}
    .pass {{ background: rgba(34, 211, 238, 0.18); color: var(--cyan); }}
    .drift {{ background: rgba(244, 114, 182, 0.18); color: #f9a8d4; }}
    .fail, .error {{ background: rgba(255, 61, 154, 0.2); color: #fb7185; }}
    .baseline {{ background: rgba(167, 139, 250, 0.2); color: var(--violet); }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 14px;
    }}
    figure {{ margin: 0; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 12px;
      background: #0c0a16;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }}
    ul {{ margin: 8px 0 0; padding-left: 18px; color: var(--muted); }}
    @media (max-width: 800px) {{
      .frames {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>SeedRegress</h1>
    <p class="meta">{html.escape(suite_name)} · ComfyUI {version} · host {html.escape(host or "unset")}</p>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    path = report_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def _card(case, image_dir: Path) -> str:
    verdict = html.escape(case.verdict)
    bits = []
    if case.phash is not None:
        bits.append(f"pHash {case.phash}")
    if case.ssim is not None:
        bits.append(f"SSIM {case.ssim:.4f}")
    if case.lpips is not None:
        bits.append(f"LPIPS {case.lpips:.4f}")
    if case.max_abs_diff is not None:
        bits.append(f"max pixel diff {case.max_abs_diff:.1f}")
    metrics = " · ".join(bits) or "No comparison metrics"
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in case.reasons + case.changes)
    frames = (
        _figure("Before", _copy_image(case.before_path, image_dir, f"{case.case_id}-before.png"))
        + _figure("After", _copy_image(case.after_path, image_dir, f"{case.case_id}-after.png"))
        + _figure("Diff", _copy_image(case.diff_path, image_dir, f"{case.case_id}-diff.png"))
    )
    error = f"<p>{html.escape(case.error)}</p>" if case.error else ""
    return f"""
    <section class="card">
      <div class="row">
        <h2>{html.escape(case.case_id)}</h2>
        <span class="pill {verdict}">{verdict}</span>
      </div>
      <p class="meta">{html.escape(metrics)}</p>
      {error}
      <div class="frames">{frames}</div>
      <ul>{reasons}</ul>
    </section>
    """


def _copy_image(source: str | None, image_dir: Path, name: str) -> str | None:
    if not source:
        return None
    src = Path(source)
    if not src.is_file():
        return None
    dest = image_dir / name
    shutil.copyfile(src, dest)
    return f"images/{name}"


def _figure(caption: str, src: str | None) -> str:
    if not src:
        body = "<div class=\"meta\">No image</div>"
    else:
        body = f'<img src="{html.escape(src)}" alt="{html.escape(caption)}">'
    return f"<figure><figcaption>{html.escape(caption)}</figcaption>{body}</figure>"
