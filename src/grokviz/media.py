"""Phase `render_media`: record short clips of the website's animations for the README and the paper.

Local explanation: serves site/ locally, opens the dashboard in headless Chromium with `?still=1` (no autoplay, no tweening,
no enlarge buttons: web/js/core/motion.js), and for every frame of a clip either sets the page's training step through
its state module (stored frames only, never blended) or moves a slider, waits until every figure has finished drawing
(`window.grokviz.anim.settled()`), and screenshots one figure. ffmpeg then turns the frames into an MP4 (H.264),
a palette-optimised GIF and a 5-frame film strip.
Global context: the clips show exactly what readers see on the site (one implementation); GIFs go to
docs/images/clips/ for the README (size-capped by tests/test_media.py), MP4s to reports/media/, film strips to
reports/figures/ for the paper. Existing outputs are kept unless --force.
Sources: Playwright screenshots (https://playwright.dev/python/docs/screenshots); ffmpeg two-pass GIF palette
(https://ffmpeg.org/ffmpeg-filters.html#palettegen, https://ffmpeg.org/ffmpeg-filters.html#paletteuse,
https://blog.pkh.me/p/21-high-quality-gif-with-ffmpeg.html); tile filter (https://ffmpeg.org/ffmpeg-filters.html#tile);
H.264 for the web, yuv420p and faststart (https://trac.ffmpeg.org/wiki/Encode/H.264); imageio-ffmpeg's bundled binary
(https://github.com/imageio/imageio-ffmpeg).
"""

from __future__ import annotations  # postponed annotations

import logging  # module logger
import shutil  # ffmpeg on PATH, clean frame folders
import struct  # frame size from the PNG header
import subprocess  # run ffmpeg
from dataclasses import dataclass  # clip specification
from pathlib import Path  # paths

from playwright.sync_api import sync_playwright  # headless browser

from grokviz.config import REPO_ROOT  # output folders
from grokviz.context import PipelineContext  # phase argument
from grokviz.figures import open_figure, serve  # dashboard in capture mode; quiet local server of site/

log = logging.getLogger("grokviz.media")
FPS = 10  # frames per second of every clip
GIF_WIDTH = 640  # px; keeps the README GIFs within their size budget


@dataclass(frozen=True)
class Clip:
    """One clip: the dashboard figure to film, the initial URL-hash state, and what changes from frame to frame —
    `steps` = range(start, stop, step) of training steps, or `slider` = (CSS selector, start, stop, step)."""

    name: str
    selector: str
    state: str
    steps: tuple[int, int, int] | None = None
    slider: tuple[str, int, int, int] | None = None


CLIPS = [  # the paper's run; a frame per 200 steps = one stored geometry frame (src/grokviz/geometry.py)
    Clip("grokking-one-view", "#fig-hero-grokking", "run=paper", steps=(1, 15402, 200)),
    Clip("circles-emerge", "#fig-embed-circles", "run=paper", steps=(1, 15402, 200)),
    Clip("spectrum-race", "#fig-freq-norm-race", "run=paper", steps=(1, 15402, 200)),
    Clip("torus-grokking", "#fig-torus-3d", "run=paper", steps=(1, 15402, 200)),
    Clip("interference-sweep", "#fig-interference", "run=paper&a=100&b=50",  # odd c: 37 = a + b is a frame
         slider=("#fig-interference .interf-ctl input[type=range]", 1, 113, 2)),
]

SET_STEP = """async (s) => { // the page's own state module (same URL, so the same instance as the figures use)
  const state = await import(new URL("assets/js/core/state.js", document.baseURI).href);
  state.set({ step: s });
  await window.grokviz.anim.settled();
}"""
SET_SLIDER = """async ([sel, v]) => { const el = document.querySelector(sel); el.value = String(v); el.dispatchEvent(new Event("input")); await window.grokviz.anim.settled(); }"""


def ffmpeg_exe() -> str:
    """Path of ffmpeg: imageio-ffmpeg's bundled binary (uv extra `media`), else ffmpeg on PATH."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # optional dependency

        return get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found: `uv sync --extra media` (or install ffmpeg)") from None
        return exe


def run_ffmpeg(*args: str) -> None:
    """Run ffmpeg quietly; raise with its error output if it fails."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", *map(str, args)]
    log.info("ffmpeg %s", " ".join(cmd[4:]))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")


def capture(page, base: str, clip: Clip, frames: Path) -> int:
    """Screenshot every frame of `clip` into `frames/frame_0000.png …`; return the number of frames."""
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    fig = open_figure(page, base, clip.selector, clip.state)  # its first draw has finished, so it cannot reset a slider later
    values = list(range(*clip.steps)) if clip.steps else list(range(*clip.slider[1:]))

    def shoot() -> set[tuple[int, int]]:
        """One pass over the clip; returns the set of frame sizes (PNG IHDR width and height, bytes 16–24)."""
        sizes = set()
        for i, v in enumerate(values):
            page.evaluate(SET_STEP, v) if clip.steps else page.evaluate(SET_SLIDER, [clip.slider[0], v])
            sizes.add(struct.unpack(">II", fig.screenshot(path=str(frames / f"frame_{i:04d}.png"))[16:24]))
        return sizes

    for _ in range(2):  # ffmpeg needs one frame size: a figure whose notes re-wrap is shot again at its tallest
        sizes = shoot()
        if len(sizes) == 1:
            break
        page.evaluate("([sel, h]) => { document.querySelector(sel).style.minHeight = h + 'px'; }", [clip.selector, max(h for _, h in sizes)])
    else:
        raise RuntimeError(f"clip {clip.name}: frame sizes still differ: {sorted(sizes)}")
    log.info("clip %s: %d frames of %d×%d px", clip.name, len(values), *sizes.pop())
    return len(values)


def encode(frames: Path, n: int, name: str, gif_dir: Path, mp4_dir: Path, strip_dir: Path) -> dict[str, Path]:
    """MP4 (H.264, web-friendly), GIF (two-pass palette) and a 5-tile film strip from the numbered frames."""
    for d in (gif_dir, mp4_dir, strip_dir):
        d.mkdir(parents=True, exist_ok=True)
    src = ["-framerate", str(FPS), "-i", str(frames / "frame_%04d.png")]
    out = {"mp4": mp4_dir / f"{name}.mp4", "gif": gif_dir / f"{name}.gif", "strip": strip_dir / f"{name}-strip.png"}
    run_ffmpeg(*src, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", out["mp4"])  # even sizes: yuv420p requires them
    palette = frames / "palette.png"
    scale = f"fps={FPS},scale={GIF_WIDTH}:-1:flags=lanczos"
    run_ffmpeg(*src, "-vf", f"{scale},palettegen=stats_mode=diff", palette)  # pass 1: one palette for the whole clip
    run_ffmpeg(*src, "-i", palette, "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
               out["gif"])  # pass 2: every frame mapped to that palette; only the changed rectangle is re-dithered
    picks = sorted({round(i * (n - 1) / 4) for i in range(5)})  # 5 evenly spaced frames, first and last included
    sel = "+".join(f"eq(n\\,{i})" for i in picks)  # select keeps a frame where the expression is non-zero
    run_ffmpeg(*src, "-vf", f"select='{sel}',scale=360:-1,tile={len(picks)}x1", "-frames:v", "1", out["strip"])
    return out


def render_clip(clip: Clip, site: Path, frames: Path, gif_dir: Path, mp4_dir: Path, strip_dir: Path) -> dict[str, Path]:
    """Record and encode one clip from the built site."""
    server, port = serve(site)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800}, color_scheme="light")
            n = capture(page, f"http://127.0.0.1:{port}", clip, frames)
            browser.close()
    finally:
        server.shutdown()
    return encode(frames, n, clip.name, gif_dir, mp4_dir, strip_dir)


def render_media(ctx: PipelineContext) -> None:
    """Phase entry point: every clip in CLIPS (skipping ones already rendered unless --force)."""
    gif_dir, mp4_dir = REPO_ROOT / "docs" / "images" / "clips", ctx.cfg.reports_dir / "media"
    for clip in CLIPS:
        if (ctx.cfg.reports_dir / "figures" / f"{clip.name}-strip.png").exists() and not ctx.force:  # the strip is written last
            log.info("clip %s exists (use --force to re-render)", clip.name)
            continue
        out = render_clip(clip, ctx.cfg.site_dir, ctx.cfg.data_dir / "media" / clip.name, gif_dir, mp4_dir, ctx.cfg.reports_dir / "figures")
        log.info("clip %s → %s", clip.name, ", ".join(f"{p.name} ({p.stat().st_size / 1e6:.1f} MB)" for p in out.values()))
