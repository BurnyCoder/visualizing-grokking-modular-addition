"""Hero clips recorded from the website (src/grokviz/media.py): size budget of the committed GIFs and an end-to-end clip.

Global context: the README shows short GIFs of the site's animations and the paper film strips made from the same
frames. GIFs live in git, so their size is capped; the end-to-end test records a 4-frame clip from the built site and
checks that the MP4, the GIF and the film strip come out with the right number of frames.
Source: imageio-ffmpeg `count_frames_and_secs` (https://github.com/imageio/imageio-ffmpeg#api), which decodes MP4 and GIF alike.
"""

import pytest  # markers, skip

from grokviz.config import REPO_ROOT, load_config  # paths

CLIPS_DIR = REPO_ROOT / "docs" / "images" / "clips"
SITE = load_config().site_dir


def test_committed_gifs_fit_the_budget():
    """Each README GIF ≤ 4 MB and all of them ≤ 15 MB, so the repository stays light."""
    gifs = sorted(CLIPS_DIR.glob("*.gif"))
    if not gifs:
        pytest.skip("no clips rendered yet")
    sizes = {g.name: g.stat().st_size for g in gifs}
    assert all(s <= 4e6 for s in sizes.values()), sizes
    assert sum(sizes.values()) <= 15e6, sizes


@pytest.mark.slow
@pytest.mark.browser
@pytest.mark.skipif(not (SITE / "data" / "manifest.json").exists(), reason="site not built with data")
def test_a_short_clip_renders(tmp_path):
    """A 4-step clip of the hero gives an MP4 and a GIF with 4 frames each, and a film strip image."""
    from imageio_ffmpeg import count_frames_and_secs

    from grokviz.media import Clip, render_clip

    clip = Clip("test-hero", "#fig-hero-grokking", "run=paper", steps=(1001, 1401, 100))
    out = render_clip(clip, SITE, tmp_path / "frames", gif_dir=tmp_path, mp4_dir=tmp_path, strip_dir=tmp_path)
    assert count_frames_and_secs(str(out["mp4"]))[0] == 4 and count_frames_and_secs(str(out["gif"]))[0] == 4
    assert out["strip"].stat().st_size > 0
