from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_demo_gif_is_an_animated_gif_asset():
    asset = ROOT / "docs" / "assets" / "demo.gif"
    content = asset.read_bytes()

    assert content.startswith((b"GIF87a", b"GIF89a"))
    assert content.count(b"\x21\xf9\x04") >= 3


def test_demo_frames_keep_the_safety_disclaimer_visible():
    frames = sorted((ROOT / "docs" / "assets").glob("demo-frame-*.svg"))

    assert len(frames) == 3
    assert any("not for diagnosis or treatment" in frame.read_text() for frame in frames)
