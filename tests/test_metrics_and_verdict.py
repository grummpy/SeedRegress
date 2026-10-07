"""Metrics on synthetic images, and the pass / drift / fail thresholds."""

from __future__ import annotations

from PIL import Image

from seedregress.metrics import abs_diff, heatmap, phash_distance, ssim
from seedregress.suite import Thresholds
from seedregress.verdict import judge


def _bars(horizontal: bool, size: int = 64) -> Image.Image:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            on = (y // 8) % 2 == 0 if horizontal else (x // 8) % 2 == 0
            pixels[x, y] = (230, 210, 245) if on else (18, 10, 36)
    return image


def _paint(size: int = 64) -> Image.Image:
    image = Image.new("RGB", (size, size), (40, 20, 70))
    pixels = image.load()
    for y in range(size):
        pixels[y % size, y] = (240, 80, 160)
        if y + 3 < size:
            pixels[y, y + 3] = (80, 220, 230)
    return image


def test_identical_images_have_zero_distance():
    image = _paint()
    assert phash_distance(image, image.copy()) == 0
    assert ssim(image, image.copy()) > 0.999
    assert float(abs_diff(image, image.copy()).max()) == 0


def test_structured_difference_is_larger_than_a_small_edit():
    base = _paint()
    small = base.copy()
    pixels = small.load()
    for y in range(4):
        for x in range(4):
            pixels[x, y] = (255, 255, 255)
    opposite = _bars(False)
    assert phash_distance(base, small) < phash_distance(_bars(True), opposite)
    assert ssim(base, small) > ssim(_bars(True), opposite)
    assert float(abs_diff(base, small).max()) > 0
    heat = heatmap(base, small)
    assert heat.size == base.size


def test_threshold_boundaries():
    thresholds = Thresholds()
    assert judge(phash=6, ssim_value=0.98, thresholds=thresholds)[0] == "pass"
    assert judge(phash=7, ssim_value=0.99, thresholds=thresholds)[0] == "drift"
    assert judge(phash=16, ssim_value=1.0, thresholds=thresholds)[0] == "fail"
    assert judge(phash=0, ssim_value=0.90, thresholds=thresholds)[0] == "fail"
    assert judge(phash=0, ssim_value=0.95, thresholds=thresholds)[0] == "drift"


def test_lpips_is_ignored_unless_enabled_and_size_change_fails():
    thresholds = Thresholds()
    assert (
        judge(phash=0, ssim_value=1.0, thresholds=thresholds, lpips_value=1.0, lpips_enabled=False)[0]
        == "pass"
    )
    failed = judge(
        phash=0,
        ssim_value=1.0,
        thresholds=thresholds,
        lpips_value=0.5,
        lpips_enabled=True,
    )
    assert failed[0] == "fail"
    mismatch = judge(phash=0, ssim_value=1.0, thresholds=thresholds, size_mismatch=True)
    assert mismatch[0] == "fail"
    assert "output size changed" in mismatch[1]


def test_real_images_land_on_the_expected_side_of_the_thresholds():
    thresholds = Thresholds()
    image = _paint()
    verdict, _reasons = judge(
        phash=phash_distance(image, image),
        ssim_value=ssim(image, image),
        thresholds=thresholds,
    )
    assert verdict == "pass"
    verdict, _reasons = judge(
        phash=phash_distance(_bars(True), _bars(False)),
        ssim_value=ssim(_bars(True), _bars(False)),
        thresholds=thresholds,
    )
    assert verdict == "fail"
