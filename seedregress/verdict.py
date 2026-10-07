"""Turn metric numbers into pass, drift, or fail."""

from __future__ import annotations

from seedregress.suite import Thresholds


def judge(
    *,
    phash: int,
    ssim_value: float,
    thresholds: Thresholds,
    lpips_value: float | None = None,
    lpips_enabled: bool = False,
    size_mismatch: bool = False,
) -> tuple[str, list[str]]:
    """Return (verdict, reasons).

    Pass requires every enabled metric to clear its pass threshold.
    Fail when any enabled metric crosses its fail threshold, or the size changed.
    Anything between those bounds is drift.
    """
    reasons: list[str] = []
    if size_mismatch:
        reasons.append("output size changed")

    failed = size_mismatch
    if phash >= thresholds.phash_fail_min:
        failed = True
        reasons.append(
            f"pHash distance {phash} ≥ fail threshold {thresholds.phash_fail_min}"
        )
    if ssim_value <= thresholds.ssim_fail_max:
        failed = True
        reasons.append(f"SSIM {ssim_value:.4f} ≤ fail threshold {thresholds.ssim_fail_max}")
    if lpips_enabled and lpips_value is not None and lpips_value >= thresholds.lpips_fail_min:
        failed = True
        reasons.append(f"LPIPS {lpips_value:.4f} ≥ fail threshold {thresholds.lpips_fail_min}")
    if failed:
        return "fail", reasons

    phash_ok = phash <= thresholds.phash_pass_max
    ssim_ok = ssim_value >= thresholds.ssim_pass_min
    lpips_ok = True
    if lpips_enabled and lpips_value is not None:
        lpips_ok = lpips_value <= thresholds.lpips_pass_max
    if phash_ok and ssim_ok and lpips_ok:
        reasons.append(
            f"pHash {phash} ≤ {thresholds.phash_pass_max}, "
            f"SSIM {ssim_value:.4f} ≥ {thresholds.ssim_pass_min}"
        )
        if lpips_enabled and lpips_value is not None:
            reasons.append(f"LPIPS {lpips_value:.4f} ≤ {thresholds.lpips_pass_max}")
        return "pass", reasons

    if not phash_ok:
        reasons.append(
            f"pHash distance {phash} is between {thresholds.phash_pass_max} and "
            f"{thresholds.phash_fail_min}"
        )
    if not ssim_ok:
        reasons.append(
            f"SSIM {ssim_value:.4f} is between {thresholds.ssim_fail_max} and "
            f"{thresholds.ssim_pass_min}"
        )
    if lpips_enabled and lpips_value is not None and not lpips_ok:
        reasons.append(
            f"LPIPS {lpips_value:.4f} is between {thresholds.lpips_pass_max} and "
            f"{thresholds.lpips_fail_min}"
        )
    return "drift", reasons
