"""pHash, SSIM, and a per-pixel diff heatmap. LPIPS is optional and off by default."""

from __future__ import annotations

import numpy
from PIL import Image


def _gray(image: Image.Image, size: tuple[int, int] | None = None) -> numpy.ndarray:
    gray = image.convert("L")
    if size is not None and gray.size != size:
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    return numpy.asarray(gray, dtype=numpy.float64)


def _dct2(block: numpy.ndarray) -> numpy.ndarray:
    """Unnormalized DCT-II on both axes."""
    height, width = block.shape
    row = numpy.arange(height).reshape(-1, 1)
    col = numpy.arange(height).reshape(1, -1)
    basis_h = numpy.cos(numpy.pi * row * (2 * col + 1) / (2 * height))
    row = numpy.arange(width).reshape(-1, 1)
    col = numpy.arange(width).reshape(1, -1)
    basis_w = numpy.cos(numpy.pi * row * (2 * col + 1) / (2 * width))
    return basis_h @ block @ basis_w.T


def phash_bits(image: Image.Image, hash_size: int = 8) -> numpy.ndarray:
    pixels = _gray(image, (hash_size * 4, hash_size * 4))
    low = _dct2(pixels)[:hash_size, :hash_size]
    return low > numpy.median(low)


def phash_distance(left: Image.Image, right: Image.Image) -> int:
    return int(numpy.count_nonzero(phash_bits(left) != phash_bits(right)))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> numpy.ndarray:
    ax = numpy.arange(size, dtype=numpy.float64) - (size // 2)
    kernel = numpy.exp(-(ax**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _filter(image: numpy.ndarray, kernel: numpy.ndarray) -> numpy.ndarray:
    pad = len(kernel) // 2
    padded = numpy.pad(image, ((0, 0), (pad, pad)), mode="reflect")
    width = image.shape[1]
    horizontal = numpy.zeros_like(image)
    for index, weight in enumerate(kernel):
        horizontal += padded[:, index : index + width] * weight
    padded_v = numpy.pad(horizontal, ((pad, pad), (0, 0)), mode="reflect")
    height = image.shape[0]
    vertical = numpy.zeros_like(image)
    for index, weight in enumerate(kernel):
        vertical += padded_v[index : index + height, :] * weight
    return vertical


def ssim(left: Image.Image, right: Image.Image) -> float:
    """Mean SSIM on grayscale. `right` is resized to `left` when the sizes differ."""
    target = left.size
    a = _gray(left) / 255.0
    b = _gray(right, target) / 255.0
    kernel = _gaussian_kernel()
    mu_a = _filter(a, kernel)
    mu_b = _filter(b, kernel)
    sigma_a = numpy.maximum(_filter(a * a, kernel) - mu_a**2, 0)
    sigma_b = numpy.maximum(_filter(b * b, kernel) - mu_b**2, 0)
    sigma_ab = _filter(a * b, kernel) - mu_a * mu_b
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float((numerator / denominator).mean())


def abs_diff(left: Image.Image, right: Image.Image) -> numpy.ndarray:
    a = numpy.asarray(left.convert("RGB"), dtype=numpy.float32)
    other = right.convert("RGB")
    if other.size != left.size:
        other = other.resize(left.size, Image.Resampling.LANCZOS)
    b = numpy.asarray(other, dtype=numpy.float32)
    return numpy.abs(a - b).mean(axis=2)


def heatmap(left: Image.Image, right: Image.Image) -> Image.Image:
    """Colorize per-pixel absolute difference. Identical pixels stay dark purple."""
    delta = abs_diff(left, right)
    amp = numpy.clip(delta * 4.0, 0, 255) / 255.0
    low = numpy.array([20.0, 8.0, 24.0])
    mid = numpy.array([34.0, 211.0, 238.0])
    high = numpy.array([255.0, 45.0, 149.0])
    t = amp[..., None]
    color = numpy.where(t < 0.5, low + (mid - low) * (t * 2), mid + (high - mid) * ((t - 0.5) * 2))
    return Image.fromarray(numpy.clip(color, 0, 255).astype(numpy.uint8), "RGB")


def lpips_available() -> tuple[bool, str]:
    try:
        import lpips  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return (
            False,
            "LPIPS is not installed. Install requirements-lpips.txt and set enable_lpips "
            "on the suite. It is off by default.",
        )
    return True, "LPIPS is installed."


def compute_lpips(left: Image.Image, right: Image.Image) -> float:
    """AlexNet LPIPS. Call only when the suite turned LPIPS on and the extra install exists."""
    import lpips
    import torch

    other = right.convert("RGB")
    if other.size != left.size:
        other = other.resize(left.size, Image.Resampling.LANCZOS)
    loss = lpips.LPIPS(net="alex")

    def tensor(image: Image.Image) -> torch.Tensor:
        array = numpy.asarray(image, dtype=numpy.float32) / 255.0
        array = array * 2.0 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        value = loss(tensor(left.convert("RGB")), tensor(other))
    return float(value.item())
