# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def _kernel(w, kind):
    if kind == "gaussian":
        sigma = w / 6.0
        t = np.arange(w) - (w - 1) / 2.0
        k = np.exp(-0.5 * (t / sigma) ** 2)
    elif kind == "hann":
        k = np.hanning(w)
    elif kind in ("exp", "exponential"):
        k = np.exp(np.linspace(-2, 0, w))
    else:  # uniform fallback
        k = np.ones(w)
    return k / k.sum()

def _mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m))

def _auto_kind(x):
    mad = _mad(x)
    if mad < 0.02:
        return "gaussian"
    if mad < 0.07:
        return "hann"
    return "exponential"

def _median_filter(x, w):
    p = w // 2
    a = np.pad(x, (p, p), mode="edge")
    s = a.strides[0]
    windows = np.lib.stride_tricks.as_strided(a, (len(x), w), (s, s))
    return np.median(windows, axis=1)

def _detrend(x, w, method="mean"):
    if method == "median":
        return x - _median_filter(x, w)
    # mean (moving‑average) detrend
    return x - np.convolve(x, np.ones(w) / w, mode="same")

def _valid_slice(y, w):
    off = w - 1
    start = off // 2
    end = -(off - start) if (off - start) else None
    return y[start:end]

def process_signal(sig, w=20, alg="enhanced", full=False):
    if w <= 0 or len(sig) < w:
        raise ValueError("Invalid window size")
    x = np.asarray(sig, dtype=float)

    if alg == "enhanced":
        x = _detrend(x, w, "mean")
        kind = _auto_kind(x)
        y = np.convolve(x, _kernel(w, kind), mode="same")
    elif alg == "robust":
        x = _detrend(x, w, "median")
        kind = _auto_kind(x)
        y = np.convolve(x, _kernel(w, kind), mode="same")
    elif alg == "simple":
        y = np.convolve(x, _kernel(w, "uniform"), mode="same")
    elif alg == "median":
        y = _median_filter(x, w)
    else:  # explicit kernel name
        y = np.convolve(x, _kernel(w, alg), mode="same")

    return y if full else _valid_slice(y, w)
# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Returns:
        Dictionary containing results and metrics
    """
    # Generate test signal
    noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)

    # Process the signal
    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics
    if len(filtered_signal) > 0:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print(f"Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")
