# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

def _prepare(x, ws):
    ws = int(ws)
    if ws <= 0:
        raise ValueError("window_size must be positive")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("input must be 1‑D")
    if arr.size < ws:
        return np.empty(0, dtype=float), ws
    return arr, ws

@lru_cache(maxsize=64)
def _exp_weights(ws):
    w = np.exp(np.linspace(-2, 0, ws))
    return w / w.sum()

def adaptive_filter(x, window_size=20):
    a, w = _prepare(x, window_size)
    if a.size == 0:
        return a
    c = np.cumsum(np.insert(a, 0, 0.0))
    return (c[w:] - c[:-w]) / w

def weighted_filter(x, window_size=20):
    a, w = _prepare(x, window_size)
    if a.size == 0:
        return a
    return np.convolve(a, _exp_weights(w), mode="valid")

def ema_filter(x, window_size=20):
    a, w = _prepare(x, window_size)
    if a.size == 0:
        return a
    alpha = 2.0 / (w + 1)
    ema = np.empty_like(a)
    ema[0] = a[0]
    for i in range(1, a.size):
        ema[i] = alpha * a[i] + (1 - alpha) * ema[i - 1]
    return ema[w - 1 :]

def enhanced_filter_with_trend_preservation(x, window_size=20):
    a, w = _prepare(x, window_size)
    if a.size == 0:
        return a
    trend = adaptive_filter(a, w)
    trend_full = np.concatenate((np.full(w - 1, trend[0]), trend))
    detrended = a - trend_full
    return weighted_filter(detrended, w)

def _hybrid_filter(x, window_size=20):
    """Average of weighted and adaptive filters – keeps length unchanged."""
    a, w = _prepare(x, window_size)
    if a.size == 0:
        return a
    weighted = weighted_filter(a, w)
    adaptive = adaptive_filter(a, w)
    # both have identical length (n‑w+1)
    return (weighted + adaptive) / 2.0

def _residual_variance(orig, filt):
    m = min(len(orig), len(filt))
    return float(np.var(orig[:m] - filt[:m])) if m else float("inf")

def _select_best_filter(x, window_size):
    cand = {
        "basic": adaptive_filter(x, window_size),
        "weighted": weighted_filter(x, window_size),
        "ema": ema_filter(x, window_size),
        "enhanced": enhanced_filter_with_trend_preservation(x, window_size),
        "hybrid": _hybrid_filter(x, window_size),
    }
    return cand[min(cand, key=lambda k: _residual_variance(x, cand[k]))]

def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    alg = (algorithm_type or "").lower()
    if alg == "basic":
        return adaptive_filter(input_signal, window_size)
    if alg == "weighted":
        return weighted_filter(input_signal, window_size)
    if alg == "ema":
        return ema_filter(input_signal, window_size)
    if alg == "enhanced":
        # use hybrid for stronger noise reduction while preserving length
        return _hybrid_filter(input_signal, window_size)
    # fallback to auto‑selection of the best filter
    return _select_best_filter(np.asarray(input_signal, dtype=float), window_size)
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
