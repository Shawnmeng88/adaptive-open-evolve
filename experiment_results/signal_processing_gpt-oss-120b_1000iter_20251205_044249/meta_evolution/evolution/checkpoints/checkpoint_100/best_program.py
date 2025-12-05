# EVOLVE-BLOCK-START
import numpy as np

def adaptive_filter(x, window_size=20):
    """
    Simple moving average filter using convolution.
    Returns the filtered signal with length len(x) - window_size + 1.
    """
    x = np.asarray(x, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if x.size < window_size:
        # Not enough data – return empty array for consistency
        return np.array([], dtype=float)
    return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


def weighted_filter(x, window_size=20):
    """
    Exponential weighted moving average.
    Emphasises recent samples while preserving overall trend.
    """
    x = np.asarray(x, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if x.size < window_size:
        return np.array([], dtype=float)
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights /= weights.sum()
    return np.convolve(x, weights, mode="valid")


def ema_filter(x, window_size=20):
    """
    Classic exponential moving average (EMA).
    Output length matches other filters (len(x) - window_size + 1).
    """
    x = np.asarray(x, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if x.size < window_size:
        return np.array([], dtype=float)
    alpha = 2.0 / (window_size + 1)
    ema = np.empty_like(x)
    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
    return ema[window_size - 1 :]


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Detrends the signal using a simple moving average, then applies
    a weighted moving average to the detrended series.
    """
    x = np.asarray(x, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if x.size < window_size:
        return np.array([], dtype=float)
    # Estimate local trend
    trend = adaptive_filter(x, window_size)
    # Pad trend to original length (repeat edge values)
    trend_full = np.concatenate((np.full(window_size - 1, trend[0]), trend))
    detrended = x - trend_full
    return weighted_filter(detrended, window_size)


def _residual_variance(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute variance of the residual (original - filtered).
    Signals are aligned to the shortest length.
    """
    min_len = min(len(original), len(filtered))
    if min_len == 0:
        return np.inf
    residual = original[:min_len] - filtered[:min_len]
    return float(np.var(residual))


def _select_best_filter(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Evaluate all available filters and return the one that yields the lowest
    residual variance (i.e., highest estimated noise reduction).
    """
    candidates = {
        "basic": adaptive_filter(x, window_size),
        "weighted": weighted_filter(x, window_size),
        "ema": ema_filter(x, window_size),
        "enhanced": enhanced_filter_with_trend_preservation(x, window_size),
    }
    # Choose filter with minimal residual variance relative to the raw signal
    best_name = min(candidates, key=lambda k: _residual_variance(x, candidates[k]))
    return candidates[best_name]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Dispatches to the requested filtering algorithm.
    Supported types:
        - "basic": simple moving average (adaptive_filter)
        - "weighted": exponential weighted moving average (weighted_filter)
        - "ema": classic exponential moving average (ema_filter)
        - "enhanced": detrended weighted filter (enhanced_filter_with_trend_preservation)
        - any other value: automatic selection of the best filter based on residual variance.
    """
    if algorithm_type == "basic":
        return adaptive_filter(input_signal, window_size)
    if algorithm_type == "weighted":
        return weighted_filter(input_signal, window_size)
    if algorithm_type == "ema":
        return ema_filter(input_signal, window_size)
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)

    # Fallback: automatic selection
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
