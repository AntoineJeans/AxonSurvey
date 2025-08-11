from collections import Counter
def imbalance_score(sample, potential_values):
    """Calculate imbalance score for dataset sampling."""

    if len(potential_values) == 0 : raise ValueError("imbalance scores potential_values can't be empty")

    counts = Counter(sample)
    total = len(sample)
    freqs = [counts[val] / total if val in counts else 0 for val in potential_values]
    mean_freq = 1 / len(potential_values)
    # Variance from uniform distribution
    variance = sum((f - mean_freq) ** 2 for f in freqs) / len(potential_values)
    # Normalize by max variance (worst case: all in one class)
    max_variance = (1 - mean_freq)**2 * (1 - 1/len(potential_values))
    score = variance / max_variance if max_variance != 0 else 0
    return score  # 0 = perfectly balanced, 1 = maximally imbalanced